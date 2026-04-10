"""Evaluate COCO tracking JSON pairs with TrackEval metric nodes via CuvisPipeline."""

from __future__ import annotations

from pathlib import Path

import click
import pytorch_lightning as pl
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from cuvis_ai_core.utils.node_registry import NodeRegistry
from torch.utils.data import DataLoader, TensorDataset

from cuvis_ai.node.json_file import TrackingResultsReader


class _FrameCountDataModule(pl.LightningDataModule):
    """Yield N empty batches so source reader nodes can advance one frame per call."""

    def __init__(self, n_frames: int) -> None:
        super().__init__()
        self._n_frames = int(n_frames)

    def setup(self, stage: str = "predict") -> None:  # noqa: ARG002
        return None

    def predict_dataloader(self) -> DataLoader:
        dataset = TensorDataset(torch.zeros(self._n_frames))
        return DataLoader(dataset, batch_size=1, collate_fn=lambda _: {})


@click.command()
@click.option(
    "--gt",
    "gt_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Ground-truth tracking JSON (COCO bbox format).",
)
@click.option(
    "--pred",
    "pred_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Prediction tracking JSON (COCO bbox format).",
)
@click.option(
    "--match-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="IoU threshold for CLEAR and Identity metrics.",
)
@click.option(
    "--plugins-manifest",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=Path("configs/plugins/trackeval.yaml"),
    show_default=True,
    help="TrackEval plugin manifest path.",
)
def main(
    gt_path: Path,
    pred_path: Path,
    match_threshold: float,
    plugins_manifest: Path,
) -> None:
    """Run TrackEval metric nodes on a pair of tracking JSON files."""
    if not (0.0 <= match_threshold <= 1.0):
        raise click.BadParameter("--match-threshold must be in [0.0, 1.0].")

    registry = NodeRegistry()
    registry.load_plugins(str(plugins_manifest))

    hota_cls = registry.get("cuvis_ai_trackeval.node.HOTAMetricNode")
    clear_cls = registry.get("cuvis_ai_trackeval.node.CLEARMetricNode")
    identity_cls = registry.get("cuvis_ai_trackeval.node.IdentityMetricNode")

    gt_reader = TrackingResultsReader(json_path=str(gt_path), name="gt_reader")
    pred_reader = TrackingResultsReader(json_path=str(pred_path), name="pred_reader")

    n_frames = min(gt_reader.num_frames, pred_reader.num_frames)
    if n_frames <= 0:
        raise click.ClickException("No frames available for evaluation.")
    if gt_reader.num_frames != pred_reader.num_frames:
        click.echo(
            f"Warning: GT frames={gt_reader.num_frames}, pred frames={pred_reader.num_frames}; "
            f"evaluating first {n_frames} aligned frames.",
            err=True,
        )

    hota_node = hota_cls(iou_threshold=match_threshold, name="hota")
    clear_node = clear_cls(match_threshold=match_threshold, name="clear")
    identity_node = identity_cls(match_threshold=match_threshold, name="identity")

    pipeline = CuvisPipeline("TrackEval", strict_runtime_io_validation=False)
    connections = [
        # HOTA
        (gt_reader.outputs.frame_id, hota_node.frame_id),
        (gt_reader.outputs.bboxes, hota_node.gt_bboxes),
        (gt_reader.outputs.track_ids, hota_node.gt_track_ids),
        (pred_reader.outputs.bboxes, hota_node.pred_bboxes),
        (pred_reader.outputs.track_ids, hota_node.pred_track_ids),
        (pred_reader.outputs.confidences, hota_node.pred_scores),
        # CLEAR
        (gt_reader.outputs.frame_id, clear_node.frame_id),
        (gt_reader.outputs.bboxes, clear_node.gt_bboxes),
        (gt_reader.outputs.track_ids, clear_node.gt_track_ids),
        (pred_reader.outputs.bboxes, clear_node.pred_bboxes),
        (pred_reader.outputs.track_ids, clear_node.pred_track_ids),
        # Identity
        (gt_reader.outputs.frame_id, identity_node.frame_id),
        (gt_reader.outputs.bboxes, identity_node.gt_bboxes),
        (gt_reader.outputs.track_ids, identity_node.gt_track_ids),
        (pred_reader.outputs.bboxes, identity_node.pred_bboxes),
        (pred_reader.outputs.track_ids, identity_node.pred_track_ids),
    ]

    if hasattr(hota_node, "pred_frame_id"):
        connections.append((pred_reader.outputs.frame_id, hota_node.pred_frame_id))
    if hasattr(clear_node, "pred_frame_id"):
        connections.append((pred_reader.outputs.frame_id, clear_node.pred_frame_id))
    if hasattr(identity_node, "pred_frame_id"):
        connections.append((pred_reader.outputs.frame_id, identity_node.pred_frame_id))

    pipeline.connect(*connections)

    predictor = Predictor(
        pipeline=pipeline,
        datamodule=_FrameCountDataModule(n_frames),
    )
    predictor.predict(max_batches=n_frames, collect_outputs=False)

    hota_results = hota_node.finalize()
    clear_results = clear_node.finalize()
    identity_results = identity_node.finalize()

    click.echo("TrackEval node results:")
    click.echo(f"  HOTA: {hota_results['hota'].item():.6f}")
    click.echo(f"  DetA: {hota_results['deta'].item():.6f}")
    click.echo(f"  AssA: {hota_results['assa'].item():.6f}")
    click.echo(f"  LocA: {hota_results['loca'].item():.6f}")
    click.echo(f"  MOTA: {clear_results['mota'].item():.6f}")
    click.echo(f"  MOTP: {clear_results['motp'].item():.6f}")
    click.echo(f"  FP: {int(clear_results['fp'].item())}")
    click.echo(f"  FN: {int(clear_results['fn'].item())}")
    click.echo(f"  IDSW: {int(clear_results['idsw'].item())}")
    click.echo(f"  IDF1: {identity_results['idf1'].item():.6f}")
    click.echo(f"  IDP: {identity_results['idp'].item():.6f}")
    click.echo(f"  IDR: {identity_results['idr'].item():.6f}")


if __name__ == "__main__":
    main()
