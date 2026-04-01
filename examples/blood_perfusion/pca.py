"""Monochrome PCA blood-perfusion video export for CU3S sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor, StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.dimensionality_reduction import PCA, TrainablePCA
from cuvis_ai.node.metrics import ExplainedVarianceMetric
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.utils.cli_helpers import resolve_run_output_dir

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")
PCA_MODES = ("per_frame", "global")
DEFAULT_CU3S_PATH = Path("data/XMR_Blood_Perfusion/Blood_Perfusion_Refl.cu3s")
DEFAULT_OUTPUT_DIR = Path("tracking_output/pca_blood_perfusion")
MONOCHROME_COMPONENTS = 1


class MonochromeToRGBNode(Node):
    """Repeat one normalized monochrome channel into RGB-compatible output."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Monochrome frames [B, H, W, 1] in [0, 1].",
        )
    }
    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB-compatible monochrome frames [B, H, W, 3] in [0, 1].",
        )
    }

    def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Repeat the single monochrome component across all RGB channels."""
        if data.ndim != 4 or data.shape[-1] != MONOCHROME_COMPONENTS:
            raise ValueError(
                f"Expected monochrome data with shape [B, H, W, 1], got {tuple(data.shape)}"
            )
        return {"rgb_image": data.repeat(1, 1, 1, 3)}


def _resolve_processing_mode(processing_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


def _resolve_pca_mode(pca_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PCA_MODES}
    resolved = lookup.get(pca_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid pca_mode '{pca_mode}'. Supported: {', '.join(PCA_MODES)}"
        )
    return resolved


def _validate_frame_window(start_frame: int, end_frame: int) -> None:
    if start_frame < 0:
        raise click.BadParameter("--start-frame must be zero or positive.")
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if end_frame != -1 and end_frame <= start_frame:
        raise click.BadParameter("--end-frame must be greater than --start-frame.")


def _validate_init_frames(init_frames: int | None) -> int | None:
    if init_frames is None:
        return None
    if init_frames <= 0:
        raise click.BadParameter("--init-frames must be a positive integer.")
    return int(init_frames)


def _probe_dataset_window(
    cu3s_file_path: str,
    processing_mode: str,
    *,
    start_frame: int,
    end_frame: int,
) -> tuple[list[int], list[int] | None, int, float, int]:
    probe_dm = SingleCu3sDataModule(
        cu3s_file_path=cu3s_file_path,
        processing_mode=processing_mode,
        batch_size=1,
    )
    probe_dm.setup(stage="predict")
    if probe_dm.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    total_available = len(probe_dm.predict_ds)
    if total_available <= 0:
        raise click.ClickException("No frames available for prediction.")
    if start_frame >= total_available:
        raise click.BadParameter(
            f"--start-frame {start_frame} is outside the available range [0, {total_available - 1}]"
        )

    effective_end = min(end_frame, total_available) if end_frame > 0 else total_available
    if effective_end <= start_frame:
        raise click.BadParameter("--end-frame must be greater than --start-frame.")

    selected_first = probe_dm.predict_ds[start_frame]
    input_channels = int(selected_first["cube"].shape[-1])

    dataset_fps = float(getattr(probe_dm.predict_ds, "fps", None) or 10.0)
    if dataset_fps <= 0:
        dataset_fps = 10.0
        logger.warning("Could not infer positive FPS from dataset; using fallback 10.0")

    selected_ids = list(range(start_frame, effective_end))
    predict_ids = None if start_frame == 0 and effective_end == total_available else selected_ids
    target_frames = effective_end - start_frame
    return selected_ids, predict_ids, target_frames, dataset_fps, input_channels


def _resolve_train_ids(
    selected_ids: list[int],
    *,
    pca_mode: str,
    init_frames: int | None,
) -> list[int]:
    if pca_mode != "global" or init_frames is None:
        return selected_ids

    resolved_count = min(init_frames, len(selected_ids))
    if resolved_count < init_frames:
        logger.warning(
            "Requested init_frames={} but only {} frames are available in the selected window; "
            "using all selected frames for initialization.",
            init_frames,
            len(selected_ids),
        )
    return selected_ids[:resolved_count]


def _save_pipeline_artifacts(
    pipeline: CuvisPipeline,
    run_output_dir: Path,
    *,
    save_pipeline_config: bool,
) -> None:
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(run_output_dir / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path=str(run_output_dir / f"{pipeline.name}.md"),
        direction="LR",
        include_node_class=True,
        wrap_markdown=True,
        show_execution_stage=True,
    )
    if save_pipeline_config:
        pipeline_path = run_output_dir / f"{pipeline.name}.yaml"
        pipeline.save_to_file(str(pipeline_path))
        logger.info("Pipeline config saved: {}", pipeline_path)


def export_pca_video(
    *,
    cu3s_file_path: str,
    output_video_path: str,
    pca_mode: str = "per_frame",
    processing_mode: str = "Reflectance",
    start_frame: int = 0,
    end_frame: int = -1,
    init_frames: int | None = None,
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    save_pipeline_config: bool = False,
) -> Path:
    """Run CU3S -> PCA -> normalized monochrome visualization -> MP4 export."""
    output_path = Path(output_video_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".log")
    log_sink_id = logger.add(log_path, level="INFO", mode="w")

    try:
        logger.info("Writing monochrome PCA logs to {}", log_path)

        resolved_mode = _resolve_processing_mode(processing_mode)
        resolved_pca_mode = _resolve_pca_mode(pca_mode)
        _validate_frame_window(start_frame, end_frame)
        resolved_init_frames = _validate_init_frames(init_frames)

        selected_ids, predict_ids, target_frames, dataset_fps, input_channels = (
            _probe_dataset_window(
                cu3s_file_path=cu3s_file_path,
                processing_mode=resolved_mode,
                start_frame=start_frame,
                end_frame=end_frame,
            )
        )
        train_ids = _resolve_train_ids(
            selected_ids,
            pca_mode=resolved_pca_mode,
            init_frames=resolved_init_frames,
        )

        if frame_rate is not None and frame_rate <= 0:
            raise click.BadParameter("--frame-rate must be > 0.")
        resolved_frame_rate = float(frame_rate) if frame_rate is not None else float(dataset_fps)

        datamodule = SingleCu3sDataModule(
            cu3s_file_path=cu3s_file_path,
            processing_mode=resolved_mode,
            batch_size=1,
            train_ids=train_ids,
            predict_ids=predict_ids,
        )

        pipeline_name = (
            "PCA_PerFrame_Projection"
            if resolved_pca_mode == "per_frame"
            else "PCA_Global_Projection"
        )
        pipeline = CuvisPipeline(pipeline_name)

        cu3s_data = CU3SDataNode(name="cu3s_data")
        if resolved_pca_mode == "per_frame":
            pca_node: Node = PCA(n_components=MONOCHROME_COMPONENTS, name="pca_per_frame")
        else:
            pca_node = TrainablePCA(
                num_channels=input_channels,
                n_components=MONOCHROME_COMPONENTS,
                whiten=False,
                init_method="svd",
                name="pca_global",
            )

        explained_variance = ExplainedVarianceMetric(
            execution_stages={ExecutionStage.ALWAYS},
            name="explained_variance_metric",
        )
        projection_normalizer = MinMaxNormalizer(
            eps=1.0e-6,
            use_running_stats=False,
            name="projection_normalizer",
        )
        monochrome_to_rgb = MonochromeToRGBNode(name="monochrome_to_rgb")
        to_video = ToVideoNode(
            output_video_path=str(output_path),
            frame_rate=resolved_frame_rate,
            frame_rotation=frame_rotation,
            name="to_video",
        )

        pipeline.connect(
            (cu3s_data.outputs.cube, pca_node.data),
            (
                pca_node.outputs.explained_variance_ratio,
                explained_variance.explained_variance_ratio,
            ),
            (pca_node.outputs.projected, projection_normalizer.data),
            (projection_normalizer.normalized, monochrome_to_rgb.data),
            (monochrome_to_rgb.rgb_image, to_video.rgb_image),
            (cu3s_data.outputs.mesu_index, to_video.frame_id),
        )

        run_output_dir = output_path.parent
        _save_pipeline_artifacts(
            pipeline,
            run_output_dir,
            save_pipeline_config=save_pipeline_config,
        )

        device = torch.device("cpu")
        pipeline.to(device)

        logger.info(
            "Starting monochrome PCA export: file={} mode={} processing_mode={} frames={} "
            "init_frames={} fps={} device={}",
            cu3s_file_path,
            resolved_pca_mode,
            resolved_mode,
            target_frames,
            len(train_ids),
            resolved_frame_rate,
            device,
        )

        stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
        stat_trainer.fit()

        predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
        predictor.predict(max_batches=target_frames, collect_outputs=False)

        if not output_path.exists():
            raise RuntimeError(f"Expected output video was not created: {output_path}")

        logger.success("Monochrome PCA export complete: {}", output_path)
        logger.info("Log file: {}", log_path)
        return output_path
    finally:
        logger.remove(log_sink_id)


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_CU3S_PATH,
    show_default=True,
    help="Path to the blood-perfusion CU3S session.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help=(
        "Parent output directory. Final run folder is "
        "<output-dir>/<out-basename or input-file-stem>."
    ),
)
@click.option(
    "--out-basename",
    type=str,
    default=None,
    help="Optional leaf run-folder name under --output-dir (must not include '/' or '\\').",
)
@click.option(
    "--pca-mode",
    type=click.Choice(PCA_MODES, case_sensitive=False),
    default="per_frame",
    show_default=True,
    help="Use independent per-frame PCA or one global PCA fit across the selected frame window.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="Reflectance",
    show_default=True,
)
@click.option("--start-frame", type=int, default=0, show_default=True)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop at this source frame index (exclusive). -1 means all frames.",
)
@click.option(
    "--init-frames",
    type=int,
    default=None,
    help=(
        "For global PCA, statistically initialize on the first N frames of the selected "
        "window, then predict over the full selected window. Defaults to the full window."
    ),
)
@click.option(
    "--frame-rate",
    type=float,
    default=None,
    help="Optional FPS override. Defaults to dataset FPS or 10.0 when missing.",
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--save-pipeline-config/--no-save-pipeline-config",
    default=False,
    show_default=True,
    help="Save pipeline YAML/PT alongside the video output.",
)
def main(
    cu3s_path: Path,
    output_dir: Path,
    out_basename: str | None,
    pca_mode: str,
    processing_mode: str,
    start_frame: int,
    end_frame: int,
    init_frames: int | None,
    frame_rate: float | None,
    frame_rotation: int | None,
    save_pipeline_config: bool,
) -> None:
    """Export a monochrome PCA visualization video for the blood-perfusion CU3S session."""
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=cu3s_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = run_output_dir / "pca_projection.mp4"

    export_pca_video(
        cu3s_file_path=str(cu3s_path),
        output_video_path=str(output_video_path),
        pca_mode=pca_mode,
        processing_mode=processing_mode,
        start_frame=start_frame,
        end_frame=end_frame,
        init_frames=init_frames,
        frame_rate=frame_rate,
        frame_rotation=frame_rotation,
        save_pipeline_config=save_pipeline_config,
    )


if __name__ == "__main__":
    main()
