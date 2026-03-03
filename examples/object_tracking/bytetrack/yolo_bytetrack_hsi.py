"""YOLO26 + ByteTrack HSI tracking pipeline."""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option("--processing-mode", type=str, default="SpectralRadiance", show_default=True)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--end-frame", type=int, default=-1, show_default=True)
@click.option("--model-name", type=str, default="yolo26n.pt", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.5, show_default=True)
@click.option("--track-thresh", type=float, default=0.5, show_default=True)
@click.option("--track-buffer", type=int, default=30, show_default=True)
@click.option("--match-thresh", type=float, default=0.8, show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./tracking_output"),
    show_default=True,
)
@click.option(
    "--plugins-dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    show_default=True,
    help="Directory containing ultralytics.yaml and bytetrack.yaml",
)
@click.option("--bf16", is_flag=True, default=False)
def main(
    cu3s_path: Path,
    processing_mode: str,
    frame_rotation: int | None,
    end_frame: int,
    model_name: str,
    confidence_threshold: float,
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    output_dir: Path,
    plugins_dir: Path | None,
    bf16: bool,
) -> None:
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir.mkdir(parents=True, exist_ok=True)

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode
    from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.video import ToVideoNode

    predict_ids = list(range(end_frame)) if end_frame > 0 else None
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=processing_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")
    target_frames = len(datamodule.predict_ds)
    dataset_fps = float(getattr(datamodule.predict_ds, "fps", None) or 10.0)

    _repo_root = Path(__file__).parents[3]
    plugins_dir_resolved = (plugins_dir or (_repo_root / "configs" / "plugins")).resolve()

    registry = NodeRegistry()
    for yaml_name in ("ultralytics.yaml", "bytetrack.yaml"):
        registry.load_plugins(str(plugins_dir_resolved / yaml_name))
        logger.info("Plugin manifest loaded: {}", plugins_dir_resolved / yaml_name)

    yolo_cls = registry.get("cuvis_ai_ultralytics.node.YOLO26Detection")
    yolo_post_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPostprocess")
    bytetrack_cls = registry.get("cuvis_ai_bytetrack.node.ByteTrack")

    pipeline = CuvisPipeline("YOLO_ByteTrack_HSI")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = CIETristimulusFalseRGBSelector(name="cie_false_rgb")
    yolo_det = yolo_cls(model_path=model_name, name="yolo26_det")
    yolo_post = yolo_post_cls(confidence_threshold=confidence_threshold, name="yolo_post")
    tracker = bytetrack_cls(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        frame_rate=int(dataset_fps),
        name="bytetrack",
    )
    bbox_overlay = BBoxesOverlayNode(name="bboxes_overlay")
    overlay_path = output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.inputs.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.inputs.wavelengths),
        (false_rgb.outputs.rgb_image, yolo_det.inputs.rgb_image),
        (yolo_det.outputs.raw_preds, yolo_post.inputs.raw_preds),
        (yolo_det.outputs.model_input_hw, yolo_post.inputs.model_input_hw),
        (yolo_det.outputs.orig_hw, yolo_post.inputs.orig_hw),
        (yolo_post.outputs.bboxes, tracker.inputs.bboxes),
        (yolo_post.outputs.category_ids, tracker.inputs.category_ids),
        (yolo_post.outputs.confidences, tracker.inputs.confidences),
        (false_rgb.outputs.rgb_image, bbox_overlay.inputs.rgb_image),
        (tracker.outputs.bboxes, bbox_overlay.inputs.bboxes),
        (tracker.outputs.track_ids, bbox_overlay.inputs.category_ids),
        (tracker.outputs.confidences, bbox_overlay.inputs.confidences),
        (bbox_overlay.outputs.rgb_with_overlay, to_video.inputs.rgb_image),
    )

    pipeline.to(device)
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )

    logger.info("Starting tracking ({} frames) on {}", target_frames, device)
    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    logger.success("Tracking complete; overlay saved to {}", overlay_path)


if __name__ == "__main__":
    main()
