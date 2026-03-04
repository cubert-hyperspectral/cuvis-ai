"""YOLO26 HSI detection: CU3S -> CIE false RGB -> YOLO26 -> bbox overlay -> MP4."""

from __future__ import annotations

from pathlib import Path

import click
import torch
from loguru import logger

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


def _resolve_processing_mode(processing_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


@click.command()
@click.option(
    "--cu3s-path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--output-video-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("./tracking_output/yolo_overlay.mp4"),
    show_default=True,
)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("../../configs/plugins/ultralytics.yaml"),
    show_default=True,
)
@click.option("--model-path", type=str, default="yolo26n.pt", show_default=True)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop after this many frames (exclusive). -1 means all frames.",
)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--confidence-threshold", type=float, default=0.5, show_default=True)
@click.option("--iou-threshold", type=float, default=0.7, show_default=True)
@click.option("--max-detections", type=int, default=300, show_default=True)
@click.option("--line-thickness", type=int, default=2, show_default=True)
@click.option("--half-precision", is_flag=True, default=False, help="Enable FP16 model input.")
def main(
    cu3s_path: Path,
    output_video_path: Path,
    plugins_yaml: Path,
    model_path: str,
    processing_mode: str,
    end_frame: int,
    frame_rotation: int | None,
    confidence_threshold: float,
    iou_threshold: float,
    max_detections: int,
    line_thickness: int,
    half_precision: bool,
) -> None:
    """Run YOLO26 detection on CIE false-RGB frames and render bbox overlays to MP4."""
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise click.BadParameter("--confidence-threshold must be in [0, 1].")
    if not (0.0 <= iou_threshold <= 1.0):
        raise click.BadParameter("--iou-threshold must be in [0, 1].")
    if max_detections <= 0:
        raise click.BadParameter("--max-detections must be > 0.")
    if line_thickness <= 0:
        raise click.BadParameter("--line-thickness must be > 0.")

    resolved_mode = _resolve_processing_mode(processing_mode)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

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
        processing_mode=resolved_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")

    if datamodule.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise click.ClickException("No frames available for prediction.")

    dataset_fps = float(getattr(datamodule.predict_ds, "fps", None) or 10.0)
    if dataset_fps <= 0:
        dataset_fps = 10.0

    plugin_manifest = plugins_yaml
    if not plugin_manifest.is_absolute():
        plugin_manifest = (Path(__file__).parent / plugin_manifest).resolve()
    if not plugin_manifest.exists():
        raise click.ClickException(f"Plugins manifest not found: {plugin_manifest}")
    if not plugin_manifest.is_file():
        raise click.ClickException(f"Plugins manifest is not a file: {plugin_manifest}")

    registry = NodeRegistry()
    registry.load_plugins(str(plugin_manifest))
    YOLO26Detection = NodeRegistry.get("cuvis_ai_ultralytics.node.YOLO26Detection")
    YOLOPreprocess = NodeRegistry.get("cuvis_ai_ultralytics.node.YOLOPreprocess")
    YOLOPostprocess = NodeRegistry.get("cuvis_ai_ultralytics.node.YOLOPostprocess")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if half_precision and device != "cuda":
        logger.warning("half-precision requested on non-CUDA device; disabling FP16.")
        half_precision = False

    logger.info("CU3S: {}", cu3s_path)
    logger.info("Processing mode: {}", resolved_mode)
    logger.info("Plugin manifest: {}", plugin_manifest)
    logger.info("Frames: {} (FPS {:.1f})", target_frames, dataset_fps)
    logger.info("YOLO model: {} | device={}", model_path, device)

    pipeline = CuvisPipeline("YOLO26_HSI_Detection")
    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = CIETristimulusFalseRGBSelector(name="cie_false_rgb")
    yolo_det = YOLO26Detection(
        model_path=model_path,
        half_precision=half_precision,
        name="yolo26_det",
    )
    yolo_pre = YOLOPreprocess(stride=yolo_det.stride, name="yolo_pre")
    yolo_post = YOLOPostprocess(
        confidence_threshold=float(confidence_threshold),
        iou_threshold=float(iou_threshold),
        max_detections=int(max_detections),
        name="yolo_post",
    )
    bbox_overlay = BBoxesOverlayNode(line_thickness=int(line_thickness), name="bboxes_overlay")
    to_video = ToVideoNode(
        output_video_path=str(output_video_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, yolo_pre.rgb_image),
        (yolo_pre.preprocessed, yolo_det.preprocessed),
        (yolo_det.raw_preds, yolo_post.raw_preds),
        (yolo_pre.model_input_hw, yolo_post.model_input_hw),
        (yolo_pre.orig_hw, yolo_post.orig_hw),
        (false_rgb.rgb_image, bbox_overlay.rgb_image),
        (yolo_post.bboxes, bbox_overlay.bboxes),
        (yolo_post.category_ids, bbox_overlay.category_ids),
        (yolo_post.confidences, bbox_overlay.confidences),
        (bbox_overlay.rgb_with_overlay, to_video.rgb_image),
    )

    pipeline_viz_dir = output_video_path.parent / "pipeline"
    pipeline_viz_dir.mkdir(parents=True, exist_ok=True)
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(pipeline_viz_dir / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path=str(pipeline_viz_dir / f"{pipeline.name}.md"),
        show_execution_stage=True,
    )

    pipeline.to(device)
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=target_frames, collect_outputs=False)

    if not output_video_path.exists():
        raise RuntimeError(f"Expected output video was not created: {output_video_path}")

    logger.success("YOLO26 detection complete")
    logger.info("Overlay video: {}", output_video_path)


if __name__ == "__main__":
    main()
