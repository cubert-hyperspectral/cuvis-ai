"""RT-DETR object detection on CU3S hyperspectral data.

Outputs:
- detection_results.json   COCO detection format (bboxes, scores, category_ids)
- detection_overlay.mp4    Range-averaged false RGB frames with bounding box overlays
- pipeline/RTDETR_Detection.{png,md}  Pipeline visualisations
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger

from cuvis_ai.node.json_writer import DetectionCocoJsonNode

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
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input CU3S hyperspectral file.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--model-id",
    type=str,
    default="PekingU/rtdetr_r50vd",
    show_default=True,
    help="HuggingFace model ID or local path for RT-DETR.",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=0.5,
    show_default=True,
    help="Minimum score to keep a detection; must be in [0, 1].",
)
@click.option(
    "--allowed-category-ids",
    type=str,
    default="0",
    show_default=True,
    help="Comma-separated COCO category IDs to retain (e.g. '0' = person).",
)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("plugins.yaml"),
    show_default=True,
    help="Plugin manifest path (resolved relative to this script if not absolute).",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./detection_output"),
    show_default=True,
)
@click.option(
    "--name-suffix",
    type=str,
    default="",
    show_default=True,
    help="Optional suffix appended to output-dir name to keep multiple test runs.",
)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop after this many frames (exclusive). -1 means all frames.",
)
@click.option("--bf16", is_flag=True, default=False, help="Enable bfloat16 autocast.")
@click.option(
    "--compile", "compile_model", is_flag=True, default=False, help="Enable torch.compile."
)
def main(
    cu3s_path: Path,
    processing_mode: str,
    frame_rotation: int | None,
    model_id: str,
    confidence_threshold: float,
    allowed_category_ids: str,
    plugins_yaml: Path,
    output_dir: Path,
    name_suffix: str,
    end_frame: int,
    bf16: bool,
    compile_model: bool,
) -> None:
    """Run RT-DETR detection on CU3S hyperspectral data via a cuvis-ai pipeline."""
    # --- Validation ---
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if not (0.0 <= confidence_threshold <= 1.0):
        raise click.BadParameter("--confidence-threshold must be in [0, 1].")

    try:
        allowed_ids = [int(x.strip()) for x in allowed_category_ids.split(",") if x.strip()]
    except ValueError as exc:
        raise click.BadParameter(
            "--allowed-category-ids must be a comma-separated list of integers."
        ) from exc

    resolved_mode = _resolve_processing_mode(processing_mode)

    suffix = name_suffix.strip()
    if suffix:
        safe_suffix = suffix.replace("\\", "_").replace("/", "_").replace(" ", "_")
        if not safe_suffix.startswith("_"):
            safe_suffix = f"_{safe_suffix}"
        output_dir = output_dir.parent / f"{output_dir.name}{safe_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("CU3S: {}", cu3s_path)
    logger.info("Processing mode: {}", resolved_mode)
    logger.info("False RGB: range-averaged (R:580-650, G:500-580, B:420-500 nm)")
    logger.info("Model: {}", model_id)
    logger.info("Confidence threshold: {}", confidence_threshold)
    logger.info("Allowed category IDs: {}", allowed_ids)
    logger.info("Output: {}", output_dir)
    logger.info("Device: {}", device)

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode
    from cuvis_ai.node.channel_selector import RangeAverageFalseRGBSelector
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.video import ToVideoNode

    # --- Plugin loading ---
    plugin_manifest = plugins_yaml
    if not plugin_manifest.is_absolute():
        plugin_manifest = Path(__file__).parent / plugin_manifest
    plugin_manifest = plugin_manifest.resolve()
    if not plugin_manifest.exists():
        raise click.ClickException(f"Plugins manifest not found: {plugin_manifest}")
    if not plugin_manifest.is_file():
        raise click.ClickException(f"Plugins manifest is not a file: {plugin_manifest}")

    registry = NodeRegistry()
    registry.load_plugins(str(plugin_manifest))
    rtdetr_detection_cls = registry.get("cuvis_ai_detr.node.RTDETRDetection")
    rtdetr_postprocess_cls = registry.get("cuvis_ai_detr.node.RTDETRPostprocess")
    category_filter_cls = registry.get("cuvis_ai_detr.node.CategoryFilterNode")
    logger.info("RT-DETR plugin loaded from {}", plugin_manifest)

    # --- DataModule ---
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
        logger.warning("Invalid FPS from dataset; falling back to 10.0.")

    logger.info("Frames to process: {} (FPS {:.1f})", target_frames, dataset_fps)

    # --- Build pipeline ---
    pipeline = CuvisPipeline("RTDETR_Detection")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = RangeAverageFalseRGBSelector(name="range_average_false_rgb")
    detection_node = rtdetr_detection_cls(
        model_id=model_id,
        name="rtdetr_detection",
    )
    if compile_model:
        detection_node.model = torch.compile(detection_node.model)  # type: ignore[assignment]
        logger.info("torch.compile applied to detection model")
    postprocess_node = rtdetr_postprocess_cls(
        confidence_threshold=confidence_threshold,
        name="rtdetr_postprocess",
    )
    category_filter = category_filter_cls(
        allowed_category_ids=allowed_ids,
        name="category_filter",
    )
    json_sink = DetectionCocoJsonNode(
        output_json_path=str(output_dir / "detection_results.json"),
        name="detection_coco_json",
    )
    overlay_node = BBoxesOverlayNode(name="bboxes_overlay")
    overlay_path = output_dir / "detection_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.inputs.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.inputs.wavelengths),
        (false_rgb.outputs.rgb_image, detection_node.inputs.rgb_image),
        (detection_node.outputs.raw_logits, postprocess_node.inputs.raw_logits),
        (detection_node.outputs.raw_boxes, postprocess_node.inputs.raw_boxes),
        (detection_node.outputs.orig_hw, postprocess_node.inputs.orig_hw),
        (postprocess_node.outputs.bboxes, category_filter.inputs.bboxes),
        (postprocess_node.outputs.category_ids, category_filter.inputs.category_ids),
        (postprocess_node.outputs.confidences, category_filter.inputs.confidences),
        (cu3s_data.outputs.frame_id, json_sink.inputs.frame_id),
        (category_filter.outputs.bboxes, json_sink.inputs.bboxes),
        (category_filter.outputs.category_ids, json_sink.inputs.category_ids),
        (category_filter.outputs.confidences, json_sink.inputs.confidences),
        (detection_node.outputs.orig_hw, json_sink.inputs.orig_hw),
        (false_rgb.outputs.rgb_image, overlay_node.inputs.rgb_image),
        (category_filter.outputs.bboxes, overlay_node.inputs.bboxes),
        (category_filter.outputs.category_ids, overlay_node.inputs.category_ids),
        (category_filter.outputs.confidences, overlay_node.inputs.confidences),
        (overlay_node.outputs.rgb_with_overlay, to_video.inputs.rgb_image),
    )

    # Pipeline visualisation
    pipeline_viz_dir = output_dir / "pipeline"
    pipeline_png_path = pipeline_viz_dir / f"{pipeline.name}.png"
    pipeline_md_path = pipeline_viz_dir / f"{pipeline.name}.md"
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(pipeline_png_path),
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path=str(pipeline_md_path),
        show_execution_stage=True,
    )

    pipeline.to(device)

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )

    logger.info("Starting detection...")
    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    json_out = output_dir / "detection_results.json"
    if not json_out.exists():
        raise RuntimeError(f"Expected detection JSON was not created: {json_out}")
    if not overlay_path.exists():
        raise RuntimeError(f"Expected overlay video was not created: {overlay_path}")

    logger.success("Detection complete")
    logger.info("Results: {}", json_out)
    logger.info("Overlay: {}", overlay_path)
    logger.info("Pipeline PNG: {}", pipeline_png_path)
    logger.info("Pipeline MD: {}", pipeline_md_path)


if __name__ == "__main__":
    main()
