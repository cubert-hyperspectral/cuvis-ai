"""SAM3 HSI object tracking: CU3S -> CIE false RGB -> pipeline-based SAM3 inference.

Outputs:
- tracking_results.json   COCO-format annotations with RLE masks, bboxes, scores
- tracking_overlay.mp4    CIE false RGB frames with coloured mask overlays
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger

from cuvis_ai.node.json_writer import TrackingCocoJsonNode

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
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Stop after this many frames (exclusive). -1 means all frames.",
)
@click.option("--prompt", type=str, default="person", show_default=True)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("./tracking_output"),
    show_default=True,
)
@click.option(
    "--name-suffix",
    type=str,
    default="",
    show_default=True,
    help="Optional suffix appended to output-dir name to keep multiple test runs.",
)
@click.option("--checkpoint-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("plugins.yaml"),
    show_default=True,
)
@click.option("--bf16", is_flag=True, default=False, help="Enable bfloat16 autocast.")
@click.option(
    "--compile", "compile_model", is_flag=True, default=False, help="Enable torch.compile."
)
@click.option(
    "--score-threshold-detection",
    type=float,
    default=0.5,
    show_default=True,
    help="SAM3 detector score threshold (lower increases recall).",
)
@click.option(
    "--new-det-thresh",
    type=float,
    default=0.7,
    show_default=True,
    help="SAM3 new-track threshold (lower admits weaker new tracks).",
)
@click.option(
    "--det-nms-thresh",
    type=float,
    default=0.1,
    show_default=True,
    help="SAM3 detector NMS IoU threshold (higher is less suppressive).",
)
@click.option(
    "--overlap-suppress-thresh",
    type=float,
    default=0.7,
    show_default=True,
    help="SAM3 overlap suppression threshold (higher is less suppressive).",
)
def main(
    cu3s_path: Path,
    processing_mode: str,
    frame_rotation: int | None,
    end_frame: int,
    prompt: str,
    output_dir: Path,
    name_suffix: str,
    checkpoint_path: Path | None,
    plugins_yaml: Path,
    bf16: bool,
    compile_model: bool,
    score_threshold_detection: float,
    new_det_thresh: float,
    det_nms_thresh: float,
    overlap_suppress_thresh: float,
) -> None:
    """Run SAM3 HSI tracking using a CuvisPipeline plus core Predictor."""
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    for option_name, value in (
        ("--score-threshold-detection", score_threshold_detection),
        ("--new-det-thresh", new_det_thresh),
        ("--det-nms-thresh", det_nms_thresh),
        ("--overlap-suppress-thresh", overlap_suppress_thresh),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise click.BadParameter(f"{option_name} must be in [0, 1].")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    resolved_mode = _resolve_processing_mode(processing_mode)
    suffix = name_suffix.strip()
    if suffix:
        safe_suffix = suffix.replace("\\", "_").replace("/", "_").replace(" ", "_")
        if not safe_suffix.startswith("_"):
            safe_suffix = f"_{safe_suffix}"
        output_dir = output_dir.parent / f"{output_dir.name}{safe_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("CU3S: {}", cu3s_path)
    logger.info("Processing mode: {}", resolved_mode)
    logger.info("False RGB: CIE tristimulus")
    logger.info("Prompt: '{}'", prompt)
    logger.info("Output: {}", output_dir)
    logger.info("Device: {}", device)

    from cuvis_ai_core.data.datasets import SingleCu3sDataModule
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.band_selection import CIETristimulusFalseRGBSelector
    from cuvis_ai.node.data import CU3SDataNode
    from cuvis_ai.node.video import ToVideoNode
    from cuvis_ai.node.visualizations import TrackingOverlayNode

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
    sam3_tracker_cls = registry.get("cuvis_ai_sam3.node.SAM3TrackerInference")
    logger.info("SAM3 plugin loaded from {}", plugin_manifest)

    pipeline = CuvisPipeline("SAM3_HSI_Tracking")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = CIETristimulusFalseRGBSelector(name="cie_false_rgb")
    sam3_tracker = sam3_tracker_cls(
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        prompt_text=prompt,
        compile_model=compile_model,
        name="sam3_tracker",
    )
    tracker_model = getattr(sam3_tracker, "_model", None)
    if tracker_model is None:
        logger.warning("Could not access SAM3 internal model; threshold overrides were skipped.")
    else:
        tracker_model.score_threshold_detection = float(score_threshold_detection)
        tracker_model.new_det_thresh = float(new_det_thresh)
        tracker_model.det_nms_thresh = float(det_nms_thresh)
        tracker_model.suppress_overlapping_based_on_recent_occlusion_threshold = float(
            overlap_suppress_thresh
        )
        logger.info(
            "SAM3 thresholds: score_threshold_detection={}, new_det_thresh={}, "
            "det_nms_thresh={}, overlap_suppress_thresh={}",
            score_threshold_detection,
            new_det_thresh,
            det_nms_thresh,
            overlap_suppress_thresh,
        )
    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(output_dir / "tracking_results.json"),
        category_name=prompt,
        name="tracking_coco_json",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.inputs.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.inputs.wavelengths),
        (false_rgb.outputs.rgb_image, sam3_tracker.inputs.rgb_frame),
        (sam3_tracker.outputs.frame_id, tracking_json.inputs.frame_id),
        (sam3_tracker.outputs.mask, tracking_json.inputs.mask),
        (sam3_tracker.outputs.object_ids, tracking_json.inputs.object_ids),
        (sam3_tracker.outputs.detection_scores, tracking_json.inputs.detection_scores),
    )

    overlay_node = TrackingOverlayNode(alpha=0.2, name="overlay")
    overlay_path = output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )
    pipeline.connect(
        (false_rgb.outputs.rgb_image, overlay_node.inputs.rgb_image),
        (sam3_tracker.outputs.mask, overlay_node.inputs.mask),
        (sam3_tracker.outputs.object_ids, overlay_node.inputs.object_ids),
        (overlay_node.outputs.rgb_with_overlay, to_video.inputs.rgb_image),
    )

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

    logger.info("Starting tracking...")
    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    json_path = output_dir / "tracking_results.json"
    if not json_path.exists():
        raise RuntimeError(f"Expected tracking JSON was not created: {json_path}")
    if not overlay_path.exists():
        raise RuntimeError(f"Expected overlay video was not created: {overlay_path}")

    logger.success("Tracking complete")
    logger.info("Results: {}", json_path)
    logger.info("Overlay: {}", overlay_path)
    logger.info("Pipeline PNG: {}", pipeline_png_path)
    logger.info("Pipeline MD: {}", pipeline_md_path)


if __name__ == "__main__":
    main()
