"""SAM3 MP4 video tracking: MP4 -> SAM3 node pipeline -> COCO JSON + overlay.

Replaces the CU3S + false-RGB chain with a direct MP4 reader (VideoIterator
backed by torchcodec) to isolate data-source effects on pipeline performance.

Outputs:
- tracking_results.json   COCO-format annotations with RLE masks, bboxes, scores
- tracking_overlay.mp4    RGB frames with coloured mask overlays

Example (50 frames):
    uv run python examples/object_tracking/sam3_video_tracker.py \\
        --video-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01-trustimulus.mp4" \\
        --output-dir "D:\\data\\XMR_notarget_Busstation\\20260226\\tracker\\mp4_test_50" \\
        --end-frame 50
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger

from cuvis_ai.data.video import VideoFrameDataModule
from cuvis_ai.node.data import VideoFrameNode


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option(
    "--video-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to an MP4 (or other cv2-compatible) video file.",
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
@click.option(
    "--max-tracker-states",
    type=int,
    default=5,
    show_default=True,
    help="Maximum number of active tracker states.",
)
@click.option(
    "--state-diagnostics/--no-state-diagnostics",
    default=False,
    show_default=True,
    help="Enable periodic internal state-size debug logs from SAM3 tracker node.",
)
@click.option(
    "--progress-log-interval",
    type=int,
    default=50,
    show_default=True,
    help="Emit a tracker progress log every N frames (0 disables periodic progress logs).",
)
@click.option(
    "--confirmed-output/--tentative-output",
    default=True,
    show_default=True,
    help="Use confirmed tracker outputs for JSON/video sinks.",
)
def main(
    video_path: Path,
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
    max_tracker_states: int,
    state_diagnostics: bool,
    progress_log_interval: int,
    confirmed_output: bool,
) -> None:
    """Run SAM3 video tracking from an MP4 using the cuvis-ai pipeline."""
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
    if max_tracker_states < 1:
        raise click.BadParameter("--max-tracker-states must be >= 1.")
    if progress_log_interval < 0:
        raise click.BadParameter("--progress-log-interval must be >= 0.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    suffix = name_suffix.strip()
    if suffix:
        safe_suffix = suffix.replace("\\", "_").replace("/", "_").replace(" ", "_")
        if not safe_suffix.startswith("_"):
            safe_suffix = f"_{safe_suffix}"
        output_dir = output_dir.parent / f"{output_dir.name}{safe_suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Video: {}", video_path)
    logger.info("Prompt: '{}'", prompt)
    logger.info("Output: {}", output_dir)
    logger.info("Device: {}", device)

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import TrackingOverlayNode
    from cuvis_ai.node.json_writer import TrackingCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode

    # -- Data module ----------------------------------------------------------
    datamodule = VideoFrameDataModule(
        video_path=str(video_path),
        end_frame=end_frame,
        batch_size=1,
    )
    datamodule.setup(stage="predict")

    if datamodule.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise click.ClickException("No frames available for prediction.")

    dataset_fps = datamodule.fps
    logger.info("Frames to process: {} (FPS {:.1f})", target_frames, dataset_fps)

    # -- SAM3 plugin ----------------------------------------------------------
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

    # -- Pipeline -------------------------------------------------------------
    pipeline = CuvisPipeline("SAM3_Video_Tracking")

    video_frame = VideoFrameNode(name="video_frame")
    sam3_tracker = sam3_tracker_cls(
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        prompt_text=prompt,
        score_threshold_detection=float(score_threshold_detection),
        new_det_thresh=float(new_det_thresh),
        det_nms_thresh=float(det_nms_thresh),
        overlap_suppress_thresh=float(overlap_suppress_thresh),
        max_tracker_states=int(max_tracker_states),
        enable_state_diagnostics=bool(state_diagnostics),
        progress_log_interval=int(progress_log_interval),
        compile_model=compile_model,
        name="sam3_tracker",
    )
    logger.info(
        "SAM3 thresholds: score_threshold_detection={}, new_det_thresh={}, "
        "det_nms_thresh={}, overlap_suppress_thresh={}, max_tracker_states={}, "
        "state_diagnostics={}, progress_log_interval={}",
        score_threshold_detection,
        new_det_thresh,
        det_nms_thresh,
        overlap_suppress_thresh,
        max_tracker_states,
        state_diagnostics,
        progress_log_interval,
    )

    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(output_dir / "tracking_results.json"),
        category_name=prompt,
        name="tracking_coco_json",
    )

    output_mask_port = (
        sam3_tracker.outputs.confirmed_mask if confirmed_output else sam3_tracker.outputs.mask
    )
    output_ids_port = (
        sam3_tracker.outputs.confirmed_object_ids
        if confirmed_output
        else sam3_tracker.outputs.object_ids
    )
    output_scores_port = (
        sam3_tracker.outputs.confirmed_detection_scores
        if confirmed_output
        else sam3_tracker.outputs.detection_scores
    )
    logger.info(
        "Sink outputs: {} tracks",
        "confirmed" if confirmed_output else "tentative",
    )

    pipeline.connect(
        (video_frame.outputs.rgb_image, sam3_tracker.inputs.rgb_frame),
        (sam3_tracker.outputs.frame_id, tracking_json.inputs.frame_id),
        (output_mask_port, tracking_json.inputs.mask),
        (output_ids_port, tracking_json.inputs.object_ids),
        (output_scores_port, tracking_json.inputs.detection_scores),
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
        (video_frame.outputs.rgb_image, overlay_node.inputs.rgb_image),
        (output_mask_port, overlay_node.inputs.mask),
        (output_ids_port, overlay_node.inputs.object_ids),
        (overlay_node.outputs.rgb_with_overlay, to_video.inputs.rgb_image),
    )

    # -- Visualize pipeline ---------------------------------------------------
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

    # -- Run ------------------------------------------------------------------
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
