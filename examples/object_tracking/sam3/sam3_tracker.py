"""Unified SAM3 tracking for CU3S or RGB video sources.

Exactly one of ``--cu3s-path`` or ``--video-path`` is required.

Outputs:
- tracking_results.json   COCO-format annotations with RLE masks, bboxes, scores
- tracking_overlay.mp4    Source RGB frames with coloured mask overlays
- profiling_summary.txt   Per-node runtime profiling summary
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click
import torch
from loguru import logger
from torch.utils.data import Subset

PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


@dataclass
class SourceContext:
    source_type: str
    datamodule: object
    source_rgb_port: Any
    source_frame_id_port: Any
    source_connections: list[tuple[object, object]]
    dataset_fps: float
    target_frames: int


def _resolve_processing_mode(processing_mode: str) -> str:
    lookup = {mode.lower(): mode for mode in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


def _resolve_run_output_dir(
    *,
    output_root: Path,
    source_path: Path,
    out_basename: str | None,
) -> Path:
    resolved_basename = source_path.stem
    if out_basename is not None:
        candidate = out_basename.strip()
        if not candidate:
            raise click.BadParameter(
                "--out-basename must not be empty or whitespace only",
                param_hint="--out-basename",
            )
        if "/" in candidate or "\\" in candidate:
            raise click.BadParameter(
                "--out-basename must be a folder name, not a path",
                param_hint="--out-basename",
            )
        resolved_basename = candidate
    return output_root / resolved_basename


def _resolve_plugin_manifest(plugins_yaml: Path) -> Path:
    plugin_manifest = plugins_yaml
    if not plugin_manifest.is_absolute():
        plugin_manifest = (Path(__file__).resolve().parents[3] / plugin_manifest).resolve()
    plugin_manifest = plugin_manifest.resolve()
    if not plugin_manifest.exists():
        raise click.ClickException(f"Plugins manifest not found: {plugin_manifest}")
    if not plugin_manifest.is_file():
        raise click.ClickException(f"Plugins manifest is not a file: {plugin_manifest}")
    return plugin_manifest


def _tracker_accepts_frame_id_input(tracker: object) -> bool:
    input_specs = getattr(tracker, "INPUT_SPECS", None) or getattr(
        type(tracker), "INPUT_SPECS", None
    )
    return isinstance(input_specs, dict) and "frame_id" in input_specs


def _write_profiling_summary(output_dir: Path, pipeline: object, total_frames: int) -> Path:
    summary = pipeline.format_profiling_summary(total_frames=total_frames)
    logger.info("\n{}", summary)
    profiling_path = output_dir / "profiling_summary.txt"
    profiling_path.write_text(summary)
    logger.info("Profiling saved: {}", profiling_path)
    return profiling_path


def _build_source_context(
    *,
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    start_frame: int,
    end_frame: int,
    single_cu3s_datamodule_cls: type | None = None,
    cu3s_data_node_cls: type | None = None,
    false_rgb_selector_cls: type | None = None,
    false_rgb_norm_mode: object | None = None,
    video_frame_datamodule_cls: type | None = None,
    video_frame_node_cls: type | None = None,
    subset_cls: type = Subset,
    false_rgb_initializer: Any | None = None,
) -> SourceContext:
    source_type = "cu3s" if cu3s_path is not None else "video"

    datamodule: object
    source_connections: list[tuple[object, object]] = []
    source_rgb_port: Any
    source_frame_id_port: Any

    if source_type == "cu3s":
        assert cu3s_path is not None

        if single_cu3s_datamodule_cls is None:
            from cuvis_ai_core.data.datasets import SingleCu3sDataModule

            single_cu3s_datamodule_cls = SingleCu3sDataModule
        if cu3s_data_node_cls is None:
            from cuvis_ai.node.data import CU3SDataNode

            cu3s_data_node_cls = CU3SDataNode
        if false_rgb_selector_cls is None:
            from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode

            false_rgb_selector_cls = CIETristimulusFalseRGBSelector
            false_rgb_norm_mode = NormMode.STATISTICAL
        if false_rgb_initializer is None:
            from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed

            false_rgb_initializer = initialize_false_rgb_sampled_fixed

        predict_ids = None
        if start_frame > 0 or end_frame > 0:
            dm_probe = single_cu3s_datamodule_cls(
                cu3s_file_path=str(cu3s_path),
                processing_mode=processing_mode,
                batch_size=1,
            )
            dm_probe.setup(stage="predict")
            if dm_probe.predict_ds is None:
                raise RuntimeError("Predict dataset was not initialized.")
            total_available = len(dm_probe.predict_ds)
            effective_end = min(end_frame, total_available) if end_frame > 0 else total_available
            predict_ids = list(range(start_frame, effective_end))

        datamodule = single_cu3s_datamodule_cls(
            cu3s_file_path=str(cu3s_path),
            processing_mode=processing_mode,
            batch_size=1,
            predict_ids=predict_ids,
        )
        datamodule.setup(stage="predict")
        if datamodule.predict_ds is None:
            raise RuntimeError("Predict dataset was not initialized.")

        cu3s_data = cu3s_data_node_cls(name="cu3s_data")
        false_rgb_kwargs: dict[str, object] = {"name": "cie_false_rgb"}
        if false_rgb_norm_mode is not None:
            false_rgb_kwargs["norm_mode"] = false_rgb_norm_mode
        false_rgb = false_rgb_selector_cls(**false_rgb_kwargs)
        sample_positions = false_rgb_initializer(
            false_rgb,
            datamodule.predict_ds,
            sample_fraction=0.05,
        )
        logger.info(
            "False-RGB sampled-fixed calibration: sample_fraction=0.05, sample_count={}",
            len(sample_positions),
        )
        source_connections.extend(
            [
                (cu3s_data.outputs.cube, false_rgb.cube),
                (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
            ]
        )
        source_rgb_port = false_rgb.rgb_image
        source_frame_id_port = cu3s_data.outputs.mesu_index
    else:
        assert video_path is not None

        if video_frame_datamodule_cls is None:
            from cuvis_ai.node.video import VideoFrameDataModule

            video_frame_datamodule_cls = VideoFrameDataModule
        if video_frame_node_cls is None:
            from cuvis_ai.node.video import VideoFrameNode

            video_frame_node_cls = VideoFrameNode

        datamodule = video_frame_datamodule_cls(
            video_path=str(video_path),
            end_frame=end_frame,
            batch_size=1,
        )
        datamodule.setup(stage="predict")
        if datamodule.predict_ds is None:
            raise RuntimeError("Predict dataset was not initialized.")

        effective_end = len(datamodule.predict_ds)
        if start_frame > 0:
            datamodule.predict_ds = subset_cls(
                datamodule.predict_ds, range(start_frame, effective_end)
            )

        video_frame = video_frame_node_cls(name="video_frame")
        source_rgb_port = video_frame.outputs.rgb_image
        source_frame_id_port = video_frame.outputs.frame_id

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise click.ClickException("No frames available for prediction.")

    dataset_fps = float(
        getattr(datamodule, "fps", None) or getattr(datamodule.predict_ds, "fps", None) or 10.0
    )
    if dataset_fps <= 0:
        dataset_fps = 10.0
        logger.warning("Invalid FPS from dataset; falling back to 10.0.")

    return SourceContext(
        source_type=source_type,
        datamodule=datamodule,
        source_rgb_port=source_rgb_port,
        source_frame_id_port=source_frame_id_port,
        source_connections=source_connections,
        dataset_fps=dataset_fps,
        target_frames=target_frames,
    )


def run_sam3_tracker(
    *,
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    frame_rotation: int | None,
    start_frame: int,
    end_frame: int,
    prompt: str,
    output_dir: Path,
    out_basename: str | None,
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    source_type = "cu3s" if cu3s_path is not None else "video"
    source_path = cu3s_path if source_type == "cu3s" else video_path
    assert source_path is not None
    run_output_dir = _resolve_run_output_dir(
        output_root=output_dir,
        source_path=source_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Source type: {}", source_type)
    if source_type == "cu3s":
        assert cu3s_path is not None
        logger.info("CU3S: {}", cu3s_path)
        logger.info("Processing mode: {}", processing_mode)
    else:
        assert video_path is not None
        logger.info("Video: {}", video_path)
    logger.info("Prompt: '{}'", prompt)
    logger.info("Output run directory: {}", run_output_dir)
    logger.info("Device: {}", device)

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import TrackingOverlayNode
    from cuvis_ai.node.json_writer import TrackingCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode

    source_context = _build_source_context(
        cu3s_path=cu3s_path,
        video_path=video_path,
        processing_mode=processing_mode,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    logger.info(
        "Frames to process: {} (FPS {:.1f}, start_frame={}, end_frame={})",
        source_context.target_frames,
        source_context.dataset_fps,
        start_frame,
        end_frame,
    )

    plugin_manifest = _resolve_plugin_manifest(plugins_yaml)
    registry = NodeRegistry()
    registry.load_plugins(str(plugin_manifest))
    sam3_tracker_cls = registry.get("cuvis_ai_sam3.node.SAM3TrackerInference")
    logger.info("SAM3 plugin loaded from {}", plugin_manifest)

    pipeline_name = (
        "SAM3_HSI_Tracking" if source_context.source_type == "cu3s" else "SAM3_Video_Tracking"
    )
    pipeline = CuvisPipeline(pipeline_name)

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
        parity_mode=True,
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
    logger.info("SAM3 mode: parity-only")

    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        category_name=prompt,
        name="tracking_coco_json",
    )

    output_mask_port = sam3_tracker.confirmed_mask if confirmed_output else sam3_tracker.mask
    output_ids_port = (
        sam3_tracker.confirmed_object_ids if confirmed_output else sam3_tracker.object_ids
    )
    output_scores_port = (
        sam3_tracker.confirmed_detection_scores
        if confirmed_output
        else sam3_tracker.detection_scores
    )
    logger.info(
        "Sink outputs: {} tracks",
        "confirmed" if confirmed_output else "tentative",
    )

    overlay_node = TrackingOverlayNode(alpha=0.2, name="overlay")
    overlay_path = run_output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=source_context.dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    connections = list(source_context.source_connections)
    if _tracker_accepts_frame_id_input(sam3_tracker):
        connections.append((source_context.source_frame_id_port, sam3_tracker.frame_id))
    connections.extend(
        [
            (source_context.source_rgb_port, sam3_tracker.rgb_frame),
            (source_context.source_frame_id_port, tracking_json.frame_id),
            (output_mask_port, tracking_json.mask),
            (output_ids_port, tracking_json.object_ids),
            (output_scores_port, tracking_json.detection_scores),
            (source_context.source_rgb_port, overlay_node.rgb_image),
            (source_context.source_frame_id_port, overlay_node.frame_id),
            (output_mask_port, overlay_node.mask),
            (output_ids_port, overlay_node.object_ids),
            (overlay_node.rgb_with_overlay, to_video.rgb_image),
        ]
    )
    pipeline.connect(*connections)

    pipeline_viz_dir = run_output_dir / "pipeline"
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
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)
    logger.info("Profiling enabled (synchronize_cuda={}, skip_first_n=3)", device == "cuda")

    predictor = Predictor(pipeline=pipeline, datamodule=source_context.datamodule)
    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )

    logger.info("Starting tracking...")
    with amp_ctx:
        predictor.predict(max_batches=source_context.target_frames, collect_outputs=False)

    json_path = run_output_dir / "tracking_results.json"
    if not json_path.exists():
        raise RuntimeError(f"Expected tracking JSON was not created: {json_path}")
    if not overlay_path.exists():
        raise RuntimeError(f"Expected overlay video was not created: {overlay_path}")

    profiling_path = _write_profiling_summary(
        run_output_dir, pipeline, source_context.target_frames
    )

    logger.success("Tracking complete -> {}", run_output_dir)
    logger.info("Results: {}", json_path)
    logger.info("Overlay: {}", overlay_path)
    logger.info("Profiling: {}", profiling_path)
    logger.info("Pipeline PNG: {}", pipeline_png_path)
    logger.info("Pipeline MD: {}", pipeline_md_path)


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a .cu3s file. False-RGB frames are generated on the fly.",
)
@click.option(
    "--video-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to an MP4 (or other cv2-compatible) video file.",
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--start-frame", type=int, default=0, show_default=True)
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
@click.option("--checkpoint-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("configs/plugins/sam3.yaml"),
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
    help="Maximum number of active tracker states (lower is faster, higher improves ID continuity for new arrivals).",
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
    help="Use confirmed tracker outputs for JSON/video sinks (more reliable, fewer false positives).",
)
def main(
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    frame_rotation: int | None,
    start_frame: int,
    end_frame: int,
    prompt: str,
    output_dir: Path,
    out_basename: str | None,
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
    if (cu3s_path is None) == (video_path is None):
        raise click.UsageError("Exactly one of --cu3s-path or --video-path must be provided.")
    if start_frame < 0:
        raise click.BadParameter(
            "--start-frame must be zero or positive", param_hint="--start-frame"
        )
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive", param_hint="--end-frame")
    if end_frame != -1 and end_frame <= start_frame:
        raise click.BadParameter(
            "--end-frame must be greater than --start-frame",
            param_hint="--end-frame",
        )
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

    resolved_mode = _resolve_processing_mode(processing_mode)
    if video_path is not None:
        logger.info("Ignoring --processing-mode because --video-path is set")

    run_sam3_tracker(
        cu3s_path=cu3s_path,
        video_path=video_path,
        processing_mode=resolved_mode,
        frame_rotation=frame_rotation,
        start_frame=start_frame,
        end_frame=end_frame,
        prompt=prompt,
        output_dir=output_dir,
        out_basename=out_basename,
        checkpoint_path=checkpoint_path,
        plugins_yaml=plugins_yaml,
        bf16=bf16,
        compile_model=compile_model,
        score_threshold_detection=score_threshold_detection,
        new_det_thresh=new_det_thresh,
        det_nms_thresh=det_nms_thresh,
        overlap_suppress_thresh=overlap_suppress_thresh,
        max_tracker_states=max_tracker_states,
        state_diagnostics=state_diagnostics,
        progress_log_interval=progress_log_interval,
        confirmed_output=confirmed_output,
    )


if __name__ == "__main__":
    main()
