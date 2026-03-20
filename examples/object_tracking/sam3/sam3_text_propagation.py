"""SAM3 text-prompt propagation for CU3S or RGB video.

Exactly one of ``--cu3s-path`` or ``--video-path`` is required.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from cuvis_ai_core.utils.node_registry import NodeRegistry
from loguru import logger
from sam3_source_context import (
    PROCESSING_MODES,
    build_source_context,
    resolve_end_frame,
    resolve_plugin_manifest,
    resolve_processing_mode,
    resolve_run_output_dir,
    validate_source_and_window,
)

from cuvis_ai.node.anomaly_visualization import TrackingOverlayNode
from cuvis_ai.node.json_writer import TrackingCocoJsonNode
from cuvis_ai.node.video import ToVideoNode


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
@click.option(
    "--prompt",
    type=str,
    default="person",
    show_default=True,
    help="Text prompt for SAM3 detector (e.g. 'person', 'car').",
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
    "--max-frames",
    type=int,
    default=None,
    help="Deprecated alias for frame window length from --start-frame.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("D:/experiments/sam3/text_propagation"),
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
    "--save-pipeline-yaml/--no-save-pipeline-yaml",
    default=True,
    show_default=True,
    help="Save pipeline YAML config in run output directory.",
)
@click.option(
    "--save-pipeline-weights/--no-save-pipeline-weights",
    default=False,
    show_default=True,
    help="Also save .pt pipeline weights next to the YAML config.",
)
@click.option("--checkpoint-path", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option(
    "--plugins-yaml",
    type=click.Path(exists=False, dir_okay=False, path_type=Path),
    default=Path("configs/plugins/sam3.yaml"),
    show_default=True,
)
@click.option("--bf16", is_flag=True, default=False)
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
def main(
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    frame_rotation: int | None,
    prompt: str,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
    output_dir: Path,
    out_basename: str | None,
    save_pipeline_yaml: bool,
    save_pipeline_weights: bool,
    checkpoint_path: Path | None,
    plugins_yaml: Path,
    bf16: bool,
    compile_model: bool,
    score_threshold_detection: float,
    new_det_thresh: float,
    det_nms_thresh: float,
    overlap_suppress_thresh: float,
    max_tracker_states: int,
) -> None:
    effective_end_frame = resolve_end_frame(
        start_frame=start_frame,
        end_frame=end_frame,
        max_frames=max_frames,
    )
    validate_source_and_window(
        cu3s_path=cu3s_path,
        video_path=video_path,
        start_frame=start_frame,
        end_frame=effective_end_frame,
    )
    source_path = cu3s_path if cu3s_path is not None else video_path
    assert source_path is not None
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=source_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
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
    if save_pipeline_weights and not save_pipeline_yaml:
        raise click.BadParameter("--save-pipeline-weights requires --save-pipeline-yaml.")

    resolved_mode = resolve_processing_mode(processing_mode)
    if video_path is not None:
        logger.info("Ignoring --processing-mode because --video-path is set")

    source_context = build_source_context(
        cu3s_path=cu3s_path,
        video_path=video_path,
        processing_mode=resolved_mode,
        start_frame=start_frame,
        end_frame=effective_end_frame,
    )
    logger.info(
        "Frames: {} (start_frame={}, end_frame={}, FPS {:.1f})",
        source_context.target_frames,
        start_frame,
        effective_end_frame,
        source_context.dataset_fps,
    )

    plugin_manifest = resolve_plugin_manifest(plugins_yaml)
    registry = NodeRegistry()
    registry.load_plugins(str(plugin_manifest))
    sam3_cls = registry.get("cuvis_ai_sam3.node.SAM3TextPropagation")

    pipeline_name = (
        "SAM3_Text_Propagation_HSI"
        if source_context.source_type == "cu3s"
        else "SAM3_Text_Propagation_Video"
    )
    input_frame_id_offset = start_frame if source_context.source_type == "video" else 0

    sam3_node = sam3_cls(
        num_frames=source_context.target_frames,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        compile_model=compile_model,
        input_frame_id_offset=input_frame_id_offset,
        prompt_text=prompt,
        score_threshold_detection=float(score_threshold_detection),
        new_det_thresh=float(new_det_thresh),
        det_nms_thresh=float(det_nms_thresh),
        overlap_suppress_thresh=float(overlap_suppress_thresh),
        max_tracker_states=int(max_tracker_states),
        name="sam3_streaming",
    )

    # -- build pipeline ------------------------------------------------
    pipeline = CuvisPipeline(pipeline_name)

    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        category_name=prompt,
        name="tracking_coco_json",
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
    connections.extend(
        [
            (source_context.source_rgb_port, sam3_node.rgb_frame),
            (source_context.source_frame_id_port, sam3_node.inputs.frame_id),
            (source_context.source_frame_id_port, tracking_json.frame_id),
            (sam3_node.mask, tracking_json.mask),
            (sam3_node.object_ids, tracking_json.object_ids),
            (sam3_node.detection_scores, tracking_json.detection_scores),
            (source_context.source_rgb_port, overlay_node.rgb_image),
            (source_context.source_frame_id_port, overlay_node.frame_id),
            (sam3_node.mask, overlay_node.mask),
            (sam3_node.object_ids, overlay_node.object_ids),
            (overlay_node.rgb_with_overlay, to_video.rgb_image),
        ]
    )
    pipeline.connect(*connections)

    pipeline_png = run_output_dir / f"{pipeline.name}.png"
    pipeline.visualize(
        format="render_graphviz", output_path=str(pipeline_png), show_execution_stage=True
    )
    pipeline_yaml = run_output_dir / f"{pipeline.name}.yaml"
    if save_pipeline_yaml:
        if save_pipeline_weights:
            pipeline.save_to_file(str(pipeline_yaml))
            logger.info(
                "Pipeline config saved (YAML + weights): {}, {}",
                pipeline_yaml,
                pipeline_yaml.with_suffix(".pt"),
            )
        else:
            with pipeline_yaml.open("w", encoding="utf-8") as f:
                yaml.dump(
                    pipeline.serialize().to_dict(),
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                )
            logger.info("Pipeline config saved (YAML only): {}", pipeline_yaml)
    else:
        logger.info("Skipping pipeline config save (--no-save-pipeline-yaml)")

    # -- predict -------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)
    predictor = Predictor(pipeline=pipeline, datamodule=source_context.datamodule)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )
    logger.info(
        "Starting text propagation (prompt='{}', {} frames)...",
        prompt,
        source_context.target_frames,
    )
    with amp_ctx:
        predictor.predict(max_batches=source_context.target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=source_context.target_frames)
    logger.info("\n{}", summary)
    (run_output_dir / "profiling_summary.txt").write_text(summary)

    logger.success("Done -> {}", run_output_dir)
    logger.info("JSON: {}", run_output_dir / "tracking_results.json")
    logger.info("Overlay: {}", overlay_path)


if __name__ == "__main__":
    main()
