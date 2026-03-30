"""SAM3 prompt-free segment-everything for CU3S or RGB video.

Exactly one of ``--cu3s-path`` or ``--video-path`` is required.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
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
from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode
from cuvis_ai.node.json_writer import CocoTrackMaskWriter
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
    default=Path("D:/experiments/sam3/segment_everything"),
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
@click.option("--bf16", is_flag=True, default=False)
@click.option(
    "--compile", "compile_model", is_flag=True, default=False, help="Enable torch.compile."
)
@click.option("--points-per-side", type=int, default=32, show_default=True)
@click.option("--points-per-batch", type=int, default=64, show_default=True)
@click.option("--pred-iou-thresh", type=float, default=0.8, show_default=True)
@click.option("--stability-score-thresh", type=float, default=0.95, show_default=True)
@click.option("--stability-score-offset", type=float, default=1.0, show_default=True)
@click.option("--mask-threshold", type=float, default=0.0, show_default=True)
@click.option("--box-nms-thresh", type=float, default=0.7, show_default=True)
@click.option("--crop-n-layers", type=int, default=0, show_default=True)
@click.option("--crop-nms-thresh", type=float, default=0.7, show_default=True)
@click.option("--crop-overlap-ratio", type=float, default=512 / 1500, show_default=True)
@click.option("--crop-n-points-downscale-factor", type=int, default=1, show_default=True)
@click.option("--min-mask-region-area", type=int, default=0, show_default=True)
@click.option(
    "--multimask-output/--single-mask-output",
    default=True,
    show_default=True,
    help="Use SAM3 multimask predictions for each sampled point.",
)
def main(
    cu3s_path: Path | None,
    video_path: Path | None,
    processing_mode: str,
    frame_rotation: int | None,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
    output_dir: Path,
    out_basename: str | None,
    checkpoint_path: Path | None,
    plugins_yaml: Path,
    bf16: bool,
    compile_model: bool,
    points_per_side: int,
    points_per_batch: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
    stability_score_offset: float,
    mask_threshold: float,
    box_nms_thresh: float,
    crop_n_layers: int,
    crop_nms_thresh: float,
    crop_overlap_ratio: float,
    crop_n_points_downscale_factor: int,
    min_mask_region_area: int,
    multimask_output: bool,
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
        ("--pred-iou-thresh", pred_iou_thresh),
        ("--stability-score-thresh", stability_score_thresh),
        ("--box-nms-thresh", box_nms_thresh),
        ("--crop-nms-thresh", crop_nms_thresh),
    ):
        if not (0.0 <= float(value) <= 1.0):
            raise click.BadParameter(f"{option_name} must be in [0, 1].")
    if points_per_side < 1:
        raise click.BadParameter("--points-per-side must be >= 1.")
    if points_per_batch < 1:
        raise click.BadParameter("--points-per-batch must be >= 1.")
    if crop_n_layers < 0:
        raise click.BadParameter("--crop-n-layers must be >= 0.")
    if crop_n_points_downscale_factor < 1:
        raise click.BadParameter("--crop-n-points-downscale-factor must be >= 1.")
    if min_mask_region_area < 0:
        raise click.BadParameter("--min-mask-region-area must be >= 0.")
    if stability_score_offset < 0:
        raise click.BadParameter("--stability-score-offset must be >= 0.")
    if crop_overlap_ratio < 0:
        raise click.BadParameter("--crop-overlap-ratio must be >= 0.")

    resolved_mode = resolve_processing_mode(processing_mode)
    if video_path is not None:
        logger.info("Ignoring --processing-mode because --video-path is set")

    source_context = build_source_context(
        cu3s_path=cu3s_path,
        video_path=video_path,
        processing_mode=resolved_mode,
        start_frame=start_frame,
        end_frame=effective_end_frame,
        false_rgb_selector_cls=CIETristimulusFalseRGBSelector if cu3s_path is not None else None,
        false_rgb_norm_mode=NormMode.RUNNING if cu3s_path is not None else None,
        false_rgb_selector_kwargs=(
            {
                "running_warmup_frames": 0,
                "freeze_running_bounds_after_frames": 1,
            }
            if cu3s_path is not None
            else None
        ),
    )

    plugin_manifest = resolve_plugin_manifest(plugins_yaml)
    registry = NodeRegistry()
    registry.load_plugins(str(plugin_manifest))
    sam3_cls = registry.get("cuvis_ai_sam3.node.SAM3SegmentEverything")

    pipeline_name = (
        "SAM3_Segment_Everything_HSI"
        if source_context.source_type == "cu3s"
        else "SAM3_Segment_Everything_Video"
    )

    sam3_node = sam3_cls(
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        compile_model=compile_model,
        points_per_side=int(points_per_side),
        points_per_batch=int(points_per_batch),
        pred_iou_thresh=float(pred_iou_thresh),
        stability_score_thresh=float(stability_score_thresh),
        stability_score_offset=float(stability_score_offset),
        mask_threshold=float(mask_threshold),
        box_nms_thresh=float(box_nms_thresh),
        crop_n_layers=int(crop_n_layers),
        crop_nms_thresh=float(crop_nms_thresh),
        crop_overlap_ratio=float(crop_overlap_ratio),
        crop_n_points_downscale_factor=int(crop_n_points_downscale_factor),
        min_mask_region_area=int(min_mask_region_area),
        multimask_output=bool(multimask_output),
        name="sam3_segment_everything",
    )

    pipeline = CuvisPipeline(pipeline_name)

    tracking_json = CocoTrackMaskWriter(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        category_name="object",
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
            (sam3_node.outputs.mask, tracking_json.mask),
            (sam3_node.object_ids, tracking_json.object_ids),
            (sam3_node.detection_scores, tracking_json.detection_scores),
            (source_context.source_rgb_port, overlay_node.rgb_image),
            (source_context.source_frame_id_port, overlay_node.frame_id),
            (sam3_node.outputs.mask, overlay_node.mask),
            (sam3_node.object_ids, overlay_node.object_ids),
            (overlay_node.rgb_with_overlay, to_video.rgb_image),
            (source_context.source_frame_id_port, to_video.frame_id),
        ]
    )
    pipeline.connect(*connections)

    pipeline_png = run_output_dir / f"{pipeline.name}.png"
    pipeline.visualize(
        format="render_graphviz", output_path=str(pipeline_png), show_execution_stage=True
    )

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
        "Starting segment everything ({} frames, source={})...",
        source_context.target_frames,
        source_context.source_type,
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
