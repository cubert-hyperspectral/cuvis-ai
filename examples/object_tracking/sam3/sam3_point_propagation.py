"""SAM3 point-prompt propagation for CU3S or RGB video.

Exactly one of ``--cu3s-path`` or ``--video-path`` is required.

Uses the center of a detection bbox as a positive point prompt.
Specify the detection with ``--detection ID@FRAME``.
"""

from __future__ import annotations

import contextlib
import json
import re
from pathlib import Path

import click
import torch
from loguru import logger
from sam3_source_context import (
    PROCESSING_MODES,
    build_source_context,
    resolve_end_frame,
    resolve_plugin_manifest,
    resolve_processing_mode,
    validate_source_and_window,
)


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


def _parse_detection_spec(spec: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)@(\d+)", spec.strip())
    if not m:
        raise click.BadParameter(
            f"Invalid detection spec '{spec}'. Expected format: ID@FRAME (e.g. 2@76)."
        )
    return int(m.group(1)), int(m.group(2))


def _extract_point_prompt(
    detection_json: Path, det_id: int, frame_idx: int
) -> tuple[list[list[float]], list[int], int]:
    data = json.loads(detection_json.read_text(encoding="utf-8"))
    images = {img["id"]: img for img in data["images"]}

    if frame_idx not in images:
        raise click.ClickException(f"Frame {frame_idx} not found in {detection_json}.")

    frame_annots = [a for a in data["annotations"] if a["image_id"] == frame_idx]
    has_track_ids = any("track_id" in a for a in frame_annots)

    if has_track_ids:
        by_track = {a["track_id"]: a for a in frame_annots if "track_id" in a}
        if det_id in by_track:
            a = by_track[det_id]
            obj_id = det_id
        else:
            raise click.ClickException(
                f"Track ID {det_id} not found on frame {frame_idx}. "
                f"Available: {sorted(by_track.keys())}"
            )
    else:
        frame_annots.sort(key=lambda ann: ann.get("score", 0.0), reverse=True)
        rank = det_id - 1
        if rank < 0 or rank >= len(frame_annots):
            raise click.ClickException(
                f"Detection rank {det_id} out of range on frame {frame_idx} "
                f"(have {len(frame_annots)} detections)."
            )
        a = frame_annots[rank]
        obj_id = det_id

    img = images[frame_idx]
    w_img, h_img = img["width"], img["height"]
    x, y, w, h = a["bbox"]
    cx = (x + w / 2) / w_img
    cy = (y + h / 2) / h_img
    logger.info(
        "  obj_id={} point: [{:.3f}, {:.3f}] score={:.3f}",
        obj_id,
        cx,
        cy,
        a.get("score", 0),
    )
    return [[cx, cy]], [1], obj_id


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
    "--detection-json", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="SpectralRadiance",
    show_default=True,
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--detection",
    "detection_spec",
    type=str,
    default="1@0",
    show_default=True,
    help="Detection to use: ID@FRAME (e.g. 2@76).",
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
    default=Path("D:/experiments/sam3/point_propagation"),
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
def main(
    cu3s_path: Path | None,
    video_path: Path | None,
    detection_json: Path,
    processing_mode: str,
    frame_rotation: int | None,
    detection_spec: str,
    start_frame: int,
    end_frame: int,
    max_frames: int | None,
    output_dir: Path,
    out_basename: str | None,
    checkpoint_path: Path | None,
    plugins_yaml: Path,
    bf16: bool,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    run_output_dir = _resolve_run_output_dir(
        output_root=output_dir,
        source_path=source_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)

    resolved_mode = resolve_processing_mode(processing_mode)
    if video_path is not None:
        logger.info("Ignoring --processing-mode because --video-path is set")

    det_id, prompt_frame_idx_source = _parse_detection_spec(detection_spec)
    if start_frame > prompt_frame_idx_source:
        raise click.ClickException(
            f"--start-frame ({start_frame}) must be <= prompt frame ({prompt_frame_idx_source})."
        )

    logger.info("Extracting point prompt from {}", detection_json)
    points, point_labels, obj_id = _extract_point_prompt(
        detection_json, det_id=det_id, frame_idx=prompt_frame_idx_source
    )

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import TrackingOverlayNode
    from cuvis_ai.node.json_writer import TrackingCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode

    source_context = build_source_context(
        cu3s_path=cu3s_path,
        video_path=video_path,
        processing_mode=resolved_mode,
        start_frame=start_frame,
        end_frame=effective_end_frame,
    )
    prompt_frame_idx_local = prompt_frame_idx_source - start_frame
    if prompt_frame_idx_local < 0 or prompt_frame_idx_local >= source_context.target_frames:
        raise click.ClickException(
            "Prompt frame is outside the streamed window after start-frame slicing: "
            f"source={prompt_frame_idx_source}, start={start_frame}, local={prompt_frame_idx_local}, "
            f"target_frames={source_context.target_frames}."
        )

    plugin_manifest = resolve_plugin_manifest(plugins_yaml)
    registry = NodeRegistry()
    registry.load_plugins(str(plugin_manifest))
    sam3_cls = registry.get("cuvis_ai_sam3.node.SAM3StreamingPropagation")

    pipeline_name = (
        "SAM3_Point_Propagation_HSI"
        if source_context.source_type == "cu3s"
        else "SAM3_Point_Propagation_Video"
    )
    pipeline = CuvisPipeline(pipeline_name)

    sam3_node = sam3_cls(
        num_frames=source_context.target_frames,
        checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        prompt_type="point",
        prompt_frame_idx=prompt_frame_idx_local,
        prompt_points=points,
        prompt_point_labels=point_labels,
        prompt_obj_id=obj_id,
        name="sam3_streaming",
    )
    tracking_json = TrackingCocoJsonNode(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        category_name="person",
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

    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)
    predictor = Predictor(pipeline=pipeline, datamodule=source_context.datamodule)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )
    logger.info(
        "Starting point propagation (obj_id={}, source prompt frame {}, local prompt frame {}, {} frames)...",
        obj_id,
        prompt_frame_idx_source,
        prompt_frame_idx_local,
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
