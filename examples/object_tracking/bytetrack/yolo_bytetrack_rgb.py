"""YOLO26 + ByteTrack RGB video tracking pipeline.

Reads an MP4 (or other cv2-compatible) video file, runs YOLO detection and
ByteTrack association, and writes COCO-format detection/tracking JSONs plus
an overlay video with frame IDs and track IDs.

Example (60 frames):
    uv run python examples/object_tracking/bytetrack/yolo_bytetrack_rgb.py --video-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01-trustimulus.mp4" --output-dir tracking_output --end-frame 60
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import click
import torch
from loguru import logger

from cuvis_ai.utils.cli_helpers import resolve_run_output_dir


@click.command()
@click.option(
    "--video-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to an MP4 (or other cv2-compatible) video file.",
)
@click.option("--frame-rotation", type=int, default=None)
@click.option("--end-frame", type=int, default=-1, show_default=True)
@click.option("--model-name", type=str, default="yolo26n.pt", show_default=True)
@click.option("--confidence-threshold", type=float, default=0.2, show_default=True)
@click.option("--iou-threshold", type=float, default=0.7, show_default=True, help="YOLO NMS IoU")
@click.option(
    "--agnostic-nms",
    is_flag=True,
    default=False,
    help="Enable class-agnostic NMS (off by default).",
)
@click.option(
    "--classes",
    type=int,
    multiple=True,
    help="Limit NMS to these class ids (repeat flag). Default: keep all classes.",
)
@click.option("--track-thresh", type=float, default=0.5, show_default=True)
@click.option("--track-buffer", type=int, default=30, show_default=True)
@click.option("--match-thresh", type=float, default=0.8, show_default=True)
@click.option(
    "--second-score-thresh",
    type=float,
    default=0.1,
    show_default=True,
    help="Low-score floor used for ByteTrack second association.",
)
@click.option(
    "--second-match-thresh",
    type=float,
    default=0.5,
    show_default=True,
    help="IoU match threshold for ByteTrack second association stage.",
)
@click.option(
    "--unconfirmed-match-thresh",
    type=float,
    default=0.7,
    show_default=True,
    help="IoU threshold when matching unconfirmed tracks.",
)
@click.option(
    "--new-track-thresh-offset",
    type=float,
    default=0.1,
    show_default=True,
    help="Offset in det_thresh = track_thresh + offset for new track activation.",
)
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
@click.option(
    "--plugins-dir",
    type=click.Path(exists=False, file_okay=False, path_type=Path),
    default=None,
    show_default=True,
    help="Directory containing ultralytics.yaml and bytetrack.yaml",
)
@click.option("--bf16", is_flag=True, default=False)
def main(
    video_path: Path,
    frame_rotation: int | None,
    end_frame: int,
    model_name: str,
    confidence_threshold: float,
    iou_threshold: float,
    agnostic_nms: bool,
    classes: tuple[int, ...],
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    second_score_thresh: float,
    second_match_thresh: float,
    unconfirmed_match_thresh: float,
    new_track_thresh_offset: float,
    output_dir: Path,
    out_basename: str | None,
    plugins_dir: Path | None,
    bf16: bool,
) -> None:
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=video_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Output run directory: {}", run_output_dir)

    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode
    from cuvis_ai.node.json_file import CocoTrackBBoxWriter, DetectionCocoJsonNode
    from cuvis_ai.node.video import ToVideoNode, VideoFrameDataModule, VideoFrameNode

    # -- Data module -----------------------------------------------------------
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
    logger.info("Video: {}", video_path)
    logger.info("Frames to process: {} (FPS {:.1f})", target_frames, dataset_fps)

    # -- Plugins ---------------------------------------------------------------
    _repo_root = Path(__file__).parents[3]
    plugins_dir_resolved = (plugins_dir or (_repo_root / "configs" / "plugins")).resolve()

    registry = NodeRegistry()
    for yaml_name in ("ultralytics.yaml", "bytetrack.yaml"):
        registry.load_plugins(str(plugins_dir_resolved / yaml_name))
        logger.info("Plugin manifest loaded: {}", plugins_dir_resolved / yaml_name)

    yolo_pre_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPreprocess")
    yolo_cls = registry.get("cuvis_ai_ultralytics.node.YOLO26Detection")
    yolo_post_cls = registry.get("cuvis_ai_ultralytics.node.YOLOPostprocess")
    bytetrack_cls = registry.get("cuvis_ai_bytetrack.node.ByteTrack")

    # -- Pipeline --------------------------------------------------------------
    pipeline = CuvisPipeline("YOLO_ByteTrack_RGB")

    video_frame = VideoFrameNode(name="video_frame")
    yolo_det = yolo_cls(model_path=model_name, name="yolo26_det")
    yolo_pre = yolo_pre_cls(stride=int(getattr(yolo_det, "stride", 32)), name="yolo_preprocess")
    yolo_post = yolo_post_cls(
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
        agnostic_nms=agnostic_nms,
        classes=list(classes) if classes else None,
        name="yolo_post",
    )
    tracker = bytetrack_cls(
        track_thresh=track_thresh,
        track_buffer=track_buffer,
        match_thresh=match_thresh,
        second_score_thresh=second_score_thresh,
        second_match_thresh=second_match_thresh,
        unconfirmed_match_thresh=unconfirmed_match_thresh,
        new_track_thresh_offset=new_track_thresh_offset,
        frame_rate=int(dataset_fps),
        name="bytetrack",
    )
    det_json = DetectionCocoJsonNode(
        output_json_path=str(run_output_dir / "detection_results.json"),
        name="detection_coco_json",
    )
    track_json = CocoTrackBBoxWriter(
        output_json_path=str(run_output_dir / "tracking_results.json"),
        name="tracking_coco_json",
    )
    bbox_overlay = BBoxesOverlayNode(name="bboxes_overlay", draw_labels=True)
    overlay_path = run_output_dir / "tracking_overlay.mp4"
    to_video = ToVideoNode(
        output_video_path=str(overlay_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        # Video → YOLO
        (video_frame.outputs.rgb_image, yolo_pre.rgb_image),
        (yolo_pre.preprocessed, yolo_det.preprocessed),
        (yolo_pre.model_input_hw, yolo_post.model_input_hw),
        (yolo_pre.orig_hw, yolo_post.orig_hw),
        (yolo_det.raw_preds, yolo_post.raw_preds),
        # YOLO → ByteTrack
        (yolo_post.bboxes, tracker.inputs.bboxes),
        (yolo_post.category_ids, tracker.inputs.category_ids),
        (yolo_post.confidences, tracker.inputs.confidences),
        # Detection JSON (pre-tracking)
        (video_frame.outputs.frame_id, det_json.frame_id),
        (yolo_post.bboxes, det_json.bboxes),
        (yolo_post.category_ids, det_json.category_ids),
        (yolo_post.confidences, det_json.confidences),
        (yolo_pre.orig_hw, det_json.orig_hw),
        # Tracking JSON (post-tracking)
        (video_frame.outputs.frame_id, track_json.frame_id),
        (tracker.outputs.bboxes, track_json.bboxes),
        (tracker.outputs.category_ids, track_json.category_ids),
        (tracker.outputs.confidences, track_json.confidences),
        (tracker.track_ids, track_json.track_ids),
        (yolo_pre.orig_hw, track_json.orig_hw),
        # Video overlay with track ID labels and frame ID
        (video_frame.outputs.rgb_image, bbox_overlay.rgb_image),
        (video_frame.outputs.frame_id, bbox_overlay.frame_id),
        (tracker.outputs.bboxes, bbox_overlay.bboxes),
        (tracker.track_ids, bbox_overlay.category_ids),
        (tracker.outputs.confidences, bbox_overlay.confidences),
        (bbox_overlay.rgb_with_overlay, to_video.rgb_image),
    )

    # -- Run -------------------------------------------------------------------
    pipeline_png = run_output_dir / f"{pipeline.name}.png"
    pipeline.visualize(
        format="render_graphviz", output_path=str(pipeline_png), show_execution_stage=True
    )

    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)

    amp_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if bf16 and device == "cuda"
        else contextlib.nullcontext()
    )

    logger.info("Starting tracking ({} frames) on {}", target_frames, device)
    with amp_ctx:
        predictor.predict(max_batches=target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=target_frames)
    logger.info("\n{}", summary)
    (run_output_dir / "profiling_summary.txt").write_text(summary)

    logger.success("Tracking complete -> {}", run_output_dir)
    logger.info("Overlay: {}", overlay_path)
    logger.info("Detections: {}", run_output_dir / "detection_results.json")
    logger.info("Tracks: {}", run_output_dir / "tracking_results.json")


if __name__ == "__main__":
    main()
