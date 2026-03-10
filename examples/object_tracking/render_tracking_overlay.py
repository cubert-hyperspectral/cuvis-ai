"""Render overlay video from tracking results (bboxes or masks).

Reads a tracking JSON and source video, applies bbox and/or mask overlays
using cuvis-ai nodes, and writes the result to an MP4 file.

Supports tracking JSON formats:
  - COCO bbox tracking (DeepEIoU / ByteTrack output with track_id)
  - Video COCO (segmentations list per annotation)

Usage:
    uv run python examples/object_tracking/render_tracking_overlay.py \
        --video-path path/to/source.mp4 \
        --tracking-json path/to/tracking_results.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode, TrackingOverlayNode
from cuvis_ai.node.json_reader import TrackingResultsReader
from cuvis_ai.node.video import ToVideoNode, VideoFrameDataset, VideoIterator


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render overlay video from tracking results (bboxes or masks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--video-path", required=True, help="Path to the source MP4 video.")
    p.add_argument("--tracking-json", required=True, help="Path to tracking results JSON.")
    p.add_argument(
        "--output-video-path",
        default=None,
        help="Output overlay MP4 path. Default: <tracking_json_dir>/overlay.mp4",
    )
    p.add_argument("--mask-alpha", type=float, default=0.4, help="Mask overlay opacity (0-1).")
    p.add_argument("--line-thickness", type=int, default=2, help="Bbox line thickness.")
    p.add_argument("--draw-labels", action="store_true", help="Render track ID labels on bboxes.")
    p.add_argument(
        "--draw-contours",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw mask contours.",
    )
    p.add_argument(
        "--draw-ids",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render object ID labels on masks.",
    )
    p.add_argument("--start-frame", type=int, default=0, help="First frame to render (inclusive).")
    p.add_argument("--end-frame", type=int, default=-1, help="Last frame to render (-1 = end).")
    p.add_argument(
        "--frame-rate", type=float, default=None, help="Output FPS (default: same as source)."
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    tracking_json = Path(args.tracking_json)
    if not tracking_json.exists():
        logger.error("Tracking JSON not found: {}", tracking_json)
        sys.exit(1)

    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error("Video not found: {}", video_path)
        sys.exit(1)

    output = (
        Path(args.output_video_path)
        if args.output_video_path
        else tracking_json.parent / "overlay.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # -- Nodes -----------------------------------------------------------------

    reader = TrackingResultsReader(json_path=str(tracking_json))
    logger.info("Tracking format: {} ({} frames)", reader.format, reader.num_frames)

    video_iter = VideoIterator(str(video_path))
    dataset = VideoFrameDataset(video_iter, end_frame=args.end_frame if args.end_frame > 0 else -1)
    fps = args.frame_rate if args.frame_rate is not None else video_iter.frame_rate

    has_bboxes = reader.format == "coco_bbox"
    has_masks = reader.format == "video_coco"

    bbox_overlay = (
        BBoxesOverlayNode(
            line_thickness=args.line_thickness,
            draw_labels=args.draw_labels,
        )
        if has_bboxes
        else None
    )

    mask_overlay = (
        TrackingOverlayNode(
            alpha=args.mask_alpha,
            draw_contours=args.draw_contours,
            draw_ids=args.draw_ids,
        )
        if has_masks
        else None
    )

    video_writer = ToVideoNode(output_video_path=str(output), frame_rate=fps)

    # -- Pipeline loop ---------------------------------------------------------

    total = min(reader.num_frames, len(dataset))
    if args.end_frame > 0:
        total = min(total, args.end_frame - args.start_frame)

    reader_cursor_start = args.start_frame
    if reader_cursor_start > 0:
        reader._cursor = reader_cursor_start

    for i in tqdm(range(total), desc="Rendering overlay", unit="frame"):
        frame_idx = args.start_frame + i
        if frame_idx >= len(dataset):
            break

        # Read video frame
        frame_data = dataset[frame_idx]
        rgb = frame_data["rgb_image"].unsqueeze(0)  # [1, H, W, 3]

        # Read tracking data
        try:
            track = reader.forward()
        except StopIteration:
            break

        # Apply bbox overlay
        if bbox_overlay is not None and track.get("bboxes") is not None:
            # Use track_ids as category_ids for per-track coloring
            category_ids = (
                track["track_ids"] if track.get("track_ids") is not None else track["category_ids"]
            )
            result = bbox_overlay.forward(
                rgb_image=rgb,
                bboxes=track["bboxes"],
                category_ids=category_ids,
                frame_id=track["frame_id"],
            )
            rgb = result["rgb_with_overlay"]

        # Apply mask overlay
        if mask_overlay is not None and track.get("mask") is not None:
            result = mask_overlay.forward(
                rgb_image=rgb,
                mask=track["mask"],
                object_ids=track.get("object_ids"),
            )
            rgb = result["rgb_with_overlay"]

        # Write frame
        video_writer.forward(rgb_image=rgb)

    video_writer.close()
    logger.info("Overlay video written to: {}", output)


if __name__ == "__main__":
    main()
