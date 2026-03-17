"""Render overlay video from tracking results (bboxes or masks).

Reads a tracking JSON and a source (MP4 video or CU3S file), applies bbox and/or mask overlays
using cuvis-ai nodes, and writes the result to an MP4 file.

Supports tracking JSON formats:
  - COCO bbox tracking (DeepEIoU / ByteTrack output with track_id)
  - Video COCO (segmentations list per annotation)

Source modes:
  - --video-path: render overlays on an existing MP4
  - --cu3s-file-path: generate false-RGB frames from a CU3S file on the fly and render overlays
    (default method: cie_tristimulus)

Usage — video mode:
    uv run python examples/object_tracking/render_tracking_overlay.py `
        --video-path path/to/source.mp4 `
        --tracking-json path/to/tracking_results.json

Usage — CU3S mode:
    uv run python examples/object_tracking/render_tracking_overlay.py `
        --cu3s-file-path path/to/recording.cu3s `
        --tracking-json path/to/tracking_results.json `
        --method cie_tristimulus
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from cuvis_ai.node.anomaly_visualization import BBoxesOverlayNode, TrackingOverlayNode
from cuvis_ai.node.json_reader import TrackingResultsReader
from cuvis_ai.node.video import ToVideoNode, VideoFrameDataset, VideoIterator
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed

_SUPPORTED_METHODS = ("cie_tristimulus",)
_PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


def _make_false_rgb_node(method: str):
    from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode

    if method != "cie_tristimulus":
        raise ValueError(f"Unknown method '{method}'. Supported: {_SUPPORTED_METHODS}")
    return CIETristimulusFalseRGBSelector(
        norm_mode=NormMode.STATISTICAL,
        name="cie_tristimulus_false_rgb",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Render overlay video from tracking results (bboxes or masks).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video-path", default=None, help="Path to the source MP4 video.")
    source.add_argument(
        "--cu3s-file-path",
        default=None,
        help="Path to a .cu3s file. False-RGB frames are generated on the fly.",
    )
    p.add_argument("--tracking-json", required=True, help="Path to tracking results JSON.")
    p.add_argument(
        "--output-video-path",
        default=None,
        help="Output overlay MP4 path. Default: <tracking_json_dir>/overlay.mp4",
    )
    p.add_argument(
        "--method",
        choices=_SUPPORTED_METHODS,
        default="cie_tristimulus",
        help="False-RGB method (CU3S mode only).",
    )
    p.add_argument(
        "--processing-mode",
        choices=_PROCESSING_MODES,
        default="Raw",
        help="CU3S processing mode (CU3S mode only).",
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


def _apply_overlays(
    rgb: torch.Tensor,
    track: dict,
    bbox_overlay: BBoxesOverlayNode | None,
    mask_overlay: TrackingOverlayNode | None,
) -> torch.Tensor:
    """Apply bbox and/or mask overlays to an rgb frame tensor."""
    if bbox_overlay is not None and track.get("bboxes") is not None:
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

    if mask_overlay is not None and track.get("mask") is not None:
        result = mask_overlay.forward(
            rgb_image=rgb,
            mask=track["mask"],
            object_ids=track.get("object_ids"),
        )
        rgb = result["rgb_with_overlay"]

    return rgb


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    tracking_json = Path(args.tracking_json)
    if not tracking_json.exists():
        logger.error("Tracking JSON not found: {}", tracking_json)
        sys.exit(1)

    output = (
        Path(args.output_video_path)
        if args.output_video_path
        else tracking_json.parent / "overlay.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # -- Shared nodes ----------------------------------------------------------

    reader = TrackingResultsReader(json_path=str(tracking_json))
    logger.info("Tracking format: {} ({} frames)", reader.format, reader.num_frames)

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

    # -- Source: CU3S ----------------------------------------------------------

    if args.cu3s_file_path is not None:
        from cuvis_ai_core.data.datasets import SingleCu3sDataModule

        from cuvis_ai.node.data import CU3SDataNode

        cu3s_path = Path(args.cu3s_file_path)
        if not cu3s_path.exists():
            logger.error("CU3S file not found: {}", cu3s_path)
            sys.exit(1)

        predict_ids = (
            list(range(args.start_frame, args.end_frame))
            if args.end_frame > 0
            else (
                list(range(args.start_frame, reader.num_frames)) if args.start_frame > 0 else None
            )
        )

        datamodule = SingleCu3sDataModule(
            cu3s_file_path=str(cu3s_path),
            processing_mode=args.processing_mode,
            batch_size=1,
            predict_ids=predict_ids,
        )
        datamodule.setup(stage="predict")

        dataset_fps = getattr(datamodule.predict_ds, "fps", None)
        if args.frame_rate is not None and args.frame_rate > 0:
            fps = float(args.frame_rate)
        elif dataset_fps is not None and dataset_fps > 0:
            fps = float(dataset_fps)
        else:
            fps = 10.0
            logger.warning(
                "Could not determine FPS from session metadata; falling back to 10.0 FPS."
            )

        cu3s_node = CU3SDataNode(name="cu3s_data")
        false_rgb = _make_false_rgb_node(args.method)
        sample_positions = initialize_false_rgb_sampled_fixed(
            false_rgb,
            datamodule.predict_ds,
            sample_fraction=0.05,
        )
        logger.info(
            "False-RGB sampled-fixed calibration: sample_fraction=0.05, sample_count={}",
            len(sample_positions),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cu3s_node.to(device)
        false_rgb.to(device)

        video_writer = ToVideoNode(output_video_path=str(output), frame_rate=fps)

        total = min(reader.num_frames, len(datamodule.predict_ds))
        loader = DataLoader(datamodule.predict_ds, batch_size=1, num_workers=0)

        for batch in tqdm(loader, total=total, desc="Rendering overlay (CU3S)", unit="frame"):
            cu3s_out = cu3s_node.forward(
                **{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            )
            rgb_out = false_rgb.forward(cube=cu3s_out["cube"], wavelengths=cu3s_out["wavelengths"])
            rgb = rgb_out["rgb_image"]  # [1, H, W, 3] float32 0-1

            try:
                track = reader.forward()
            except StopIteration:
                break

            rgb = _apply_overlays(rgb, track, bbox_overlay, mask_overlay)
            video_writer.forward(rgb_image=rgb)

        video_writer.close()
        logger.info("Overlay video written to: {}", output)
        return

    # -- Source: MP4 -----------------------------------------------------------

    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error("Video not found: {}", video_path)
        sys.exit(1)

    video_iter = VideoIterator(str(video_path))
    dataset = VideoFrameDataset(video_iter, end_frame=args.end_frame if args.end_frame > 0 else -1)
    fps = args.frame_rate if args.frame_rate is not None else video_iter.frame_rate

    video_writer = ToVideoNode(output_video_path=str(output), frame_rate=fps)

    total = min(reader.num_frames, len(dataset))
    if args.end_frame > 0:
        total = min(total, args.end_frame - args.start_frame)

    if args.start_frame > 0:
        reader._cursor = args.start_frame

    for i in tqdm(range(total), desc="Rendering overlay", unit="frame"):
        frame_idx = args.start_frame + i
        if frame_idx >= len(dataset):
            break

        frame_data = dataset[frame_idx]
        rgb = frame_data["rgb_image"].unsqueeze(0)  # [1, H, W, 3]

        try:
            track = reader.forward()
        except StopIteration:
            break

        rgb = _apply_overlays(rgb, track, bbox_overlay, mask_overlay)
        video_writer.forward(rgb_image=rgb)

    video_writer.close()
    logger.info("Overlay video written to: {}", output)


if __name__ == "__main__":
    main()
