"""Render tracking overlays on video or CU3S using one pipeline run."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from loguru import logger
from torch.utils.data import Subset

from cuvis_ai.node.anomaly_visualization import (
    BBoxesOverlayNode,
    TrackingOverlayNode,
    TrackingPointerOverlayNode,
)
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.json_reader import TrackingResultsReader
from cuvis_ai.node.video import ToVideoNode, VideoFrameDataModule, VideoFrameNode
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed

if TYPE_CHECKING:
    from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector

from cuvis_ai_core.data.datasets import SingleCu3sDataModule

_SUPPORTED_METHODS = ("cie_tristimulus",)
_PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")
_OVERLAY_MODES = ("mask", "bbox", "pointer", "mask_pointer")


@dataclass
class SourceContext:
    """Runtime source context used to wire and execute the overlay pipeline."""

    source_type: str
    datamodule: Any
    source_connections: list[tuple[object, object]]
    source_rgb_port: Any
    source_frame_id_port: Any
    dataset_fps: float
    target_frames: int


def _make_false_rgb_node(method: str) -> CIETristimulusFalseRGBSelector:
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
        "--cu3s-path",
        default=None,
        help="Path to a .cu3s file. False-RGB frames are generated on the fly.",
    )
    p.add_argument("--tracking-json", required=True, help="Path to tracking results JSON.")
    p.add_argument(
        "--overlay-mode",
        choices=_OVERLAY_MODES,
        default="mask",
        help="Overlay mode to render. Select explicitly; default is mask.",
    )
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
    p.add_argument(
        "--end-frame",
        type=int,
        default=-1,
        help="Stop at this source frame index (exclusive). -1 means source end.",
    )
    p.add_argument(
        "--frame-rate", type=float, default=None, help="Output FPS (default: same as source)."
    )
    p.add_argument(
        "--overlay-title",
        default=None,
        help="Optional static title rendered at the top center of each frame.",
    )
    return p.parse_args(argv)


def _validate_frame_window(start_frame: int, end_frame: int) -> None:
    if start_frame < 0:
        raise ValueError("--start-frame must be zero or positive.")
    if end_frame == 0 or end_frame < -1:
        raise ValueError("--end-frame must be -1 or positive.")
    if end_frame != -1 and end_frame <= start_frame:
        raise ValueError("--end-frame must be greater than --start-frame.")


def _resolve_frame_rate(requested_fps: float | None, dataset_fps: float | None) -> float:
    if requested_fps is not None and requested_fps > 0:
        return float(requested_fps)
    if dataset_fps is not None and float(dataset_fps) > 0:
        return float(dataset_fps)
    logger.warning("Could not determine FPS from source metadata; falling back to 10.0 FPS.")
    return 10.0


def _build_cu3s_source_context(args: argparse.Namespace) -> SourceContext:
    cu3s_path = Path(args.cu3s_path)
    if not cu3s_path.exists():
        raise FileNotFoundError(f"CU3S file not found: {cu3s_path}")

    probe_dm = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=args.processing_mode,
        batch_size=1,
    )
    probe_dm.setup(stage="predict")
    if probe_dm.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")
    total_available = len(probe_dm.predict_ds)
    if total_available <= 0:
        raise RuntimeError("No frames available in CU3S source.")

    if args.start_frame >= total_available:
        raise ValueError(
            f"--start-frame ({args.start_frame}) must be < available CU3S frames ({total_available})."
        )

    effective_end = min(args.end_frame, total_available) if args.end_frame > 0 else total_available
    if effective_end <= args.start_frame:
        raise ValueError(
            f"Selected CU3S frame window is empty: start={args.start_frame}, end={effective_end}."
        )

    predict_ids: list[int] | None = None
    if args.start_frame > 0 or effective_end < total_available:
        predict_ids = list(range(args.start_frame, effective_end))

    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=args.processing_mode,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")
    if datamodule.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise RuntimeError("No frames available for selected CU3S frame window.")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = _make_false_rgb_node(args.method)
    initialize_false_rgb_sampled_fixed(
        false_rgb,
        datamodule.predict_ds,
        sample_fraction=0.05,
    )

    fps = _resolve_frame_rate(
        requested_fps=args.frame_rate,
        dataset_fps=getattr(datamodule.predict_ds, "fps", None),
    )
    return SourceContext(
        source_type="cu3s",
        datamodule=datamodule,
        source_connections=[
            (cu3s_data.outputs.cube, false_rgb.cube),
            (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        ],
        source_rgb_port=false_rgb.rgb_image,
        source_frame_id_port=cu3s_data.outputs.mesu_index,
        dataset_fps=fps,
        target_frames=target_frames,
    )


def _build_video_source_context(args: argparse.Namespace) -> SourceContext:
    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    datamodule = VideoFrameDataModule(
        video_path=str(video_path),
        end_frame=args.end_frame if args.end_frame > 0 else -1,
        batch_size=1,
    )
    datamodule.setup(stage="predict")
    if datamodule.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    total_available = len(datamodule.predict_ds)
    if total_available <= 0:
        raise RuntimeError("No frames available in video source.")
    if args.start_frame >= total_available:
        raise ValueError(
            f"--start-frame ({args.start_frame}) must be < available video frames ({total_available})."
        )

    if args.start_frame > 0:
        datamodule.predict_ds = Subset(
            datamodule.predict_ds, range(args.start_frame, total_available)
        )

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise RuntimeError("No frames available for selected video frame window.")

    video_frame = VideoFrameNode(name="video_frame")
    fps = _resolve_frame_rate(
        requested_fps=args.frame_rate,
        dataset_fps=(
            getattr(datamodule, "fps", None) or getattr(datamodule.predict_ds, "fps", None)
        ),
    )
    return SourceContext(
        source_type="video",
        datamodule=datamodule,
        source_connections=[],
        source_rgb_port=video_frame.outputs.rgb_image,
        source_frame_id_port=video_frame.outputs.frame_id,
        dataset_fps=fps,
        target_frames=target_frames,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_frame_window(args.start_frame, args.end_frame)

    tracking_json = Path(args.tracking_json)
    if not tracking_json.exists():
        logger.error("Tracking JSON not found: {}", tracking_json)
        sys.exit(1)

    source_path = Path(args.video_path if args.video_path else args.cu3s_path)
    output = (
        Path(args.output_video_path)
        if args.output_video_path
        else tracking_json.parent / f"{source_path.stem}.mp4"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    expected_format = "coco_bbox" if args.overlay_mode == "bbox" else "video_coco"
    reader = TrackingResultsReader(
        json_path=str(tracking_json),
        required_format=expected_format,
    )

    try:
        source_context = (
            _build_cu3s_source_context(args)
            if args.cu3s_path is not None
            else _build_video_source_context(args)
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        logger.error("{}", exc)
        sys.exit(1)

    pipeline = CuvisPipeline("Tracking_Overlay_Render")
    to_video = ToVideoNode(
        output_video_path=str(output),
        frame_rate=source_context.dataset_fps,
        overlay_title=args.overlay_title,
    )

    connections = list(source_context.source_connections)
    connections.append((source_context.source_frame_id_port, reader.inputs.frame_id))

    if args.overlay_mode == "bbox":
        overlay = BBoxesOverlayNode(
            line_thickness=args.line_thickness,
            draw_labels=args.draw_labels,
            name="bbox_overlay",
        )
        connections.extend(
            [
                (source_context.source_rgb_port, overlay.rgb_image),
                (reader.outputs.bboxes, overlay.bboxes),
                (reader.outputs.track_ids, overlay.category_ids),
                (reader.outputs.confidences, overlay.confidences),
                (reader.outputs.frame_id, overlay.frame_id),
                (overlay.rgb_with_overlay, to_video.rgb_image),
            ]
        )
    elif args.overlay_mode == "mask":
        overlay = TrackingOverlayNode(
            alpha=args.mask_alpha,
            draw_contours=args.draw_contours,
            draw_ids=args.draw_ids,
            name="mask_overlay",
        )
        connections.extend(
            [
                (source_context.source_rgb_port, overlay.rgb_image),
                (reader.outputs.mask, overlay.mask),
                (reader.outputs.object_ids, overlay.object_ids),
                (reader.outputs.frame_id, overlay.frame_id),
                (overlay.rgb_with_overlay, to_video.rgb_image),
            ]
        )
    elif args.overlay_mode == "pointer":
        pointer = TrackingPointerOverlayNode(name="pointer_overlay")
        connections.extend(
            [
                (source_context.source_rgb_port, pointer.rgb_image),
                (reader.outputs.mask, pointer.mask),
                (reader.outputs.object_ids, pointer.object_ids),
                (reader.outputs.frame_id, pointer.frame_id),
                (pointer.rgb_with_overlay, to_video.rgb_image),
            ]
        )
    elif args.overlay_mode == "mask_pointer":
        mask_overlay = TrackingOverlayNode(
            alpha=args.mask_alpha,
            draw_contours=args.draw_contours,
            draw_ids=args.draw_ids,
            name="mask_overlay",
        )
        pointer = TrackingPointerOverlayNode(name="pointer_overlay")
        connections.extend(
            [
                (source_context.source_rgb_port, mask_overlay.rgb_image),
                (reader.outputs.mask, mask_overlay.mask),
                (reader.outputs.object_ids, mask_overlay.object_ids),
                (reader.outputs.frame_id, mask_overlay.frame_id),
                (mask_overlay.rgb_with_overlay, pointer.rgb_image),
                (reader.outputs.mask, pointer.mask),
                (reader.outputs.object_ids, pointer.object_ids),
                (reader.outputs.frame_id, pointer.frame_id),
                (pointer.rgb_with_overlay, to_video.rgb_image),
            ]
        )
    else:
        logger.error("Unsupported overlay mode: {}", args.overlay_mode)
        sys.exit(1)

    pipeline.connect(*connections)

    pipeline_png = output.parent / f"{pipeline.name}.png"
    pipeline.visualize(
        format="render_graphviz",
        output_path=str(pipeline_png),
        show_execution_stage=True,
    )

    pipeline_yaml = output.parent / f"{pipeline.name}.yaml"
    with pipeline_yaml.open("w", encoding="utf-8") as stream:
        yaml.dump(pipeline.serialize().to_dict(), stream, default_flow_style=False, sort_keys=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)

    predictor = Predictor(pipeline=pipeline, datamodule=source_context.datamodule)
    predictor.predict(max_batches=source_context.target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=source_context.target_frames)
    profiling_path = output.parent / "profiling_summary.txt"
    profiling_path.write_text(summary, encoding="utf-8")

    if not output.exists():
        raise RuntimeError(f"Expected output video was not created: {output}")

    logger.info("Overlay video written to: {}", output)


if __name__ == "__main__":
    main()
