"""Export a CU3S sequence to false-RGB MP4 using cuvis.ai nodes.

Supports multiple false-RGB generation methods:
- range_average: per-channel wavelength-range averaging (default)
- cie_tristimulus: CIE 1931 XYZ -> sRGB conversion
- camera_emulation: Gaussian camera sensitivity curves
- baseline: fixed wavelength band selection (650/550/450 nm)

Pipeline:
1. CU3SDataNode: normalizes CU3S tensors and extracts wavelengths
2. False RGB node (method-dependent): spectral -> 3-channel conversion
3. ToVideoNode: stream-write RGB frames to MP4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cuvis
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataset
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from loguru import logger
from torch.utils.data import DataLoader

from cuvis_ai.node.band_selection import (
    BandSelectorBase,
    BaselineFalseRGBSelector,
    CameraEmulationFalseRGBSelector,
    CIETristimulusFalseRGBSelector,
    RangeAverageFalseRGBSelector,
)
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.video import ToVideoNode

SUPPORTED_METHODS = ("range_average", "cie_tristimulus", "camera_emulation", "baseline")


def _resolve_processing_mode(
    processing_mode: str | cuvis.ProcessingMode,
) -> cuvis.ProcessingMode:
    """Resolve CLI/user input into a valid cuvis ProcessingMode enum."""
    if isinstance(processing_mode, cuvis.ProcessingMode):
        return processing_mode

    mode_text = str(processing_mode).strip()
    mode_text = mode_text.removeprefix("cuvis.")
    mode_text = mode_text.removeprefix("ProcessingMode.")

    supported_mode_names = (
        "Raw",
        "DarkSubtract",
        "Preview",
        "Reflectance",
        "SpectralRadiance",
    )
    mode_lookup = {
        name.lower(): getattr(cuvis.ProcessingMode, name) for name in supported_mode_names
    }

    resolved_mode = mode_lookup.get(mode_text.lower())
    if resolved_mode is None:
        raise ValueError(
            "Invalid processing_mode. Supported values: " + ", ".join(supported_mode_names)
        )
    return resolved_mode


def _create_false_rgb_node(
    method: str,
    *,
    red_low: float = 580.0,
    red_high: float = 650.0,
    green_low: float = 500.0,
    green_high: float = 580.0,
    blue_low: float = 420.0,
    blue_high: float = 500.0,
    r_peak: float = 610.0,
    g_peak: float = 540.0,
    b_peak: float = 460.0,
    r_sigma: float = 40.0,
    g_sigma: float = 35.0,
    b_sigma: float = 30.0,
) -> BandSelectorBase:
    """Create the appropriate false RGB node for the given method."""
    if method == "range_average":
        return RangeAverageFalseRGBSelector(
            red_range=(red_low, red_high),
            green_range=(green_low, green_high),
            blue_range=(blue_low, blue_high),
            name="range_average_false_rgb",
        )
    elif method == "cie_tristimulus":
        return CIETristimulusFalseRGBSelector(name="cie_tristimulus_false_rgb")
    elif method == "camera_emulation":
        return CameraEmulationFalseRGBSelector(
            r_peak=r_peak,
            g_peak=g_peak,
            b_peak=b_peak,
            r_sigma=r_sigma,
            g_sigma=g_sigma,
            b_sigma=b_sigma,
            name="camera_emulation_false_rgb",
        )
    elif method == "baseline":
        return BaselineFalseRGBSelector(name="baseline_false_rgb")
    else:
        raise ValueError(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")


def export_false_rgb_video(
    cu3s_file_path: str,
    output_video_path: str,
    method: str = "range_average",
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    max_num_frames: int = -1,
    batch_size: int = 1,
    processing_mode: str | cuvis.ProcessingMode = "Raw",
    red_low: float = 580.0,
    red_high: float = 650.0,
    green_low: float = 500.0,
    green_high: float = 580.0,
    blue_low: float = 420.0,
    blue_high: float = 500.0,
    r_peak: float = 610.0,
    g_peak: float = 540.0,
    b_peak: float = 460.0,
    r_sigma: float = 40.0,
    g_sigma: float = 35.0,
    b_sigma: float = 30.0,
) -> Path:
    """Run CU3S -> false RGB -> MP4 export pipeline."""
    resolved_processing_mode = _resolve_processing_mode(processing_mode)
    dataset = SingleCu3sDataset(
        cu3s_file_path=cu3s_file_path,
        processing_mode=resolved_processing_mode,
    )

    dataset_fps = getattr(dataset, "fps", None)
    if frame_rate is not None and frame_rate > 0:
        resolved_frame_rate = float(frame_rate)
    elif dataset_fps is not None and dataset_fps > 0:
        resolved_frame_rate = float(dataset_fps)
    else:
        resolved_frame_rate = 10.0
        logger.warning("Could not determine FPS from session metadata; falling back to 10.0 FPS.")

    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)

    pipeline = CuvisPipeline("SAM3_FalseRGB_Export")

    cu3s_data = CU3SDataNode(name="cu3s_data")
    false_rgb = _create_false_rgb_node(
        method,
        red_low=red_low,
        red_high=red_high,
        green_low=green_low,
        green_high=green_high,
        blue_low=blue_low,
        blue_high=blue_high,
        r_peak=r_peak,
        g_peak=g_peak,
        b_peak=b_peak,
        r_sigma=r_sigma,
        g_sigma=g_sigma,
        b_sigma=b_sigma,
    )
    to_video = ToVideoNode(
        output__video_path=output_video_path,
        frame_rate=resolved_frame_rate,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, to_video.rgb_image),
    )

    total_frames = len(dataset)
    target_frames = total_frames if max_num_frames < 0 else min(total_frames, int(max_num_frames))
    if target_frames <= 0:
        raise ValueError("max_num_frames must be -1 (all) or a positive integer.")

    logger.info(f"Starting export of {target_frames}/{total_frames} frames from {cu3s_file_path}")
    logger.info(
        "Video settings: "
        f"method={method}, "
        f"frame_rate={resolved_frame_rate}, "
        f"dataset_fps={dataset_fps}, "
        f"frame_rotation={frame_rotation}, "
        f"processing_mode={resolved_processing_mode}, "
        f"max_num_frames={max_num_frames}"
    )
    if method == "range_average":
        logger.info(
            "Channel ranges: "
            f"R=[{red_low}, {red_high}] "
            f"G=[{green_low}, {green_high}] "
            f"B=[{blue_low}, {blue_high}]"
        )
    elif method == "camera_emulation":
        logger.info(
            "Camera emulation: "
            f"R peak={r_peak} sigma={r_sigma}, "
            f"G peak={g_peak} sigma={g_sigma}, "
            f"B peak={b_peak} sigma={b_sigma}"
        )

    processed_frames = 0
    try:
        for batch in dataloader:
            remaining = target_frames - processed_frames
            if remaining <= 0:
                break

            batch_count = int(batch["cube"].shape[0])
            if batch_count > remaining:
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor) and value.shape[0] == batch_count:
                        batch[key] = value[:remaining]
                batch_count = remaining

            # Some CU3S sessions provide uint8 RAW cubes; cast to uint16 for CU3SDataNode IO contract.
            if batch["cube"].dtype != torch.uint16:
                batch["cube"] = batch["cube"].to(torch.uint16)

            pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)
            processed_frames += batch_count

            if processed_frames % 100 == 0 or processed_frames == target_frames:
                logger.info(f"Processed {processed_frames}/{target_frames} frames")
    finally:
        to_video.close()

    output_path = Path(output_video_path)
    if not output_path.exists():
        raise RuntimeError(f"Expected output video was not created: {output_video_path}")

    # Save pipeline config (YAML + .pt weights) alongside the video.
    pipeline_config_path = output_path.with_suffix(".yaml")
    pipeline.save_to_file(str(pipeline_config_path))
    logger.info(f"Pipeline config saved: {pipeline_config_path}")

    logger.success(f"Video export complete: {output_video_path}")
    return output_path


def export_compare_all(
    cu3s_file_path: str,
    output_dir: str,
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    max_num_frames: int = -1,
    batch_size: int = 1,
    processing_mode: str | cuvis.ProcessingMode = "Raw",
) -> dict[str, Path]:
    """Export false RGB videos for all methods into a comparison directory."""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    results = {}
    for method in SUPPORTED_METHODS:
        output_path = output_dir_path / f"{method}.mp4"
        logger.info(f"--- Exporting method: {method} ---")
        results[method] = export_false_rgb_video(
            cu3s_file_path=cu3s_file_path,
            output_video_path=str(output_path),
            method=method,
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            max_num_frames=max_num_frames,
            batch_size=batch_size,
            processing_mode=processing_mode,
        )

    logger.success(f"All methods exported to {output_dir_path}")
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export CU3S sequence to false-RGB MP4.",
    )
    parser.add_argument("--cu3s-file-path", type=str, required=True, help="Path to .cu3s file")
    parser.add_argument(
        "--output-video-path",
        type=str,
        default=None,
        help="Target MP4 path (single method mode)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="range_average",
        choices=SUPPORTED_METHODS,
        help=f"False RGB method: {', '.join(SUPPORTED_METHODS)} (default: range_average)",
    )
    parser.add_argument(
        "--compare-all",
        type=str,
        default=None,
        metavar="OUTPUT_DIR",
        help="Export all methods to this directory for comparison",
    )
    parser.add_argument(
        "--frame-rate",
        type=float,
        default=None,
        help="Output FPS (default: use session FPS, fallback 10.0 if unavailable)",
    )
    parser.add_argument(
        "--frame-rotation",
        type=int,
        default=None,
        help="Optional rotation in degrees; +90=anticlockwise, -90=clockwise",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Dataloader batch size")
    parser.add_argument(
        "--max-num-frames",
        type=int,
        default=-1,
        help="Maximum frames to write (-1 = all frames)",
    )
    parser.add_argument(
        "--processing-mode",
        type=str,
        default="Raw",
        help="CUVIS processing mode (Raw, DarkSubtract, Preview, Reflectance, SpectralRadiance)",
    )
    # range_average-specific args
    parser.add_argument("--red-low", type=float, default=580.0)
    parser.add_argument("--red-high", type=float, default=650.0)
    parser.add_argument("--green-low", type=float, default=500.0)
    parser.add_argument("--green-high", type=float, default=580.0)
    parser.add_argument("--blue-low", type=float, default=420.0)
    parser.add_argument("--blue-high", type=float, default=500.0)
    # camera_emulation-specific args
    parser.add_argument("--r-peak", type=float, default=610.0, help="Red peak wavelength (nm)")
    parser.add_argument("--g-peak", type=float, default=540.0, help="Green peak wavelength (nm)")
    parser.add_argument("--b-peak", type=float, default=460.0, help="Blue peak wavelength (nm)")
    parser.add_argument("--r-sigma", type=float, default=40.0, help="Red Gaussian sigma (nm)")
    parser.add_argument("--g-sigma", type=float, default=35.0, help="Green Gaussian sigma (nm)")
    parser.add_argument("--b-sigma", type=float, default=30.0, help="Blue Gaussian sigma (nm)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.compare_all:
        export_compare_all(
            cu3s_file_path=args.cu3s_file_path,
            output_dir=args.compare_all,
            frame_rate=args.frame_rate,
            frame_rotation=args.frame_rotation,
            max_num_frames=args.max_num_frames,
            batch_size=args.batch_size,
            processing_mode=args.processing_mode,
        )
    else:
        if not args.output_video_path:
            raise ValueError("--output-video-path is required when not using --compare-all")
        export_false_rgb_video(
            cu3s_file_path=args.cu3s_file_path,
            output_video_path=args.output_video_path,
            method=args.method,
            frame_rate=args.frame_rate,
            frame_rotation=args.frame_rotation,
            max_num_frames=args.max_num_frames,
            batch_size=args.batch_size,
            processing_mode=args.processing_mode,
            red_low=args.red_low,
            red_high=args.red_high,
            green_low=args.green_low,
            green_high=args.green_high,
            blue_low=args.blue_low,
            blue_high=args.blue_high,
            r_peak=args.r_peak,
            g_peak=args.g_peak,
            b_peak=args.b_peak,
            r_sigma=args.r_sigma,
            g_sigma=args.g_sigma,
            b_sigma=args.b_sigma,
        )


if __name__ == "__main__":
    main()
