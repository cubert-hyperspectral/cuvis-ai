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

Examples
--------
CLI — single method export (CIE tristimulus, SpectralRadiance mode)::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-file-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-video-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.mp4" `
        --method cie_tristimulus `
        --processing-mode SpectralRadiance

CLI — compare all methods side-by-side::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-file-path path/to/recording.cu3s `
        --compare-all path/to/output_dir

Python API::

    from examples.object_tracking.export_cu3s_false_rgb_video import export_false_rgb_video

    export_false_rgb_video(
        cu3s_file_path=r"D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s",
        output_video_path=r"D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.mp4",
        method="cie_tristimulus",
        processing_mode="SpectralRadiance",
    )
"""

from __future__ import annotations

from pathlib import Path

import click
import torch
from cuvis_ai.node.band_selection import (
    BandSelectorBase,
    BaselineFalseRGBSelector,
    CameraEmulationFalseRGBSelector,
    CIETristimulusFalseRGBSelector,
    RangeAverageFalseRGBSelector,
)
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from loguru import logger

from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.video import ToVideoNode

SUPPORTED_METHODS = ("range_average", "cie_tristimulus", "camera_emulation", "baseline")
PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")


def _resolve_processing_mode(processing_mode: str) -> str:
    """Resolve CLI/user input into a canonical processing mode string."""
    lookup = {name.lower(): name for name in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing_mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


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
        raise click.BadParameter(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")


def export_false_rgb_video(
    cu3s_file_path: str,
    output_video_path: str,
    method: str = "range_average",
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    max_num_frames: int = -1,
    batch_size: int = 1,
    processing_mode: str = "Raw",
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
    resolved_mode = _resolve_processing_mode(processing_mode)

    predict_ids = list(range(max_num_frames)) if max_num_frames > 0 else None
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=cu3s_file_path,
        processing_mode=resolved_mode,
        batch_size=batch_size,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")

    if datamodule.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise ValueError("No frames available. Check max_num_frames or the CU3S file.")

    dataset_fps = getattr(datamodule.predict_ds, "fps", None)
    if frame_rate is not None and frame_rate > 0:
        resolved_frame_rate = float(frame_rate)
    elif dataset_fps is not None and dataset_fps > 0:
        resolved_frame_rate = float(dataset_fps)
    else:
        resolved_frame_rate = 10.0
        logger.warning("Could not determine FPS from session metadata; falling back to 10.0 FPS.")

    pipeline = CuvisPipeline("SAM3_FalseRGB_Export")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        output_video_path=output_video_path,
        frame_rate=resolved_frame_rate,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    pipeline.connect(
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, to_video.rgb_image),
    )
    pipeline.to(device)

    logger.info(
        f"Starting export of {target_frames} frames from {cu3s_file_path} [device={device}]"
    )
    logger.info(
        "Video settings: "
        f"method={method}, "
        f"frame_rate={resolved_frame_rate}, "
        f"dataset_fps={dataset_fps}, "
        f"frame_rotation={frame_rotation}, "
        f"processing_mode={resolved_mode}, "
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

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=None, collect_outputs=False)

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
    processing_mode: str = "Raw",
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


@click.command()
@click.option(
    "--cu3s-file-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to .cu3s file.",
)
@click.option(
    "--output-video-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Target MP4 path (single method mode).",
)
@click.option(
    "--method",
    type=click.Choice(SUPPORTED_METHODS, case_sensitive=False),
    default="range_average",
    show_default=True,
)
@click.option(
    "--compare-all",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Export all methods to this directory for comparison.",
)
@click.option(
    "--frame-rate",
    type=float,
    default=None,
    help="Output FPS (default: use session FPS, fallback 10.0).",
)
@click.option(
    "--frame-rotation",
    type=int,
    default=None,
    help="Rotation in degrees; +90=anticlockwise, -90=clockwise.",
)
@click.option("--batch-size", type=int, default=1, show_default=True)
@click.option(
    "--max-num-frames",
    type=int,
    default=-1,
    show_default=True,
    help="Maximum frames to write (-1 = all frames).",
)
@click.option(
    "--processing-mode",
    type=click.Choice(PROCESSING_MODES, case_sensitive=False),
    default="Raw",
    show_default=True,
)
# range_average-specific options
@click.option("--red-low", type=float, default=580.0, show_default=True)
@click.option("--red-high", type=float, default=650.0, show_default=True)
@click.option("--green-low", type=float, default=500.0, show_default=True)
@click.option("--green-high", type=float, default=580.0, show_default=True)
@click.option("--blue-low", type=float, default=420.0, show_default=True)
@click.option("--blue-high", type=float, default=500.0, show_default=True)
# camera_emulation-specific options
@click.option(
    "--r-peak", type=float, default=610.0, show_default=True, help="Red peak wavelength (nm)."
)
@click.option(
    "--g-peak", type=float, default=540.0, show_default=True, help="Green peak wavelength (nm)."
)
@click.option(
    "--b-peak", type=float, default=460.0, show_default=True, help="Blue peak wavelength (nm)."
)
@click.option(
    "--r-sigma", type=float, default=40.0, show_default=True, help="Red Gaussian sigma (nm)."
)
@click.option(
    "--g-sigma", type=float, default=35.0, show_default=True, help="Green Gaussian sigma (nm)."
)
@click.option(
    "--b-sigma", type=float, default=30.0, show_default=True, help="Blue Gaussian sigma (nm)."
)
def main(
    cu3s_file_path: Path,
    output_video_path: Path | None,
    method: str,
    compare_all: Path | None,
    frame_rate: float | None,
    frame_rotation: int | None,
    batch_size: int,
    max_num_frames: int,
    processing_mode: str,
    red_low: float,
    red_high: float,
    green_low: float,
    green_high: float,
    blue_low: float,
    blue_high: float,
    r_peak: float,
    g_peak: float,
    b_peak: float,
    r_sigma: float,
    g_sigma: float,
    b_sigma: float,
) -> None:
    """Export CU3S sequence to false-RGB MP4."""
    if compare_all:
        export_compare_all(
            cu3s_file_path=str(cu3s_file_path),
            output_dir=str(compare_all),
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            max_num_frames=max_num_frames,
            batch_size=batch_size,
            processing_mode=processing_mode,
        )
    else:
        if not output_video_path:
            raise click.UsageError("--output-video-path is required when not using --compare-all")
        export_false_rgb_video(
            cu3s_file_path=str(cu3s_file_path),
            output_video_path=str(output_video_path),
            method=method,
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            max_num_frames=max_num_frames,
            batch_size=batch_size,
            processing_mode=processing_mode,
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


if __name__ == "__main__":
    main()
