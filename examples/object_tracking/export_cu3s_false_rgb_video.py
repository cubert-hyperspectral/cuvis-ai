"""Export a CU3S sequence to false-RGB MP4 using CIE tristimulus conversion.

Pipeline:
1. CU3SDataNode: normalizes CU3S tensors and extracts wavelengths
2. CIETristimulusFalseRGBSelector: spectral -> 3-channel conversion
3. ToVideoNode: stream-write RGB frames to MP4 (optionally overlays frame ID)

Examples
--------
CLI — single method export (CIE tristimulus, SpectralRadiance mode)::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-file-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-video-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.mp4" `
        --method cie_tristimulus `
        --processing-mode SpectralRadiance

CLI — with frame ID overlay::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-file-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-video-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.mp4" `
        --method cie_tristimulus `
        --overlay-frame-id

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
        overlay_frame_id=True,
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from loguru import logger

from cuvis_ai.node.channel_selector import (
    ChannelSelectorBase,
    CIETristimulusFalseRGBSelector,
    NormMode,
)
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.video import ToVideoNode

SUPPORTED_METHODS = ("cie_tristimulus",)
PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")
NORMALIZATION_MODES = ("sampled_fixed", "running", "per_frame")


def _resolve_processing_mode(processing_mode: str) -> str:
    """Resolve CLI/user input into a canonical processing mode string."""
    lookup = {name.lower(): name for name in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing_mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


def _resolve_normalization_mode(normalization_mode: str) -> str:
    """Resolve CLI/user input into a canonical normalization mode string."""
    lookup = {name.lower(): name for name in NORMALIZATION_MODES}
    resolved = lookup.get(normalization_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            "Invalid normalization_mode "
            f"'{normalization_mode}'. Supported: {', '.join(NORMALIZATION_MODES)}"
        )
    return resolved


def _validate_sample_fraction(sample_fraction: float) -> float:
    """Validate the sampled-fixed calibration fraction."""
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError(f"sample_fraction must be in (0, 1], got {sample_fraction}")
    return float(sample_fraction)


def _uniform_sample_positions(total_frames: int, sample_fraction: float) -> list[int]:
    """Return deterministic, uniformly spaced frame positions in [0, total_frames)."""
    if total_frames <= 0:
        raise ValueError("total_frames must be > 0")
    fraction = _validate_sample_fraction(sample_fraction)
    sample_count = max(1, int(math.ceil(total_frames * fraction)))
    if sample_count >= total_frames:
        return list(range(total_frames))
    if sample_count == 1:
        return [0]
    # Even spacing across the full range, including both ends.
    return [int((i * (total_frames - 1)) // (sample_count - 1)) for i in range(sample_count)]


def _build_statistical_sample_stream(
    predict_ds: Any,
    sample_positions: list[int],
) -> Any:
    """Yield sampled BHWC cubes and wavelengths for selector statistical initialization."""
    for pos in sample_positions:
        sample = predict_ds[pos]
        cube_raw = sample["cube"]
        if isinstance(cube_raw, torch.Tensor):
            cube_t = cube_raw.to(dtype=torch.float32)
        else:
            cube_t = torch.from_numpy(np.asarray(cube_raw)).to(dtype=torch.float32)
        if cube_t.ndim != 3:
            raise ValueError(
                f"Expected sampled cube with shape [H, W, C], got {tuple(cube_t.shape)}"
            )

        wavelengths_raw = sample["wavelengths"]
        if isinstance(wavelengths_raw, torch.Tensor):
            wavelengths_np = wavelengths_raw.detach().cpu().numpy().ravel()
        else:
            wavelengths_np = np.asarray(wavelengths_raw).ravel()

        yield {
            "cube": cube_t.unsqueeze(0),  # [1, H, W, C]
            "wavelengths": wavelengths_np,
        }


def _resolve_selector_norm_mode(normalization_mode: str) -> NormMode:
    """Map export normalization mode to selector norm mode."""
    if normalization_mode == "sampled_fixed":
        return NormMode.STATISTICAL
    if normalization_mode == "running":
        return NormMode.RUNNING
    return NormMode.PER_FRAME


def _create_false_rgb_node(
    method: str,
    *,
    norm_mode: str | NormMode = NormMode.RUNNING,
    freeze_running_bounds_after_frames: int | None = 20,
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
) -> ChannelSelectorBase:
    """Create the CIE tristimulus false RGB node."""
    if method != "cie_tristimulus":
        raise click.BadParameter(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")
    return CIETristimulusFalseRGBSelector(
        norm_mode=norm_mode,
        freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
        name="cie_tristimulus_false_rgb",
    )


def export_false_rgb_video(
    cu3s_file_path: str,
    output_video_path: str,
    method: str = "cie_tristimulus",
    frame_rate: float | None = None,
    frame_rotation: int | None = None,
    max_num_frames: int = -1,
    batch_size: int = 1,
    processing_mode: str = "Raw",
    normalization_mode: str = "sampled_fixed",
    sample_fraction: float = 0.05,
    freeze_running_bounds_after_frames: int | None = 20,
    save_pipeline_config: bool = False,
    overlay_frame_id: bool = False,
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
    resolved_norm_mode = _resolve_normalization_mode(normalization_mode)
    resolved_sample_fraction = _validate_sample_fraction(sample_fraction)

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
    selector_norm_mode = _resolve_selector_norm_mode(resolved_norm_mode)
    false_rgb = _create_false_rgb_node(
        method,
        norm_mode=selector_norm_mode,
        freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
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

    sampled_positions: list[int] = []
    sampled_mesu_ids: list[int] = []
    if resolved_norm_mode == "sampled_fixed":
        sampled_positions = _uniform_sample_positions(target_frames, resolved_sample_fraction)
        sampled_mesu_ids = [
            int(datamodule.predict_ds.measurement_indices[pos]) for pos in sampled_positions
        ]
        sample_stream = _build_statistical_sample_stream(datamodule.predict_ds, sampled_positions)
        false_rgb.statistical_initialization(sample_stream)

    to_video = ToVideoNode(
        output_video_path=output_video_path,
        frame_rate=resolved_frame_rate,
        frame_rotation=frame_rotation,
        name="to_video",
    )

    connections = [
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        (false_rgb.rgb_image, to_video.rgb_image),
    ]
    if overlay_frame_id:
        connections.append((cu3s_data.outputs.mesu_index, to_video.frame_id))
    pipeline.connect(*connections)

    pipeline_png = Path(output_video_path).parent / f"{pipeline.name}.png"
    pipeline.visualize(
        format="render_graphviz", output_path=str(pipeline_png), show_execution_stage=True
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
        f"normalization_mode={resolved_norm_mode}, "
        f"sample_fraction={resolved_sample_fraction}, "
        f"freeze_running_bounds_after_frames={freeze_running_bounds_after_frames}, "
        f"save_pipeline_config={save_pipeline_config}, "
        f"max_num_frames={max_num_frames}, "
        f"overlay_frame_id={overlay_frame_id}"
    )
    if resolved_norm_mode == "sampled_fixed":
        logger.info(
            "Sampled-fixed calibration: "
            f"sample_count={len(sampled_positions)}, "
            f"sample_pos_span={sampled_positions[0]}..{sampled_positions[-1]}, "
            f"sample_mesu_span={sampled_mesu_ids[0]}..{sampled_mesu_ids[-1]}"
        )
    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=None, collect_outputs=False)

    output_path = Path(output_video_path)
    if not output_path.exists():
        raise RuntimeError(f"Expected output video was not created: {output_video_path}")

    if save_pipeline_config:
        # Save pipeline config (YAML + .pt weights) alongside the video.
        pipeline_config_path = output_path.with_suffix(".yaml")
        pipeline.save_to_file(str(pipeline_config_path))
        logger.info(f"Pipeline config saved: {pipeline_config_path}")
    else:
        logger.info("Skipping pipeline config save (--no-save-pipeline-config).")

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
    normalization_mode: str = "sampled_fixed",
    sample_fraction: float = 0.05,
    freeze_running_bounds_after_frames: int | None = 20,
    save_pipeline_config: bool = False,
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
            normalization_mode=normalization_mode,
            sample_fraction=sample_fraction,
            freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
            save_pipeline_config=save_pipeline_config,
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
    default="cie_tristimulus",
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
    "--overlay-frame-id",
    is_flag=True,
    default=False,
    help="Render the measurement index (frame ID) as text in the top-left corner of each frame.",
)
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
@click.option(
    "--normalization-mode",
    type=click.Choice(NORMALIZATION_MODES, case_sensitive=False),
    default="sampled_fixed",
    show_default=True,
    help="RGB normalization strategy for export.",
)
@click.option(
    "--sample-fraction",
    type=float,
    default=0.05,
    show_default=True,
    help="Fraction of frames used for sampled-fixed calibration (0,1].",
)
@click.option(
    "--freeze-running-bounds-after",
    type=int,
    default=20,
    show_default=True,
    help="Freeze running normalization bounds after N frames (<=0 disables freezing).",
)
@click.option(
    "--save-pipeline-config/--no-save-pipeline-config",
    default=False,
    show_default=True,
    help="Save pipeline config files (.yaml + .pt) next to the output video.",
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
    overlay_frame_id: bool,
    max_num_frames: int,
    processing_mode: str,
    normalization_mode: str,
    sample_fraction: float,
    freeze_running_bounds_after: int,
    save_pipeline_config: bool,
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
    try:
        _validate_sample_fraction(sample_fraction)
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="--sample-fraction") from exc

    freeze_running_bounds_after_frames = (
        None if freeze_running_bounds_after <= 0 else freeze_running_bounds_after
    )

    if compare_all:
        export_compare_all(
            cu3s_file_path=str(cu3s_file_path),
            output_dir=str(compare_all),
            frame_rate=frame_rate,
            frame_rotation=frame_rotation,
            max_num_frames=max_num_frames,
            batch_size=batch_size,
            processing_mode=processing_mode,
            normalization_mode=normalization_mode,
            sample_fraction=sample_fraction,
            freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
            save_pipeline_config=save_pipeline_config,
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
            normalization_mode=normalization_mode,
            sample_fraction=sample_fraction,
            freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
            save_pipeline_config=save_pipeline_config,
            overlay_frame_id=overlay_frame_id,
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
