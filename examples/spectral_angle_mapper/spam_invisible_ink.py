"""SPAM invisible-ink highlighting pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import torch
from loguru import logger

from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.anomaly_visualization import MaskOverlayNode
from cuvis_ai.node.channel_selector import CIETristimulusFalseRGBSelector, NormMode
from cuvis_ai.node.compositing import ROIZoomNode
from cuvis_ai.node.conversion import DecisionToMask, ScoreToLogit
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.mask_ops import MaskRobustifier, MaskToBBoxKalman
from cuvis_ai.node.numpy_file import NpyReader
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.spectral_extractor import MaskedMeanSpectrum
from cuvis_ai.node.spectrum_plot import SpectrumPlotNode
from cuvis_ai.node.video import ToVideoNode
from cuvis_ai.utils.cli_helpers import compute_real_fps_from_dataset, resolve_run_output_dir
from cuvis_ai.utils.false_rgb_sampling import initialize_false_rgb_sampled_fixed
from cuvis_ai.utils.xml_plugin_parser import parse_numeric_text, read_xml_inputs
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor

PROCESSING_MODE = "SpectralRadiance"
FALSE_RGB_SAMPLE_FRACTION = 0.05


def _parse_float(raw: str, *, label: str) -> float:
    try:
        return parse_numeric_text(raw, label=label)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


def _parse_reference_spectrum(raw: str) -> np.ndarray:
    parts = [piece.strip() for piece in raw.split(";")]
    values = [float(piece) for piece in parts if piece]
    if not values:
        raise click.ClickException("ReferenceSpectrum is empty in SAM XML")
    return np.asarray([values], dtype=np.float64)


def _extract_sam_from_inputs(
    input_values: dict[str, str], *, xml_path: Path
) -> tuple[np.ndarray, float, float, float]:
    for key in ("ReferenceSpectrum", "SAM_Threshold", "SAM_MinWL", "SAM_MaxWL"):
        if key not in input_values:
            raise click.ClickException(f"Missing '{key}' in SAM XML: {xml_path}")

    spectrum = _parse_reference_spectrum(input_values["ReferenceSpectrum"])
    threshold = _parse_float(input_values["SAM_Threshold"], label=f"{xml_path.name}:SAM_Threshold")
    wl_min = _parse_float(input_values["SAM_MinWL"], label=f"{xml_path.name}:SAM_MinWL")
    wl_max = _parse_float(input_values["SAM_MaxWL"], label=f"{xml_path.name}:SAM_MaxWL")
    return spectrum, threshold, wl_min, wl_max


def _save_generated_reference_npy(
    output_dir: Path,
    sam_xml_path: Path,
    spectrum: np.ndarray,
    threshold: float,
    wl_min: float,
    wl_max: float,
) -> Path:
    npy_path = output_dir / f"{sam_xml_path.stem}.npy"
    cfg_path = output_dir / f"{sam_xml_path.stem}_config.json"

    np.save(npy_path, spectrum)
    cfg = {
        "threshold": threshold,
        "wl_min": wl_min,
        "wl_max": wl_max,
        "spectrum_length": int(spectrum.shape[1]),
        "background_method": "cie_tristimulus",
        "normalization_mode": "sampled_fixed",
        "sample_fraction": FALSE_RGB_SAMPLE_FRACTION,
    }
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return npy_path


def _first_wavelengths_1d(datamodule: SingleCu3sDataModule) -> np.ndarray:
    if datamodule.predict_ds is None or len(datamodule.predict_ds) == 0:
        raise RuntimeError("Predict dataset is empty")

    sample = datamodule.predict_ds[0]
    raw = sample["wavelengths"]
    if isinstance(raw, torch.Tensor):
        return raw.detach().cpu().numpy().ravel()
    return np.asarray(raw).ravel()


def _count_selected_channels(wavelengths: np.ndarray, wl_min: float, wl_max: float) -> int:
    keep = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    return int(keep.sum())


def _parse_overlay_color(raw: str) -> tuple[float, float, float]:
    parts = [piece.strip() for piece in raw.split(",")]
    if len(parts) != 3:
        raise click.BadParameter(
            "--overlay-color must contain exactly 3 comma-separated values: R,G,B."
        )

    try:
        channels = tuple(float(piece) for piece in parts)
    except ValueError as exc:
        raise click.BadParameter(f"--overlay-color values must be numeric, got: {raw!r}") from exc

    if all(0.0 <= channel <= 1.0 for channel in channels):
        return channels
    if all(0.0 <= channel <= 255.0 for channel in channels):
        return tuple(channel / 255.0 for channel in channels)

    raise click.BadParameter(
        "--overlay-color must be either all values in [0,1] or all values in [0,255]."
    )


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input CU3S file",
)
@click.option(
    "--sam-xml-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="SPAM XML (SpectralRadiance) with SAM parameters",
)
@click.option(
    "--no-ink-xml",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help=(
        "XML describing the 'no ink' reference spectrum to display on the "
        "signature plot (legend: 'hoodie without ink'). Separate from --sam-xml-path, "
        "which drives detection."
    ),
)
@click.option(
    "--reference-npy",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Optional pre-generated reference spectrum .npy (default: generated from --sam-xml-path)",
)
@click.option("--overlay-alpha", type=float, default=1.0, show_default=True)
@click.option(
    "--overlay-color",
    type=str,
    default="255,0,0",
    show_default=True,
    help="Overlay color as R,G,B in either [0,255] or [0,1].",
)
@click.option("--frame-rotation", type=int, default=None)
@click.option(
    "--start-frame",
    type=int,
    default=0,
    show_default=True,
    help="First frame index, inclusive. Requires --end-frame when non-zero.",
)
@click.option(
    "--end-frame",
    type=int,
    default=-1,
    show_default=True,
    help="Last frame index exclusive (-1 = all)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("D:/experiments/20260323/spam_ink_highlight"),
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
    help=(
        "Optional leaf run-folder name under --output-dir (must not include '/' or '\\'). "
        "Pass '.' to write artifacts directly into --output-dir."
    ),
)
@click.option(
    "--save-profiling-summary/--no-save-profiling-summary",
    default=False,
    show_default=True,
    help=(
        "When enabled, profile each pipeline stage and write profiling_summary.txt "
        "into the run directory. Off by default to keep runs quiet and avoid the "
        "per-frame CUDA sync overhead."
    ),
)
@click.option(
    "--save-pipeline-config/--no-save-pipeline-config",
    default=True,
    show_default=True,
    help=(
        "When enabled, write the pipeline config YAML (<pipeline>.yaml) into the run "
        "directory. The .pt weights sidecar is never written."
    ),
)
@click.option(
    "--sam-threshold",
    type=float,
    default=None,
    help="Override SAM threshold from XML. Must be in [0.0, 1.0].",
)
@click.option(
    "--zoom-size",
    type=int,
    default=320,
    show_default=True,
    help="Edge length (px) of the square ROI zoom video frame.",
)
@click.option(
    "--robust-min-area",
    type=int,
    default=10,
    show_default=True,
    help="Drop mask components smaller than this many pixels.",
)
@click.option(
    "--robust-opening",
    type=int,
    default=0,
    show_default=True,
    help="Side length (px) of the morphological opening kernel. 0 or 1 disables.",
)
@click.option(
    "--kalman-max-predict",
    type=int,
    default=20,
    show_default=True,
    help="Maximum consecutive empty frames to predict the ROI via Kalman before dropping the track.",
)
@click.option(
    "--kalman-min-hits",
    type=int,
    default=3,
    show_default=True,
    help="Consecutive measurement frames required to confirm an ROI track (suppresses transient false positives).",
)
@click.option(
    "--plot-width",
    type=int,
    default=960,
    show_default=True,
    help="Pixel width of the signature plot video.",
)
@click.option(
    "--plot-height",
    type=int,
    default=720,
    show_default=True,
    help="Pixel height of the signature plot video.",
)
@click.option(
    "--plot-dpi",
    type=int,
    default=150,
    show_default=True,
    help="Matplotlib DPI for the signature plot figure.",
)
@click.option(
    "--plot-tracked-hold",
    type=int,
    default=15,
    show_default=True,
    help="Frames to keep the last tracked spectrum visible after the mask goes empty.",
)
def main(
    cu3s_path: Path,
    sam_xml_path: Path,
    no_ink_xml: Path,
    reference_npy: Path | None,
    overlay_alpha: float,
    overlay_color: str,
    frame_rotation: int | None,
    start_frame: int,
    end_frame: int,
    output_dir: Path,
    out_basename: str | None,
    save_profiling_summary: bool,
    save_pipeline_config: bool,
    sam_threshold: float | None,
    zoom_size: int,
    robust_min_area: int,
    robust_opening: int,
    kalman_max_predict: int,
    kalman_min_hits: int,
    plot_width: int,
    plot_height: int,
    plot_dpi: int,
    plot_tracked_hold: int,
) -> None:
    """Run SPAM pipeline and export overlay video."""
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if start_frame < 0:
        raise click.BadParameter("--start-frame must be >= 0.")
    if start_frame > 0 and end_frame == -1:
        raise click.BadParameter("--start-frame requires --end-frame to be set (> start).")
    if end_frame != -1 and end_frame <= start_frame:
        raise click.BadParameter("--end-frame must be greater than --start-frame.")
    if not (0.0 <= overlay_alpha <= 1.0):
        raise click.BadParameter("--overlay-alpha must be in [0, 1].")
    overlay_color_rgb = _parse_overlay_color(overlay_color)

    sam_inputs = read_xml_inputs(sam_xml_path)
    spectrum, threshold, wl_min, wl_max = _extract_sam_from_inputs(
        sam_inputs, xml_path=sam_xml_path
    )
    if sam_threshold is not None:
        if not (0.0 <= sam_threshold <= 1.0):
            raise click.BadParameter("--sam-threshold must be in [0.0, 1.0].")
        logger.info(
            "Overriding XML SAM threshold ({}) with CLI value ({})", threshold, sam_threshold
        )
        threshold = sam_threshold
    if not (0.0 <= threshold <= 1.0):
        raise click.ClickException(f"SAM_Threshold must be in [0,1], got {threshold}")
    if wl_min > wl_max:
        raise click.ClickException(f"SAM_MinWL must be <= SAM_MaxWL, got {wl_min}>{wl_max}")

    predict_ids = list(range(start_frame, end_frame)) if end_frame > 0 else None
    datamodule = SingleCu3sDataModule(
        cu3s_file_path=str(cu3s_path),
        processing_mode=PROCESSING_MODE,
        batch_size=1,
        predict_ids=predict_ids,
    )
    datamodule.setup(stage="predict")
    if datamodule.predict_ds is None:
        raise RuntimeError("Predict dataset was not initialized.")

    target_frames = len(datamodule.predict_ds)
    if target_frames <= 0:
        raise click.ClickException("No frames available for prediction.")

    # FPS precedence: real fps from capture_time span > nominal session.fps > 10.0.
    # session.fps is nominal and often wildly off real wall-clock cadence for
    # spectral cameras, causing exports to play in fast-forward.
    session_fps = getattr(datamodule.predict_ds, "fps", None)
    real_fps = compute_real_fps_from_dataset(datamodule.predict_ds)
    if real_fps is not None and real_fps > 0:
        dataset_fps = float(real_fps)
        logger.info(
            "FPS source: measurement.capture_time span = {:.3f} (session.fps nominal = {})",
            dataset_fps,
            session_fps,
        )
    elif session_fps is not None and float(session_fps) > 0:
        dataset_fps = float(session_fps)
        logger.info("FPS source: session.fps nominal = {:.3f}", dataset_fps)
    else:
        dataset_fps = 10.0
        logger.warning("Could not infer FPS from dataset; using fallback 10.0")

    wavelengths = _first_wavelengths_1d(datamodule)
    num_channels = _count_selected_channels(wavelengths, wl_min=wl_min, wl_max=wl_max)
    if num_channels <= 0:
        raise click.ClickException("Bandpass selected zero channels; adjust --wl-min/--wl-max.")

    run_output_dir = resolve_run_output_dir(
        output_root=output_dir,
        source_path=cu3s_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = run_output_dir / f"{cu3s_path.stem}.mp4"
    zoom_video_path = run_output_dir / f"{cu3s_path.stem}_zoom.mp4"
    spectrum_video_path = run_output_dir / f"{cu3s_path.stem}_spectrum.mp4"
    log_file_path = run_output_dir / f"{cu3s_path.stem}.log"
    logger.add(str(log_file_path), level="INFO", mode="w")
    logger.info("Output run directory: {}", run_output_dir)
    logger.info("Log file: {}", log_file_path)

    resolved_reference_npy = (
        reference_npy
        if reference_npy is not None
        else _save_generated_reference_npy(
            output_dir=run_output_dir,
            sam_xml_path=sam_xml_path,
            spectrum=spectrum,
            threshold=threshold,
            wl_min=wl_min,
            wl_max=wl_max,
        )
    )

    pipeline = CuvisPipeline("SPAM_Invisible_Ink")
    cu3s_data = CU3SDataNode(name="cu3s_data")
    ref_spectrum = NpyReader(file_path=str(resolved_reference_npy), name="ref_spectrum")
    cube_bandpass = BandpassByWavelength(
        min_wavelength_nm=wl_min,
        max_wavelength_nm=wl_max,
        name="cube_bandpass",
    )
    sig_bandpass = BandpassByWavelength(
        min_wavelength_nm=wl_min,
        max_wavelength_nm=wl_max,
        name="sig_bandpass",
    )
    false_rgb = CIETristimulusFalseRGBSelector(
        norm_mode=NormMode.STATISTICAL,
        name="cie_tristimulus_false_rgb",
    )
    initialize_false_rgb_sampled_fixed(
        false_rgb, datamodule.predict_ds, sample_fraction=FALSE_RGB_SAMPLE_FRACTION
    )

    spam = SpectralAngleMapper(num_channels=num_channels, name="spam")
    score_to_logit = ScoreToLogit(init_scale=-1.0, init_bias=1.0 - threshold, name="score_to_logit")
    decider = BinaryDecider(threshold=0.5, name="decider")
    to_mask = DecisionToMask(name="to_mask")
    robust = MaskRobustifier(
        opening_kernel=robust_opening,
        closing_kernel=robust_opening,
        min_area=robust_min_area,
        keep_largest=True,
        name="robust_mask",
    )
    overlay = MaskOverlayNode(alpha=overlay_alpha, overlay_color=overlay_color_rgb, name="overlay")
    bbox = MaskToBBoxKalman(
        max_predict_frames=kalman_max_predict,
        min_hits=kalman_min_hits,
        name="roi_kalman",
    )
    roi_zoom = ROIZoomNode(
        zoom_height=zoom_size,
        zoom_width=zoom_size,
        name="roi_zoom",
    )
    to_video_main = ToVideoNode(
        output_video_path=str(output_video_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video_main",
    )
    to_video_zoom = ToVideoNode(
        output_video_path=str(zoom_video_path),
        frame_rate=dataset_fps,
        name="to_video_zoom",
    )
    to_video_spectrum = ToVideoNode(
        output_video_path=str(spectrum_video_path),
        frame_rate=dataset_fps,
        name="to_video_spectrum",
    )

    # Reference / tracked wavelengths for the signature plot.
    bp_keep = (wavelengths >= wl_min) & (wavelengths <= wl_max)
    full_wavelengths = np.asarray(wavelengths, dtype=np.float32)
    bp_wavelengths = full_wavelengths[bp_keep]

    # Separate "no-ink" reference for the plot legend (decoupled from the SAM
    # detection reference). Parsed once at startup; same XML schema as the SAM
    # plugin, so existing helpers apply.
    no_ink_inputs = read_xml_inputs(no_ink_xml)
    if "ReferenceSpectrum" not in no_ink_inputs:
        raise click.ClickException(f"Missing 'ReferenceSpectrum' in no-ink XML: {no_ink_xml}")
    no_ink_spectrum_array = _parse_reference_spectrum(no_ink_inputs["ReferenceSpectrum"])
    if no_ink_spectrum_array.shape[1] != bp_wavelengths.size:
        raise click.ClickException(
            f"No-ink ReferenceSpectrum length {no_ink_spectrum_array.shape[1]} "
            f"does not match plot reference wavelengths ({bp_wavelengths.size}). "
            f"Check that {no_ink_xml} uses the same wavelength grid as the camera."
        )
    no_ink_npy_path = run_output_dir / f"{no_ink_xml.stem}.npy"
    np.save(no_ink_npy_path, no_ink_spectrum_array)

    no_ink_reader = NpyReader(file_path=str(no_ink_npy_path), name="no_ink_spectrum")
    mean_spec = MaskedMeanSpectrum(name="masked_mean_spectrum")
    plot = SpectrumPlotNode(
        wavelengths=full_wavelengths,
        reference_wavelengths=bp_wavelengths,
        plot_width=plot_width,
        plot_height=plot_height,
        dpi=plot_dpi,
        tracked_label="hoodie inked (tracked)",
        reference_label="hoodie without ink",
        tracked_hold_frames=plot_tracked_hold,
        y_fixed_range=(0.0, 15.0),
        y_num_ticks=15,
        name="spectrum_plot",
    )

    connections: list[tuple[object, object]] = [
        (cu3s_data.outputs.cube, cube_bandpass.data),
        (cu3s_data.outputs.wavelengths, cube_bandpass.wavelengths),
        (cu3s_data.outputs.mesu_index, ref_spectrum.frame_id),
        (ref_spectrum.data, sig_bandpass.data),
        (cu3s_data.outputs.wavelengths, sig_bandpass.wavelengths),
        (cube_bandpass.filtered, spam.cube),
        (sig_bandpass.filtered, spam.spectral_signature),
        (spam.best_scores, score_to_logit.scores),
        (score_to_logit.logits, decider.logits),
        (decider.decisions, to_mask.decisions),
        (spam.identity_mask, to_mask.identity_mask),
        (cu3s_data.outputs.cube, false_rgb.cube),
        (cu3s_data.outputs.wavelengths, false_rgb.wavelengths),
        # Robustify mask once, feed the cleaned version everywhere downstream.
        (to_mask.mask, robust.inputs.mask),
        (false_rgb.rgb_image, overlay.rgb_image),
        (robust.outputs.mask, overlay.mask),
        # ROI tracking + zoom (emitted as its own video stream).
        (robust.outputs.mask, bbox.mask),
        (overlay.rgb_with_overlay, roi_zoom.source),
        (bbox.bbox, roi_zoom.bbox),
        (bbox.valid, roi_zoom.valid),
        # Masked-mean spectrum → plot (emitted as its own video stream).
        (cu3s_data.outputs.cube, mean_spec.cube),
        (robust.outputs.mask, mean_spec.mask),
        (mean_spec.mean_spectrum, plot.tracked_spectrum),
        (cu3s_data.outputs.mesu_index, no_ink_reader.frame_id),
        (no_ink_reader.data, plot.reference_spectrum),
        (mean_spec.valid, plot.valid),
        (cu3s_data.outputs.mesu_index, plot.frame_id),
        # Three independent video outputs — fuse externally.
        (overlay.rgb_with_overlay, to_video_main.rgb_image),
        (roi_zoom.zoom, to_video_zoom.rgb_image),
        (plot.rgb_image, to_video_spectrum.rgb_image),
    ]

    pipeline.connect(*connections)

    pipeline.visualize(
        format="render_mermaid",
        output_path=str(run_output_dir / f"{pipeline.name}.md"),
        show_execution_stage=True,
    )

    if save_pipeline_config:
        pipeline_yaml = run_output_dir / f"{pipeline.name}.yaml"
        pipeline.save_to_file(str(pipeline_yaml), save_weights=False)
        logger.info("Pipeline config YAML: {}", pipeline_yaml)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    if save_profiling_summary:
        pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)

    logger.info(
        (
            "Starting SPAM run: frames={}, mode={}, wl=[{}, {}], threshold={}, channels={}, "
            "device={}, sam_xml={}, background={}, normalization_mode={}, sample_fraction={}, "
            "overlay_alpha={}, overlay_color={}"
        ),
        target_frames,
        PROCESSING_MODE,
        wl_min,
        wl_max,
        threshold,
        num_channels,
        device,
        sam_xml_path,
        "cie_tristimulus",
        "sampled_fixed",
        FALSE_RGB_SAMPLE_FRACTION,
        overlay_alpha,
        overlay_color_rgb,
    )

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=target_frames, collect_outputs=False)

    if save_profiling_summary:
        summary = pipeline.format_profiling_summary(total_frames=target_frames)
        (run_output_dir / "profiling_summary.txt").write_text(summary, encoding="utf-8")
        logger.info("Profiling summary: {}", run_output_dir / "profiling_summary.txt")

    logger.success("SPAM run completed: {}", run_output_dir)
    logger.info("Main overlay video: {}", output_video_path)
    logger.info("Zoom video: {}", zoom_video_path)
    logger.info("Spectrum video: {}", spectrum_video_path)


if __name__ == "__main__":
    main()
