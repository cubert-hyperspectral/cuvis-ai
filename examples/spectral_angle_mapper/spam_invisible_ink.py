"""SPAM invisible-ink highlighting pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

import click
import numpy as np
import torch
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from loguru import logger

from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.anomaly_visualization import MaskOverlayNode
from cuvis_ai.node.channel_selector import FastRGBSelector
from cuvis_ai.node.conversion import DecisionToMask, ScoreToLogit
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.numpy_reader import NpyReader
from cuvis_ai.node.preprocessors import BandpassByWavelength
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai.node.video import ToVideoNode

PROCESSING_MODE = "SpectralRadiance"


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _read_xml_inputs(xml_path: Path) -> dict[str, str]:
    root = ET.parse(xml_path).getroot()
    values: dict[str, str] = {}
    for node in root.iter():
        if _local_name(node.tag) != "input":
            continue
        key = (node.attrib.get("id") or "").strip()
        if not key:
            continue
        values[key] = (node.text or "").strip()
    return values


def _parse_float(raw: str, *, label: str) -> float:
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise click.ClickException(f"{label} must be numeric, got '{raw}'") from exc


def _parse_reference_spectrum(raw: str) -> np.ndarray:
    parts = [piece.strip() for piece in raw.split(";")]
    values = [float(piece) for piece in parts if piece]
    if not values:
        raise click.ClickException("ReferenceSpectrum is empty in SAM XML")
    return np.asarray([values], dtype=np.float64)


def _extract_fast_rgb_from_inputs(
    input_values: dict[str, str], *, xml_path: Path
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], float]:
    for key in ("RedWL", "GreenWL", "BlueWL", "Width", "Normalize"):
        if key not in input_values:
            raise click.ClickException(f"Missing '{key}' in XML: {xml_path}")

    red_wl = _parse_float(input_values["RedWL"], label=f"{xml_path.name}:RedWL")
    green_wl = _parse_float(input_values["GreenWL"], label=f"{xml_path.name}:GreenWL")
    blue_wl = _parse_float(input_values["BlueWL"], label=f"{xml_path.name}:BlueWL")
    width = _parse_float(input_values["Width"], label=f"{xml_path.name}:Width")
    normalize = _parse_float(input_values["Normalize"], label=f"{xml_path.name}:Normalize")

    half = width / 2.0
    return (
        (red_wl - half, red_wl + half),
        (green_wl - half, green_wl + half),
        (blue_wl - half, blue_wl + half),
        normalize,
    )


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
    red_range: tuple[float, float],
    green_range: tuple[float, float],
    blue_range: tuple[float, float],
    normalization_strength: float,
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
        "red_range": list(red_range),
        "green_range": list(green_range),
        "blue_range": list(blue_range),
        "normalization_strength": normalization_strength,
        "spectrum_length": int(spectrum.shape[1]),
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
    help="SPAM XML (SpectralRadiance) with SAM + primary FastRGB params",
)
@click.option(
    "--rgb-xml-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Auxiliary FastRGB XML for overlay background (e.g., 00_RGB.xml)",
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
)
def main(
    cu3s_path: Path,
    sam_xml_path: Path,
    rgb_xml_path: Path,
    reference_npy: Path | None,
    overlay_alpha: float,
    overlay_color: str,
    frame_rotation: int | None,
    end_frame: int,
    output_dir: Path,
) -> None:
    """Run SPAM pipeline and export overlay video."""
    if end_frame == 0 or end_frame < -1:
        raise click.BadParameter("--end-frame must be -1 or a positive integer.")
    if not (0.0 <= overlay_alpha <= 1.0):
        raise click.BadParameter("--overlay-alpha must be in [0, 1].")
    overlay_color_rgb = _parse_overlay_color(overlay_color)

    sam_inputs = _read_xml_inputs(sam_xml_path)
    spectrum, threshold, wl_min, wl_max = _extract_sam_from_inputs(
        sam_inputs, xml_path=sam_xml_path
    )
    if not (0.0 <= threshold <= 1.0):
        raise click.ClickException(f"SAM_Threshold must be in [0,1], got {threshold}")
    if wl_min > wl_max:
        raise click.ClickException(f"SAM_MinWL must be <= SAM_MaxWL, got {wl_min}>{wl_max}")

    aux_inputs = _read_xml_inputs(rgb_xml_path)
    aux_red, aux_green, aux_blue, aux_norm = _extract_fast_rgb_from_inputs(
        aux_inputs, xml_path=rgb_xml_path
    )

    predict_ids = list(range(end_frame)) if end_frame > 0 else None
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

    dataset_fps = float(getattr(datamodule.predict_ds, "fps", None) or 10.0)
    if dataset_fps <= 0:
        dataset_fps = 10.0
        logger.warning("Could not infer positive FPS from dataset; using fallback 10.0")

    wavelengths = _first_wavelengths_1d(datamodule)
    num_channels = _count_selected_channels(wavelengths, wl_min=wl_min, wl_max=wl_max)
    if num_channels <= 0:
        raise click.ClickException("Bandpass selected zero channels; adjust --wl-min/--wl-max.")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / "spam_result.mp4"
    resolved_reference_npy = (
        reference_npy
        if reference_npy is not None
        else _save_generated_reference_npy(
            output_dir=output_dir,
            sam_xml_path=sam_xml_path,
            spectrum=spectrum,
            red_range=aux_red,
            green_range=aux_green,
            blue_range=aux_blue,
            normalization_strength=aux_norm,
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
    fast_rgb_aux = FastRGBSelector(
        red_range=aux_red,
        green_range=aux_green,
        blue_range=aux_blue,
        normalization_strength=aux_norm,
        name="fast_rgb_aux",
    )

    spam = SpectralAngleMapper(num_channels=num_channels, name="spam")
    score_to_logit = ScoreToLogit(init_scale=-1.0, init_bias=1.0 - threshold, name="score_to_logit")
    decider = BinaryDecider(threshold=0.5, name="decider")
    to_mask = DecisionToMask(name="to_mask")
    overlay = MaskOverlayNode(alpha=overlay_alpha, overlay_color=overlay_color_rgb, name="overlay")
    to_video = ToVideoNode(
        output_video_path=str(output_video_path),
        frame_rate=dataset_fps,
        frame_rotation=frame_rotation,
        name="to_video",
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
        (cu3s_data.outputs.cube, fast_rgb_aux.cube),
        (cu3s_data.outputs.wavelengths, fast_rgb_aux.wavelengths),
        (fast_rgb_aux.rgb_image, overlay.rgb_image),
        (to_mask.mask, overlay.mask),
        (overlay.rgb_with_overlay, to_video.rgb_image),
    ]

    pipeline.connect(*connections)

    pipeline.visualize(
        format="render_graphviz",
        output_path=str(output_dir / f"{pipeline.name}.png"),
        show_execution_stage=True,
    )
    pipeline.visualize(
        format="render_mermaid",
        output_path=str(output_dir / f"{pipeline.name}.md"),
        show_execution_stage=True,
    )
    pipeline_yaml = output_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(str(pipeline_yaml))
    logger.info(
        "Pipeline config saved (YAML + weights): {}, {}",
        pipeline_yaml,
        pipeline_yaml.with_suffix(".pt"),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline.to(device)
    pipeline.set_profiling(enabled=True, synchronize_cuda=(device == "cuda"), skip_first_n=3)

    logger.info(
        (
            "Starting SPAM run: frames={}, mode={}, wl=[{}, {}], threshold={}, channels={}, "
            "device={}, sam_xml={}, rgb_xml={}, background={}, overlay_alpha={}, overlay_color={}"
        ),
        target_frames,
        PROCESSING_MODE,
        wl_min,
        wl_max,
        threshold,
        num_channels,
        device,
        sam_xml_path,
        rgb_xml_path,
        "aux",
        overlay_alpha,
        overlay_color_rgb,
    )

    predictor = Predictor(pipeline=pipeline, datamodule=datamodule)
    predictor.predict(max_batches=target_frames, collect_outputs=False)

    summary = pipeline.format_profiling_summary(total_frames=target_frames)
    (output_dir / "profiling_summary.txt").write_text(summary, encoding="utf-8")

    logger.success("SPAM run completed: {}", output_dir)
    logger.info("Overlay video: {}", output_video_path)


if __name__ == "__main__":
    main()
