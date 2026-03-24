"""Extract SPAM reference spectrum and config from a cuvis user-plugin XML."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import click
import numpy as np


def _local_name(tag: str) -> str:
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_float(value: str, *, label: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError(f"{label} is empty")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric, got '{text}'") from exc


def _read_inputs(root: ET.Element) -> dict[str, str]:
    values: dict[str, str] = {}
    for node in root.iter():
        if _local_name(node.tag) != "input":
            continue
        key = (node.attrib.get("id") or "").strip()
        if not key:
            continue
        values[key] = (node.text or "").strip()
    return values


def _parse_reference_spectrum(raw: str) -> np.ndarray:
    parts = [segment.strip() for segment in raw.split(";")]
    values = [float(part) for part in parts if part]
    if not values:
        raise ValueError("ReferenceSpectrum contains no values")
    return np.asarray([values], dtype=np.float64)


def _extract_config(input_values: dict[str, str]) -> tuple[np.ndarray, dict[str, Any]]:
    missing = [
        key
        for key in (
            "ReferenceSpectrum",
            "SAM_Threshold",
            "SAM_MinWL",
            "SAM_MaxWL",
            "RedWL",
            "GreenWL",
            "BlueWL",
            "Width",
            "Normalize",
        )
        if key not in input_values
    ]
    if missing:
        raise ValueError(f"Missing required XML input(s): {', '.join(missing)}")

    spectrum = _parse_reference_spectrum(input_values["ReferenceSpectrum"])

    threshold = _parse_float(input_values["SAM_Threshold"], label="SAM_Threshold")
    wl_min = _parse_float(input_values["SAM_MinWL"], label="SAM_MinWL")
    wl_max = _parse_float(input_values["SAM_MaxWL"], label="SAM_MaxWL")
    red_wl = _parse_float(input_values["RedWL"], label="RedWL")
    green_wl = _parse_float(input_values["GreenWL"], label="GreenWL")
    blue_wl = _parse_float(input_values["BlueWL"], label="BlueWL")
    width = _parse_float(input_values["Width"], label="Width")
    normalize = _parse_float(input_values["Normalize"], label="Normalize")

    half_width = width / 2.0
    config = {
        "threshold": threshold,
        "wl_min": wl_min,
        "wl_max": wl_max,
        "red_range": [red_wl - half_width, red_wl + half_width],
        "green_range": [green_wl - half_width, green_wl + half_width],
        "blue_range": [blue_wl - half_width, blue_wl + half_width],
        "normalization_strength": normalize,
        "spectrum_length": int(spectrum.shape[1]),
    }
    return spectrum, config


@click.command()
@click.option(
    "--xml-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to SPAM plugin XML file",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Output directory (default: XML parent directory)",
)
def main(xml_path: Path, output_dir: Path | None) -> None:
    """Create `<xml-stem>.npy` and `<xml-stem>_config.json` from XML inputs."""
    root = ET.parse(xml_path).getroot()
    input_values = _read_inputs(root)
    spectrum, config = _extract_config(input_values)

    target_dir = xml_path.parent if output_dir is None else output_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    base = xml_path.stem
    npy_path = target_dir / f"{base}.npy"
    cfg_path = target_dir / f"{base}_config.json"

    np.save(npy_path, spectrum)
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    click.echo(f"Saved reference spectrum: {npy_path}")
    click.echo(f"Saved config: {cfg_path}")
    click.echo(f"Spectrum shape: {tuple(spectrum.shape)}")


if __name__ == "__main__":
    main()

