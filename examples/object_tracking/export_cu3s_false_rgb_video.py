"""Export a CU3S sequence to false-RGB MP4.

Pipeline:
1. CU3SDataNode: normalizes CU3S tensors and extracts wavelengths
2. False-RGB selector: spectral -> 3-channel conversion
3. ToVideoNode: stream-write RGB frames to MP4 (optionally overlays frame ID)

Examples
--------
CLI — single method export (CIE tristimulus, SpectralRadiance mode)::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-dir "D:\\experiments\\sam3\\false_rgb_export" `
        --method cie_tristimulus `
        --processing-mode SpectralRadiance

CLI — single method export (CIR: NIR->R, Red->G, Green->B)::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-dir "D:\\experiments\\sam3\\false_rgb_export" `
        --out-basename "auto_013_cir" `
        --method cir `
        --nir-nm 860 --red-nm 670 --green-nm 560 `
        --processing-mode SpectralRadiance

CLI — with frame ID overlay::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-dir "D:\\experiments\\sam3\\false_rgb_export" `
        --method cie_tristimulus `
        --overlay-frame-id

CLI — cuvis-plugin XML parity (fast_rgb config from plugin XML)::

    uv run python examples/object_tracking/export_cu3s_false_rgb_video.py `
        --cu3s-path "D:\\data\\XMR_notarget_Busstation\\20260226\\Auto_013+01.cu3s" `
        --output-dir "D:\\experiments\\sam3\\false_rgb_export" `
        --method cuvis-plugin `
        --plugin-xml-path "C:\\Users\\nima.ghorbani\\CuvisNEXT\\invisible_ink.xml"

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
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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
    CIRSelector,
    FastRGBSelector,
    NormMode,
)
from cuvis_ai.node.data import CU3SDataNode
from cuvis_ai.node.video import ToVideoNode

SUPPORTED_METHODS = ("cie_tristimulus", "cir", "fast_rgb", "cuvis-plugin")
METHOD_ALIASES: dict[str, str] = {"fastrgb": "fast_rgb"}
METHOD_CHOICES = SUPPORTED_METHODS + tuple(METHOD_ALIASES.keys())
FAST_RGB_METHODS = ("fast_rgb", "cuvis-plugin")
PROCESSING_MODES = ("Raw", "DarkSubtract", "Preview", "Reflectance", "SpectralRadiance")
NORMALIZATION_MODES = ("sampled_fixed", "running", "per_frame", "live_running_fixed")


@dataclass(frozen=True)
class PluginFastRGBConfig:
    """Resolved cuvis-plugin fast_rgb parameters from user-plugin XML."""

    red_range: tuple[float, float]
    green_range: tuple[float, float]
    blue_range: tuple[float, float]
    normalization_strength: float


def _xml_local_name(tag: str) -> str:
    """Return local XML tag name independent of namespace."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def _parse_numeric_text(text: str | None, *, label: str) -> float:
    """Parse a numeric XML text payload with descriptive errors."""
    payload = (text or "").strip()
    if not payload:
        raise ValueError(f"{label} is empty")
    try:
        return float(payload)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric, got '{payload}'") from exc


def _find_plugin_fast_rgb_nodes(root: ET.Element) -> tuple[ET.Element, ET.Element]:
    """Find the first <configuration> containing a <fast_rgb> node."""
    for config_node in root.iter():
        if _xml_local_name(config_node.tag) != "configuration":
            continue
        for child in config_node.iter():
            if _xml_local_name(child.tag) == "fast_rgb":
                return config_node, child
    raise ValueError("No <fast_rgb> node found in plugin XML.")


def _evaluate_plugin_operator(
    operator_node: ET.Element,
    resolve_ref: Any,
) -> float:
    """Evaluate a plugin <operator> tree to a scalar float."""
    operator_type = operator_node.attrib.get("type", "").strip().lower()
    operands: list[float] = []

    for child in list(operator_node):
        child_tag = _xml_local_name(child.tag)
        if child_tag == "variable":
            ref = (child.attrib.get("ref") or "").strip()
            if not ref:
                raise ValueError("Operator variable reference is missing 'ref' attribute.")
            operands.append(float(resolve_ref(ref)))
        elif child_tag == "value":
            operands.append(_parse_numeric_text(child.text, label="<value>"))
        elif child_tag == "operator":
            operands.append(_evaluate_plugin_operator(child, resolve_ref))

    if len(operands) != 2:
        raise ValueError(
            f"Operator '{operator_type}' expects exactly 2 operands, got {len(operands)}."
        )
    left, right = operands

    if operator_type == "add":
        return left + right
    if operator_type == "subtract":
        return left - right
    if operator_type == "multiply":
        return left * right
    if operator_type == "divide":
        if abs(right) <= 1.0e-12:
            raise ValueError("Division by zero while evaluating plugin XML expression.")
        return left / right
    raise ValueError(f"Unsupported operator type '{operator_type}' in plugin XML.")


def _parse_plugin_fast_rgb_config(plugin_xml_path: Path) -> PluginFastRGBConfig:
    """Parse cuvis user-plugin XML and resolve fast_rgb ranges + normalization."""
    try:
        root = ET.parse(plugin_xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML format: {exc}") from exc

    config_node, fast_rgb_node = _find_plugin_fast_rgb_nodes(root)

    input_nodes: dict[str, ET.Element] = {}
    evaluate_nodes: dict[str, ET.Element] = {}
    for node in config_node.iter():
        node_tag = _xml_local_name(node.tag)
        node_id = (node.attrib.get("id") or "").strip()
        if not node_id:
            continue
        if node_tag == "input":
            input_nodes[node_id] = node
        elif node_tag == "evaluate":
            evaluate_nodes[node_id] = node

    cache: dict[str, float] = {}
    active_refs: set[str] = set()

    def resolve_ref(ref: str) -> float:
        if ref in cache:
            return cache[ref]
        if ref in active_refs:
            raise ValueError(f"Cyclic reference detected while resolving '{ref}'.")

        active_refs.add(ref)
        try:
            if ref in input_nodes:
                value = _parse_numeric_text(input_nodes[ref].text, label=f"input '{ref}'")
            elif ref in evaluate_nodes:
                evaluate_node = evaluate_nodes[ref]
                operator_node = next(
                    (
                        child
                        for child in list(evaluate_node)
                        if _xml_local_name(child.tag) == "operator"
                    ),
                    None,
                )
                if operator_node is None:
                    raise ValueError(f"evaluate '{ref}' does not contain an <operator> node.")
                value = _evaluate_plugin_operator(operator_node, resolve_ref)
            else:
                raise ValueError(f"Reference '{ref}' not found in plugin XML inputs/evaluations.")
            cache[ref] = float(value)
            return float(value)
        finally:
            active_refs.discard(ref)

    def resolve_fast_rgb_attr(attr_name: str) -> float:
        attr_value = (fast_rgb_node.attrib.get(attr_name) or "").strip()
        if not attr_value:
            raise ValueError(f"<fast_rgb> attribute '{attr_name}' is required.")
        try:
            return float(attr_value)
        except ValueError:
            return resolve_ref(attr_value)

    red_low = resolve_fast_rgb_attr("red_min")
    red_high = resolve_fast_rgb_attr("red_max")
    green_low = resolve_fast_rgb_attr("green_min")
    green_high = resolve_fast_rgb_attr("green_max")
    blue_low = resolve_fast_rgb_attr("blue_min")
    blue_high = resolve_fast_rgb_attr("blue_max")
    normalization_strength = resolve_fast_rgb_attr("normalization")

    if red_low > red_high:
        raise ValueError(f"Invalid red range: red_min ({red_low}) > red_max ({red_high}).")
    if green_low > green_high:
        raise ValueError(
            f"Invalid green range: green_min ({green_low}) > green_max ({green_high})."
        )
    if blue_low > blue_high:
        raise ValueError(f"Invalid blue range: blue_min ({blue_low}) > blue_max ({blue_high}).")

    return PluginFastRGBConfig(
        red_range=(float(red_low), float(red_high)),
        green_range=(float(green_low), float(green_high)),
        blue_range=(float(blue_low), float(blue_high)),
        normalization_strength=float(normalization_strength),
    )


def _resolve_plugin_xml_path(plugin_xml_path: str | Path | None) -> Path:
    """Resolve and validate the plugin XML path for cuvis-plugin mode."""
    if plugin_xml_path is None:
        raise click.BadParameter(
            "--plugin-xml-path is required when --method cuvis-plugin",
            param_hint="--plugin-xml-path",
        )

    path = Path(plugin_xml_path).expanduser().resolve()
    if not path.exists():
        raise click.BadParameter(
            f"Plugin XML not found: {path}",
            param_hint="--plugin-xml-path",
        )
    if not path.is_file():
        raise click.BadParameter(
            f"Plugin XML is not a file: {path}",
            param_hint="--plugin-xml-path",
        )
    return path


def _resolve_processing_mode(processing_mode: str) -> str:
    """Resolve CLI/user input into a canonical processing mode string."""
    lookup = {name.lower(): name for name in PROCESSING_MODES}
    resolved = lookup.get(processing_mode.strip().lower())
    if resolved is None:
        raise click.BadParameter(
            f"Invalid processing_mode '{processing_mode}'. Supported: {', '.join(PROCESSING_MODES)}"
        )
    return resolved


def _resolve_method(method: str) -> str:
    """Resolve CLI/user input into a canonical export method string."""
    normalized = method.strip().lower()
    if normalized in SUPPORTED_METHODS:
        return normalized
    alias = METHOD_ALIASES.get(normalized)
    if alias is not None:
        return alias
    raise click.BadParameter(f"Unknown method '{method}'. Supported: {', '.join(METHOD_CHOICES)}")


def _resolve_run_output_dir(
    *,
    output_root: Path,
    source_path: Path,
    out_basename: str | None,
) -> Path:
    """Resolve per-run output directory from --output-dir and --out-basename."""
    resolved_basename = source_path.stem
    if out_basename is not None:
        candidate = out_basename.strip()
        if not candidate:
            raise click.BadParameter(
                "--out-basename must not be empty or whitespace only",
                param_hint="--out-basename",
            )
        if "/" in candidate or "\\" in candidate:
            raise click.BadParameter(
                "--out-basename must be a folder name, not a path",
                param_hint="--out-basename",
            )
        resolved_basename = candidate
    return output_root / resolved_basename


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
    if normalization_mode in {"running", "live_running_fixed"}:
        return NormMode.RUNNING
    return NormMode.PER_FRAME


def _resolve_fast_rgb_normalization_strength(
    processing_mode: str,
    override: float | None,
) -> float:
    """Resolve FastRGB normalization strength from mode defaults or CLI override."""
    if override is not None:
        return float(override)
    if processing_mode == "Reflectance":
        return 0.0
    return 0.75


def _create_false_rgb_node(
    method: str,
    *,
    norm_mode: str | NormMode = NormMode.RUNNING,
    freeze_running_bounds_after_frames: int | None = 20,
    running_warmup_frames: int = 10,
    fast_rgb_normalization_strength: float = 0.75,
    nir_nm: float = 860.0,
    red_nm: float = 670.0,
    green_nm: float = 560.0,
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
    """Create a false-RGB selector node for the requested method."""
    if method == "cie_tristimulus":
        return CIETristimulusFalseRGBSelector(
            norm_mode=norm_mode,
            freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
            running_warmup_frames=running_warmup_frames,
            name="cie_tristimulus_false_rgb",
        )
    if method == "cir":
        return CIRSelector(
            nir_nm=nir_nm,
            red_nm=red_nm,
            green_nm=green_nm,
            norm_mode=norm_mode,
            freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
            running_warmup_frames=running_warmup_frames,
            name="cir_false_rgb",
        )
    if method in {"fast_rgb", "cuvis-plugin"}:
        return FastRGBSelector(
            red_range=(red_low, red_high),
            green_range=(green_low, green_high),
            blue_range=(blue_low, blue_high),
            normalization_strength=fast_rgb_normalization_strength,
            name="fast_rgb_false_rgb",
        )
    raise click.BadParameter(f"Unknown method '{method}'. Supported: {SUPPORTED_METHODS}")


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
    running_warmup_frames: int = 10,
    fast_rgb_normalization_strength: float | None = None,
    save_pipeline_config: bool = False,
    overlay_frame_id: bool = False,
    nir_nm: float = 860.0,
    red_nm: float = 670.0,
    green_nm: float = 560.0,
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
    plugin_xml_path: str | None = None,
) -> Path:
    """Run CU3S -> false RGB -> MP4 export pipeline."""
    resolved_method = _resolve_method(method)
    resolved_mode = _resolve_processing_mode(processing_mode)
    resolved_plugin_xml_path: Path | None = None
    plugin_config: PluginFastRGBConfig | None = None
    effective_red_low = float(red_low)
    effective_red_high = float(red_high)
    effective_green_low = float(green_low)
    effective_green_high = float(green_high)
    effective_blue_low = float(blue_low)
    effective_blue_high = float(blue_high)
    effective_fast_rgb_normalization_strength: float | None = None
    if resolved_method == "cuvis-plugin":
        resolved_plugin_xml_path = _resolve_plugin_xml_path(plugin_xml_path)
        try:
            plugin_config = _parse_plugin_fast_rgb_config(resolved_plugin_xml_path)
        except ValueError as exc:
            raise click.BadParameter(str(exc), param_hint="--plugin-xml-path") from exc

        effective_red_low, effective_red_high = plugin_config.red_range
        effective_green_low, effective_green_high = plugin_config.green_range
        effective_blue_low, effective_blue_high = plugin_config.blue_range

        resolved_norm_mode = normalization_mode
        resolved_sample_fraction = float(sample_fraction)
        effective_running_warmup_frames = running_warmup_frames
        effective_freeze_running_bounds_after_frames = freeze_running_bounds_after_frames
        effective_fast_rgb_normalization_strength = (
            float(fast_rgb_normalization_strength)
            if fast_rgb_normalization_strength is not None
            else float(plugin_config.normalization_strength)
        )
    elif resolved_method == "fast_rgb":
        resolved_norm_mode = normalization_mode
        resolved_sample_fraction = float(sample_fraction)
        effective_running_warmup_frames = running_warmup_frames
        effective_freeze_running_bounds_after_frames = freeze_running_bounds_after_frames
        effective_fast_rgb_normalization_strength = _resolve_fast_rgb_normalization_strength(
            processing_mode=resolved_mode,
            override=fast_rgb_normalization_strength,
        )
    else:
        resolved_norm_mode = _resolve_normalization_mode(normalization_mode)
        if resolved_norm_mode == "sampled_fixed":
            resolved_sample_fraction = _validate_sample_fraction(sample_fraction)
        else:
            resolved_sample_fraction = float(sample_fraction)

        effective_running_warmup_frames = int(running_warmup_frames)
        effective_freeze_running_bounds_after_frames = freeze_running_bounds_after_frames
        if resolved_norm_mode == "live_running_fixed":
            # Live-stable mode: no warmup per-frame normalization and fixed
            # normalization bounds from the first frame onward.
            effective_running_warmup_frames = 0
            effective_freeze_running_bounds_after_frames = 1

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
    selector_norm_mode = (
        _resolve_selector_norm_mode(resolved_norm_mode)
        if resolved_method not in FAST_RGB_METHODS
        else NormMode.PER_FRAME
    )
    selector_method = "fast_rgb" if resolved_method in FAST_RGB_METHODS else resolved_method
    false_rgb = _create_false_rgb_node(
        selector_method,
        norm_mode=selector_norm_mode,
        freeze_running_bounds_after_frames=effective_freeze_running_bounds_after_frames,
        running_warmup_frames=effective_running_warmup_frames,
        fast_rgb_normalization_strength=(
            effective_fast_rgb_normalization_strength
            if effective_fast_rgb_normalization_strength is not None
            else 0.75
        ),
        nir_nm=nir_nm,
        red_nm=red_nm,
        green_nm=green_nm,
        red_low=effective_red_low,
        red_high=effective_red_high,
        green_low=effective_green_low,
        green_high=effective_green_high,
        blue_low=effective_blue_low,
        blue_high=effective_blue_high,
        r_peak=r_peak,
        g_peak=g_peak,
        b_peak=b_peak,
        r_sigma=r_sigma,
        g_sigma=g_sigma,
        b_sigma=b_sigma,
    )

    sampled_positions: list[int] = []
    sampled_mesu_ids: list[int] = []
    if resolved_method not in FAST_RGB_METHODS and resolved_norm_mode == "sampled_fixed":
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
        f"method={resolved_method}, "
        f"method_input={method}, "
        f"plugin_xml_path={resolved_plugin_xml_path}, "
        f"cir_nm=({nir_nm},{red_nm},{green_nm}), "
        f"fast_rgb_ranges_nm=((R:{effective_red_low},{effective_red_high}),"
        f"(G:{effective_green_low},{effective_green_high}),"
        f"(B:{effective_blue_low},{effective_blue_high})), "
        f"fast_rgb_normalization_strength={effective_fast_rgb_normalization_strength}, "
        f"frame_rate={resolved_frame_rate}, "
        f"dataset_fps={dataset_fps}, "
        f"frame_rotation={frame_rotation}, "
        f"processing_mode={resolved_mode}, "
        f"normalization_mode={resolved_norm_mode}, "
        f"sample_fraction={resolved_sample_fraction}, "
        f"running_warmup_frames={effective_running_warmup_frames}, "
        f"freeze_running_bounds_after_frames={effective_freeze_running_bounds_after_frames}, "
        f"save_pipeline_config={save_pipeline_config}, "
        f"max_num_frames={max_num_frames}, "
        f"overlay_frame_id={overlay_frame_id}"
    )
    if resolved_method in FAST_RGB_METHODS:
        logger.info(
            "FastRGB parity mode: ignoring legacy normalization controls "
            "(normalization_mode/sample_fraction/freeze_running_bounds_after/running_warmup_frames)."
        )
    if resolved_method == "cuvis-plugin" and plugin_config is not None:
        logger.info(
            "Resolved cuvis-plugin XML: "
            f"red_range={plugin_config.red_range}, "
            f"green_range={plugin_config.green_range}, "
            f"blue_range={plugin_config.blue_range}, "
            f"xml_normalization_strength={plugin_config.normalization_strength}"
        )
    if resolved_method not in FAST_RGB_METHODS and resolved_norm_mode == "sampled_fixed":
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


@click.command()
@click.option(
    "--cu3s-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to .cu3s file.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("D:/experiments/sam3/false_rgb_export"),
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
    "--method",
    type=click.Choice(METHOD_CHOICES, case_sensitive=False),
    default="cie_tristimulus",
    show_default=True,
)
@click.option(
    "--plugin-xml-path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help=(
        "Path to a cuvis user-plugin XML containing <fast_rgb> configuration. "
        "Required when --method cuvis-plugin."
    ),
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
    "--running-warmup-frames",
    type=int,
    default=10,
    show_default=True,
    help="Running-mode warmup frames using per-frame normalization (0 disables warmup).",
)
@click.option(
    "--fast-rgb-normalization-strength",
    type=float,
    default=None,
    help=(
        "Optional FastRGB normalization strength override. "
        "For cuvis-plugin mode this overrides XML normalization. "
        "Default: 0.75 for Raw/DarkSubtract/Preview/SpectralRadiance, "
        "0.0 (static scaling) for Reflectance."
    ),
)
@click.option(
    "--save-pipeline-config/--no-save-pipeline-config",
    default=False,
    show_default=True,
    help="Save pipeline config files (.yaml + .pt) next to the output video.",
)
# cir-specific options
@click.option(
    "--nir-nm", type=float, default=860.0, show_default=True, help="CIR NIR wavelength (nm)."
)
@click.option(
    "--red-nm", type=float, default=670.0, show_default=True, help="CIR Red wavelength (nm)."
)
@click.option(
    "--green-nm",
    type=float,
    default=560.0,
    show_default=True,
    help="CIR Green wavelength (nm).",
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
    cu3s_path: Path,
    output_dir: Path,
    out_basename: str | None,
    method: str,
    plugin_xml_path: Path | None,
    frame_rate: float | None,
    frame_rotation: int | None,
    batch_size: int,
    overlay_frame_id: bool,
    max_num_frames: int,
    processing_mode: str,
    normalization_mode: str,
    sample_fraction: float,
    freeze_running_bounds_after: int,
    running_warmup_frames: int,
    fast_rgb_normalization_strength: float | None,
    save_pipeline_config: bool,
    nir_nm: float,
    red_nm: float,
    green_nm: float,
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
    resolved_method = _resolve_method(method)
    if resolved_method == "cuvis-plugin":
        _resolve_plugin_xml_path(plugin_xml_path)

    if resolved_method not in FAST_RGB_METHODS:
        resolved_norm_mode = _resolve_normalization_mode(normalization_mode)
        if resolved_norm_mode == "sampled_fixed":
            try:
                _validate_sample_fraction(sample_fraction)
            except ValueError as exc:
                raise click.BadParameter(str(exc), param_hint="--sample-fraction") from exc

        if running_warmup_frames < 0:
            raise click.BadParameter(
                "--running-warmup-frames must be >= 0",
                param_hint="--running-warmup-frames",
            )

    freeze_running_bounds_after_frames = (
        None if freeze_running_bounds_after <= 0 else freeze_running_bounds_after
    )

    run_output_dir = _resolve_run_output_dir(
        output_root=output_dir,
        source_path=cu3s_path,
        out_basename=out_basename,
    )
    run_output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = run_output_dir / f"{cu3s_path.stem}.mp4"
    logger.info("Output run directory: {}", run_output_dir)

    export_false_rgb_video(
        cu3s_file_path=str(cu3s_path),
        output_video_path=str(output_video_path),
        method=method,
        plugin_xml_path=str(plugin_xml_path) if plugin_xml_path is not None else None,
        frame_rate=frame_rate,
        frame_rotation=frame_rotation,
        max_num_frames=max_num_frames,
        batch_size=batch_size,
        processing_mode=processing_mode,
        normalization_mode=normalization_mode,
        sample_fraction=sample_fraction,
        freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
        running_warmup_frames=running_warmup_frames,
        fast_rgb_normalization_strength=fast_rgb_normalization_strength,
        save_pipeline_config=save_pipeline_config,
        overlay_frame_id=overlay_frame_id,
        nir_nm=nir_nm,
        red_nm=red_nm,
        green_nm=green_nm,
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
