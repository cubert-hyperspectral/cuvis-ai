"""The ``normalize_output`` switch of ``ChannelSelectorBase``.

Seven selector paths go through the base normalisation (FixedWavelength, RangeAverage,
HighContrast, CIR, CIETristimulus, CameraEmulation and the supervised selectors); the switch
turns it off for all of them: the raw composed bands come back, the running bounds and the
frame counter do not move, no gamma is applied, and no statistical fit pass is scheduled.
FastRGB and the index selectors render their own output: for them the five base
normalisation hparams are neither saved nor shown, and a value that differs from the class
default warns once. The normalised-difference selectors rebuild from their own saved hparams.
"""

from __future__ import annotations

import inspect
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import numpy as np
import pytest
import torch
from loguru import logger

from cuvis_ai.node import channel_selector as cs
from cuvis_ai.node.channel_selector import (
    CameraEmulationFalseRGBSelector,
    ChannelSelectorBase,
    CIETristimulusRGBSelector,
    CIRSelector,
    FastRGBSelector,
    FixedWavelengthSelector,
    GNDVISelector,
    HighContrastSelector,
    NBRSelector,
    NDRESelector,
    NDVISelector,
    NDWISelector,
    RangeAverageFalseRGBSelector,
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
    SupervisedWindowedSelector,
)
from cuvis_ai_core.pipeline.factory import PipelineBuilder

pytestmark = pytest.mark.unit

NORMALISATION_KEYS = (
    "norm_mode",
    "apply_gamma",
    "running_warmup_frames",
    "freeze_running_bounds_after_frames",
    "normalize_output",
)

B, H, W, C = 2, 4, 5, 30
WAVELENGTHS = np.linspace(400.0, 1000.0, C).astype(np.float32)

# Selectors whose forward goes through the base normalisation and that construct with
# their defaults. The supervised selectors need a fit and are tested separately.
BASE_NORMALISED = [
    CIRSelector,
    RangeAverageFalseRGBSelector,
    HighContrastSelector,
    CIETristimulusRGBSelector,
    CameraEmulationFalseRGBSelector,
    FixedWavelengthSelector,
]
SUPERVISED = [SupervisedCIRSelector, SupervisedWindowedSelector, SupervisedFullSpectrumSelector]
NORMALISED_DIFFERENCE = [NDVISelector, NDWISelector, NBRSelector, GNDVISelector, NDRESelector]
BYPASSING = [
    FastRGBSelector,
    *NORMALISED_DIFFERENCE,
    cs.EVISelector,
    cs.EVI2Selector,
    cs.SAVISelector,
    cs.MSAVISelector,
    cs.CIRedEdgeSelector,
    cs.MCARISelector,
    cs.PRISelector,
]


def _cube(seed: int = 0) -> torch.Tensor:
    """A cube with values far above 1, so a normalised output differs from the raw one."""
    torch.manual_seed(seed)
    return torch.rand(B, H, W, C) * 500.0 + 10.0


@contextmanager
def _captured_warnings() -> Generator[list[str], None, None]:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


# ---------------------------------------------------------------------------
# The switch on the base-normalised selectors
# ---------------------------------------------------------------------------


@torch.no_grad()
@pytest.mark.parametrize("cls", BASE_NORMALISED)
def test_switch_off_returns_raw_bands_without_touching_state(cls: type) -> None:
    """Off: the raw composed bands, no gamma (default apply_gamma=True), no running state."""
    node = cls(normalize_output=False)
    cube = _cube()

    first = node.forward(cube=cube, wavelengths=WAVELENGTHS)
    second = node.forward(cube=cube, wavelengths=WAVELENGTHS)
    expected = node._compute_raw_rgb(cube, WAVELENGTHS)

    assert torch.equal(first["rgb_image"], expected)
    assert torch.equal(second["rgb_image"], expected)
    assert torch.isnan(node.running_min).all()
    assert int(node._norm_frame_count.item()) == 0
    assert first["band_info"]["normalized_output"] is False


@torch.no_grad()
@pytest.mark.parametrize("cls", BASE_NORMALISED)
def test_switch_on_normalises_to_unit_range(cls: type) -> None:
    node = cls()
    out = node.forward(cube=_cube(), wavelengths=WAVELENGTHS)

    assert out["rgb_image"].min() >= 0.0
    assert out["rgb_image"].max() <= 1.0 + 1e-6
    assert out["band_info"]["normalized_output"] is True


@torch.no_grad()
def test_statistical_mode_off_schedules_no_fit_and_bypasses_fitted_bounds() -> None:
    assert CIRSelector(norm_mode="statistical").requires_initial_fit is True

    node = CIRSelector(norm_mode="statistical", normalize_output=False)
    assert node.requires_initial_fit is False

    cube = _cube()
    node.statistical_initialization(iter([{"cube": cube, "wavelengths": WAVELENGTHS}]))
    assert node._statistically_initialized is True
    assert not torch.isnan(node.running_min).any()

    out = node.forward(cube=cube, wavelengths=WAVELENGTHS)
    assert torch.equal(out["rgb_image"], node._compute_raw_rgb(cube, WAVELENGTHS))


@torch.no_grad()
def test_supervised_selector_still_fits_and_returns_the_learned_bands_raw() -> None:
    n_bands = 12
    wavelengths = np.linspace(450.0, 950.0, n_bands).astype(np.float32)
    torch.manual_seed(1)
    cube = torch.rand(2, 6, 6, n_bands) * 100.0
    mask = torch.zeros(2, 6, 6, 1)
    mask[:, :3, :, :] = 1.0
    cube[:, :3, :, 7] += 50.0  # the positive class lights up one band

    node = SupervisedCIRSelector(num_spectral_bands=n_bands, normalize_output=False)
    assert node.requires_initial_fit is True

    node.statistical_initialization(
        iter([{"cube": cube, "mask": mask, "wavelengths": wavelengths}])
    )
    assert node._statistically_initialized is True

    out = node.forward(cube=cube, wavelengths=wavelengths)
    indices = node.selected_indices.tolist()
    expected = torch.stack([cube[..., i] for i in indices], dim=-1)
    assert torch.equal(out["rgb_image"], expected)
    assert out["band_info"]["normalized_output"] is False
    assert out["band_info"]["band_indices"] == indices


@torch.no_grad()
def test_cie_band_info_reports_the_gamma_that_was_applied() -> None:
    on = CIETristimulusRGBSelector().forward(cube=_cube(), wavelengths=WAVELENGTHS)
    off = CIETristimulusRGBSelector(normalize_output=False).forward(
        cube=_cube(), wavelengths=WAVELENGTHS
    )
    assert on["band_info"]["apply_gamma"] is True
    assert off["band_info"]["apply_gamma"] is False


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_normalize_output_must_be_a_bool(value: Any) -> None:
    with pytest.raises(ValueError, match="normalize_output must be a bool"):
        CIRSelector(normalize_output=value)
    with pytest.raises(ValueError, match="normalize_output must be a bool"):
        FixedWavelengthSelector(normalize_output=value)


# ---------------------------------------------------------------------------
# hparams: captured on the seven paths, absent on the thirteen bypassing classes
# ---------------------------------------------------------------------------


def _construct(cls: type, **overrides: Any) -> ChannelSelectorBase:
    if issubclass(cls, cs.SupervisedSelectorBase):
        return cls(num_spectral_bands=C, **overrides)
    return cls(**overrides)


@pytest.mark.parametrize("cls", BASE_NORMALISED + SUPERVISED)
def test_base_normalised_selectors_save_the_switch_and_rebuild(cls: type) -> None:
    assert cls._USES_BASE_NORMALIZATION is True
    assert _construct(cls).hparams["normalize_output"] is True

    node = _construct(cls, normalize_output=False)
    assert node.hparams["normalize_output"] is False
    assert set(NORMALISATION_KEYS) <= set(node.hparams)

    rebuilt = cls(**node.hparams)
    assert rebuilt.normalize_output is False


@pytest.mark.parametrize("cls", BYPASSING)
def test_bypassing_selectors_do_not_save_the_normalisation_keys(cls: type) -> None:
    assert cls._USES_BASE_NORMALIZATION is False
    assert not set(NORMALISATION_KEYS) & set(cls().hparams)


def test_every_concrete_selector_is_in_exactly_one_group() -> None:
    concrete = {
        name
        for name, obj in vars(cs).items()
        if inspect.isclass(obj)
        and issubclass(obj, ChannelSelectorBase)
        and obj is not ChannelSelectorBase
        and obj is not cs.SupervisedSelectorBase
        and not inspect.isabstract(obj)
        and not name.startswith("_")
    }
    grouped = {c.__name__ for c in BASE_NORMALISED + SUPERVISED + BYPASSING}
    assert concrete == grouped


# ---------------------------------------------------------------------------
# Bypassing classes: class defaults, one warning on a non-default value
# ---------------------------------------------------------------------------


def test_bypassing_selector_keeps_its_class_defaults_silently() -> None:
    with _captured_warnings() as messages:
        node = NDVISelector(norm_mode="per_frame", apply_gamma=False)  # what ndvi.yaml passes
        FastRGBSelector()
    assert messages == []
    assert node.norm_mode == cs.NormMode.PER_FRAME
    assert node.apply_gamma is False
    assert node.normalize_output is True


@torch.no_grad()
def test_bypassing_selector_warns_once_and_drops_a_non_default_value() -> None:
    with _captured_warnings() as messages:
        node = NDVISelector(apply_gamma=True)
    assert len(messages) == 1
    assert "NDVISelector" in messages[0] and "apply_gamma" in messages[0]
    assert node.apply_gamma is False

    with _captured_warnings() as messages:
        statistical = NDVISelector(norm_mode="statistical", normalize_output=False)
    assert len(messages) == 2
    assert any("norm_mode" in m for m in messages)
    assert any("normalize_output" in m for m in messages)
    assert statistical.norm_mode == cs.NormMode.PER_FRAME
    assert statistical.normalize_output is True
    assert statistical.requires_initial_fit is False

    cube = _cube()
    assert torch.equal(
        statistical.forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"],
        NDVISelector().forward(cube=cube, wavelengths=WAVELENGTHS)["rgb_image"],
    )

    with _captured_warnings() as messages:
        fast = FastRGBSelector(norm_mode="running")
    assert len(messages) == 1
    assert "FastRGBSelector" in messages[0] and "norm_mode" in messages[0]
    assert fast.norm_mode == cs.NormMode.PER_FRAME


# ---------------------------------------------------------------------------
# Saved YAML round trips
# ---------------------------------------------------------------------------


def _single_node_config(name: str, class_name: str, hparams: dict) -> dict:
    return {
        "metadata": {"name": name},
        "nodes": [{"name": name, "class_name": class_name, "hparams": hparams}],
        "connections": [],
    }


def _node(pipeline: Any, name: str) -> ChannelSelectorBase:
    return next(n for n in pipeline.nodes if n.name == name)


@torch.no_grad()
def test_switch_off_survives_the_pipeline_yaml_round_trip(tmp_path) -> None:
    config = _single_node_config(
        "cir", "cuvis_ai.node.channel_selector.CIRSelector", {"normalize_output": False}
    )
    pipeline = PipelineBuilder().build_from_config(config)
    yaml_path = tmp_path / "cir.yaml"
    pipeline.save_to_file(yaml_path, save_weights=False)

    rebuilt = _node(PipelineBuilder().build_from_config(yaml_path), "cir")
    assert rebuilt.normalize_output is False
    cube = _cube()
    out = rebuilt.forward(cube=cube, wavelengths=WAVELENGTHS)
    assert torch.equal(out["rgb_image"], rebuilt._compute_raw_rgb(cube, WAVELENGTHS))


def test_ndvi_preset_node_round_trips_without_a_warning(tmp_path) -> None:
    """The hparams of the shipped ndvi.yaml, saved by cuvis-ai and loaded again."""
    config = _single_node_config(
        "ndvi",
        "cuvis_ai.node.channel_selector.NDVISelector",
        {"nir_nm": 827.0, "red_nm": 668.0, "norm_mode": "per_frame"},
    )
    yaml_path = tmp_path / "ndvi.yaml"
    with _captured_warnings() as messages:
        pipeline = PipelineBuilder().build_from_config(config)
        pipeline.save_to_file(yaml_path, save_weights=False)
        rebuilt = _node(PipelineBuilder().build_from_config(yaml_path), "ndvi")
    assert messages == []
    assert rebuilt.nir_nm == 827.0 and rebuilt.red_nm == 668.0
    assert not set(NORMALISATION_KEYS) & set(rebuilt.hparams)
