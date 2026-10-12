from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import pytest
import torch
from loguru import logger

from cuvis_ai.node.channel_selector import (
    GNDVISelector,
    NBRSelector,
    NDRESelector,
    NDVISelector,
    NDWISelector,
    NormMode,
)

pytestmark = pytest.mark.unit

NORMALISED_DIFFERENCE = [NDVISelector, NDWISelector, NBRSelector, GNDVISelector, NDRESelector]

# The selector's own wavelength names (primary, secondary), as its constructor declares them.
OWN_NAMES = {
    NDVISelector: ("nir_nm", "red_nm"),
    NDWISelector: ("green_nm", "nir_nm"),
    NBRSelector: ("nir_nm", "swir_nm"),
    GNDVISelector: ("nir_nm", "green_nm"),
    NDRESelector: ("nir_nm", "red_edge_nm"),
}
# Not the default of any of the five classes.
PRIMARY_NM, SECONDARY_NM = 777.0, 555.0


@contextmanager
def _captured_warnings() -> Generator[list[str], None, None]:
    messages: list[str] = []
    handler = logger.add(lambda m: messages.append(m.record["message"]), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _hparams_saved_before_the_fix(cls: type) -> dict:
    """What ``save_to_file`` wrote for the node on cuvis-ai 0.18.2: the selector's own
    wavelength names beside the base's derived ``primary_nm`` and ``secondary_nm``."""
    primary, secondary = OWN_NAMES[cls]
    return {
        primary: PRIMARY_NM,
        secondary: SECONDARY_NM,
        "primary_nm": PRIMARY_NM,
        "secondary_nm": SECONDARY_NM,
        "colormap_min": -1.0,
        "colormap_max": 1.0,
        "eps": 1.0e-6,
        "band_tolerance_nm": 50.0,
        "norm_mode": "per_frame",
        "apply_gamma": False,
        "freeze_running_bounds_after_frames": 20,
        "running_warmup_frames": 10,
    }


@torch.no_grad()
def test_ndvi_selector_resolves_nearest_bands_and_returns_raw_index() -> None:
    cube = torch.tensor(
        [
            [
                [
                    [0.1, 0.2, 0.8, 0.9],
                    [0.1, 0.5, 0.5, 0.1],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    wavelengths = np.array([660.0, 670.0, 830.0, 840.0], dtype=np.float32)

    node = NDVISelector()
    result = node.forward(cube=cube, wavelengths=wavelengths)

    expected_index = torch.tensor([[[[0.6], [0.0]]]], dtype=torch.float32)
    assert torch.allclose(result["index_image"], expected_index, atol=1e-6, rtol=1e-6)

    rgb = result["rgb_image"]
    assert rgb.shape == (1, 1, 2, 3)
    assert torch.allclose(
        rgb[0, 0, 0],
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        rgb[0, 0, 1],
        torch.tensor([0.0, 0.5, 1.0]),
        atol=1e-6,
        rtol=1e-6,
    )

    band_info = result["band_info"]
    assert band_info["strategy"] == "ndvi"
    assert band_info["band_labels"] == ["nir", "red"]
    assert band_info["band_indices"] == [2, 1]
    assert band_info["requested_wavelengths_nm"] == [827.0, 668.0]
    assert band_info["resolved_wavelengths_nm"] == [830.0, 670.0]
    assert band_info["rendering"] == "hsv_colormap"
    assert band_info["colormap"] == "hsv"
    assert band_info["colormap_min"] == -0.7
    assert band_info["colormap_max"] == 0.5


@torch.no_grad()
def test_ndvi_selector_divide_by_zero_returns_zero_index() -> None:
    cube = torch.tensor(
        [
            [
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    wavelengths = np.array([650.0, 668.0, 827.0], dtype=np.float32)

    node = NDVISelector(
        nir_nm=827.0,
        red_nm=668.0,
        colormap_min=-1.0,
        colormap_max=1.0,
        eps=1.0e-6,
    )
    result = node.forward(cube=cube, wavelengths=wavelengths)

    expected_index = torch.tensor([[[[0.0], [1.0]]]], dtype=torch.float32)
    assert torch.allclose(result["index_image"], expected_index, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        result["rgb_image"][0, 0, 0],
        torch.tensor([0.0, 1.0, 1.0]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert torch.allclose(
        result["rgb_image"][0, 0, 1],
        torch.tensor([1.0, 0.0, 0.0]),
        atol=1e-6,
        rtol=1e-6,
    )


@torch.no_grad()
def test_ndvi_selector_applies_hsv_colormap_across_index_range() -> None:
    cube = torch.tensor(
        [
            [
                [
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [1.0, 4.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    wavelengths = np.array([668.0, 827.0], dtype=np.float32)

    node = NDVISelector(colormap_min=-1.0, colormap_max=1.0)
    result = node.forward(cube=cube, wavelengths=wavelengths)

    expected_rgb = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [0.8, 0.0, 1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result["rgb_image"], expected_rgb, atol=1e-5, rtol=1e-5)


def test_ndvi_selector_defaults_to_per_frame_and_no_gamma() -> None:
    node = NDVISelector()

    assert node.norm_mode == NormMode.PER_FRAME
    assert node.apply_gamma is False
    assert node.colormap == "hsv"
    assert node.colormap_min == -0.7
    assert node.colormap_max == 0.5


@pytest.mark.parametrize("cls", NORMALISED_DIFFERENCE)
def test_rebuilds_from_its_saved_hparams(cls: type) -> None:
    """The saved hparams carry only the selector's own wavelength names, so a node saved
    by cuvis-ai (``pipeline.save`` writes ``node.hparams``) can be built again from them."""
    node = cls()
    rebuilt = cls(**node.hparams)
    assert rebuilt.primary_nm == node.primary_nm
    assert rebuilt.secondary_nm == node.secondary_nm
    assert "primary_nm" not in node.hparams and "secondary_nm" not in node.hparams


@torch.no_grad()
@pytest.mark.parametrize("cls", NORMALISED_DIFFERENCE)
def test_loads_the_hparams_a_pipeline_saved_before_the_fix(cls: type) -> None:
    """A YAML written by an earlier cuvis-ai carries ``primary_nm`` and ``secondary_nm`` beside
    the selector's own names: it builds silently, reads its own names and re-saves without
    the two."""
    with _captured_warnings() as messages:
        node = cls(**_hparams_saved_before_the_fix(cls))
    assert messages == []
    assert (node.primary_nm, node.secondary_nm) == (PRIMARY_NM, SECONDARY_NM)
    assert "primary_nm" not in node.hparams and "secondary_nm" not in node.hparams

    wavelengths = np.linspace(400.0, 1000.0, 30, dtype=np.float32)
    out = node.forward(cube=torch.rand(1, 2, 3, 30), wavelengths=wavelengths)
    assert out["band_info"]["requested_wavelengths_nm"] == [PRIMARY_NM, SECONDARY_NM]

    rebuilt = cls(**node.hparams)
    assert (rebuilt.primary_nm, rebuilt.secondary_nm) == (PRIMARY_NM, SECONDARY_NM)


@pytest.mark.parametrize("cls", NORMALISED_DIFFERENCE)
@pytest.mark.parametrize("legacy", ["primary_nm", "secondary_nm"])
def test_each_legacy_operand_key_is_dropped_on_its_own(cls: type, legacy: str) -> None:
    saved = _hparams_saved_before_the_fix(cls)
    del saved["secondary_nm" if legacy == "primary_nm" else "primary_nm"]
    with _captured_warnings() as messages:
        node = cls(**saved)
    assert messages == []
    assert (node.primary_nm, node.secondary_nm) == (PRIMARY_NM, SECONDARY_NM)
    assert legacy not in node.hparams


@pytest.mark.parametrize("cls", NORMALISED_DIFFERENCE)
def test_a_differing_legacy_value_loses_to_the_own_name_with_one_warning(cls: type) -> None:
    saved = _hparams_saved_before_the_fix(cls)
    saved["primary_nm"] = PRIMARY_NM + 100.0
    with _captured_warnings() as messages:
        node = cls(**saved)
    assert len(messages) == 1
    assert cls.__name__ in messages[0] and "primary_nm" in messages[0]
    assert (node.primary_nm, node.secondary_nm) == (PRIMARY_NM, SECONDARY_NM)


@pytest.mark.parametrize("cls", NORMALISED_DIFFERENCE)
def test_a_null_legacy_value_is_dropped_silently(cls: type) -> None:
    saved = _hparams_saved_before_the_fix(cls)
    saved["primary_nm"] = None
    saved["secondary_nm"] = None
    with _captured_warnings() as messages:
        node = cls(**saved)
    assert messages == []
    assert (node.primary_nm, node.secondary_nm) == (PRIMARY_NM, SECONDARY_NM)
