"""CIETristimulusRGBSelector: output shape and range, band info, NIR rejection, CMF caching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import CIETristimulusRGBSelector

pytestmark = pytest.mark.unit


def _cube_and_wavelengths() -> tuple[torch.Tensor, np.ndarray]:
    wavelengths = np.arange(400.0, 1000.0, 20.0, dtype=np.float32)  # 30 bands, 400 to 980 nm
    cube = torch.rand(2, 4, 5, wavelengths.size, generator=torch.Generator().manual_seed(0))
    return cube, wavelengths


def test_cie_selector_renders_unit_range_rgb_with_band_info() -> None:
    cube, wavelengths = _cube_and_wavelengths()
    node = CIETristimulusRGBSelector(apply_gamma=False)
    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]
    assert rgb.shape == (2, 4, 5, 3)
    assert float(rgb.min()) >= 0.0 and float(rgb.max()) <= 1.0

    info = out["band_info"]
    assert info["strategy"] == "cie_tristimulus" and info["illuminant"] == "D65"
    assert info["apply_gamma"] is False
    assert info["sensor_bands_total"] == 30
    assert info["wavelength_range_nm"] == [400.0, 980.0]
    cmf_sum = np.sum(
        [
            np.interp(wavelengths, node._CMF_WAVELENGTHS, bar, left=0.0, right=0.0)
            for bar in (node._X_BAR, node._Y_BAR, node._Z_BAR)
        ],
        axis=0,
    )
    assert info["sensor_bands_visible"] == int((cmf_sum > 1e-6).sum())
    assert 18 <= info["sensor_bands_visible"] <= 20


def test_cie_selector_gives_no_colour_to_nir_only_energy() -> None:
    wavelengths = np.arange(400.0, 1000.0, 20.0, dtype=np.float32)
    cube = torch.zeros(1, 2, 2, wavelengths.size)
    cube[..., torch.from_numpy(wavelengths > 800.0)] = 1.0
    raw = CIETristimulusRGBSelector()._compute_raw_rgb(cube, wavelengths)
    assert torch.equal(raw, torch.zeros_like(raw))


def test_cie_selector_caches_the_cmf_weights_per_wavelength_grid() -> None:
    cube, wavelengths = _cube_and_wavelengths()
    node = CIETristimulusRGBSelector()
    node.forward(cube=cube, wavelengths=wavelengths)
    weights = node._cmf_weights
    assert weights is not None and tuple(weights.shape) == (3, 30)

    node.forward(cube=cube, wavelengths=wavelengths)
    assert node._cmf_weights is weights

    node.forward(cube=cube[..., :15], wavelengths=wavelengths[:15])
    assert node._cmf_weights is not weights and tuple(node._cmf_weights.shape) == (3, 15)
    assert (
        node.forward(cube=cube[..., :15], wavelengths=wavelengths[:15])["band_info"][
            "sensor_bands_total"
        ]
        == 15
    )
