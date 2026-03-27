from __future__ import annotations

import numpy as np
import torch

from cuvis_ai.node.channel_selector import FastRGBSelector


def _make_cube(values: list[float]) -> torch.Tensor:
    """Create a tiny BHWC cube with identical spectra for each pixel."""
    spectrum = torch.tensor(values, dtype=torch.float32).view(1, 1, 1, -1)
    return spectrum.repeat(1, 2, 2, 1)


def test_fast_rgb_dynamic_normalization_matches_parity_formula() -> None:
    wavelengths = np.array([430.0, 500.0, 560.0, 620.0, 700.0], dtype=np.float32)
    cube = _make_cube([10.0, 20.0, 30.0, 40.0, 50.0])
    node = FastRGBSelector(
        red_range=(580.0, 650.0),  # -> idx 3
        green_range=(500.0, 580.0),  # -> idx 1,2
        blue_range=(420.0, 500.0),  # -> idx 0,1
        normalization_strength=0.75,
    )

    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]

    # Raw channel means: R=40, G=25, B=15
    # Global mean = (40+25+15)/3 = 26.666..., factor=(0.75*128)/mean = 3.6
    expected_u8 = torch.tensor([144, 90, 54], dtype=torch.uint8).view(1, 1, 1, 3)
    expected = expected_u8.to(torch.float32) / 255.0
    expected = expected.repeat(1, 2, 2, 1)

    assert torch.allclose(rgb, expected, atol=1e-7, rtol=0.0)
    assert out["band_info"]["strategy"] == "fast_rgb"
    assert out["band_info"]["normalization_strength"] == 0.75


def test_fast_rgb_static_scaling_matches_reflectance_mode() -> None:
    wavelengths = np.array([430.0, 500.0, 560.0, 620.0, 700.0], dtype=np.float32)
    cube = _make_cube([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    node = FastRGBSelector(
        red_range=(580.0, 650.0),  # -> idx 3
        green_range=(500.0, 580.0),  # -> idx 1,2
        blue_range=(420.0, 500.0),  # -> idx 0,1
        normalization_strength=0.0,
    )

    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]

    # Static factor = 255/10000 = 0.0255
    # Raw: R=4000,G=2500,B=1500 -> scaled [102.0,63.75,38.25] -> round [102,64,38]
    expected_u8 = torch.tensor([102, 64, 38], dtype=torch.uint8).view(1, 1, 1, 3)
    expected = expected_u8.to(torch.float32) / 255.0
    expected = expected.repeat(1, 2, 2, 1)

    assert torch.allclose(rgb, expected, atol=1e-7, rtol=0.0)
    assert out["band_info"]["normalization_strength"] == 0.0


def test_fast_rgb_invalid_channel_range_outputs_zeros() -> None:
    wavelengths = np.array([430.0, 500.0, 560.0, 620.0, 700.0], dtype=np.float32)
    cube = _make_cube([100.0, 200.0, 300.0, 400.0, 500.0])
    node = FastRGBSelector(
        red_range=(580.0, 650.0),
        green_range=(500.0, 580.0),
        blue_range=(900.0, 950.0),  # No overlap with wavelengths -> invalid channel
        normalization_strength=0.75,
    )

    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]
    info = out["band_info"]

    assert torch.allclose(rgb[..., 2], torch.zeros_like(rgb[..., 2]))
    assert info["band_indices"][2] == []
    assert info["missing_channels"] == ["blue"]
