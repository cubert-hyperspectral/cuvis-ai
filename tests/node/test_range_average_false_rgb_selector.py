from __future__ import annotations

import numpy as np
import torch

from cuvis_ai.node.band_selection import BandSelectorBase, RangeAverageFalseRGBSelector


def test_range_average_false_rgb_selector_averages_bands_in_ranges(create_test_cube) -> None:
    red_range = (610.0, 750.0)
    green_range = (500.0, 610.0)
    blue_range = (420.0, 500.0)

    node = RangeAverageFalseRGBSelector(
        red_range=red_range,
        green_range=green_range,
        blue_range=blue_range,
    )

    cube, wavelengths_2d = create_test_cube(
        batch_size=1,
        height=2,
        width=2,
        num_channels=5,
        mode="random",
        dtype=torch.float32,
    )
    wavelengths = wavelengths_2d[0].cpu().numpy()

    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]
    info = out["band_info"]

    # Compute expected band indices from generated wavelengths.
    red_idx = [i for i, w in enumerate(wavelengths) if red_range[0] <= w <= red_range[1]]
    green_idx = [i for i, w in enumerate(wavelengths) if green_range[0] <= w <= green_range[1]]
    blue_idx = [i for i, w in enumerate(wavelengths) if blue_range[0] <= w <= blue_range[1]]

    red = cube[..., red_idx].mean(dim=-1)
    green = cube[..., green_idx].mean(dim=-1)
    blue = cube[..., blue_idx].mean(dim=-1)
    raw = torch.stack([red, green, blue], dim=-1)
    rgb_min = raw.amin(dim=(1, 2), keepdim=True)
    rgb_max = raw.amax(dim=(1, 2), keepdim=True)
    denom = (rgb_max - rgb_min).clamp_min(1e-8)
    expected = ((raw - rgb_min) / denom).clamp(0.0, 1.0)
    expected = BandSelectorBase._srgb_gamma(expected)

    assert torch.allclose(rgb, expected, atol=1e-6, rtol=1e-6)
    assert info["strategy"] == "range_average_false_rgb"
    assert info["band_indices"] == [red_idx, green_idx, blue_idx]
    assert info["missing_channels"] == []


def test_range_average_false_rgb_selector_handles_missing_channel_ranges(create_test_cube) -> None:
    node = RangeAverageFalseRGBSelector(
        red_range=(580.0, 650.0),
        green_range=(500.0, 580.0),
        blue_range=(900.0, 950.0),  # intentionally out of range
    )

    wavelengths = np.array([430, 500, 560, 620, 700], dtype=np.int32)
    cube, _ = create_test_cube(
        batch_size=1,
        height=3,
        width=4,
        num_channels=5,
        mode="random",
        dtype=torch.float32,
    )

    out = node.forward(cube=cube, wavelengths=wavelengths)
    rgb = out["rgb_image"]
    info = out["band_info"]

    assert rgb.shape == (1, 3, 4, 3)
    assert torch.allclose(rgb[..., 2], torch.zeros_like(rgb[..., 2]))  # Blue channel is missing
    assert info["band_indices"][2] == []
    assert info["missing_channels"] == ["blue"]


def test_range_average_false_rgb_selector_accepts_batched_wavelengths(create_test_cube) -> None:
    node = RangeAverageFalseRGBSelector()

    cube, wavelengths = create_test_cube(
        batch_size=2,
        height=4,
        width=4,
        num_channels=6,
        mode="random",
        dtype=torch.float32,
    )

    out = node.forward(cube=cube, wavelengths=wavelengths)
    assert out["rgb_image"].shape == (2, 4, 4, 3)
    assert isinstance(out["band_info"], dict)
