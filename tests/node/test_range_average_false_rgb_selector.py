from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import ChannelSelectorBase, RangeAverageFalseRGBSelector
from examples.object_tracking.export_cu3s_false_rgb_video import _create_false_rgb_node


def _make_three_band_cube(r: float, g: float, b: float) -> torch.Tensor:
    """Build a tiny 3-band BHWC cube with per-channel spatial variation."""
    return torch.tensor(
        [
            [
                [[r, g, b], [r + 1.0, g + 1.0, b + 1.0]],
                [[r + 2.0, g + 2.0, b + 2.0], [r + 3.0, g + 3.0, b + 3.0]],
            ]
        ],
        dtype=torch.float32,
    )


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
    # Running mode warmup uses per-frame percentile normalization.
    flat = raw.reshape(-1, 3)
    lo = torch.quantile(flat, ChannelSelectorBase._NORM_QUANTILE_LOW, dim=0).view(1, 1, 1, 3)
    hi = torch.quantile(flat, ChannelSelectorBase._NORM_QUANTILE_HIGH, dim=0).view(1, 1, 1, 3)
    denom = (hi - lo).clamp_min(1e-8)
    expected = ((raw - lo) / denom).clamp(0.0, 1.0)
    expected = ChannelSelectorBase._srgb_gamma(expected)

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


def test_running_bounds_freeze_after_20_frames_blocks_late_outlier_updates() -> None:
    wavelengths = np.array([620.0, 530.0, 450.0], dtype=np.float32)
    node = RangeAverageFalseRGBSelector(
        red_range=(618.0, 622.0),
        green_range=(528.0, 532.0),
        blue_range=(448.0, 452.0),
    )

    base = _make_three_band_cube(10.0, 20.0, 30.0)
    for _ in range(20):
        node.forward(cube=base, wavelengths=wavelengths)
    max_after_20 = node.running_max.clone()

    late_outlier = _make_three_band_cube(10.0, 20.0, 300.0)
    node.forward(cube=late_outlier, wavelengths=wavelengths)

    assert torch.allclose(node.running_max, max_after_20, atol=1e-7, rtol=0.0)


def test_running_bounds_with_no_freeze_updates_on_late_outlier() -> None:
    wavelengths = np.array([620.0, 530.0, 450.0], dtype=np.float32)
    node = RangeAverageFalseRGBSelector(
        red_range=(618.0, 622.0),
        green_range=(528.0, 532.0),
        blue_range=(448.0, 452.0),
        freeze_running_bounds_after_frames=None,
    )

    base = _make_three_band_cube(10.0, 20.0, 30.0)
    for _ in range(20):
        node.forward(cube=base, wavelengths=wavelengths)
    max_after_20 = node.running_max.clone()

    late_outlier = _make_three_band_cube(10.0, 20.0, 300.0)
    node.forward(cube=late_outlier, wavelengths=wavelengths)

    assert node.running_max[2] > max_after_20[2]


def test_running_mode_warmup_outputs_per_frame_percentile_even_with_freeze_enabled() -> None:
    wavelengths = np.array([620.0, 530.0, 450.0], dtype=np.float32)
    node = RangeAverageFalseRGBSelector(
        red_range=(618.0, 622.0),
        green_range=(528.0, 532.0),
        blue_range=(448.0, 452.0),
    )

    prefill = _make_three_band_cube(5.0, 10.0, 15.0)
    for _ in range(ChannelSelectorBase._WARMUP_FRAMES - 1):
        node.forward(cube=prefill, wavelengths=wavelengths)

    target = _make_three_band_cube(40.0, 80.0, 120.0)
    raw_target = node._compute_raw_rgb(target, wavelengths)
    expected = node._per_frame_percentile_normalize(raw_target)
    expected = ChannelSelectorBase._srgb_gamma(expected)

    out = node.forward(cube=target, wavelengths=wavelengths)
    assert torch.allclose(out["rgb_image"], expected, atol=1e-6, rtol=1e-6)


def test_freeze_running_bounds_after_frames_validation() -> None:
    with pytest.raises(ValueError, match="freeze_running_bounds_after_frames"):
        RangeAverageFalseRGBSelector(freeze_running_bounds_after_frames=0)
    with pytest.raises(ValueError, match="freeze_running_bounds_after_frames"):
        RangeAverageFalseRGBSelector(freeze_running_bounds_after_frames=-5)
    with pytest.raises(ValueError, match="freeze_running_bounds_after_frames"):
        RangeAverageFalseRGBSelector(freeze_running_bounds_after_frames=2.5)
    with pytest.raises(ValueError, match="freeze_running_bounds_after_frames"):
        RangeAverageFalseRGBSelector(freeze_running_bounds_after_frames="20")


def test_export_false_rgb_node_propagates_freeze_running_bounds_parameter() -> None:
    node = _create_false_rgb_node(
        "cie_tristimulus",
        freeze_running_bounds_after_frames=7,
    )
    assert node.freeze_running_bounds_after_frames == 7
