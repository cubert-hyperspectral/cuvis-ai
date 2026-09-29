"""RangeAverageFalseRGBSelector: band averaging, band info and the per-grid weight cache."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import RangeAverageFalseRGBSelector

pytestmark = pytest.mark.unit


def _node() -> RangeAverageFalseRGBSelector:
    return RangeAverageFalseRGBSelector(norm_mode="per_frame", apply_gamma=False)


def test_range_average_selector_averages_the_bands_of_each_range() -> None:
    wavelengths = np.array([430.0, 470.0, 520.0, 560.0, 600.0, 640.0, 900.0], dtype=np.float32)
    cube = torch.zeros(1, 2, 2, wavelengths.size)
    cube[..., :] = torch.tensor([1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 100.0])
    node = _node()

    # red 580-650 -> (9 + 11) / 2, green 500-580 -> (5 + 7) / 2, blue 420-500 -> (1 + 3) / 2
    raw = node._compute_raw_rgb(cube, wavelengths)
    torch.testing.assert_close(raw[0, 0, 0], torch.tensor([10.0, 6.0, 2.0]))

    out = node.forward(cube=cube, wavelengths=wavelengths)
    assert out["rgb_image"].shape == (1, 2, 2, 3)
    info = out["band_info"]
    assert info["strategy"] == "range_average_false_rgb"
    assert info["band_indices"] == [[4, 5], [2, 3], [0, 1]]
    assert info["band_wavelengths_nm"] == [[600.0, 640.0], [520.0, 560.0], [430.0, 470.0]]
    assert info["ranges_nm"] == {
        "red": [580.0, 650.0],
        "green": [500.0, 580.0],
        "blue": [420.0, 500.0],
    }
    assert info["aggregation"] == "mean"
    assert info["missing_channels"] == []


def test_range_average_selector_reports_a_range_without_bands() -> None:
    wavelengths = np.array([520.0, 560.0, 600.0], dtype=np.float32)
    cube = torch.rand(1, 2, 2, wavelengths.size)
    node = _node()
    raw = node._compute_raw_rgb(cube, wavelengths)
    assert torch.equal(raw[..., 2], torch.zeros(1, 2, 2))
    info = node.forward(cube=cube, wavelengths=wavelengths)["band_info"]
    assert info["missing_channels"] == ["blue"]
    assert info["band_indices"] == [[2], [0, 1], []]


def test_range_average_selector_rebuilds_its_weights_only_when_the_grid_changes() -> None:
    wavelengths = np.array([430.0, 520.0, 600.0], dtype=np.float32)
    cube = torch.rand(1, 2, 2, wavelengths.size)
    node = _node()
    node.forward(cube=cube, wavelengths=wavelengths)
    weights = node._avg_weights
    assert weights is not None and tuple(weights.shape) == (3, 3)
    node.forward(cube=cube, wavelengths=wavelengths)
    assert node._avg_weights is weights

    node.forward(cube=cube[..., :2], wavelengths=wavelengths[:2])
    assert node._avg_weights is not weights and tuple(node._avg_weights.shape) == (3, 2)
