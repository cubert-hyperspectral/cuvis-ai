from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.channel_selector import NDVISelector, NormMode

pytestmark = pytest.mark.unit


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
