from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.colormap import ScalarHSVColormapNode, render_scalar_hsv_colormap

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_render_scalar_hsv_colormap_matches_expected_reference_colors() -> None:
    normalized = torch.tensor(
        [
            [
                [
                    [0.0],
                    [0.5],
                    [0.8],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    rgb = render_scalar_hsv_colormap(normalized)

    expected = torch.tensor(
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
    assert torch.allclose(rgb, expected, atol=1e-6, rtol=1e-6)


@torch.no_grad()
def test_scalar_hsv_colormap_node_applies_custom_value_range_and_clamps() -> None:
    data = torch.tensor(
        [
            [
                [
                    [-1.0],
                    [0.0],
                    [1.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )

    node = ScalarHSVColormapNode(value_min=-1.0, value_max=1.0)
    result = node.forward(data=data)

    expected = torch.tensor(
        [
            [
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 1.0],
                    [1.0, 0.0, 0.0],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(result["rgb_image"], expected, atol=1e-6, rtol=1e-6)


def test_scalar_hsv_colormap_node_validates_shape_and_range() -> None:
    with pytest.raises(ValueError, match="value_max"):
        ScalarHSVColormapNode(value_min=1.0, value_max=1.0)

    node = ScalarHSVColormapNode()
    with pytest.raises(ValueError, match=r"\[B, H, W, 1\]"):
        node.forward(data=torch.zeros((1, 2, 2, 3), dtype=torch.float32))
