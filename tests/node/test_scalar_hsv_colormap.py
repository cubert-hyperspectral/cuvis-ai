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


def _reference_hsv_colormap(normalized: torch.Tensor) -> torch.Tensor:
    """The unrolled six-sector implementation shipped up to 0.17.2 (exact-output reference)."""
    hue = normalized.clamp(0.0, 1.0)
    h6 = hue * 6.0
    sector = torch.floor(h6).to(torch.int64) % 6
    frac = h6 - torch.floor(h6)
    one = torch.ones_like(hue)
    zero = torch.zeros_like(hue)
    q = 1.0 - frac
    t = frac
    red = torch.zeros_like(hue)
    green = torch.zeros_like(hue)
    blue = torch.zeros_like(hue)
    for k, (r, g, b) in enumerate(
        (
            (one, t, zero),
            (q, one, zero),
            (zero, one, t),
            (zero, q, one),
            (t, zero, one),
            (one, zero, q),
        )
    ):
        mask = sector == k
        red = torch.where(mask, r, red)
        green = torch.where(mask, g, green)
        blue = torch.where(mask, b, blue)
    return torch.cat([red, green, blue], dim=-1).clamp_(0.0, 1.0)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_render_scalar_hsv_colormap_is_bit_identical_to_the_unrolled_reference(
    dtype: torch.dtype,
) -> None:
    """Sector edges, random values, NaN and infinities render exactly as before."""
    edges = torch.arange(0, 7, dtype=dtype) / 6.0
    specials = torch.tensor([float("nan"), float("inf"), float("-inf"), -0.5, 1.5], dtype=dtype)
    values = torch.cat([edges, specials, torch.rand(2048, dtype=dtype)]).reshape(1, 1, -1, 1)
    out = render_scalar_hsv_colormap(values)
    expected = _reference_hsv_colormap(values)
    assert out.dtype == expected.dtype
    assert torch.equal(out, expected) or (
        torch.equal(torch.nan_to_num(out, nan=-1.0), torch.nan_to_num(expected, nan=-1.0))
        and torch.equal(out.isnan(), expected.isnan())
    )


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
