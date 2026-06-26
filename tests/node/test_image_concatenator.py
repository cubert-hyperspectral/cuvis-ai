from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.compositing import ImageConcatenator

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_horizontal_equal_size_matches_cat() -> None:
    a = torch.rand(1, 4, 5, 3)
    b = torch.rand(1, 4, 7, 3)
    out = ImageConcatenator().forward(images=[a, b])["rgb_image"]
    assert out.dtype == torch.float32
    assert out.shape == (1, 4, 12, 3)
    assert torch.allclose(out, torch.cat([a, b], dim=2))


@torch.no_grad()
def test_vertical_equal_size_matches_cat() -> None:
    a = torch.rand(2, 4, 6, 3)
    b = torch.rand(2, 3, 6, 3)
    out = ImageConcatenator(axis="vertical").forward(images=[a, b])["rgb_image"]
    assert out.shape == (2, 7, 6, 3)
    assert torch.allclose(out, torch.cat([a, b], dim=1))


@torch.no_grad()
def test_gap_inserts_bg_separator() -> None:
    a = torch.zeros(1, 4, 3, 3)
    b = torch.zeros(1, 4, 2, 3)
    out = ImageConcatenator(gap=2, bg_color=(1.0, 0.0, 0.0)).forward(images=[a, b])["rgb_image"]
    # widths: 3 + gap(2) + 2 = 7
    assert out.shape == (1, 4, 7, 3)
    gap_cols = out[0, :, 3:5, :]
    assert torch.allclose(gap_cols, torch.tensor([1.0, 0.0, 0.0]).expand_as(gap_cols))


@torch.no_grad()
def test_unequal_cross_axis_is_padded_and_aligned() -> None:
    tall = torch.rand(1, 6, 4, 3)
    short = torch.rand(1, 2, 4, 3)
    out = ImageConcatenator(bg_color=(0.5, 0.5, 0.5), align="center").forward(images=[tall, short])[
        "rgb_image"
    ]
    # horizontal strip: common height = max(6, 2) = 6, width 4 + 4 = 8
    assert out.shape == (1, 6, 8, 3)
    short_panel = out[0, :, 4:8, :]
    # center align: 2 padding rows top, 2 bottom, image rows 2..4
    bg = torch.tensor([0.5, 0.5, 0.5])
    assert torch.allclose(short_panel[:2], bg.expand(2, 4, 3))
    assert torch.allclose(short_panel[4:], bg.expand(2, 4, 3))
    assert torch.allclose(short_panel[2:4], short[0])


@torch.no_grad()
def test_fan_in_order_is_preserved() -> None:
    red = torch.zeros(1, 2, 1, 3)
    red[..., 0] = 1.0
    green = torch.zeros(1, 2, 1, 3)
    green[..., 1] = 1.0
    out = ImageConcatenator().forward(images=[red, green])["rgb_image"]
    assert torch.allclose(out[0, :, 0, :], torch.tensor([1.0, 0.0, 0.0]).expand(2, 3))
    assert torch.allclose(out[0, :, 1, :], torch.tensor([0.0, 1.0, 0.0]).expand(2, 3))


@torch.no_grad()
def test_output_is_clamped() -> None:
    out = ImageConcatenator().forward(images=[torch.full((1, 2, 2, 3), 5.0)])["rgb_image"]
    assert float(out.max()) <= 1.0


@torch.no_grad()
def test_errors() -> None:
    with pytest.raises(ValueError, match="no images"):
        ImageConcatenator().forward(images=[])
    with pytest.raises(ValueError, match="batch"):
        ImageConcatenator().forward(images=[torch.rand(1, 2, 2, 3), torch.rand(2, 2, 2, 3)])
    with pytest.raises(ValueError, match=r"\[B, H, W, 3\]"):
        ImageConcatenator().forward(images=[torch.rand(1, 2, 2)])
    with pytest.raises(ValueError, match="axis"):
        ImageConcatenator(axis="diagonal")
    with pytest.raises(ValueError, match="align"):
        ImageConcatenator(align="middle")
