"""Tests for LabelOverlay: alpha-blend a colourised label map onto RGB."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.compositing import LabelOverlay

B, H, W = 1, 6, 10
ALPHA = 0.5


def _base_and_label():
    """Uniform grey base; label_rgb red on a known rectangle, black elsewhere."""
    rgb = torch.full((B, H, W, 3), 0.5, dtype=torch.float32)
    label = torch.zeros(B, H, W, 3, dtype=torch.float32)
    label[0, 1:4, 3:7] = torch.tensor([1.0, 0.0, 0.0])
    fg = torch.zeros(B, H, W, dtype=torch.bool)
    fg[0, 1:4, 3:7] = True
    return rgb, label, fg[0]


def test_blends_foreground_leaves_background():
    rgb, label, fg = _base_and_label()
    out = LabelOverlay(alpha=ALPHA).forward(rgb_image=rgb, label_rgb=label)["frame"]

    assert out.shape == rgb.shape and out.dtype == torch.float32
    # Foreground = (1-a)*base + a*label
    expected_fg = (1 - ALPHA) * 0.5 + ALPHA * torch.tensor([1.0, 0.0, 0.0])
    assert torch.allclose(out[0][fg][0], expected_fg, atol=1e-6)
    # Background unchanged
    assert torch.allclose(out[0][~fg], rgb[0][~fg], atol=1e-6)


def test_uint8_label_matches_float_label():
    """A uint8 [0,255] label map normalises to the same result as float [0,1]."""
    rgb, label_f, _ = _base_and_label()
    label_u8 = (label_f * 255).round().to(torch.uint8)
    out_f = LabelOverlay(alpha=ALPHA).forward(rgb_image=rgb, label_rgb=label_f)["frame"]
    out_u8 = LabelOverlay(alpha=ALPHA).forward(rgb_image=rgb, label_rgb=label_u8)["frame"]
    assert torch.allclose(out_f, out_u8, atol=1e-6)


def test_shape_mismatch_raises():
    rgb, label, _ = _base_and_label()
    bad = label[:, :, :-1]  # width mismatch
    with pytest.raises(ValueError, match="label_rgb"):
        LabelOverlay().forward(rgb_image=rgb, label_rgb=bad)


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError, match="alpha"):
        LabelOverlay(alpha=2.0)
