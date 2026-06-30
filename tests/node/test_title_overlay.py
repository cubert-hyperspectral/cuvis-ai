"""Tests for TitleOverlay: burns a per-instance caption into RGB frames."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.compositing import TitleOverlay

B, H, W = 2, 24, 80


def _frame(value: float = 0.3) -> torch.Tensor:
    return torch.full((B, H, W, 3), value, dtype=torch.float32)


def test_caption_changes_pixels_and_preserves_shape():
    """The constructor caption is drawn in, changing pixels but keeping shape/dtype/range."""
    frame = _frame()
    out = TitleOverlay(text="compartment").forward(frame=frame)["frame"]

    assert out.shape == frame.shape
    assert out.dtype == torch.float32
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert not torch.allclose(out, frame), "caption did not modify any pixels"


def test_explicit_text_arg_overrides_constructor():
    """Passing text= to forward overrides the constructor default."""
    frame = _frame()
    node = TitleOverlay(text="default")
    out_default = node.forward(frame=frame)["frame"]
    out_override = node.forward(frame=frame, text="other caption here")["frame"]

    # Both draw something; the two captions differ so the rendered pixels differ.
    assert not torch.allclose(out_default, frame)
    assert not torch.allclose(out_default, out_override)


def test_empty_text_leaves_frame_effectively_unchanged():
    """An empty caption draws no visible box/text, so the frame is preserved."""
    frame = _frame()
    out = TitleOverlay(text="").forward(frame=frame)["frame"]
    assert torch.allclose(out, frame, atol=1e-6)


def test_invalid_box_alpha_rejected():
    with pytest.raises(ValueError, match="box_alpha"):
        TitleOverlay(box_alpha=1.5)
