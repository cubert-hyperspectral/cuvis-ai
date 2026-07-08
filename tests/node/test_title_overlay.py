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


def test_per_frame_caption_port_draws_distinct_captions():
    """A per-frame caption list captions each frame independently, overriding the default."""
    frame = _frame()
    out = TitleOverlay(text="default").forward(frame=frame, caption=["first", "second column"])[
        "frame"
    ]

    assert out.shape == frame.shape
    # Each frame got a (different-length) caption, so neither matches the bare frame...
    assert not torch.allclose(out[0], frame[0])
    assert not torch.allclose(out[1], frame[1])
    # ...and the two distinct captions render differently from each other.
    assert not torch.allclose(out[0], out[1])


def test_caption_port_takes_priority_over_text_arg():
    """The caption port wins over both the text= arg and the constructor default."""
    frame = _frame()
    node = TitleOverlay(text="ctor")
    from_port = node.forward(frame=frame, caption=["A", "B"], text="ignored")["frame"]
    from_text = node.forward(frame=frame, text="ignored")["frame"]
    assert not torch.allclose(from_port, from_text)


def test_per_frame_empty_caption_is_passthrough_for_that_frame():
    """An empty per-frame caption leaves only that frame unchanged."""
    frame = _frame()
    out = TitleOverlay().forward(frame=frame, caption=["", "labelled"])["frame"]
    assert torch.allclose(out[0], frame[0], atol=1e-6)
    assert not torch.allclose(out[1], frame[1])


def test_caption_length_mismatch_raises():
    """A caption list whose length != batch size is a wiring error, not a silent crop."""
    frame = _frame()  # batch of 2
    with pytest.raises(ValueError, match="caption has 1 entries but the batch has 2"):
        TitleOverlay().forward(frame=frame, caption=["only one"])


def test_invalid_box_alpha_rejected():
    with pytest.raises(ValueError, match="box_alpha"):
        TitleOverlay(box_alpha=1.5)


def test_missing_truetype_font_falls_back_to_default(monkeypatch: pytest.MonkeyPatch):
    """When arial.ttf is unavailable, PIL's default font is used and drawing still works."""
    from cuvis_ai.node import compositing

    real_truetype = compositing.ImageFont.truetype

    def fake_truetype(font=None, *args: object, **kwargs: object):
        # Fail only the named-font lookup; load_default's internal call passes a buffer.
        if isinstance(font, str):
            raise OSError("no truetype font")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(compositing.ImageFont, "truetype", fake_truetype)
    node = TitleOverlay(text="fallback")
    out = node.forward(frame=_frame())["frame"]
    assert out.shape == (B, H, W, 3)
    assert not torch.allclose(out, _frame())
