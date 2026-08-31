"""Tests for LogitsToClassMap: per-pixel argmax of segmentation logits."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.conversion import LogitsToClassMap

pytestmark = pytest.mark.unit


def test_argmax_matches_handbuilt_and_torch():
    """class_map is the per-pixel argmax over the class axis (golden reference)."""
    # [B=1, H=1, W=3, C=3]: winners are classes 0, 2, 1.
    logits = torch.tensor([[[[2.0, 1.0, 0.5], [0.1, 0.2, 5.0], [0.0, 3.0, 1.0]]]])
    out = LogitsToClassMap().forward(logits=logits)["class_map"]

    assert torch.equal(out, torch.tensor([[[0, 2, 1]]], dtype=torch.int32))
    assert torch.equal(out, logits.argmax(dim=-1).to(torch.int32))


def test_port_contract_shape_and_dtype():
    """Output drops the class axis -> [B, H, W] int32."""
    out = LogitsToClassMap().forward(logits=torch.randn(2, 4, 5, 3))["class_map"]

    assert out.shape == (2, 4, 5)
    assert out.dtype == torch.int32
    assert int(out.max()) <= 2 and int(out.min()) >= 0


def test_two_class_equals_foreground_probability_half():
    """For 2 classes, argmax == (P(shell) >= 0.5) -- the SegMetrics foreground rule."""
    logits = torch.randn(1, 8, 8, 2)
    class_map = LogitsToClassMap().forward(logits=logits)["class_map"]
    fg_by_prob = (torch.softmax(logits, dim=-1)[..., 1] >= 0.5).to(torch.int32)

    assert torch.equal(class_map, fg_by_prob)
