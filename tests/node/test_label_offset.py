"""Tests for LabelOffset: add a constant to every label in an integer map."""

from __future__ import annotations

import torch

from cuvis_ai.node.mask_ops import LabelOffset


def test_default_offset_is_one():
    """0-based cluster ids become 1-based so 0 stops colliding with background."""
    class_map = torch.tensor([[[0, 1, 2]]], dtype=torch.int32)
    out = LabelOffset().forward(class_map=class_map)["class_map"]
    assert out.tolist() == [[[1, 2, 3]]]


def test_custom_offset():
    class_map = torch.tensor([[[0, 5]]], dtype=torch.int32)
    out = LabelOffset(offset=10).forward(class_map=class_map)["class_map"]
    assert out.tolist() == [[[10, 15]]]


def test_output_dtype_int32():
    out = LabelOffset().forward(class_map=torch.zeros(1, 2, 2, dtype=torch.int32))["class_map"]
    assert out.dtype == torch.int32


def test_shape_preserved_batched():
    class_map = torch.zeros(3, 4, 5, dtype=torch.int32)
    out = LabelOffset(offset=2).forward(class_map=class_map)["class_map"]
    assert out.shape == (3, 4, 5)
    assert torch.equal(out, torch.full((3, 4, 5), 2, dtype=torch.int32))
