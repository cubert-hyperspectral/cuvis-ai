"""Tests for LegendStrip: append a class-colour legend below an RGB frame."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.compositing import LegendStrip

B, H, W = 1, 30, 120
ENTRIES = [("a", (255, 0, 0)), ("b", (0, 255, 0)), ("c", (0, 0, 255))]
TILE, PAD = 22, 6


def _frame() -> torch.Tensor:
    return torch.full((B, H, W, 3), 0.2, dtype=torch.float32)


def _legend_h(n_entries: int, n_columns: int) -> int:
    n_rows = (n_entries + n_columns - 1) // n_columns
    return n_rows * TILE + 2 * PAD


def test_appends_strip_below_frame():
    node = LegendStrip(entries=ENTRIES, n_columns=3)
    out = node.forward(frame=_frame())["frame"]
    expected_h = H + _legend_h(len(ENTRIES), 3)
    assert out.shape == (B, expected_h, W, 3)
    assert out.dtype == torch.float32
    # Top region is the untouched frame.
    assert torch.allclose(out[:, :H], _frame())


def test_n_columns_wrapping_increases_height():
    """One column wraps every entry onto its own row, growing the strip."""
    one_col = LegendStrip(entries=ENTRIES, n_columns=1).forward(frame=_frame())["frame"]
    three_col = LegendStrip(entries=ENTRIES, n_columns=3).forward(frame=_frame())["frame"]
    assert one_col.shape[1] == H + _legend_h(len(ENTRIES), 1)
    assert one_col.shape[1] > three_col.shape[1]


def test_instance_counts_run_with_label_rgb():
    """Supplying label_rgb exercises the connected-component count path without error."""
    label = torch.zeros(B, H, W, 3, dtype=torch.float32)
    label[0, 2:6, 2:6] = torch.tensor([1.0, 0.0, 0.0])  # one red blob -> count 1 for entry 'a'
    out = LegendStrip(entries=ENTRIES, n_columns=3).forward(frame=_frame(), label_rgb=label)[
        "frame"
    ]
    assert out.shape[1] == H + _legend_h(len(ENTRIES), 3)


def test_empty_entries_rejected():
    with pytest.raises(ValueError, match="entries"):
        LegendStrip(entries=[])
