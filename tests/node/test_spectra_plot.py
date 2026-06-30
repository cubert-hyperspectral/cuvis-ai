"""Tests for SpectraPlot: multi-series per-class spectra rendered to an RGB frame."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.spectrum_plot import SpectraPlot

B, N, C = 1, 4, 39
PW, PH = 240, 180


def _inputs():
    sig = torch.rand(B, N, C, dtype=torch.float32)
    wl = torch.linspace(980, 1664, C).to(torch.int32)
    return sig, wl


def test_renders_frame_shape_dtype_range():
    sig, wl = _inputs()
    out = SpectraPlot(plot_width=PW, plot_height=PH).forward(signatures=sig, wavelengths=wl)
    img = out["rgb_image"]
    assert img.shape == (B, PH, PW, 3)
    assert img.dtype == torch.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    # Lines were drawn on a white background, so some pixels are non-white.
    assert (img < 0.95).any()


def test_valid_skips_rows_changes_render():
    """Dropping rows via `valid` produces a different (sparser) plot."""
    sig, wl = _inputs()
    node = SpectraPlot(plot_width=PW, plot_height=PH)
    full = node.forward(signatures=sig, wavelengths=wl)["rgb_image"]
    only_one = torch.zeros(B, N, dtype=torch.bool)
    only_one[0, 0] = True
    sparse = node.forward(signatures=sig, wavelengths=wl, valid=only_one)["rgb_image"]
    assert not torch.allclose(full, sparse)


def test_palette_independent_of_row_count():
    """A short palette wraps modulo, so N rows render fine with fewer colours."""
    sig, wl = _inputs()
    out = SpectraPlot(palette=[(255, 0, 0), (0, 0, 255)], plot_width=PW, plot_height=PH).forward(
        signatures=sig, wavelengths=wl
    )
    assert out["rgb_image"].shape == (B, PH, PW, 3)


def test_batch_renders_each_element():
    sig = torch.rand(2, N, C, dtype=torch.float32)
    wl = torch.linspace(980, 1664, C).to(torch.int32)
    out = SpectraPlot(plot_width=PW, plot_height=PH).forward(signatures=sig, wavelengths=wl)
    assert out["rgb_image"].shape == (2, PH, PW, 3)


def test_invalid_plot_size_rejected():
    with pytest.raises(ValueError, match="plot dimensions"):
        SpectraPlot(plot_width=16)
