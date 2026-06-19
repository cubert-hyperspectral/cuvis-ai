"""Tests for DisplayNormalizer (stateless sRGB gamma companding)."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.channel_selector import ChannelSelectorBase
from cuvis_ai.node.normalization import DisplayNormalizer

pytestmark = pytest.mark.unit


def test_matches_selector_srgb_gamma() -> None:
    """Output must equal the selector's _srgb_gamma bit-for-bit."""
    x = torch.rand(2, 4, 4, 3)
    out = DisplayNormalizer().forward(data=x)["normalized"]
    assert torch.allclose(out, ChannelSelectorBase._srgb_gamma(x))


def test_unit_range_preserved() -> None:
    """[0, 1] input stays within [0, 1]."""
    x = torch.linspace(0.0, 1.0, steps=64).reshape(1, 8, 8, 1).repeat(1, 1, 1, 3)
    out = DisplayNormalizer().forward(data=x)["normalized"]
    assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0 + 1e-6


def test_lifts_midtones() -> None:
    """sRGB companding lifts midtones above the linear value."""
    x = torch.full((1, 2, 2, 3), 0.25)
    out = DisplayNormalizer().forward(data=x)["normalized"]
    assert float(out.mean()) > 0.25


def test_any_channel_count() -> None:
    """Gamma is element-wise, so any channel count works."""
    out = DisplayNormalizer().forward(data=torch.rand(1, 4, 4, 6))["normalized"]
    assert out.shape == (1, 4, 4, 6)


def test_stateless_no_buffers_or_params() -> None:
    node = DisplayNormalizer()
    assert list(node.buffers()) == []
    assert list(node.parameters()) == []


def test_differentiable() -> None:
    x = torch.rand(1, 4, 4, 3, requires_grad=True)
    DisplayNormalizer().forward(data=x)["normalized"].sum().backward()
    assert x.grad is not None
