"""Tests that MinMaxNormalizer maps inputs to [0, 1] when used without running stats."""

from __future__ import annotations

import numpy as np
import torch

from cuvis_ai.node.normalization import MinMaxNormalizer


@torch.no_grad()
def test_minmax_normalizer_unit_range_random() -> None:
    """Random BHWC input is mapped into [0, 1] per-sample and per-channel."""
    B, H, W, C = 2, 5, 6, 7
    x = torch.randn(B, H, W, C) * 5.0 - 2.5  # roughly in [-7.5, 7.5]

    node = MinMaxNormalizer(eps=1e-6, use_running_stats=False)
    out = node.forward(data=x)["normalized"]

    assert out.shape == x.shape

    flat = out.view(B, -1, C).cpu().numpy()
    mins = flat.min(axis=1)
    maxs = flat.max(axis=1)

    # Allow small numerical tolerance around [0, 1]
    assert np.all(mins >= -1e-6)
    assert np.all(maxs <= 1.0 + 1e-6)


@torch.no_grad()
def test_minmax_normalizer_constant_channels_stable() -> None:
    """Constant channels remain finite and within [0, 1]."""
    B, H, W, C = 1, 4, 4, 3
    x = torch.zeros(B, H, W, C)
    x[..., 0] = 5.0  # constant channel

    node = MinMaxNormalizer(eps=1e-6, use_running_stats=False)
    out = node.forward(data=x)["normalized"]

    assert out.shape == x.shape
    np_out = out.cpu().numpy()
    assert np.isfinite(np_out).all()
    assert np_out.min() >= -1e-6
    assert np_out.max() <= 1.0 + 1e-6
