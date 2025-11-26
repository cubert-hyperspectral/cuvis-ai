"""Unit tests for PerPixelUnitNorm node."""

from __future__ import annotations

import numpy as np
import torch

from cuvis_ai.node.normalization import PerPixelUnitNorm


def _random_cube(
    batches: int = 2, height: int = 8, width: int = 9, channels: int = 16, seed: int = 123
) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    return torch.randn((batches, height, width, channels), generator=rng, dtype=torch.float32)


@torch.no_grad()
def test_shape_and_properties() -> None:
    x = _random_cube(3, 10, 12, 20)
    node = PerPixelUnitNorm(eps=1e-8)

    out = node.forward(data=x)["normalized"]

    assert out.shape == x.shape

    B, H, W, C = out.shape
    flat = out.view(B * H * W, C)
    means = flat.mean(dim=1).cpu().numpy()
    norms = flat.norm(p=2, dim=1).cpu().numpy()

    assert np.isfinite(means).all() and np.isfinite(norms).all()
    assert float(np.mean(np.abs(means))) < 1e-3
    assert abs(float(np.mean(norms)) - 1.0) < 1e-2


@torch.no_grad()
def test_accepts_hwc_tensor() -> None:
    x_bhwc = _random_cube(1, 5, 6, 7)
    x_hwc = x_bhwc[0]
    node = PerPixelUnitNorm(eps=1e-8)

    # Node contract expects BHWC; callers must add batch dim for HWC tensors
    out = node.forward(data=x_hwc.unsqueeze(0))["normalized"]

    assert out.shape == x_bhwc.shape


@torch.no_grad()
def test_zero_variance_pixels_stable() -> None:
    x = _random_cube(2, 6, 7, 10)
    x[0, 0, 0, :] = 5.0
    node = PerPixelUnitNorm(eps=1e-6)

    out = node.forward(data=x)["normalized"]

    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, 0, 0, :], torch.zeros_like(out[0, 0, 0, :]))
