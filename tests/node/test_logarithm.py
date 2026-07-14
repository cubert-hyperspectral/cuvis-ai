from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.pretreatments import Logarithm

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_logarithm_log10_matches_numpy() -> None:
    rng = np.random.default_rng(13)
    cube = rng.random((1, 2, 3, 6)).astype(np.float64)
    ref = np.log10(np.clip(cube, 1e-8, None))
    got = Logarithm(mode="log10").forward(cube=torch.tensor(cube, dtype=torch.float32))["cube"]
    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


@torch.no_grad()
def test_logarithm_ln_matches_numpy_and_clamps() -> None:
    cube = np.array([[[[0.0, 1e-12, 0.5, 1.0]]]], dtype=np.float64)
    ref = np.log(np.clip(cube, 1e-8, None))
    got = Logarithm(mode="ln").forward(cube=torch.tensor(cube, dtype=torch.float32))["cube"]
    assert torch.isfinite(got).all()
    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


@torch.no_grad()
def test_logarithm_negate_gives_absorbance() -> None:
    rng = np.random.default_rng(7)
    cube = rng.random((1, 2, 3, 6)).astype(np.float64)
    ref = -np.log10(np.clip(cube, 1e-8, None))  # absorbance A = -log10(R)
    got = Logarithm(mode="log10", negate=True).forward(
        cube=torch.tensor(cube, dtype=torch.float32)
    )["cube"]
    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


def test_logarithm_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError):
        Logarithm(mode="log2")
