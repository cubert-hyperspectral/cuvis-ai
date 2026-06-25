from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.pretreatments import SpectralDerivative

pytestmark = pytest.mark.unit


@torch.no_grad()
@pytest.mark.parametrize("order", [1, 2])
def test_spectral_derivative_matches_numpy_gradient(order: int) -> None:
    rng = np.random.default_rng(5)
    C = 9
    wavelengths = np.sort(rng.choice(np.arange(400, 1000), size=C, replace=False)).astype(np.int32)
    cube = rng.random((1, 2, 3, C)).astype(np.float64)
    wl_f = wavelengths.astype(np.float64)

    ref = cube
    for _ in range(order):
        ref = np.gradient(ref, wl_f, axis=-1)

    node = SpectralDerivative(order=order)
    got = node.forward(cube=torch.tensor(cube, dtype=torch.float32), wavelengths=wavelengths)[
        "cube"
    ]

    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)
