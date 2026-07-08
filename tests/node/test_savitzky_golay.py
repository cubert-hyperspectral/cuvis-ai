from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.signal import savgol_filter

from cuvis_ai.node.pretreatments import SavitzkyGolay

pytestmark = pytest.mark.unit


@torch.no_grad()
@pytest.mark.parametrize("deriv", [0, 1, 2])
def test_savgol_matches_scipy_on_interior(deriv: int) -> None:
    rng = np.random.default_rng(0)
    window_length, polyorder, delta = 11, 2, 1.0
    x = rng.random((1, 2, 3, 21)).astype(np.float64)
    wavelengths = np.arange(21, dtype=np.int32)

    ref = savgol_filter(
        x, window_length, polyorder, deriv=deriv, delta=delta, axis=-1, mode="nearest"
    )

    node = SavitzkyGolay(
        window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode="nearest"
    )
    got = node.forward(cube=torch.tensor(x, dtype=torch.float32), wavelengths=wavelengths)["cube"]

    pad = window_length // 2
    interior = slice(pad, 21 - pad)
    assert torch.allclose(
        got[..., interior],
        torch.tensor(ref[..., interior], dtype=torch.float32),
        atol=1e-5,
    )


@torch.no_grad()
@pytest.mark.parametrize("mode", ["nearest", "mirror", "constant"])
@pytest.mark.parametrize(("deriv", "delta"), [(0, 1.0), (1, 4.0)])
def test_savgol_matches_scipy_including_edges(mode: str, deriv: int, delta: float) -> None:
    rng = np.random.default_rng(1)
    window_length, polyorder = 7, 2
    x = rng.random((1, 2, 3, 21)).astype(np.float64)
    wavelengths = np.arange(21, dtype=np.int32)

    ref = savgol_filter(
        x, window_length, polyorder, deriv=deriv, delta=delta, axis=-1, mode=mode, cval=0.0
    )

    node = SavitzkyGolay(
        window_length=window_length, polyorder=polyorder, deriv=deriv, delta=delta, mode=mode
    )
    got = node.forward(cube=torch.tensor(x, dtype=torch.float32), wavelengths=wavelengths)["cube"]

    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


@torch.no_grad()
def test_savgol_preserves_shape() -> None:
    node = SavitzkyGolay()
    cube = torch.rand(2, 4, 5, 21)
    out = node.forward(cube=cube, wavelengths=np.arange(21, dtype=np.int32))["cube"]
    assert out.shape == cube.shape
