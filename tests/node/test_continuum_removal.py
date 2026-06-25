from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.pretreatments import ContinuumRemoval

pytestmark = pytest.mark.unit


def _upper_hull_ref(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference upper convex hull via Andrew's monotone chain (numpy)."""
    pts = list(zip(x.tolist(), y.tolist(), strict=True))
    hull: list[tuple[float, float]] = []

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for p in pts:
        while len(hull) >= 2 and cross(hull[-2], hull[-1], p) >= 0:
            hull.pop()
        hull.append(p)
    hx = np.array([h[0] for h in hull])
    hy = np.array([h[1] for h in hull])
    return np.interp(x, hx, hy)


@torch.no_grad()
def test_continuum_removal_matches_hull_reference() -> None:
    rng = np.random.default_rng(3)
    C = 17
    wavelengths = np.sort(rng.choice(np.arange(400, 1000), size=C, replace=False)).astype(np.int32)
    cube = rng.random((1, 3, 4, C)).astype(np.float64)

    flat = cube.reshape(-1, C)
    hull = np.stack([_upper_hull_ref(wavelengths.astype(np.float64), s) for s in flat])
    ref = (flat / np.clip(hull, 1e-8, None)).reshape(cube.shape)

    node = ContinuumRemoval()
    got = node.forward(cube=torch.tensor(cube, dtype=torch.float32), wavelengths=wavelengths)[
        "cube"
    ]

    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)


@torch.no_grad()
def test_continuum_removal_flat_spectrum_is_unity() -> None:
    wavelengths = np.array([500, 600, 700, 800], dtype=np.int32)
    cube = torch.full((1, 1, 1, 4), 0.5)
    out = ContinuumRemoval().forward(cube=cube, wavelengths=wavelengths)["cube"]
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)
