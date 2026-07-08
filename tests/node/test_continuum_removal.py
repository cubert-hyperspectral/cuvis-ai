from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.spatial import ConvexHull

from cuvis_ai.node.pretreatments import ContinuumRemoval

pytestmark = pytest.mark.unit


def _upper_hull_ref(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Reference upper convex hull via scipy's Qhull wrapper."""
    vertices = ConvexHull(np.column_stack([x, y])).vertices  # counter-clockwise
    # Walking counter-clockwise from the rightmost vertex to the leftmost one
    # traverses exactly the upper chain of the hull.
    idx = int(np.argmax(x[vertices]))
    end = int(np.argmin(x[vertices]))
    chain = [vertices[idx]]
    while idx != end:
        idx = (idx + 1) % len(vertices)
        chain.append(vertices[idx])
    chain = chain[::-1]  # ascending x for np.interp
    return np.interp(x, x[chain], y[chain])


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
