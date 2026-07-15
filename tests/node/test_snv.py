from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.pretreatments import SNVCorrection

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_snv_matches_numpy_reference() -> None:
    rng = np.random.default_rng(11)
    cube = rng.random((2, 3, 4, 7)).astype(np.float64)

    mean = cube.mean(axis=-1, keepdims=True)
    # torch.std defaults to the sample standard deviation (ddof=1).
    std = cube.std(axis=-1, ddof=1, keepdims=True)
    ref = (cube - mean) / np.clip(std, 1e-8, None)

    got = SNVCorrection().forward(cube=torch.tensor(cube, dtype=torch.float32))["cube"]
    assert torch.allclose(got, torch.tensor(ref, dtype=torch.float32), atol=1e-5)
