from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.optimize import nnls

from cuvis_ai.node.unmixing import NNLSUnmixing

pytestmark = pytest.mark.unit


@torch.no_grad()
def test_nnls_matches_scipy_per_pixel() -> None:
    torch.manual_seed(0)

    components, channels = 3, 8
    # Well-conditioned endmember matrix [K, C].
    endmembers = torch.rand(components, channels, dtype=torch.float32) + 0.5
    a_cxk = endmembers.transpose(0, 1)  # [C, K], matches scipy's M

    # Random non-negative abundances, build b = A x exactly so the NNLS fit is tight.
    height, width = 4, 5
    abundances = torch.rand(height * width, components, dtype=torch.float32)
    b = abundances @ endmembers  # [P, C]
    cube = b.reshape(1, height, width, channels)

    node = NNLSUnmixing(max_iter=2000, tol=1e-8)
    out = node.forward(cube=cube, endmembers=endmembers)

    abund_out = out["abundances"].reshape(-1, components).numpy()
    scores_out = out["scores"].reshape(-1).numpy()

    m = a_cxk.numpy()  # [C, K]
    for p in range(0, height * width, 3):
        y = b[p].numpy()
        x_scipy, res_scipy = nnls(m, y)
        assert np.allclose(abund_out[p], x_scipy, atol=1e-4)
        assert abs(scores_out[p] - res_scipy) < 1e-5


@torch.no_grad()
def test_nnls_output_shapes_and_class_mask() -> None:
    components, channels = 3, 6
    endmembers = torch.eye(components, channels, dtype=torch.float32) + 0.1
    cube = torch.rand(2, 3, 4, channels, dtype=torch.float32)

    node = NNLSUnmixing(max_iter=500)
    out = node.forward(cube=cube, endmembers=endmembers)

    assert out["abundances"].shape == (2, 3, 4, components)
    assert out["scores"].shape == (2, 3, 4, 1)
    assert out["class_mask"].shape == (2, 3, 4)
    assert out["class_mask"].dtype == torch.int32
    # Argmax is 1-based and within [1, K] when min_total is 0.
    assert out["class_mask"].min().item() >= 1
    assert out["class_mask"].max().item() <= components


@torch.no_grad()
def test_nnls_min_total_masks_background() -> None:
    components, channels = 3, 5
    endmembers = torch.rand(components, channels, dtype=torch.float32) + 0.5
    # All-zero cube -> zero abundances -> total below any positive min_total.
    cube = torch.zeros(1, 2, 2, channels, dtype=torch.float32)

    node = NNLSUnmixing(max_iter=200, min_total=1e-3)
    out = node.forward(cube=cube, endmembers=endmembers)

    assert torch.all(out["class_mask"] == 0)
