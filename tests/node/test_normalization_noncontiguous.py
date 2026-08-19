"""Regression tests: normalizers must accept non-contiguous BHWC inputs.

Cropped/sliced upstream tensors (e.g. a spatial lane-crop node) are
non-contiguous; flattening them with ``Tensor.view`` raised a stride error.
The normalizers now flatten with ``reshape``, which handles such inputs.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch

from cuvis_ai.node.normalization import (
    MinMaxNormalizer,
    PerPixelUnitNorm,
    SigmoidNormalizer,
    ZScoreNormalizer,
)

pytestmark = pytest.mark.unit


@torch.no_grad()
@pytest.mark.parametrize(
    ("node_cls", "kwargs"),
    [
        (MinMaxNormalizer, {"use_running_stats": False}),
        (SigmoidNormalizer, {}),
        (ZScoreNormalizer, {}),
        (PerPixelUnitNorm, {}),
    ],
)
def test_normalizers_accept_non_contiguous_input(node_cls: type, kwargs: dict[str, Any]) -> None:
    """A spatially cropped (non-contiguous) BHWC tensor normalizes without errors."""
    x = torch.rand(1, 64, 64, 8)[:, 5:60, 3:50, :]
    assert not x.is_contiguous()

    out = node_cls(**kwargs).forward(data=x)["normalized"]

    assert out.shape == x.shape
    assert torch.isfinite(out).all()
