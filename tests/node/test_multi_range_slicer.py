from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.deciders import MultiRangeSlicer

pytestmark = pytest.mark.unit


@torch.no_grad()
@pytest.mark.parametrize("right", [False, True])
def test_multi_range_slicer_matches_numpy_digitize(right: bool) -> None:
    edges = [0.25, 0.5, 0.75]
    # Include exact edge values to exercise the right/left semantics.
    scores_np = np.array(
        [[[0.0], [0.25], [0.3], [0.5], [0.6], [0.75], [0.9], [1.0]]],
        dtype=np.float32,
    )[np.newaxis, ...]  # [B=1, H=1, W=8, 1]
    scores = torch.from_numpy(scores_np)

    node = MultiRangeSlicer(edges=edges, right=right)
    result = node.forward(scores=scores)

    # torch.bucketize(..., right=R) == numpy.digitize(..., right=not R)
    expected = np.digitize(scores_np.squeeze(-1), bins=edges, right=not right).astype(np.int32)

    assert result["class_mask"].dtype == torch.int32
    assert result["class_mask"].shape == (1, 1, 8)
    assert torch.equal(result["class_mask"], torch.from_numpy(expected))


@torch.no_grad()
def test_multi_range_slicer_default_edges() -> None:
    node = MultiRangeSlicer()
    assert node.edges == [0.25, 0.5, 0.75]
    assert node.right is False
