"""Unit tests for DecisionToMask."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.conversion import DecisionToMask

pytestmark = pytest.mark.unit


def test_single_identity_passthrough() -> None:
    decisions = torch.tensor([[[[True], [False], [True]]]], dtype=torch.bool)  # [1,1,3,1]
    identity_mask = torch.ones((1, 1, 3), dtype=torch.int32)

    out = DecisionToMask().forward(decisions=decisions, identity_mask=identity_mask)["mask"]
    expected = decisions.squeeze(-1).to(torch.int32)
    assert torch.equal(out, expected)


def test_multi_identity_masking() -> None:
    decisions = torch.tensor(
        [
            [
                [[True], [False], [True]],
                [[False], [True], [True]],
            ]
        ],
        dtype=torch.bool,
    )  # [1,2,3,1]
    identity_mask = torch.tensor(
        [
            [
                [1, 2, 3],
                [3, 2, 1],
            ]
        ],
        dtype=torch.int32,
    )  # [1,2,3]

    out = DecisionToMask().forward(decisions=decisions, identity_mask=identity_mask)["mask"]
    expected = torch.tensor([[[1, 0, 3], [0, 2, 1]]], dtype=torch.int32)
    assert torch.equal(out, expected)


def test_shape_output() -> None:
    decisions = torch.zeros((2, 4, 5, 1), dtype=torch.bool)
    identity_mask = torch.ones((2, 4, 5), dtype=torch.int32)

    out = DecisionToMask().forward(decisions=decisions, identity_mask=identity_mask)["mask"]
    assert out.shape == (2, 4, 5)
    assert out.dtype == torch.int32
