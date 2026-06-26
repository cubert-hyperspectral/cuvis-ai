from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.mask_ops import MajorityVoteByBlob

pytestmark = pytest.mark.unit


def _reference_vote(identity: np.ndarray, blobs: np.ndarray) -> np.ndarray:
    """numpy reference: per-blob majority of nonzero identities (ties -> lowest id)."""
    out = np.zeros_like(blobs, dtype=np.int32)
    for blob_id in np.unique(blobs):
        if blob_id == 0:
            continue
        region = blobs == blob_id
        votes = identity[region]
        votes = votes[votes > 0]
        if votes.size == 0:
            continue
        out[region] = np.bincount(votes).argmax()
    return out


@torch.no_grad()
def test_majority_vote_matches_reference() -> None:
    blobs = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [2, 2, 3, 3],
            [2, 2, 3, 3],
        ],
        dtype=np.int32,
    )
    # blob 1 -> {2,2,3,3} tie -> 2 (lowest); blob 2 -> all 0 (unassigned) -> 0;
    # blob 3 -> {5,5,5,7} -> 5
    identity = np.array(
        [
            [9, 2, 3, 9],
            [9, 2, 3, 9],
            [0, 0, 5, 5],
            [0, 0, 5, 7],
        ],
        dtype=np.int32,
    )
    node = MajorityVoteByBlob()
    out = node.forward(
        identity_mask=torch.from_numpy(identity)[None],
        blob_mask=torch.from_numpy(blobs)[None],
    )
    expected = _reference_vote(identity, blobs)

    assert out["mask"].dtype == torch.int32
    assert out["mask"].shape == (1, 4, 4)
    assert torch.equal(out["mask"][0], torch.from_numpy(expected))
    # blob 2 had no labelled pixels -> stays background
    assert int(out["mask"][0, 2, 0]) == 0
    # tie in blob 1 resolves to the lowest identity
    assert int(out["mask"][0, 0, 1]) == 2


@torch.no_grad()
def test_majority_vote_background_only() -> None:
    blobs = torch.zeros((1, 3, 3), dtype=torch.int32)
    identity = torch.full((1, 3, 3), 4, dtype=torch.int32)
    out = MajorityVoteByBlob().forward(identity_mask=identity, blob_mask=blobs)
    assert torch.count_nonzero(out["mask"]) == 0
