"""RXGlobal.update rejects at fit time what forward rejects at inference time."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.anomaly.rx_detector import RXGlobal

pytestmark = pytest.mark.unit


def test_update_rejects_a_batch_that_is_not_bhwc() -> None:
    """A 3-D cube or a non-BHWC crop fails before any statistics are accumulated."""
    node = RXGlobal(num_channels=3)
    with pytest.raises(ValueError, match="BHWC"):
        node.update(torch.zeros(4, 8, 3))
    assert node._welford.count == 0


def test_update_accepts_a_bhwc_batch() -> None:
    """The guard leaves the normal (B, H, W, C) path untouched."""
    node = RXGlobal(num_channels=3)
    node.update(torch.randn(2, 4, 4, 3))
    assert node._welford.count == 32
