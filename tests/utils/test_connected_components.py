"""Tests for the shared OpenCV connected-components labeling helper."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.utils.connected_components import label_connected_components

pytestmark = pytest.mark.unit


def _two_blob_mask() -> np.ndarray:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    return mask


def test_tensor_input_labels_two_components() -> None:
    labels = label_connected_components(torch.from_numpy(_two_blob_mask()))
    assert labels.dtype == torch.int32
    assert labels.shape == (8, 8)
    assert int(labels.max()) == 2
    # Each blob carries a single label; background stays 0.
    assert int(labels[1, 1]) != int(labels[5, 5])
    assert int(labels[0, 0]) == 0


def test_numpy_input_returns_cpu_labels() -> None:
    labels = label_connected_components(_two_blob_mask())
    assert isinstance(labels, torch.Tensor)
    assert labels.device.type == "cpu"
    assert int(labels.max()) == 2


def test_empty_mask_yields_all_background() -> None:
    labels = label_connected_components(np.zeros((4, 5), dtype=np.uint8))
    assert labels.shape == (4, 5)
    assert torch.all(labels == 0)


def test_invalid_connectivity_rejected() -> None:
    with pytest.raises(ValueError, match="connectivity"):
        label_connected_components(_two_blob_mask(), connectivity=6)


def test_non_2d_mask_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        label_connected_components(np.zeros((2, 3, 4), dtype=np.uint8))
