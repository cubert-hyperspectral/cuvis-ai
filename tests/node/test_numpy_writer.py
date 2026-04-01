"""Tests for NumpyFeatureWriterNode."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cuvis_ai.node.numpy_writer import NumpyFeatureWriterNode


def test_write_single_frame(tmp_path: Path) -> None:
    """Writes a [1, N, D] tensor as [N, D] .npy file."""
    node = NumpyFeatureWriterNode(output_dir=str(tmp_path / "feats"), prefix="emb")
    features = torch.randn(1, 5, 512)
    frame_id = torch.tensor([3], dtype=torch.int64)
    node.forward(features=features, frame_id=frame_id)

    npy_path = tmp_path / "feats" / "emb_000003.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (5, 512)
    np.testing.assert_allclose(arr, features.squeeze(0).numpy(), atol=1e-6)


def test_write_zero_detections(tmp_path: Path) -> None:
    """Zero detections → [0, D] .npy file."""
    node = NumpyFeatureWriterNode(output_dir=str(tmp_path / "feats"))
    features = torch.zeros(1, 0, 512)
    frame_id = torch.tensor([0], dtype=torch.int64)
    node.forward(features=features, frame_id=frame_id)

    npy_path = tmp_path / "feats" / "features_000000.npy"
    assert npy_path.exists()
    arr = np.load(npy_path)
    assert arr.shape == (0, 512)


def test_output_dir_created(tmp_path: Path) -> None:
    """Output directory is created on first write."""
    out_dir = tmp_path / "nested" / "deep" / "feats"
    assert not out_dir.exists()
    node = NumpyFeatureWriterNode(output_dir=str(out_dir))
    node.forward(features=torch.randn(1, 2, 64), frame_id=torch.tensor([1], dtype=torch.int64))
    assert out_dir.exists()
