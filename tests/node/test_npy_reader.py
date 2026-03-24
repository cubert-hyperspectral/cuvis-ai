"""Unit tests for NpyReader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.numpy_reader import NpyReader

pytestmark = pytest.mark.unit


def test_loads_correct_shape(tmp_path) -> None:
    reference = np.ones((1, 61), dtype=np.float32)
    npy_path = tmp_path / "reference.npy"
    np.save(npy_path, reference)

    node = NpyReader(file_path=str(npy_path))
    out = node.forward()["data"]

    assert out.shape == (1, 1, 1, 61)
    assert out.dtype == torch.float32


def test_output_consistent_across_calls(tmp_path) -> None:
    reference = np.random.default_rng(123).normal(size=(2, 61)).astype(np.float32)
    npy_path = tmp_path / "reference.npy"
    np.save(npy_path, reference)

    node = NpyReader(file_path=str(npy_path))
    out_a = node.forward()["data"]
    out_b = node.forward()["data"]

    assert torch.equal(out_a, out_b)


def test_device_transfer(tmp_path) -> None:
    reference = np.arange(61, dtype=np.float32).reshape(1, 61)
    npy_path = tmp_path / "reference.npy"
    np.save(npy_path, reference)

    node = NpyReader(file_path=str(npy_path))
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    node = node.to(target_device)
    out = node.forward()["data"]

    assert out.device.type == target_device.type
