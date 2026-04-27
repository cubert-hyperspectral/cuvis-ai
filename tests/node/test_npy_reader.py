"""Unit tests for NpyReader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.numpy_file import NpyReader, _pad_to_bhwc4

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


def test_pad_to_bhwc4_accepts_3d_and_4d_arrays() -> None:
    arr_3d = np.zeros((2, 3, 4), dtype=np.float32)
    arr_4d = np.zeros((1, 2, 3, 4), dtype=np.float32)

    assert _pad_to_bhwc4(arr_3d).shape == (1, 2, 3, 4)
    assert _pad_to_bhwc4(arr_4d).shape == (1, 2, 3, 4)


def test_reader_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="input file not found"):
        NpyReader(file_path=str(tmp_path / "missing.npy"))


def test_pad_to_bhwc4_rejects_arrays_above_4d() -> None:
    with pytest.raises(ValueError, match="1-4 dimensions"):
        _pad_to_bhwc4(np.zeros((1, 1, 1, 1, 1), dtype=np.float32))
