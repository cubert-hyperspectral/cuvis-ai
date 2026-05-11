"""Unit tests for NpyReader."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.numpy_file import NpyReader, _pad_to_bhwc4

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stream(items: list[dict]) -> list[dict]:
    """Thin stand-in for InputStream — just a plain list of dicts."""
    return items


# ---------------------------------------------------------------------------
# File-mode (existing behaviour — must not regress)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Buffer mode — file_path=None
# ---------------------------------------------------------------------------


def test_buffer_mode_starts_empty() -> None:
    """NpyReader(file_path=None) constructs without error."""
    node = NpyReader(file_path=None)
    assert node.file_path is None
    assert node._data_buf.numel() == 0


def test_buffer_mode_forward_raises_before_population() -> None:
    """forward() before load_from_array / stat_init raises a clear error."""
    node = NpyReader(file_path=None)
    with pytest.raises(RuntimeError, match="buffer is empty"):
        node.forward()


# ---------------------------------------------------------------------------
# load_from_array
# ---------------------------------------------------------------------------


def test_load_from_array_numpy_2d() -> None:
    """[N, C] array → buffer shaped [N, 1, 1, C]."""
    sigs = np.random.default_rng(0).normal(size=(3, 39)).astype(np.float32)
    node = NpyReader(file_path=None)
    node.load_from_array(sigs)

    out = node.forward()["data"]
    assert out.shape == (3, 1, 1, 39)
    assert out.dtype == torch.float32
    assert torch.allclose(out[:, 0, 0, :], torch.from_numpy(sigs))


def test_load_from_array_tensor() -> None:
    """Accepts torch.Tensor directly."""
    sigs = torch.rand(2, 10)
    node = NpyReader(file_path=None)
    node.load_from_array(sigs)

    out = node.forward()["data"]
    assert out.shape == (2, 1, 1, 10)


def test_load_from_array_1d() -> None:
    """Single [C] vector → buffer shaped [1, 1, 1, C]."""
    sig = np.ones(39, dtype=np.float32)
    node = NpyReader(file_path=None)
    node.load_from_array(sig)

    out = node.forward()["data"]
    assert out.shape == (1, 1, 1, 39)


def test_load_from_array_marks_statistically_initialized() -> None:
    node = NpyReader(file_path=None)
    assert not node._statistically_initialized
    node.load_from_array(np.ones((1, 5), dtype=np.float32))
    assert node._statistically_initialized


def test_load_from_array_consistent_across_calls() -> None:
    sigs = np.random.default_rng(7).normal(size=(3, 39)).astype(np.float32)
    node = NpyReader(file_path=None)
    node.load_from_array(sigs)

    out_a = node.forward()["data"]
    out_b = node.forward()["data"]
    assert torch.equal(out_a, out_b)


# ---------------------------------------------------------------------------
# statistical_initialization
# ---------------------------------------------------------------------------


def test_stat_init_from_signatures_key() -> None:
    """Stream items with 'signatures' [B,N,C] key are squeezed → [N,1,1,C]."""
    sigs = torch.rand(1, 3, 39)  # [B=1, N=3, C=39]
    stream = _make_stream([{"signatures": sigs}])

    node = NpyReader(file_path=None)
    node.statistical_initialization(stream)

    out = node.forward()["data"]
    # batch dim is squeezed: [3, 39] → _pad_to_bhwc4 → [3, 1, 1, 39]
    assert out.shape == (3, 1, 1, 39)
    assert torch.allclose(out[:, 0, 0, :], sigs[0])


def test_stat_init_from_data_key() -> None:
    """Stream items with 'data' key are consumed correctly."""
    data = torch.rand(1, 2, 10)
    stream = _make_stream([{"data": data}])

    node = NpyReader(file_path=None)
    node.statistical_initialization(stream)

    out = node.forward()["data"]
    assert out.shape == (2, 1, 1, 10)


def test_stat_init_empty_stream_raises() -> None:
    node = NpyReader(file_path=None)
    with pytest.raises(RuntimeError, match="no usable tensors"):
        node.statistical_initialization(_make_stream([]))


def test_stat_init_skips_empty_tensors() -> None:
    """Items with numel==0 (unannotated frames) are silently skipped."""
    empty = torch.empty(1, 0, 39)  # no objects detected on this frame
    good = torch.rand(1, 3, 39)
    stream = _make_stream([{"signatures": empty}, {"signatures": good}])

    node = NpyReader(file_path=None)
    node.statistical_initialization(stream)

    out = node.forward()["data"]
    # Only `good` is kept; batch dim squeezed → [3, 1, 1, 39]
    assert out.shape == (3, 1, 1, 39)
    assert torch.allclose(out[:, 0, 0, :], good[0])


def test_stat_init_averages_multiple_items() -> None:
    """Multiple valid stream items are mean-averaged."""
    a = torch.ones(1, 1, 4)
    b = torch.full((1, 1, 4), 3.0)
    stream = _make_stream([{"signatures": a}, {"signatures": b}])

    node = NpyReader(file_path=None)
    node.statistical_initialization(stream)

    out = node.forward()["data"]
    expected = torch.full((1, 1, 1, 4), 2.0)
    assert torch.allclose(out, expected)


def test_stat_init_raises_all_empty() -> None:
    """If all items are empty tensors, raise RuntimeError."""
    stream = _make_stream(
        [
            {"signatures": torch.empty(1, 0, 39)},
            {"signatures": torch.empty(1, 0, 39)},
        ]
    )
    node = NpyReader(file_path=None)
    with pytest.raises(RuntimeError, match="no usable tensors"):
        node.statistical_initialization(stream)


# ---------------------------------------------------------------------------
# Serialization round-trip (buffer mode)
# ---------------------------------------------------------------------------


def test_buffer_mode_state_dict_round_trip() -> None:
    """save + load via state_dict preserves buffer values."""
    sigs = np.random.default_rng(42).normal(size=(3, 39)).astype(np.float32)

    node_a = NpyReader(file_path=None)
    node_a.load_from_array(sigs)

    state = node_a.state_dict()

    node_b = NpyReader(file_path=None)
    node_b.load_state_dict(state, strict=False)

    out_a = node_a.forward()["data"]
    out_b = node_b.forward()["data"]
    assert torch.equal(out_a, out_b)
