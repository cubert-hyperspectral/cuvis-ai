"""Tests for ZScoreNormalizer's global running-stats mode (and per-sample compat)."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.normalization import ZScoreNormalizer

pytestmark = pytest.mark.unit


def _stream(frames: list[torch.Tensor]):
    """Yield port-based batches the way StatisticalTrainer's input stream does."""
    for x in frames:
        yield {"data": x}


@torch.no_grad()
def test_default_per_sample_behavior_unchanged() -> None:
    """Default construction keeps the historical per-sample formula bit-for-bit."""
    x = torch.randn(2, 5, 6, 7)
    node = ZScoreNormalizer()  # use_running_stats defaults to False
    out = node.forward(data=x)["normalized"]
    mean = x.mean(dim=[1, 2], keepdim=True)
    std = x.std(dim=[1, 2], keepdim=True, unbiased=False)
    assert torch.equal(out, (x - mean) / (std + 1e-6))
    assert node.requires_initial_fit is False


@torch.no_grad()
def test_running_scalar_statistics() -> None:
    """Global scalar mode: one mean/std over all pixels, bands, and frames."""
    g = torch.Generator().manual_seed(0)
    frames = [torch.randn(2, 4, 4, 3, generator=g) * 3.0 + 5.0 for _ in range(4)]
    node = ZScoreNormalizer(use_running_stats=True)
    assert node.requires_initial_fit is True
    node.statistical_initialization(_stream(frames))

    all_values = torch.cat([f.reshape(-1) for f in frames])
    assert torch.allclose(node.zscore_mean, all_values.mean(), atol=1e-5)
    assert torch.allclose(node.zscore_std, all_values.std(unbiased=True), atol=1e-4)
    assert node.zscore_mean.shape == ()

    out = node.forward(data=frames[0])["normalized"]
    expected = (frames[0] - node.zscore_mean) / (node.zscore_std + node.eps)
    assert torch.equal(out, expected)


@torch.no_grad()
def test_running_per_band_statistics() -> None:
    """Per-band mode: one mean/std per spectral channel, buffers shaped (C,)."""
    g = torch.Generator().manual_seed(1)
    offsets = torch.tensor([0.0, 10.0, -5.0])  # distinct per-band distributions
    frames = [torch.randn(2, 4, 4, 3, generator=g) + offsets for _ in range(4)]
    node = ZScoreNormalizer(use_running_stats=True, per_band=True, num_channels=3)
    node.statistical_initialization(_stream(frames))

    flat = torch.cat([f.reshape(-1, 3) for f in frames])
    assert node.zscore_mean.shape == (3,)
    assert torch.allclose(node.zscore_mean, flat.mean(dim=0), atol=1e-5)
    assert torch.allclose(node.zscore_std, flat.std(dim=0, unbiased=True), atol=1e-4)

    # Normalized output is ~zero-mean per band despite the distinct offsets.
    out = node.forward(data=frames[0])["normalized"]
    assert out.reshape(-1, 3).mean(dim=0).abs().max() < 0.2


@torch.no_grad()
def test_max_init_frames_caps_the_stream() -> None:
    """Only the first N frames feed the statistics; a shifted tail is ignored."""
    g = torch.Generator().manual_seed(2)
    head = [torch.randn(2, 4, 4, 2, generator=g) for _ in range(2)]  # 4 frames
    tail = [torch.randn(2, 4, 4, 2, generator=g) + 100.0]  # would wreck the mean
    node = ZScoreNormalizer(use_running_stats=True, max_init_frames=4)
    node.statistical_initialization(_stream(head + tail))

    head_values = torch.cat([f.reshape(-1) for f in head])
    assert torch.allclose(node.zscore_mean, head_values.mean(), atol=1e-5)
    assert node.zscore_mean.abs().item() < 1.0  # the +100 tail did not leak in


@torch.no_grad()
def test_max_init_frames_slices_partial_batch() -> None:
    """A cap that lands mid-batch consumes only the needed leading samples."""
    frames = [torch.zeros(4, 2, 2, 1), torch.ones(4, 2, 2, 1) * 8.0]
    # Cap of 6 frames = all of batch 1 (zeros) + first 2 samples of batch 2 (8s).
    node = ZScoreNormalizer(use_running_stats=True, max_init_frames=6)
    node.statistical_initialization(_stream(frames))
    expected_mean = (0.0 * 16 + 8.0 * 8) / 24  # 16 zero pixels + 8 eight-pixels
    assert torch.allclose(node.zscore_mean, torch.tensor(expected_mean), atol=1e-6)


@torch.no_grad()
def test_forward_raises_before_initialization() -> None:
    """Running-stats mode is strict about initialization order."""
    node = ZScoreNormalizer(use_running_stats=True)
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node.forward(data=torch.randn(1, 4, 4, 3))


def test_constructor_validation() -> None:
    """per_band needs num_channels; max_init_frames must be positive."""
    with pytest.raises(ValueError, match="num_channels"):
        ZScoreNormalizer(use_running_stats=True, per_band=True)
    with pytest.raises(ValueError, match="max_init_frames"):
        ZScoreNormalizer(use_running_stats=True, max_init_frames=0)


@torch.no_grad()
def test_state_dict_round_trip_restores_buffers() -> None:
    """Fitted statistics travel through state_dict like any other buffers."""
    g = torch.Generator().manual_seed(3)
    frames = [torch.randn(2, 4, 4, 3, generator=g) for _ in range(2)]
    n1 = ZScoreNormalizer(use_running_stats=True, per_band=True, num_channels=3)
    n1.statistical_initialization(_stream(frames))
    n2 = ZScoreNormalizer(use_running_stats=True, per_band=True, num_channels=3)
    n2.load_state_dict(n1.state_dict())
    assert torch.equal(n1.zscore_mean, n2.zscore_mean)
    assert torch.equal(n1.zscore_std, n2.zscore_std)
