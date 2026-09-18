"""Tests for the running-bounds helper shared by the running normalization modes.

RunningBounds advances one step per frame, never per batch, so running_warmup_frames and
freeze_running_bounds_after_frames of FixedWavelengthSelector and PercentileNormalizer mean
frames whatever the batch size.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node._running_bounds import RunningBounds, per_frame_quantiles, running_normalize

pytestmark = pytest.mark.unit


def _bounds(warmup: int, freeze: int | None, c: int = 3) -> RunningBounds:
    return RunningBounds(
        torch.full((c,), float("nan")),
        torch.full((c,), float("nan")),
        torch.zeros((), dtype=torch.long),
        warmup_frames=warmup,
        freeze_after_frames=freeze,
    )


def _frames(n: int, c: int = 3, seed: int = 7) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 4, 4, c, generator=g) * 500.0


_LO = torch.zeros(3)
_HI = torch.ones(3)
_Q = {"quantile_low": 0.005, "quantile_high": 0.995, "eps": 1e-8}


class TestStep:
    def test_step_counts_one_frame(self) -> None:
        bounds = _bounds(warmup=0, freeze=None)
        assert [bounds.step(_LO, _HI).count for _ in range(3)] == [1, 2, 3]
        assert int(bounds.frame_count.item()) == 3

    def test_first_step_initialises_nan_bounds(self) -> None:
        bounds = _bounds(warmup=0, freeze=None)
        lo, hi = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0])
        bounds.step(lo, hi)
        assert torch.equal(bounds.running_min, lo)
        assert torch.equal(bounds.running_max, hi)

    def test_accumulates_min_of_lows_and_max_of_highs(self) -> None:
        bounds = _bounds(warmup=0, freeze=None)
        bounds.step(torch.tensor([1.0, 5.0, 3.0]), torch.tensor([10.0, 20.0, 30.0]))
        bounds.step(torch.tensor([2.0, 4.0, 6.0]), torch.tensor([11.0, 19.0, 60.0]))
        assert torch.equal(bounds.running_min, torch.tensor([1.0, 4.0, 3.0]))
        assert torch.equal(bounds.running_max, torch.tensor([11.0, 20.0, 60.0]))

    def test_warmup_boundary_is_inclusive(self) -> None:
        bounds = _bounds(warmup=2, freeze=None)
        assert [bounds.step(_LO, _HI).in_warmup for _ in range(3)] == [True, True, False]

    def test_freeze_none_never_freezes(self) -> None:
        bounds = _bounds(warmup=0, freeze=None)
        for i in range(50):
            assert bounds.step(_LO - i, _HI + i).should_update is True
        assert torch.equal(bounds.running_max, _HI + 49)

    def test_freeze_boundary_at_the_exact_frame(self) -> None:
        bounds = _bounds(warmup=0, freeze=3)
        assert [bounds.step(_LO, _HI).should_update for _ in range(3)] == [True, True, True]
        phase = bounds.step(_LO - 100.0, _HI + 100.0)
        assert (phase.count, phase.should_update) == (4, False)
        assert torch.equal(bounds.running_min, _LO)
        assert torch.equal(bounds.running_max, _HI)


class TestPerFrameQuantiles:
    def test_each_frame_gets_its_own_quantiles(self) -> None:
        data = _frames(4)
        lo, hi = per_frame_quantiles(data, 0.005, 0.995)
        assert lo.shape == (4, 3) and hi.shape == (4, 3)
        for b in range(4):
            flat = data[b].reshape(-1, 3)
            assert torch.allclose(lo[b], torch.quantile(flat, 0.005, dim=0))
            assert torch.allclose(hi[b], torch.quantile(flat, 0.995, dim=0))


class TestRunningNormalize:
    def test_batch_matches_the_same_frames_one_at_a_time(self) -> None:
        frames = _frames(6)
        batched = _bounds(warmup=2, freeze=4)
        out_batched = running_normalize(frames, batched, **_Q)
        sequential = _bounds(warmup=2, freeze=4)
        out_seq = torch.cat(
            [running_normalize(frames[b : b + 1], sequential, **_Q) for b in range(6)]
        )
        assert out_batched.shape == frames.shape
        assert torch.allclose(out_batched, out_seq)
        assert torch.allclose(batched.running_min, sequential.running_min)
        assert torch.allclose(batched.running_max, sequential.running_max)
        assert int(batched.frame_count.item()) == 6

    def test_warmup_frames_use_their_own_quantiles_later_frames_the_bounds(self) -> None:
        frames = _frames(3)
        bounds = _bounds(warmup=2, freeze=None)
        out = running_normalize(frames, bounds, **_Q)
        lo, hi = per_frame_quantiles(frames, 0.005, 0.995)
        for b in range(2):
            expected = (frames[b] - lo[b]) / (hi[b] - lo[b]).clamp_min(1e-8)
            assert torch.allclose(out[b], expected.clamp(0.0, 1.0))
        span = bounds.running_max - bounds.running_min
        expected = (frames[2] - bounds.running_min) / span
        assert torch.allclose(out[2], expected.clamp(0.0, 1.0))
