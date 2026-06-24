"""Tests for PercentileNormalizer (per-channel n-channel normalization).

Covers constructor validation, the three modes, channel-count guarding, the
statistical fit path, frame-count persistence across a state_dict round-trip, and
the critical bit-for-bit parity with the normalization machinery still living in
``ChannelSelectorBase`` (the safety net for the later selector migration).
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.channel_selector import ChannelSelectorBase
from cuvis_ai.node.normalization import DisplayNormalizer, PercentileNormalizer

pytestmark = pytest.mark.unit


class _RefSelector(ChannelSelectorBase):
    """Minimal concrete selector exposing the base normalization helpers.

    ``ChannelSelectorBase.forward`` is abstract; the parity tests only call the
    inherited normalization helpers (``_per_frame_normalize``,
    ``_running_normalize``, ``_apply_accumulated_stats``, ``_normalize_rgb``), so
    this stub just satisfies the abstract method.
    """

    def forward(self, *args, **kwargs):  # pragma: no cover - unused in parity tests
        raise NotImplementedError


def _raw(B: int = 1, H: int = 4, W: int = 4, C: int = 3, scale: float = 500.0) -> torch.Tensor:
    """Deterministic raw (unnormalized) BHWC radiance-like tensor."""
    g = torch.Generator().manual_seed(1234 + C)
    return torch.rand(B, H, W, C, generator=g) * scale


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    @pytest.mark.parametrize("n", [0, -1])
    def test_non_positive_n_channels_raises(self, n: int) -> None:
        with pytest.raises(ValueError, match="n_channels"):
            PercentileNormalizer(n_channels=n)

    def test_bool_n_channels_raises(self) -> None:
        with pytest.raises(ValueError, match="n_channels"):
            PercentileNormalizer(n_channels=True)  # type: ignore[arg-type]

    def test_invalid_norm_mode_raises(self) -> None:
        with pytest.raises(ValueError):
            PercentileNormalizer(n_channels=3, norm_mode="bogus")

    def test_bad_quantiles_raise(self) -> None:
        with pytest.raises(ValueError, match="quantile"):
            PercentileNormalizer(n_channels=3, quantile_low=0.9, quantile_high=0.1)

    def test_bad_warmup_raises(self) -> None:
        with pytest.raises(ValueError, match="running_warmup_frames"):
            PercentileNormalizer(n_channels=3, running_warmup_frames=-1)

    def test_only_statistical_requires_fit(self) -> None:
        assert PercentileNormalizer(n_channels=3, norm_mode="statistical").requires_initial_fit
        assert not PercentileNormalizer(n_channels=3, norm_mode="running").requires_initial_fit
        assert not PercentileNormalizer(n_channels=3, norm_mode="per_frame").requires_initial_fit


# ---------------------------------------------------------------------------
# Channel handling
# ---------------------------------------------------------------------------


class TestChannels:
    def test_channel_mismatch_raises(self) -> None:
        node = PercentileNormalizer(n_channels=3, norm_mode="per_frame")
        with pytest.raises(ValueError, match="expected 3 channels"):
            node.forward(data=_raw(C=6))

    @pytest.mark.parametrize("c", [1, 4, 6, 9])
    def test_n_channel_output_shape_and_range(self, c: int) -> None:
        node = PercentileNormalizer(n_channels=c, norm_mode="per_frame")
        out = node.forward(data=_raw(B=2, H=8, W=8, C=c))["normalized"]
        assert out.shape == (2, 8, 8, c)
        assert out.dtype == torch.float32
        assert float(out.min()) >= -1e-6
        assert float(out.max()) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# Statistical mode
# ---------------------------------------------------------------------------


class TestStatistical:
    def test_forward_before_fit_raises(self) -> None:
        node = PercentileNormalizer(n_channels=3, norm_mode="statistical")
        with pytest.raises(RuntimeError, match="statistical_initialization"):
            node.forward(data=_raw())

    def test_empty_stream_raises(self) -> None:
        node = PercentileNormalizer(n_channels=3, norm_mode="statistical")
        with pytest.raises(RuntimeError, match="received no data"):
            node.statistical_initialization(iter([]))

    def test_fit_then_apply_in_range(self) -> None:
        node = PercentileNormalizer(n_channels=3, norm_mode="statistical")
        batches = [{"data": _raw(scale=300.0)}, {"data": _raw(scale=900.0)}]
        node.statistical_initialization(iter(batches))
        assert not torch.isnan(node.running_min).any()
        out = node.forward(data=_raw(scale=600.0))["normalized"]
        assert float(out.min()) >= -1e-6 and float(out.max()) <= 1.0 + 1e-6

    def test_fit_accumulates_min_of_lows_max_of_highs(self) -> None:
        node = PercentileNormalizer(n_channels=3, norm_mode="statistical")
        b1, b2 = _raw(scale=100.0), _raw(scale=1000.0)
        node.statistical_initialization(iter([{"data": b1}, {"data": b2}]))
        f1l = torch.quantile(b1.reshape(-1, 3).float(), 0.005, dim=0)
        f2l = torch.quantile(b2.reshape(-1, 3).float(), 0.005, dim=0)
        f1h = torch.quantile(b1.reshape(-1, 3).float(), 0.995, dim=0)
        f2h = torch.quantile(b2.reshape(-1, 3).float(), 0.995, dim=0)
        assert torch.allclose(node.running_min, torch.minimum(f1l, f2l))
        assert torch.allclose(node.running_max, torch.maximum(f1h, f2h))


# ---------------------------------------------------------------------------
# Frame-count + bounds persistence (state_dict round-trip)
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_frame_count_and_bounds_survive_state_dict(self) -> None:
        node = PercentileNormalizer(n_channels=3, norm_mode="running", running_warmup_frames=2)
        for _ in range(5):
            node.forward(data=_raw())
        assert int(node._norm_frame_count.item()) == 5

        restored = PercentileNormalizer(n_channels=3, norm_mode="running", running_warmup_frames=2)
        restored.load_state_dict(node.state_dict())
        assert int(restored._norm_frame_count.item()) == 5
        assert torch.allclose(restored.running_min, node.running_min)
        assert torch.allclose(restored.running_max, node.running_max)

    def test_freeze_stops_updating_bounds(self) -> None:
        node = PercentileNormalizer(
            n_channels=3,
            norm_mode="running",
            running_warmup_frames=0,
            freeze_running_bounds_after_frames=2,
        )
        node.forward(data=_raw(scale=100.0))
        node.forward(data=_raw(scale=100.0))
        frozen_min = node.running_min.clone()
        frozen_max = node.running_max.clone()
        # A much wider frame after freeze must not move the bounds.
        node.forward(data=_raw(scale=5000.0))
        assert torch.allclose(node.running_min, frozen_min)
        assert torch.allclose(node.running_max, frozen_max)


# ---------------------------------------------------------------------------
# CRITICAL parity with ChannelSelectorBase (the migration safety net)
# ---------------------------------------------------------------------------


class TestParityWithSelector:
    """PercentileNormalizer(n_channels=3) must reproduce the selector's
    normalization bit-for-bit, per mode. This proves the later selector
    migration is behaviour-preserving."""

    def test_per_frame_parity(self) -> None:
        raw = _raw(B=2, C=3)
        norm = PercentileNormalizer(n_channels=3, norm_mode="per_frame")
        ref = _RefSelector(norm_mode="per_frame")
        assert torch.allclose(norm.forward(data=raw)["normalized"], ref._per_frame_normalize(raw))

    def test_running_parity_across_frames(self) -> None:
        norm = PercentileNormalizer(n_channels=3, norm_mode="running")
        ref = _RefSelector(norm_mode="running")
        for i in range(15):  # spans warmup (10) and post-warmup accumulation
            raw = _raw(B=1, C=3, scale=200.0 + 50.0 * i)
            got = norm.forward(data=raw)["normalized"]
            want = ref._running_normalize(raw)
            assert torch.allclose(got, want), f"frame {i} diverged"

    def test_statistical_apply_parity(self) -> None:
        norm = PercentileNormalizer(n_channels=3, norm_mode="statistical")
        ref = _RefSelector(norm_mode="statistical")
        lo = torch.tensor([1.0, 2.0, 3.0])
        hi = torch.tensor([10.0, 20.0, 30.0])
        for n in (norm, ref):
            n.running_min.copy_(lo)
            n.running_max.copy_(hi)
        norm._statistically_initialized = True  # mirror ref state flag if present
        ref._statistically_initialized = True
        raw = _raw(C=3)
        assert torch.allclose(
            norm.forward(data=raw)["normalized"], ref._apply_accumulated_stats(raw)
        )

    @pytest.mark.parametrize("mode", ["per_frame", "running", "statistical"])
    def test_chain_with_display_matches_selector_with_gamma(self, mode: str) -> None:
        """selector(apply_gamma=True) == PercentileNormalizer -> DisplayNormalizer."""
        raw = _raw(C=3)
        norm = PercentileNormalizer(n_channels=3, norm_mode=mode)
        display = DisplayNormalizer()
        ref = _RefSelector(norm_mode=mode, apply_gamma=True)
        if mode == "statistical":
            lo, hi = torch.tensor([1.0, 2.0, 3.0]), torch.tensor([10.0, 20.0, 30.0])
            for n in (norm, ref):
                n.running_min.copy_(lo)
                n.running_max.copy_(hi)
            norm._statistically_initialized = True
            ref._statistically_initialized = True
        chained = display.forward(data=norm.forward(data=raw)["normalized"])["normalized"]
        assert torch.allclose(chained, ref._normalize_rgb(raw))
