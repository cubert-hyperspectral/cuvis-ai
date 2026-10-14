"""PercentileNormalizer's pooled statistical fit, clamp choice and invert: golden percentiles vs
numpy, unclamped-above / floored-below behaviour, subsampling and cap, state-dict round-trip,
validation, and bit parity with the score-range calibration they replace."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cuvis_ai.node.normalization import PercentileNormalizer
from cuvis_ai_core.node.fit_utils import random_cap
from cuvis_ai_core.node.metric_utils import subsample_hw

pytestmark = pytest.mark.unit

H, W = 23, 17


def _stream(n: int = 3, c: int = 1, seed: int = 0) -> list[dict[str, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    return [{"data": torch.rand(1, H, W, c, generator=g) * 5.0 + 1.0} for _ in range(n)]


def _node(low: float = 1.0, high: float = 99.0, floor: bool = True, **kw) -> PercentileNormalizer:
    """The score-range calibration as PercentileNormalizer options (low / high in percent)."""
    return PercentileNormalizer(
        n_channels=kw.pop("n_channels", 1),
        norm_mode="statistical",
        statistical_fit="pooled",
        quantile_low=low / 100.0,
        quantile_high=high / 100.0,
        clamp="floor" if floor else "none",
        eps=kw.pop("eps", 1e-9),
        **kw,
    )


def _numpy_bounds(stream, s: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    vals = np.concatenate(
        [b["data"][:, ::s, ::s, :].reshape(-1, b["data"].shape[-1]).numpy() for b in stream]
    )
    return np.percentile(vals, low, axis=0), np.percentile(vals, high, axis=0)


# ----- 1. golden --------------------------------------------------------------------------------


def test_fitted_bounds_match_numpy_percentiles_of_the_subsampled_stream():
    node = _node(fit_subsample=2)
    node.statistical_initialization(iter(_stream()))
    lo, hi = _numpy_bounds(_stream(), 2, 1.0, 99.0)
    assert np.allclose(node.running_min.numpy(), lo, atol=1e-5)
    assert np.allclose(node.running_max.numpy(), hi, atol=1e-5)
    x = _stream(1, seed=9)[0]["data"]
    out = node(data=x)["normalized"]
    expected = np.clip((x.numpy() - lo) / (hi - lo), 0, None)
    assert np.allclose(out.numpy(), expected, atol=1e-5)


def test_per_channel_bounds():
    node = _node(n_channels=2, low=5.0, high=95.0, fit_subsample=1)
    stream = _stream(2, c=2, seed=3)
    for batch in stream:
        batch["data"][..., 1] *= 10.0  # second channel on another scale
    node.statistical_initialization(iter(stream))
    lo, hi = _numpy_bounds(stream, 1, 5.0, 95.0)
    assert np.allclose(node.running_min.numpy(), lo, atol=1e-4)
    assert np.allclose(node.running_max.numpy(), hi, atol=1e-4)


def test_unclamped_above_and_floored_below():
    node = _node(fit_subsample=1)
    node.statistical_initialization(iter(_stream()))
    hi, lo = float(node.running_max[0]), float(node.running_min[0])
    x = torch.tensor([[[[hi * 10.0], [lo - 1.0], [hi]]]])  # far above / below / at the top
    out = node(data=x)["normalized"][0, 0, :, 0]
    assert out[0] > 9.0  # an anomaly far above the normal range keeps its magnitude
    assert out[1] == 0.0  # below the low percentile is "normal": floored
    assert torch.isclose(out[2], torch.tensor(1.0), atol=1e-5)
    raw = _node(fit_subsample=1, floor=False)
    raw.statistical_initialization(iter(_stream()))
    assert raw(data=x)["normalized"][0, 0, 1, 0] < 0.0


def test_cap_is_seeded_and_bounds_stay_close():
    small = _node(fit_subsample=1, max_fit_values=200, seed=1)
    small.statistical_initialization(iter(_stream(4)))
    again = _node(fit_subsample=1, max_fit_values=200, seed=1)
    again.statistical_initialization(iter(_stream(4)))
    assert torch.equal(small.running_min, again.running_min)
    assert torch.equal(small.running_max, again.running_max)
    full = _node(fit_subsample=1)
    full.statistical_initialization(iter(_stream(4)))
    assert abs(float(small.running_max[0] - full.running_max[0])) < 0.5


# ----- 2. lifecycle -----------------------------------------------------------------------------


def test_requires_fit_and_rejects_channel_mismatch_and_empty_stream():
    node = _node()
    assert node.requires_initial_fit is True
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node(data=torch.rand(1, H, W, 1))
    with pytest.raises(ValueError, match="channels"):
        node.statistical_initialization(iter([{"data": torch.rand(1, 4, 4, 2)}]))
    with pytest.raises(RuntimeError, match="no data"):
        node.statistical_initialization(iter([]))
    assert node._fitted() is False


def test_state_dict_round_trip():
    fitted = _node(fit_subsample=1)
    fitted.statistical_initialization(iter(_stream()))
    fresh = _node(fit_subsample=1)
    fresh.load_state_dict(fitted.state_dict())
    x = torch.rand(1, H, W, 1, generator=torch.Generator().manual_seed(5)) * 8
    assert torch.equal(fresh(data=x)["normalized"], fitted(data=x)["normalized"])


def test_invert_equals_normalising_the_negated_map():
    """invert=True on a log-likelihood-like map equals the plain node on the negated map."""
    loglik = [{"data": -b["data"]} for b in _stream(seed=3)]
    inv = _node(fit_subsample=2, invert=True)
    inv.statistical_initialization(iter(loglik))
    ref = _node(fit_subsample=2)
    ref.statistical_initialization(iter([{"data": -b["data"]} for b in loglik]))
    assert torch.equal(inv.running_min, ref.running_min)
    assert torch.equal(inv.running_max, ref.running_max)
    x = -_stream(1, seed=11)[0]["data"]
    assert torch.equal(inv(data=x)["normalized"], ref(data=-x)["normalized"])
    # higher log-likelihood (more normal) gives a lower output
    assert inv(data=x + 1.0)["normalized"].mean() < inv(data=x)["normalized"].mean()


def test_invert_with_the_batch_bounds_fit():
    stream = _stream(seed=4)
    inv = PercentileNormalizer(n_channels=1, norm_mode="statistical", invert=True)
    inv.statistical_initialization(iter([{"data": -b["data"]} for b in stream]))
    ref = PercentileNormalizer(n_channels=1, norm_mode="statistical")
    ref.statistical_initialization(iter(stream))
    assert torch.equal(inv.running_min, ref.running_min)
    x = _stream(1, seed=12)[0]["data"]
    assert torch.equal(inv(data=-x)["normalized"], ref(data=x)["normalized"])


def test_hparams_json_round_trip_and_rebuild():
    node = _node(low=2, high=98, floor=False, fit_subsample=3, name="cal")
    hp = node.hparams
    json.dumps(hp)
    assert hp["statistical_fit"] == "pooled" and hp["clamp"] == "none"
    assert hp["fit_subsample"] == 3 and hp["invert"] is False
    node.statistical_initialization(iter(_stream()))
    rebuilt = PercentileNormalizer(**hp)
    rebuilt.load_state_dict(node.state_dict())
    x = _stream(1, seed=6)[0]["data"]
    assert torch.equal(rebuilt(data=x)["normalized"], node(data=x)["normalized"])


# ----- 3. defaults and validation ---------------------------------------------------------------


def test_defaults_keep_the_batch_bounds_fit_and_the_full_clamp():
    node = PercentileNormalizer(n_channels=1, norm_mode="statistical")
    assert node.hparams["statistical_fit"] == "batch_bounds" and node.hparams["clamp"] == "both"
    node.statistical_initialization(iter(_stream()))
    out = node(data=_stream(1, seed=7)[0]["data"] * 3.0)["normalized"]
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0


@pytest.mark.parametrize(
    "kw",
    [
        {"statistical_fit": "pooled"},
        {"clamp": "floor"},
        {"clamp": "none"},
        {"invert": True},
    ],
)
@pytest.mark.parametrize("mode", ["per_frame", "running"])
def test_statistical_options_refuse_the_other_modes(kw, mode):
    with pytest.raises(ValueError, match="statistical"):
        PercentileNormalizer(n_channels=1, norm_mode=mode, **kw)


@pytest.mark.parametrize("kw", [{"fit_subsample": 2}, {"max_fit_values": 10}, {"seed": 1}])
@pytest.mark.parametrize("mode", ["per_frame", "running", "statistical"])
def test_pooled_fit_options_refuse_the_other_fits(kw, mode):
    """The pooled fit's stride, cap and seed would be ignored by any other fit, so they raise."""
    with pytest.raises(ValueError, match="pooled"):
        PercentileNormalizer(n_channels=1, norm_mode=mode, **kw)


@pytest.mark.parametrize(
    "bad",
    [
        {"statistical_fit": "median"},
        {"clamp": "top"},
        {"invert": 1},
        {"fit_subsample": 0},
        {"max_fit_values": 1},
        {"seed": -1},
        {"fit_subsample": 2.0},
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        PercentileNormalizer(n_channels=1, norm_mode="statistical", **bad)


# ----- 4. parity with the score-range calibration ------------------------------------------------


def _reference_fit(stream, low, high, s, cap, seed, invert):
    """The former ScoreRangeNormalizer fit, verbatim."""
    gen = torch.Generator().manual_seed(seed)
    chunks, total = [], 0
    for batch in stream:
        v = subsample_hw(batch["data"], s).reshape(-1, batch["data"].shape[-1]).float()
        chunks.append(-v if invert else v)
        total += v.shape[0]
        if total > 2 * cap:
            chunks = [random_cap(torch.cat(chunks, dim=0), cap, gen)]
            total = chunks[0].shape[0]
    vals = random_cap(torch.cat(chunks, dim=0), cap, gen)
    q = torch.tensor([low / 100.0, high / 100.0], dtype=vals.dtype, device=vals.device)
    pct = torch.quantile(vals, q, dim=0)
    return pct[0], pct[1]


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("floor", [True, False])
@pytest.mark.parametrize("cap", [4_000_000, 300, 50])
def test_bits_match_the_score_range_calibration(invert, floor, cap):
    stream = _stream(5, seed=21)
    if cap == 50:
        # 30 rows per frame at stride 4: the fourth frame takes the pool to 120 > 2 * cap (reduced
        # to 50 while streaming), the fifth to 80, which the final cap takes down to 50.
        assert subsample_hw(stream[0]["data"], 4).reshape(-1).numel() == 30
    lo, hi = _reference_fit(stream, 1.0, 99.0, 4, cap, 0, invert)
    node = _node(floor=floor, fit_subsample=4, max_fit_values=cap, seed=0, invert=invert)
    node.statistical_initialization(iter(stream))
    assert torch.equal(node.running_min, lo) and torch.equal(node.running_max, hi)
    x = _stream(2, seed=22)[1]["data"] * 1.7
    xi = -x if invert else x
    expected = (xi - lo) / (hi - lo).clamp_min(1e-9)
    if floor:
        expected = expected.clamp_min(0.0)
    assert torch.equal(node(data=x)["normalized"], expected)
