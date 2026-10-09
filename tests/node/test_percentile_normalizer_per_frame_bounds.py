"""PercentileNormalizer per-frame bounds: frame quantiles, joint channels and quantisation.

Golden parity with numpy's percentiles, the per-channel variant, the 8-bit truncation, degenerate
frames, the unchanged default, hparam validation, and bit-for-bit parity with the per-frame
percentile stretch of the walnut false-RGB path these options replace.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cuvis_ai.node.normalization import PercentileNormalizer, _sorted_quantiles

pytestmark = pytest.mark.unit

B, H, W, C = 2, 17, 13, 3


def _data(seed: int = 0, h: int = H, w: int = W) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    scale = torch.tensor([1.0, 3.0, 0.5])  # anisotropic channels: joint != per-channel
    return torch.rand(B, h, w, C, generator=g) * scale * 4000.0 + 300.0


def _stretch(**kwargs) -> PercentileNormalizer:
    """The per-frame stretch of the false-RGB path (2nd / 98th quantile, joint channels)."""
    params = {
        "n_channels": C,
        "norm_mode": "per_frame",
        "per_frame_quantiles": True,
        "joint_channels": True,
        "quantile_low": 0.02,
        "quantile_high": 0.98,
        "eps": 1e-6,
    }
    params.update(kwargs)
    return PercentileNormalizer(**params)


def _numpy_joint(frame: np.ndarray, lo_q: float, hi_q: float) -> np.ndarray:
    lo, hi = np.percentile(frame, lo_q), np.percentile(frame, hi_q)
    return np.clip((frame - lo) / max(hi - lo, 1e-6), 0, 1)


# ----- 1. golden references -------------------------------------------------------------------


def test_joint_quantile_bounds_match_numpy_percentiles():
    data = _data()
    out = _stretch()(data=data)["normalized"]
    for b in range(B):
        expected = _numpy_joint(data[b].numpy().astype(np.float64), 2.0, 98.0)
        assert np.allclose(out[b].numpy(), expected, atol=1e-5)


def test_per_channel_quantile_bounds_match_numpy_per_channel():
    data = _data(1)
    node = _stretch(joint_channels=False, quantile_low=0.01, quantile_high=0.99)
    out = node(data=data)["normalized"]
    for b in range(B):
        frame = data[b].numpy().astype(np.float64)
        lo = np.percentile(frame.reshape(-1, C), 1.0, axis=0)
        hi = np.percentile(frame.reshape(-1, C), 99.0, axis=0)
        expected = np.clip((frame - lo) / (hi - lo), 0, 1)
        assert np.allclose(out[b].numpy(), expected, atol=1e-5)
    joint = _stretch(quantile_low=0.01, quantile_high=0.99)(data=data)["normalized"]
    assert not torch.allclose(joint, out)


def test_joint_min_max_scales_all_channels_with_one_pair_of_bounds():
    data = _data(4)
    out = PercentileNormalizer(n_channels=C, norm_mode="per_frame", joint_channels=True)(data=data)[
        "normalized"
    ]
    for b in range(B):
        lo, hi = data[b].min(), data[b].max()
        assert torch.allclose(out[b], (data[b] - lo) / (hi - lo), atol=1e-6)


def test_quantize_255_reproduces_the_uint8_image_path():
    data = _data(2)
    out = _stretch(quantize_levels=255)(data=data)["normalized"]
    for b in range(B):
        stretched = _numpy_joint(data[b].numpy().astype(np.float64), 2.0, 98.0)
        as_png = (stretched * 255).astype(np.uint8).astype(np.float64) / 255.0  # truncation
        assert np.allclose(out[b].numpy(), as_png, atol=1e-6)


def test_frames_are_independent_and_a_constant_frame_is_finite():
    data = _data(3)
    node = _stretch()
    batched = node(data=data)["normalized"]
    for b in range(B):
        assert torch.equal(batched[b], node(data=data[b : b + 1])["normalized"][0])
    flat = node(data=torch.full((1, H, W, C), 7.0))["normalized"]
    assert torch.isfinite(flat).all() and torch.equal(flat, torch.zeros_like(flat))


# ----- 2. unchanged defaults and the other modes ----------------------------------------------


def test_default_per_frame_mode_is_unchanged_per_channel_min_max():
    data = _data(5)
    out = PercentileNormalizer(n_channels=C, norm_mode="per_frame")(data=data)["normalized"]
    lo = data.amin(dim=(1, 2), keepdim=True)
    hi = data.amax(dim=(1, 2), keepdim=True)
    assert torch.equal(out, ((data - lo) / (hi - lo).clamp_min(1e-8)).clamp(0.0, 1.0))


@pytest.mark.parametrize("mode", ["running", "statistical", "per_frame"])
def test_quantize_levels_applies_in_every_mode(mode):
    data = _data(6)
    plain = PercentileNormalizer(n_channels=C, norm_mode=mode)
    quantized = PercentileNormalizer(n_channels=C, norm_mode=mode, quantize_levels=4)
    if mode == "statistical":
        for node in (plain, quantized):
            node.statistical_initialization(iter([{"data": data}]))
    expected = torch.floor(plain(data=data)["normalized"] * 4) / 4
    assert torch.equal(quantized(data=data)["normalized"], expected)


# ----- 3. port contract, validation, serialization --------------------------------------------


def test_port_contract():
    node = _stretch()
    out = node(data=_data())
    assert set(out) == {"normalized"}
    assert out["normalized"].shape == (B, H, W, C)
    assert out["normalized"].dtype == node.OUTPUT_SPECS["normalized"].dtype
    assert 0.0 <= out["normalized"].min() and out["normalized"].max() <= 1.0
    assert node.requires_initial_fit is False


@pytest.mark.parametrize(
    "bad",
    [
        {"norm_mode": "running", "per_frame_quantiles": True},
        {"norm_mode": "statistical", "joint_channels": True},
        {"per_frame_quantiles": 1},
        {"joint_channels": "yes"},
        {"quantize_levels": 0},
        {"quantize_levels": True},
        {"quantize_levels": 2.5},
    ],
)
def test_invalid_hparams_raise(bad):
    params = {"n_channels": C, "norm_mode": "per_frame"}
    params.update(bad)
    with pytest.raises(ValueError):
        PercentileNormalizer(**params)


def test_hparams_json_round_trip():
    hp = _stretch(quantize_levels=255, name="stretch").hparams
    json.dumps(hp)
    assert hp["per_frame_quantiles"] is True and hp["joint_channels"] is True
    assert hp["quantize_levels"] == 255 and hp["quantile_low"] == 0.02
    default = PercentileNormalizer(n_channels=C).hparams
    assert default["per_frame_quantiles"] is False and default["quantize_levels"] is None


# ----- 4. bit-for-bit parity with the stretch these options replace ---------------------------


def _reference_percentile(flat: torch.Tensor, q_percent: float) -> torch.Tensor:
    """The stretch's percentile, verbatim: one sort, position ``q / 100 * (n - 1)``."""
    n = flat.numel()
    s, _ = torch.sort(flat)
    pos = torch.tensor(q_percent / 100.0 * (n - 1), dtype=s.dtype, device=s.device)
    lo = pos.floor().long().clamp(0, n - 1)
    hi = pos.ceil().long().clamp(0, n - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo.to(s.dtype))


def _reference_stretch(data, low, high, quantize_levels, eps):
    """The joint-channel per-frame percentile stretch of the walnut false-RGB path, verbatim."""
    frames = []
    for frame in data:
        flat = frame.reshape(-1)
        lo, hi = _reference_percentile(flat, low), _reference_percentile(flat, high)
        out = ((frame - lo) / (hi - lo).clamp_min(eps)).clamp(0.0, 1.0)
        if quantize_levels is not None:
            out = torch.floor(out * quantize_levels) / quantize_levels
        frames.append(out)
    return torch.stack(frames, dim=0)


@pytest.mark.parametrize("size", [(17, 13), (250, 270), (1000, 1080)])
def test_catalog_stretch_is_bit_identical(size):
    """2nd / 98th percentile, joint, 255 levels, eps 1e-6: the configuration of the catalog."""
    data = _data(7, *size)
    out = _stretch(quantize_levels=255)(data=data)["normalized"]
    assert torch.equal(out, _reference_stretch(data, 2.0, 98.0, 255, 1e-6))


@pytest.mark.parametrize("n", [1, 2, 7, 1000, 1000 * 1080 * 3])
@pytest.mark.parametrize("q", [0.0, 2.0, 37.5, 50.0, 98.0, 100.0])
def test_sorted_quantiles_equal_the_percent_formulation_bit_for_bit(n, q):
    g = torch.Generator().manual_seed(n)
    flat = torch.rand(n, generator=g) * 4000.0
    if n > 2:
        flat[: n // 3] = flat[0]  # ties, as in a clipped or quantised frame
    lo, hi = _sorted_quantiles(flat, (q / 100.0, (100.0 - q) / 100.0))
    assert torch.equal(lo, _reference_percentile(flat, q))
    assert torch.equal(hi, _reference_percentile(flat, 100.0 - q))
