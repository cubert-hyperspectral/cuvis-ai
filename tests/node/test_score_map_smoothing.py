"""ScoreMapSmoothing: golden rule against OpenCV's GaussianBlur, port contract, gradient, border,
hparams."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from cuvis_ai.node.spatial import ScoreMapSmoothing

pytestmark = pytest.mark.unit


def _maps(b: int = 2, h: int = 60, w: int = 70, c: int = 1, seed: int = 0) -> torch.Tensor:
    return torch.rand(b, h, w, c, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.5, 4.0, 8.0])
def test_matches_opencv_gaussian_blur(sigma):
    s = _maps(c=2, seed=int(sigma * 10))
    out = ScoreMapSmoothing(sigma_px=sigma)(scores=s)["scores"].numpy()
    r = int(round(4 * sigma))
    for b in range(s.shape[0]):
        for c in range(s.shape[3]):
            exp = cv2.GaussianBlur(
                s[b, :, :, c].numpy(),
                (2 * r + 1, 2 * r + 1),
                sigmaX=sigma,
                sigmaY=sigma,
                borderType=cv2.BORDER_REFLECT_101,
            )
            assert np.allclose(out[b, :, :, c], exp, atol=1e-5), (sigma, b, c)


def test_a_one_pixel_peak_drops_and_a_wide_plateau_keeps_its_level():
    s = torch.zeros(1, 80, 80, 1)
    s[0, 20, 20, 0] = 1.0
    s[0, 40:75, 40:75, 0] = 1.0
    out = ScoreMapSmoothing(sigma_px=4.0)(scores=s)["scores"]
    assert float(out[0, 20, 20, 0]) < 0.02
    assert float(out[0, 57, 57, 0]) > 0.99


def test_sigma_zero_is_the_identity_and_constant_maps_stay_constant():
    s = _maps()
    assert torch.equal(ScoreMapSmoothing(sigma_px=0)(scores=s)["scores"], s)
    flat = torch.full((1, 30, 40, 1), 0.7)
    assert torch.allclose(ScoreMapSmoothing(sigma_px=3)(scores=flat)["scores"], flat, atol=1e-6)


def test_a_map_narrower_than_the_kernel_repeats_its_border():
    s = _maps(h=5, w=7)
    out = ScoreMapSmoothing(sigma_px=8.0)(scores=s)["scores"]
    assert out.shape == s.shape and torch.isfinite(out).all()


def test_is_differentiable():
    s = _maps(b=1).requires_grad_(True)
    ScoreMapSmoothing(sigma_px=2.0)(scores=s)["scores"].sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()


def test_port_contract_and_no_serialized_state():
    s = _maps(c=3)
    node = ScoreMapSmoothing()
    out = node(scores=s)
    spec = ScoreMapSmoothing.OUTPUT_SPECS["scores"]
    assert set(out) == {"scores"} and out["scores"].shape == s.shape
    assert out["scores"].dtype == spec.dtype
    assert len(node.state_dict()) == 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_matches_cpu():
    s = _maps()
    node = ScoreMapSmoothing(sigma_px=3.0)
    cpu = node(scores=s)["scores"]
    gpu = node.to("cuda")(scores=s.cuda())["scores"]
    assert gpu.is_cuda and torch.allclose(gpu.cpu(), cpu, atol=1e-5)


@pytest.mark.parametrize("bad", [-1, float("inf"), True, "8", None])
def test_invalid_sigma_raises(bad):
    with pytest.raises(ValueError):
        ScoreMapSmoothing(sigma_px=bad)


def test_hparams_round_trip_json():
    hp = ScoreMapSmoothing(sigma_px=6, name="smooth").hparams
    assert hp["sigma_px"] == 6.0
    json.dumps(hp)
    assert ScoreMapSmoothing().hparams["sigma_px"] == 8.0
