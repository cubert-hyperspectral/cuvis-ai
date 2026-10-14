"""GridSubsample and ScoreUpsample: golden slicing / interpolation, port contract, hparams."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

from cuvis_ai.node.spatial import GridSubsample, ScoreUpsample

pytestmark = pytest.mark.unit


def _cube(b: int = 2, h: int = 13, w: int = 10, c: int = 5, seed: int = 0) -> torch.Tensor:
    return torch.rand(b, h, w, c, generator=torch.Generator().manual_seed(seed))


# ----- GridSubsample ----------------------------------------------------------------------------


@pytest.mark.parametrize("stride", [1, 2, 4])
def test_grid_is_the_stride_slice(stride):
    cube = _cube()
    out = GridSubsample(stride=stride)(cube=cube)["cube"]
    assert torch.equal(out, cube[:, ::stride, ::stride, :])


def test_grid_port_contract_and_odd_sizes():
    out = GridSubsample(stride=4)(cube=_cube(h=13, w=10))["cube"]
    assert out.shape == (2, 4, 3, 5) and out.dtype == torch.float32  # ceil(13/4), ceil(10/4)


def test_grid_is_differentiable():
    cube = _cube().requires_grad_()
    GridSubsample(stride=3)(cube=cube)["cube"].sum().backward()
    assert cube.grad is not None and cube.grad[:, ::3, ::3].eq(1).all()


@pytest.mark.parametrize("bad", [0, -2, 1.5, True])
def test_grid_invalid_stride_raises(bad):
    with pytest.raises(ValueError):
        GridSubsample(stride=bad)


def test_grid_hparams_round_trip_json():
    hp = GridSubsample(stride=4, name="grid").hparams
    assert hp["stride"] == 4
    json.dumps(hp)


# ----- ScoreUpsample ----------------------------------------------------------------------------


def test_upsample_matches_bilinear_interpolation_of_a_single_map():
    """The reference op: F.interpolate(m[None, None], size, "bilinear", align_corners=False)."""
    m = torch.rand(25, 27, generator=torch.Generator().manual_seed(1))
    ref = torch.zeros(1, 100, 108, 61)
    out = ScoreUpsample()(scores=m[None, :, :, None], reference=ref)["scores"]
    expected = F.interpolate(m[None, None], size=(100, 108), mode="bilinear", align_corners=False)[
        0, 0
    ]
    assert out.shape == (1, 100, 108, 1) and torch.equal(out[0, :, :, 0], expected)


def test_upsample_batch_and_channels_follow_the_reference_size():
    scores = torch.rand(2, 4, 5, 3, generator=torch.Generator().manual_seed(2))
    out = ScoreUpsample(mode="nearest")(scores=scores, reference=torch.zeros(2, 8, 10, 1))["scores"]
    assert out.shape == (2, 8, 10, 3) and out.dtype == torch.float32
    assert torch.equal(out[:, ::2, ::2, :], scores)  # nearest x2 repeats every sample


def test_grid_then_upsample_round_trips_a_constant_map():
    cube = torch.full((1, 16, 12, 1), 3.5)
    grid = GridSubsample(stride=4)(cube=cube)["cube"]
    back = ScoreUpsample()(scores=grid, reference=cube)["scores"]
    assert torch.allclose(back, cube)


def test_upsample_is_differentiable():
    scores = torch.rand(1, 3, 3, 1).requires_grad_()
    ScoreUpsample(mode="bicubic")(scores=scores, reference=torch.zeros(1, 7, 7, 2))[
        "scores"
    ].sum().backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()


def test_upsample_invalid_mode_raises():
    with pytest.raises(ValueError):
        ScoreUpsample(mode="area")


def test_upsample_hparams_round_trip_json():
    hp = ScoreUpsample(mode="bilinear", name="up").hparams
    assert hp["mode"] == "bilinear"
    json.dumps(hp)
