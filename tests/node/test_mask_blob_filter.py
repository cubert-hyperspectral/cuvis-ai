"""MaskBlobFilter: identical to the chain MaskMinArea -> MaskBlobGate(SpectralObjectMask), the size
test alone without a cube, the empty-mask shortcut, port contract, hparams."""

from __future__ import annotations

import json

import pytest
import torch

from cuvis_ai.node.mask_ops import MaskBlobFilter, MaskBlobGate, MaskMinArea
from cuvis_ai.node.spectral_angle_mapper import SpectralObjectMask

pytestmark = pytest.mark.unit

H, W, C = 40, 40, 12  # odd sample counts on the median grid: torch's median = numpy's


def _scene(seed: int, b: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """A belt cube with two objects, and a sparse random mask with marks on belt and objects."""
    g = torch.Generator().manual_seed(seed)
    belt = torch.linspace(0.2, 0.6, C)
    cube = belt * (1 + 0.01 * torch.randn(b, H, W, C, generator=g))
    cube[:, 4:12, 4:16, :] = torch.linspace(0.6, 0.2, C)
    cube[:, 24:36, 20:30, :] = torch.linspace(0.4, 0.4, C)
    mask = torch.rand(b, H, W, 1, generator=g) < 0.08
    mask[:, 5:11, 6:14, 0] = True  # a mark on the first object
    return cube.to(torch.float32), mask


def _chain(mask, cube, area, k):
    m = MaskMinArea(min_area=area, cell=4)(decisions=mask)["decisions"]
    obj = SpectralObjectMask(min_angle_deg=6.0, stride=4, median_stride=8)(cube=cube)["decisions"]
    return MaskBlobGate(min_px=k, cell=4)(decisions=m, mask=obj)["decisions"]


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(("area", "k"), [(1, 1), (8, 4), (30, 16), (60, 30)])
def test_equals_the_three_node_chain(seed, area, k):
    cube, mask = _scene(seed)
    node = MaskBlobFilter(
        min_area=area, min_object_px=k, min_angle_deg=6.0, cell=4, median_stride=8
    )
    out = node(decisions=mask, cube=cube)["decisions"]
    assert torch.equal(out, _chain(mask, cube, area, k))


def test_marks_on_the_empty_belt_go_and_marks_on_objects_stay():
    cube, _ = _scene(0, b=1)
    mask = torch.zeros(1, H, W, 1, dtype=torch.bool)
    mask[0, 5:11, 6:14, 0] = True  # on the first object
    mask[0, 18:24, 34:40, 0] = True  # on the belt
    out = MaskBlobFilter(min_area=8, min_object_px=4)(decisions=mask, cube=cube)["decisions"]
    exp = torch.zeros_like(mask)
    exp[0, 5:11, 6:14, 0] = True
    assert torch.equal(out, exp)


def test_without_a_cube_only_the_size_test_runs():
    cube, mask = _scene(5)
    out = MaskBlobFilter(min_area=30)(decisions=mask)["decisions"]
    assert torch.equal(out, MaskMinArea(min_area=30, cell=4)(decisions=mask)["decisions"])
    out0 = MaskBlobFilter(min_area=30, min_object_px=0)(decisions=mask, cube=cube)["decisions"]
    assert torch.equal(out0, out)


def test_an_empty_mask_is_returned_without_work():
    cube, _ = _scene(6)
    empty = torch.zeros(2, H, W, 1, dtype=torch.bool)
    out = MaskBlobFilter()(decisions=empty, cube=cube)["decisions"]
    assert out.shape == empty.shape and not out.any()


def test_port_contract_and_no_serialized_state():
    cube, mask = _scene(7)
    node = MaskBlobFilter(min_area=8, min_object_px=4)
    out = node(decisions=mask, cube=cube)
    assert set(out) == {"decisions"} and out["decisions"].shape == mask.shape
    assert out["decisions"].dtype == MaskBlobFilter.OUTPUT_SPECS["decisions"].dtype
    assert len(node.state_dict()) == 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_equals_cpu():
    cube, mask = _scene(8)
    node = MaskBlobFilter(min_area=8, min_object_px=4)
    cpu = node(decisions=mask, cube=cube)["decisions"]
    gpu = node(decisions=mask.cuda(), cube=cube.cuda())["decisions"]
    assert gpu.is_cuda and torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize(
    "bad",
    [
        {"min_area": -1},
        {"min_object_px": -1},
        {"cell": 0},
        {"median_stride": 0},
        {"min_angle_deg": -1},
        {"min_angle_deg": True},
        {"min_area": 2.5},
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        MaskBlobFilter(**bad)


def test_hparams_round_trip_json():
    hp = MaskBlobFilter(min_area=100, min_object_px=8, name="blobs").hparams
    assert hp["min_area"] == 100 and hp["min_object_px"] == 8
    json.dumps(hp)
    d = MaskBlobFilter().hparams
    assert (
        d["min_area"],
        d["min_object_px"],
        d["min_angle_deg"],
        d["cell"],
        d["median_stride"],
    ) == (
        250,
        16,
        6.0,
        4,
        8,
    )
