"""SpectralObjectMask: golden rule against a numpy angle map (the median of the median-stride grid,
the angle on the stride grid), brightness invariance, the belt vs a different spectrum, port
contract, hparams."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest
import torch

from cuvis_ai.node.spectral_angle_mapper import SpectralObjectMask

pytestmark = pytest.mark.unit

H, W, C = 40, 40, 12  # odd sample counts on every grid used here: torch's median = numpy's


def _belt_cube(seed: int = 0, b: int = 1) -> torch.Tensor:
    """A belt spectrum with a little noise everywhere, a few pixels of another material."""
    g = torch.Generator().manual_seed(seed)
    belt = torch.linspace(0.2, 0.6, C)
    cube = belt * (1 + 0.01 * torch.randn(b, H, W, C, generator=g))
    cube[:, 8:16, 8:16, :] = torch.linspace(0.6, 0.2, C)  # an object: the opposite slope
    return cube.to(torch.float32)


def _reference(
    cube: np.ndarray, stride: int, deg: float, median_stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """The per-band median of cube[::m, ::m]; the angle of cube[::s, ::s] to it, to H x W."""
    c4 = cube[::stride, ::stride].astype(np.float32)
    x = c4.reshape(-1, c4.shape[-1])
    ref = np.median(cube[::median_stride, ::median_stride].reshape(-1, cube.shape[-1]), axis=0)
    cos = (x @ ref) / (np.linalg.norm(x, axis=1) * np.linalg.norm(ref) + 1e-6)
    ang = np.degrees(np.arccos(np.clip(cos, -1, 1))).reshape(c4.shape[:2])
    full = (
        cv2.resize(
            (ang > deg).astype(np.uint8),
            (cube.shape[1], cube.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        > 0
    )
    return ang, full


@pytest.mark.parametrize("stride", [1, 2, 4])
@pytest.mark.parametrize("median_stride", [1, 8])
def test_matches_the_numpy_angle_map(stride, median_stride):
    cube = _belt_cube(seed=stride)
    node = SpectralObjectMask(min_angle_deg=6.0, stride=stride, median_stride=median_stride)
    out = node(cube=cube)
    ang, full = _reference(cube[0].numpy(), stride, 6.0, median_stride)
    # float32 arccos is ill-conditioned near 0 deg (the belt): a few thousandths of a degree
    assert np.allclose(out["angle"][0, ..., 0].numpy(), ang, atol=1e-2)
    assert np.array_equal(out["decisions"][0, ..., 0].numpy(), full)


def test_the_object_is_marked_and_the_belt_is_not():
    out = SpectralObjectMask(min_angle_deg=6.0, stride=1)(cube=_belt_cube())["decisions"][0, ..., 0]
    assert out[8:16, 8:16].all()
    rest = out.clone()
    rest[8:16, 8:16] = False
    assert not rest.any()


def test_brightness_does_not_make_an_object():
    cube = _belt_cube()
    cube[0, 25:35, 25:35, :] *= 0.3  # a shadow: the belt's spectrum, darker
    out = SpectralObjectMask(min_angle_deg=6.0, stride=1)(cube=cube)["decisions"][0, ..., 0]
    assert not out[25:35, 25:35].any()


def test_frames_of_a_batch_have_their_own_median():
    cube = _belt_cube(b=2)
    cube[1] = cube[1].flip(-1)  # frame 1: everything mirrored in the band axis
    out = SpectralObjectMask(stride=1)(cube=cube)["decisions"]
    assert out[0, 8:16, 8:16].all() and out[1, 8:16, 8:16].all()
    assert out[0].sum() == out[1].sum() == 64


def test_port_contract():
    cube = _belt_cube(b=2)
    node = SpectralObjectMask(stride=4)
    out = node(cube=cube)
    assert set(out) == {"decisions", "angle"}
    assert out["decisions"].shape == (2, H, W, 1)
    assert out["decisions"].dtype == SpectralObjectMask.OUTPUT_SPECS["decisions"].dtype
    assert out["angle"].shape == (2, H // 4, W // 4, 1) and out["angle"].dtype == torch.float32
    assert len(node.state_dict()) == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"min_angle_deg": -1},
        {"min_angle_deg": 200},
        {"min_angle_deg": True},
        {"stride": 0},
        {"stride": 1.5},
        {"stride": True},
        {"median_stride": 0},
        {"median_stride": 2.0},
    ],
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        SpectralObjectMask(**bad)


def test_hparams_round_trip_json():
    hp = SpectralObjectMask(min_angle_deg=5, stride=2, median_stride=4, name="objects").hparams
    assert hp["min_angle_deg"] == 5.0 and hp["stride"] == 2 and hp["median_stride"] == 4
    json.dumps(hp)
    d = SpectralObjectMask().hparams
    assert d["min_angle_deg"] == 6.0 and d["stride"] == 4 and d["median_stride"] == 8


# ---------------------------------------------------------------- threshold="otsu", fill, dilate_px
def _objects_reference(
    angle: np.ndarray,
    threshold: str,
    floor: float,
    ceiling: float,
    fill: bool,
    dilate: int,
    deg: float = 6.0,
) -> np.ndarray:
    """From the node's own angle map [h, w]: OpenCV's Otsu level (0.1 deg steps, clipped), closing
    3 x 3 + scipy hole filling, nearest neighbour to H x W, OpenCV dilation."""
    from scipy import ndimage

    thr = deg
    if threshold == "otsu":
        q = np.clip(angle * 10, 0, 255).astype(np.uint8)
        t, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = min(max(t / 10.0, floor), ceiling)
    o = angle > thr
    if fill:
        o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
        o = ndimage.binary_fill_holes(o)
    full = cv2.resize(o.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
    if dilate:
        full = cv2.dilate(full.astype(np.uint8), np.ones((2 * dilate + 1,) * 2, np.uint8)) > 0
    return full


def _ring_cube(seed: int = 0) -> torch.Tensor:
    """The belt cube with a second object that has a belt-coloured hole (a ring)."""
    cube = _belt_cube(seed)
    cube[0, 20:36, 20:36, :] = torch.linspace(0.5, 0.3, C)
    cube[0, 26:30, 26:30, :] = torch.linspace(0.2, 0.6, C)  # the hole: belt spectrum
    return cube


@pytest.mark.parametrize("stride", [1, 4])
@pytest.mark.parametrize(
    ("threshold", "floor", "ceiling", "fill", "dilate"),
    [
        ("otsu", 3.0, 12.0, False, 0),
        ("otsu", 2.0, 12.0, True, 4),
        ("otsu", 0.0, 1.0, False, 0),
        ("fixed", 3.0, 12.0, True, 0),
        ("fixed", 3.0, 12.0, False, 3),
    ],
)
def test_otsu_fill_dilate_match_the_opencv_reference(
    stride, threshold, floor, ceiling, fill, dilate
):
    node = SpectralObjectMask(
        stride=stride,
        threshold=threshold,
        otsu_floor_deg=floor,
        otsu_ceiling_deg=ceiling,
        fill=fill,
        dilate_px=dilate,
    )
    out = node(cube=_ring_cube(stride))
    exp = _objects_reference(
        out["angle"][0, ..., 0].numpy(), threshold, floor, ceiling, fill, dilate
    )
    assert np.array_equal(out["decisions"][0, ..., 0].numpy(), exp)


def test_fill_closes_the_hole_of_an_object():
    plain = SpectralObjectMask(stride=1)(cube=_ring_cube())["decisions"][0, ..., 0]
    filled = SpectralObjectMask(stride=1, fill=True)(cube=_ring_cube())["decisions"][0, ..., 0]
    assert not plain[26:30, 26:30].any() and filled[26:30, 26:30].all()
    assert filled[20:36, 20:36].all()


def test_otsu_level_is_clipped_to_the_floor_and_ceiling():
    a = SpectralObjectMask(stride=1, threshold="otsu", otsu_floor_deg=0.0, otsu_ceiling_deg=0.5)
    b = SpectralObjectMask(stride=1, threshold="fixed", min_angle_deg=0.5)
    cube = _belt_cube()
    assert torch.equal(a(cube=cube)["decisions"], b(cube=cube)["decisions"])  # ceiling 0.5 wins


def test_defaults_keep_the_fixed_threshold():
    node = SpectralObjectMask()
    assert (node.threshold, node.fill, node.dilate_px) == ("fixed", False, 0)


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_otsu_fill_dilate_cuda_equals_cpu():
    node = SpectralObjectMask(threshold="otsu", otsu_floor_deg=2.0, fill=True, dilate_px=4)
    cube = _ring_cube()
    cpu = node(cube=cube)["decisions"]
    gpu = node(cube=cube.cuda())["decisions"]
    assert gpu.is_cuda and torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize(
    "bad",
    [
        {"threshold": "mean"},
        {"otsu_floor_deg": -1},
        {"otsu_ceiling_deg": 200},
        {"otsu_floor_deg": 8.0, "otsu_ceiling_deg": 4.0},
        {"fill": 1},
        {"dilate_px": -1},
        {"dilate_px": 2.0},
        {"dilate_px": True},
    ],
)
def test_invalid_new_hparams_raise(bad):
    with pytest.raises(ValueError):
        SpectralObjectMask(**bad)


def test_new_hparams_round_trip_json():
    hp = SpectralObjectMask(threshold="otsu", otsu_floor_deg=2.0, fill=True, dilate_px=4).hparams
    assert (
        hp["threshold"],
        hp["otsu_floor_deg"],
        hp["otsu_ceiling_deg"],
        hp["fill"],
        hp["dilate_px"],
    ) == ("otsu", 2.0, 12.0, True, 4)
    json.dumps(hp)
