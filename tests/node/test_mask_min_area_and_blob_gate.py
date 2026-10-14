"""MaskMinArea: golden rule against scipy's 8-connected labelling, channels, frames, port
contract, hparams."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scipy import ndimage

from cuvis_ai.node.mask_ops import MaskBlobGate, MaskMinArea

pytestmark = pytest.mark.unit

H, W = 20, 24


def _cell_reference(m: np.ndarray, w: np.ndarray, threshold: float, cell: int) -> np.ndarray:
    """The cell rule by scipy: cells occupied where any pixel is set, 8-connected cell blobs, a blob
    kept when its pixel weights sum to the threshold, the kept cells cut back to the mask."""
    h, wd = m.shape
    hc, wc = -(-h // cell), -(-wd // cell)
    pad = lambda a: np.pad(a, ((0, hc * cell - h), (0, wc * cell - wd)))  # noqa: E731
    occ = pad(m).reshape(hc, cell, wc, cell).any(axis=(1, 3))
    sums = pad(w.astype(np.float64)).reshape(hc, cell, wc, cell).sum(axis=(1, 3))
    lab, nb = ndimage.label(occ, structure=np.ones((3, 3)))
    keep = np.zeros(nb + 1, bool)
    if nb:
        keep[1:] = ndimage.sum(sums, lab, index=np.arange(1, nb + 1)) >= threshold
    cells = keep[lab]
    full = np.repeat(np.repeat(cells, cell, axis=0), cell, axis=1)[:h, :wd]
    return m & full


def _reference(m: np.ndarray, n: int) -> np.ndarray:
    """Blobs of a 2-D mask with >= n pixels, 8-connected, by scipy."""
    lab, nb = ndimage.label(m, structure=np.ones((3, 3)))
    if not nb:
        return np.zeros_like(m)
    area = ndimage.sum(np.ones_like(m), lab, index=np.arange(1, nb + 1))
    keep = np.zeros(nb + 1, bool)
    keep[1:] = area >= n
    return keep[lab]


def test_drops_the_small_blobs_and_keeps_the_large_ones():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    m[0, 1:3, 1:3, 0] = True  # 4 px
    m[0, 10:15, 10:16, 0] = True  # 30 px
    out = MaskMinArea(min_area=10, cell=1)(decisions=m)["decisions"]
    exp = torch.zeros_like(m)
    exp[0, 10:15, 10:16, 0] = True
    assert torch.equal(out, exp)
    # the same on the default 4 x 4 cell grid: the two marks are more than a cell apart
    out = MaskMinArea(min_area=10)(decisions=m)["decisions"]
    exp = torch.zeros_like(m)
    exp[0, 10:15, 10:16, 0] = True
    assert torch.equal(out, exp)


def test_diagonal_neighbours_belong_to_one_blob():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    for k in range(6):  # a diagonal line of 6 px: one 8-connected blob
        m[0, 2 + k, 2 + k, 0] = True
    for cell in (1, 4):
        assert torch.equal(MaskMinArea(min_area=6, cell=cell)(decisions=m)["decisions"], m)
        assert not MaskMinArea(min_area=7, cell=cell)(decisions=m)["decisions"].any()


def test_marks_whose_cells_touch_form_one_blob():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    m[0, 0:2, 0:2, 0] = True  # 4 px in cell (0, 0)
    m[0, 0:2, 6:8, 0] = True  # 4 px in cell (0, 1): 4 px gap, the cells touch
    assert not MaskMinArea(min_area=5, cell=1)(decisions=m)["decisions"].any()
    assert torch.equal(MaskMinArea(min_area=5, cell=4)(decisions=m)["decisions"], m)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 17, 60])
def test_cell_one_matches_scipy_on_random_masks(n):
    g = torch.Generator().manual_seed(n)
    m = torch.rand(3, H, W, 2, generator=g) < 0.35
    out = MaskMinArea(min_area=n, cell=1)(decisions=m)["decisions"].numpy()
    for b in range(3):
        for c in range(2):
            exp = m[b, :, :, c].numpy() if n <= 1 else _reference(m[b, :, :, c].numpy(), n)
            assert np.array_equal(out[b, :, :, c], exp), (b, c)


@pytest.mark.parametrize("cell", [2, 3, 4])
@pytest.mark.parametrize("n", [5, 30, 120])
def test_cell_grid_matches_the_scipy_cell_rule(cell, n):
    g = torch.Generator().manual_seed(10 * cell + n)
    m = torch.rand(2, H, W, 1, generator=g) < 0.06  # sparse marks: several cell blobs
    out = MaskMinArea(min_area=n, cell=cell)(decisions=m)["decisions"].numpy()
    for b in range(2):
        mm = m[b, :, :, 0].numpy()
        assert np.array_equal(out[b, :, :, 0], _cell_reference(mm, mm, n, cell)), b


def test_frames_and_channels_are_filtered_on_their_own():
    m = torch.zeros(2, H, W, 2, dtype=torch.bool)
    m[0, 0:4, 0:4, 0] = True  # 16 px in frame 0, channel 0
    m[1, 0:2, 0:2, 1] = True  # 4 px in frame 1, channel 1
    out = MaskMinArea(min_area=10)(decisions=m)["decisions"]
    assert out[0, :, :, 0].sum() == 16 and not out[1].any() and not out[0, :, :, 1].any()


def test_does_not_mutate_its_input_and_handles_empty_masks():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    m[0, 1:3, 1:3, 0] = True
    before = m.clone()
    MaskMinArea(min_area=10)(decisions=m)
    assert torch.equal(m, before)
    empty = MaskMinArea()(decisions=torch.zeros(2, H, W, 1, dtype=torch.bool))["decisions"]
    assert empty.shape == (2, H, W, 1) and not empty.any()
    none = MaskMinArea()(decisions=torch.zeros(0, H, W, 1, dtype=torch.bool))["decisions"]
    assert none.shape == (0, H, W, 1)


def test_holds_no_serialized_state():
    assert len(MaskMinArea().state_dict()) == 0


def test_port_contract():
    m = torch.rand(2, H, W, 1, generator=torch.Generator().manual_seed(3)) < 0.5
    out = MaskMinArea()(decisions=m)
    spec = MaskMinArea.OUTPUT_SPECS["decisions"]
    assert set(out) == {"decisions"}
    assert out["decisions"].shape == m.shape and out["decisions"].dtype == spec.dtype


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_input_comes_back_on_cuda_and_equal():
    m = torch.rand(2, H, W, 1, generator=torch.Generator().manual_seed(5)) < 0.4
    cpu = MaskMinArea(min_area=8)(decisions=m)["decisions"]
    gpu = MaskMinArea(min_area=8)(decisions=m.cuda())["decisions"]
    assert gpu.is_cuda and torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize("bad", [-1, 2.5, True, "250", None])
def test_invalid_min_area_raises(bad):
    with pytest.raises(ValueError):
        MaskMinArea(min_area=bad)


def test_hparams_round_trip_json():
    hp = MaskMinArea(min_area=100, name="speck_filter").hparams
    assert hp["min_area"] == 100
    json.dumps(hp)
    assert MaskMinArea().hparams["min_area"] == 250


# ----- MaskBlobGate -------------------------------------------------------------------------


def _gate_reference(m: np.ndarray, g: np.ndarray, k: int) -> np.ndarray:
    lab, nb = ndimage.label(m, structure=np.ones((3, 3)))
    if not nb:
        return np.zeros_like(m)
    hits = ndimage.sum(g, lab, index=np.arange(1, nb + 1))
    keep = np.zeros(nb + 1, bool)
    keep[1:] = hits >= k
    return keep[lab]


def test_blob_gate_keeps_the_blobs_on_the_gate_only():
    m = torch.zeros(1, H, W, 1, dtype=torch.bool)
    m[0, 1:5, 1:5, 0] = True  # on empty background
    m[0, 10:16, 10:16, 0] = True  # around an object
    g = torch.zeros(1, H, W, 1, dtype=torch.bool)
    g[0, 12:14, 12:14, 0] = True  # 4 object pixels inside the second blob
    out = MaskBlobGate(min_px=4, cell=1)(decisions=m, mask=g)["decisions"]
    exp = torch.zeros_like(m)
    exp[0, 10:16, 10:16, 0] = True
    assert torch.equal(out, exp)
    assert not MaskBlobGate(min_px=5, cell=1)(decisions=m, mask=g)["decisions"].any()


@pytest.mark.parametrize("k", [1, 3, 10])
def test_blob_gate_cell_one_matches_scipy(k):
    gen = torch.Generator().manual_seed(k)
    m = torch.rand(2, H, W, 1, generator=gen) < 0.35
    g = torch.rand(2, H, W, 2, generator=gen) < 0.2  # two gate channels: any counts
    out = MaskBlobGate(min_px=k, cell=1)(decisions=m, mask=g)["decisions"].numpy()
    for b in range(2):
        exp = _gate_reference(m[b, :, :, 0].numpy(), g[b].any(-1).numpy(), k)
        assert np.array_equal(out[b, :, :, 0], exp), b


@pytest.mark.parametrize("k", [1, 3, 10])
def test_blob_gate_cell_grid_matches_the_scipy_cell_rule(k):
    gen = torch.Generator().manual_seed(100 + k)
    m = torch.rand(2, H, W, 1, generator=gen) < 0.06
    g = torch.rand(2, H, W, 1, generator=gen) < 0.3
    out = MaskBlobGate(min_px=k)(decisions=m, mask=g)["decisions"].numpy()
    for b in range(2):
        mm, gg = m[b, :, :, 0].numpy(), g[b, :, :, 0].numpy()
        assert np.array_equal(out[b, :, :, 0], _cell_reference(mm, mm & gg, k, 4)), b


@pytest.mark.parametrize("k", [1, 3, 10])
def test_blob_gate_invert_keeps_exactly_the_other_blobs(k):
    gen = torch.Generator().manual_seed(200 + k)
    m = torch.rand(2, H, W, 1, generator=gen) < 0.06
    g = torch.rand(2, H, W, 1, generator=gen) < 0.3
    kept = MaskBlobGate(min_px=k)(decisions=m, mask=g)["decisions"]
    out = MaskBlobGate(min_px=k, invert=True)(decisions=m, mask=g)["decisions"]
    assert torch.equal(out | kept, m) and not (out & kept).any()
    for b in range(2):
        mm, gg = m[b, :, :, 0].numpy(), g[b, :, :, 0].numpy()
        assert np.array_equal(out[b, :, :, 0].numpy(), mm & ~_cell_reference(mm, mm & gg, k, 4)), b


def test_blob_gate_invert_puts_back_the_marks_a_cut_removed():
    before = torch.zeros(1, H, W, 1, dtype=torch.bool)
    before[0, 1:6, 1:6, 0] = True  # trimmed by the cut: stays as cut
    before[0, 12:18, 14:22, 0] = True  # removed by the cut: comes back whole
    cut = before.clone()
    cut[0, 1:6, 4:6, 0] = False
    cut[0, 12:18, 14:22, 0] = False
    back = MaskBlobGate(min_px=1, invert=True)(decisions=before, mask=cut)["decisions"]
    exp = torch.zeros_like(before)
    exp[0, 12:18, 14:22, 0] = True
    assert torch.equal(back, exp)
    assert torch.equal(back | cut, cut | exp)
    # nothing left after the cut: every mark comes back; zero gate pixels needed: nothing passes
    assert torch.equal(
        MaskBlobGate(min_px=1, invert=True)(decisions=before, mask=torch.zeros_like(before))[
            "decisions"
        ],
        before,
    )
    assert not MaskBlobGate(min_px=0, invert=True)(decisions=before, mask=cut)["decisions"].any()


def test_blob_gate_zero_is_the_identity_and_shapes_are_checked():
    m = torch.rand(1, H, W, 1, generator=torch.Generator().manual_seed(1)) < 0.4
    g = torch.zeros(1, H, W, 1, dtype=torch.bool)
    assert torch.equal(MaskBlobGate(min_px=0)(decisions=m, mask=g)["decisions"], m)
    with pytest.raises(ValueError):
        MaskBlobGate()(decisions=m, mask=torch.zeros(1, H, W + 1, 1, dtype=torch.bool))


def test_blob_gate_port_contract_and_hparams():
    m = torch.rand(2, H, W, 1, generator=torch.Generator().manual_seed(2)) < 0.4
    out = MaskBlobGate()(decisions=m, mask=m)
    assert set(out) == {"decisions"} and out["decisions"].shape == m.shape
    assert out["decisions"].dtype == torch.bool
    with pytest.raises(ValueError):
        MaskBlobGate(min_px=-1)
    with pytest.raises(ValueError):
        MaskBlobGate(invert=1)
    hp = MaskBlobGate(min_px=8, name="objgate").hparams
    assert hp["min_px"] == 8 and hp["invert"] is False
    assert MaskBlobGate(min_px=1, invert=True).hparams["invert"] is True
    json.dumps(hp)
