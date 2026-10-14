"""MaskPeakGate: golden rule against an independent cell-grid reference (scipy), the cut scenario
(the piece with the mark's peak stays, a halo piece far below it goes), blobs outside every
reference blob stay, frames and channels on their own, the empty / ratio-0 shortcuts, port
contract, CUDA == CPU, hparams."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from scipy import ndimage

from cuvis_ai.node.mask_ops import MaskPeakGate

pytestmark = pytest.mark.unit

H, W, CELL = 48, 56, 4


def _pad(m: np.ndarray, fill) -> np.ndarray:
    p = np.full((-(-m.shape[0] // CELL) * CELL, -(-m.shape[1] // CELL) * CELL), fill, dtype=m.dtype)
    p[: m.shape[0], : m.shape[1]] = m
    return p.reshape(p.shape[0] // CELL, CELL, p.shape[1] // CELL, CELL)


def _reference(dec: np.ndarray, ref: np.ndarray, s: np.ndarray, ratio: float) -> np.ndarray:
    """Blobs = 8-connected occupied cells of dec; a blob stays if its peak >= ratio x the highest
    peak of the reference blobs (cells) it touches, or if it touches none."""
    occ, rocc = _pad(dec, False).any(axis=(1, 3)), _pad(ref, False).any(axis=(1, 3))
    pk = _pad(np.where(dec, s, -np.inf), -np.inf).max(axis=(1, 3))
    rpk = _pad(np.where(ref, s, -np.inf), -np.inf).max(axis=(1, 3))
    lab, n = ndimage.label(occ, structure=np.ones((3, 3)))
    rlab, _ = ndimage.label(rocc, structure=np.ones((3, 3)))
    keep = np.zeros(occ.shape, bool)
    for k in range(1, n + 1):
        cells = lab == k
        parents = set(np.unique(rlab[cells]).tolist()) - {0}
        if not parents or pk[cells].max() >= ratio * max(rpk[rlab == q].max() for q in parents):
            keep |= cells
    full = np.repeat(np.repeat(keep, CELL, 0), CELL, 1)[: dec.shape[0], : dec.shape[1]]
    return dec & full


def _scene(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Two reference marks, random scores, the decisions = random pieces of the marks plus one
    piece outside every mark."""
    g = np.random.default_rng(seed)
    ref = np.zeros((H, W), bool)
    ref[5:20, 4:30] = True
    ref[28:44, 30:52] = True
    s = g.uniform(1.0, 2.0, (H, W)).astype(np.float32)
    dec = ref & (g.random((H, W)) < 0.3)
    dec[1:3, 40:44] = True
    return dec, ref, s


def _t(m: np.ndarray, dtype=torch.bool) -> torch.Tensor:
    return torch.from_numpy(m)[None, ..., None].to(dtype)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("ratio", [0.3, 0.6, 0.8, 1.0])
def test_matches_the_cell_reference(seed, ratio):
    dec, ref, s = _scene(seed)
    out = MaskPeakGate(ratio=ratio)(
        decisions=_t(dec), reference=_t(ref), scores=_t(s, torch.float32)
    )
    assert np.array_equal(out["decisions"][0, ..., 0].numpy(), _reference(dec, ref, s, ratio))


def test_the_peak_piece_stays_and_the_halo_piece_goes():
    ref = np.zeros((H, W), bool)
    ref[8:32, 8:48] = True  # one mark: an FO plus its halo
    s = np.full((H, W), 1.2, np.float32)
    s[12:20, 12:20] = 5.0  # the FO's peak
    dec = ref.copy()
    dec[:, 24:32] = False  # the cut: a left piece (with the peak) and a right piece (halo)
    out = MaskPeakGate(ratio=0.8)(decisions=_t(dec), reference=_t(ref), scores=_t(s, torch.float32))
    exp = dec.copy()
    exp[:, 32:] = False
    assert np.array_equal(out["decisions"][0, ..., 0].numpy(), exp)


def test_a_blob_outside_every_reference_blob_stays():
    dec = np.zeros((H, W), bool)
    dec[2:6, 2:6] = True
    ref = np.zeros((H, W), bool)
    ref[30:40, 30:40] = True
    s = np.ones((H, W), np.float32)
    s[30:40, 30:40] = 9.0
    out = MaskPeakGate(ratio=1.0)(decisions=_t(dec), reference=_t(ref), scores=_t(s, torch.float32))
    assert np.array_equal(out["decisions"][0, ..., 0].numpy(), dec)


def test_frames_and_channels_on_their_own():
    a, ra, sa = _scene(5)
    b, rb, sb = _scene(6)
    dec = torch.stack(
        [torch.from_numpy(np.stack([a, b], -1)), torch.from_numpy(np.stack([b, a], -1))]
    )
    ref = torch.stack([torch.from_numpy(ra), torch.from_numpy(rb)])[..., None]
    s = torch.stack([torch.from_numpy(sa), torch.from_numpy(sb)])[..., None]
    out = MaskPeakGate(ratio=0.7)(decisions=dec, reference=ref, scores=s)["decisions"].numpy()
    assert np.array_equal(out[0, ..., 0], _reference(a, ra, sa, 0.7))
    assert np.array_equal(out[0, ..., 1], _reference(b, ra, sa, 0.7))
    assert np.array_equal(out[1, ..., 0], _reference(b, rb, sb, 0.7))
    assert np.array_equal(out[1, ..., 1], _reference(a, rb, sb, 0.7))


def test_an_empty_mask_and_ratio_zero_return_the_input():
    dec, ref, s = _scene(7)
    empty = torch.zeros(1, H, W, 1, dtype=torch.bool)
    node = MaskPeakGate()
    out = node(decisions=empty, reference=_t(ref), scores=_t(s, torch.float32))["decisions"]
    assert out.shape == empty.shape and not out.any()
    out0 = MaskPeakGate(ratio=0)(decisions=_t(dec), reference=_t(ref), scores=_t(s, torch.float32))
    assert torch.equal(out0["decisions"], _t(dec))


def test_mismatched_shapes_raise():
    dec, ref, s = _scene(8)
    with pytest.raises(ValueError):
        MaskPeakGate()(decisions=_t(dec), reference=_t(ref[:-4]), scores=_t(s, torch.float32))


def test_port_contract_and_no_serialized_state():
    dec, ref, s = _scene(9)
    node = MaskPeakGate()
    out = node(decisions=_t(dec), reference=_t(ref), scores=_t(s, torch.float32))
    assert set(out) == {"decisions"} and out["decisions"].shape == (1, H, W, 1)
    assert out["decisions"].dtype == MaskPeakGate.OUTPUT_SPECS["decisions"].dtype
    assert len(node.state_dict()) == 0


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_equals_cpu():
    dec, ref, s = _scene(10)
    node = MaskPeakGate(ratio=0.8)
    cpu = node(decisions=_t(dec), reference=_t(ref), scores=_t(s, torch.float32))["decisions"]
    gpu = node(
        decisions=_t(dec).cuda(), reference=_t(ref).cuda(), scores=_t(s, torch.float32).cuda()
    )["decisions"]
    assert gpu.is_cuda and torch.equal(gpu.cpu(), cpu)


@pytest.mark.parametrize(
    "bad", [{"ratio": -0.1}, {"ratio": 1.5}, {"ratio": True}, {"cell": 0}, {"cell": 2.0}]
)
def test_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        MaskPeakGate(**bad)


def test_hparams_round_trip_json():
    hp = MaskPeakGate(ratio=0.5, cell=2, name="peaks").hparams
    assert hp["ratio"] == 0.5 and hp["cell"] == 2
    json.dumps(hp)
    d = MaskPeakGate().hparams
    assert (d["ratio"], d["cell"]) == (0.8, 4)
