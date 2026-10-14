"""ImageTiler / GridStitcher: golden reference, port contract, round trip, gradients, errors."""

from __future__ import annotations

import pytest
import torch
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.tiling import GridStitcher, ImageTiler

pytestmark = pytest.mark.unit


def _image(b: int = 2, h: int = 6, w: int = 8, c: int = 3) -> torch.Tensor:
    return torch.arange(b * h * w * c, dtype=torch.float32).reshape(b, h, w, c)


def _reference_tiles(x: torch.Tensor, t: int) -> torch.Tensor:
    """Independent reference: explicit slicing at round(i * H / T), image-major then row-major."""
    _, h, w, _ = x.shape
    hs = [round(i * h / t) for i in range(t + 1)]
    ws = [round(j * w / t) for j in range(t + 1)]
    return torch.cat(
        [
            x[bi : bi + 1, hs[i] : hs[i + 1], ws[j] : ws[j + 1], :]
            for bi in range(x.shape[0])
            for i in range(t)
            for j in range(t)
        ],
        dim=0,
    )


def _reference_stitch(tiles: torch.Tensor, t: int) -> torch.Tensor:
    """Independent reference: concatenate the columns of each tile row, then the rows."""
    per_image = []
    for bi in range(tiles.shape[0] // (t * t)):
        block = tiles[bi * t * t : (bi + 1) * t * t]
        rows = [
            torch.cat([block[i * t + j : i * t + j + 1] for j in range(t)], dim=2) for i in range(t)
        ]
        per_image.append(torch.cat(rows, dim=1))
    return torch.cat(per_image, dim=0)


@pytest.mark.parametrize("t", [1, 2, 3])
def test_tiler_golden(t):
    x = _image(h=6 * t, w=4 * t, c=5)
    out = ImageTiler(tiles=t)(image=x)["tiles"]
    assert torch.equal(out, _reference_tiles(x, t))


@pytest.mark.parametrize("t", [1, 2, 3])
def test_stitcher_golden(t):
    torch.manual_seed(0)
    grids = torch.rand(2 * t * t, 4, 5, 7)
    out = GridStitcher(tiles=t)(tiles=grids)["grid"]
    assert torch.equal(out, _reference_stitch(grids, t))


@pytest.mark.parametrize("t", [1, 2, 4])
def test_round_trip_is_identity(t):
    x = torch.rand(3, 8 * t, 5 * t, 2)
    tiles = ImageTiler(tiles=t)(image=x)["tiles"]
    assert torch.equal(GridStitcher(tiles=t)(tiles=tiles)["grid"], x)


def test_camera_frame_split_matches_slicing_reference():
    """A 1000 x 1080 camera frame with T = 2: the equal-tile reshape equals explicit slicing."""
    x = torch.rand(1, 1000, 1080, 3)
    out = ImageTiler(tiles=2)(image=x)["tiles"]
    assert out.shape == (4, 500, 540, 3)
    assert torch.equal(out, _reference_tiles(x, 2))


def test_port_contract():
    x = _image()
    cases = (
        (ImageTiler(tiles=2), {"image": x}, "tiles"),
        (GridStitcher(tiles=2), {"tiles": torch.rand(8, 3, 3, 4)}, "grid"),
    )
    for node, kw, port in cases:
        out = node(**kw)
        assert set(out) == set(node.OUTPUT_SPECS)
        spec = node.OUTPUT_SPECS[port]
        assert isinstance(spec, PortSpec)
        assert out[port].dtype == spec.dtype and out[port].ndim == len(spec.shape)
    assert ImageTiler(tiles=2)(image=x)["tiles"].shape == (8, 3, 4, 3)
    assert GridStitcher(tiles=2)(tiles=torch.rand(8, 3, 3, 4))["grid"].shape == (2, 6, 6, 4)


def test_gradient_flows():
    x = torch.rand(1, 4, 4, 2, requires_grad=True)
    y = GridStitcher(tiles=2)(tiles=ImageTiler(tiles=2)(image=x)["tiles"])["grid"]
    (y * torch.arange(y.numel()).reshape(y.shape)).sum().backward()
    assert torch.equal(x.grad, torch.arange(x.numel(), dtype=torch.float32).reshape(x.shape))


def test_errors():
    with pytest.raises(ValueError, match="positive int"):
        ImageTiler(tiles=0)
    with pytest.raises(ValueError, match="positive int"):
        GridStitcher(tiles=1.5)
    with pytest.raises(ValueError, match="equal tiles"):
        ImageTiler(tiles=3)(image=torch.rand(1, 10, 9, 3))
    with pytest.raises(ValueError, match="whole number"):
        GridStitcher(tiles=2)(tiles=torch.rand(6, 2, 2, 1))


def test_hparams_recorded():
    assert ImageTiler(tiles=3).tiles == 3
    assert GridStitcher(tiles=3).tiles == 3
