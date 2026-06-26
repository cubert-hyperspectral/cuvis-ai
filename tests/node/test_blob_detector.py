from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from cuvis_ai.node.blob_detector import BlobDetector

pytestmark = pytest.mark.unit


def _cube_from_brightness(bright: np.ndarray, channels: int = 3) -> torch.Tensor:
    """Make a [1, H, W, C] cube whose per-pixel band mean equals ``bright``."""
    stacked = np.repeat(bright[:, :, None].astype(np.float32), channels, axis=2)
    return torch.from_numpy(stacked[None])


def _components(label_map: np.ndarray) -> set[frozenset[tuple[int, int]]]:
    """Set of pixel-coordinate sets, one per nonzero label, for relabel-invariant compare."""
    out: set[frozenset[tuple[int, int]]] = set()
    for lab in np.unique(label_map):
        if lab == 0:
            continue
        ys, xs = np.where(label_map == lab)
        out.add(frozenset(zip(xs.tolist(), ys.tolist(), strict=True)))
    return out


def _scene() -> np.ndarray:
    """10x10 brightness: two 3x3 squares, one 3x4 square, plus a 1-pixel speck."""
    bright = np.zeros((10, 10), dtype=np.float32)
    bright[1:4, 1:4] = 1.0  # A, 9 px
    bright[1:4, 6:9] = 1.0  # B, 9 px
    bright[6:9, 1:5] = 1.0  # C, 12 px
    bright[9, 9] = 1.0  # speck, 1 px
    return bright


@torch.no_grad()
def test_blob_detector_matches_scipy_label() -> None:
    bright = _scene()
    cube = _cube_from_brightness(bright)
    node = BlobDetector(
        brightness="band_mean",
        threshold_method="fixed",
        threshold=0.5,
        opening_kernel=0,
        closing_kernel=0,
        min_area=4,
        connectivity=8,
    )
    out = node.forward(cube=cube)

    fg = bright >= 0.5
    labeled, _ = ndimage.label(fg, structure=np.ones((3, 3), dtype=int))
    # scipy components surviving the same area filter
    ref = {c for c in _components(labeled) if len(c) >= 4}

    got = _components(out["mask"][0].numpy())
    assert out["mask"].dtype == torch.int32
    assert got == ref
    assert int(out["count"].item()) == 3  # speck dropped by min_area


@torch.no_grad()
def test_blob_detector_keep_largest_pins_count() -> None:
    cube = _cube_from_brightness(_scene())
    node = BlobDetector(
        threshold_method="fixed", threshold=0.5, opening_kernel=0, closing_kernel=0, keep_largest=2
    )
    out = node.forward(cube=cube)
    assert int(out["count"].item()) == 2
    # the largest component (12 px) must survive
    sizes = sorted(int((out["mask"][0] == k).sum()) for k in range(1, 3))
    assert max(sizes) == 12


@torch.no_grad()
def test_blob_detector_bbox_and_centroid() -> None:
    bright = np.zeros((8, 8), dtype=np.float32)
    bright[2:5, 3:6] = 1.0  # single square, rows 2..4, cols 3..5
    node = BlobDetector(
        threshold_method="fixed", threshold=0.5, opening_kernel=0, closing_kernel=0, min_area=1
    )
    out = node.forward(cube=_cube_from_brightness(bright))
    assert int(out["count"].item()) == 1
    # xyxy: x0,y0 inclusive, x1,y1 exclusive (max + 1)
    assert out["bboxes"][0, 0].tolist() == [3.0, 2.0, 6.0, 5.0]
    assert out["centroids"][0, 0].tolist() == [4.0, 3.0]


@torch.no_grad()
def test_blob_detector_empty_foreground() -> None:
    cube = _cube_from_brightness(np.zeros((6, 6), dtype=np.float32))
    out = BlobDetector(threshold_method="fixed", threshold=0.9).forward(cube=cube)
    assert int(out["count"].item()) == 0
    assert out["mask"].shape == (1, 6, 6)
    assert out["bboxes"].shape == (1, 0, 4)


def test_blob_detector_rejects_bad_params() -> None:
    with pytest.raises(ValueError):
        BlobDetector(brightness="luminance")
    with pytest.raises(ValueError):
        BlobDetector(threshold_method="kmeans")
    with pytest.raises(ValueError):
        BlobDetector(connectivity=6)
