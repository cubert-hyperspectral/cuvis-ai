"""Shared connected-components labeling helper.

cuvis.ai has no native torch connected-component labeling (CCL) op, so nodes
that need instance labels round-trip a single 2-D frame through OpenCV's
``cv2.connectedComponents``.  Centralizing that here keeps the
CPU round-trip in one place and lets ``MaskRobustifier`` and ``ShapeMorphology``
share an identical labeling path.

The cell-grid helpers below label a coarse grid instead of every pixel: the pixel counts (or
score maxima) per ``cell x cell`` cell are computed on the device, only the small grid goes
through OpenCV, and the decision per blob comes back as a cell mask. Areas and overlap counts
stay exact pixel counts; the only difference to labelling every pixel is that marks whose cells
touch form one blob (gaps of up to ``2 x cell - 1`` pixels). ``cell=1`` labels every pixel.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


def label_connected_components(
    mask_2d: np.ndarray | torch.Tensor,
    connectivity: int = 8,
) -> torch.Tensor:
    """Label connected components of a 2-D binary mask via OpenCV.

    Any nonzero pixel is treated as foreground.  Returns an int32 label map
    where ``0`` is background and components are numbered ``1..N``.

    Parameters
    ----------
    mask_2d : numpy.ndarray or torch.Tensor
        2-D mask ``[H, W]``; any nonzero value is foreground.
    connectivity : int
        Pixel connectivity for ``cv2.connectedComponents``; either
        ``4`` or ``8``.  Default ``8``.

    Returns
    -------
    torch.Tensor
        Int32 label map ``[H, W]`` on the same device as ``mask_2d`` (CPU for a
        numpy input), with ``0`` background and ``1..N`` instances.
    """
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")

    if isinstance(mask_2d, torch.Tensor):
        device = mask_2d.device
        binary_np = (mask_2d != 0).to(torch.uint8).cpu().numpy()
    else:
        device = None
        binary_np = (np.asarray(mask_2d) != 0).astype(np.uint8)

    if binary_np.ndim != 2:
        raise ValueError("mask_2d must be 2-D [H, W]")

    if not binary_np.any():
        labels_np = np.zeros_like(binary_np, dtype=np.int32)
    else:
        _, labels_np = cv2.connectedComponents(binary_np, connectivity=connectivity)
        labels_np = labels_np.astype(np.int32, copy=False)

    labels = torch.from_numpy(labels_np)
    if device is not None:
        labels = labels.to(device=device)
    return labels


__all__ = ["label_connected_components"]


def cell_sums(x: Tensor, cell: int) -> Tensor:
    """Per-cell sums of a [N, C, H, W] float tensor (the last partial cells included)."""
    if cell == 1:
        return x
    return F.avg_pool2d(x, kernel_size=cell, stride=cell, ceil_mode=True, divisor_override=1)


def label_cell_grid(occupied: np.ndarray) -> tuple[int, np.ndarray]:
    """8-connected labels of a 2-D bool grid; 16-bit labels (3x faster) whenever they cannot
    overflow: an h x w grid has at most ceil(h / 2) * ceil(w / 2) blobs."""
    h, w = occupied.shape
    ltype = cv2.CV_16U if ((h + 1) // 2) * ((w + 1) // 2) < 65535 else cv2.CV_32S
    return cv2.connectedComponents(occupied.astype(np.uint8), connectivity=8, ltype=ltype)


def keep_blobs(occupied: np.ndarray, tests: list[tuple[np.ndarray, float]]) -> np.ndarray:
    """Cells of the 8-connected blobs of ``occupied`` that pass every (weights, threshold) test:
    the blob's summed weights reach the threshold. Sums run over the occupied cells only."""
    n, lab = label_cell_grid(occupied)
    out = np.zeros(occupied.size, bool)
    if n <= 1:
        return out.reshape(occupied.shape)
    sel = occupied.ravel()
    ls = lab.ravel()[sel]  # the blob of each occupied cell (labels >= 1)
    keep = np.ones(n, bool)
    for weights, threshold in tests:
        keep &= np.bincount(ls, weights=weights.ravel()[sel], minlength=n) >= threshold
    out[sel] = keep[ls]
    return out.reshape(occupied.shape)


def cell_max(x: Tensor, cell: int) -> Tensor:
    """Per-cell maxima of a [N, C, H, W] float tensor (the last partial cells included)."""
    if cell == 1:
        return x
    return F.max_pool2d(x, kernel_size=cell, stride=cell, ceil_mode=True)


def segment_max(labels: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    """Per-label maxima of values (labels in [0, n)); -inf for labels without values. A sort and a
    reduceat: much faster than np.maximum.at."""
    out = np.full(n, -np.inf)
    if labels.size:
        order = np.argsort(labels, kind="stable")
        lab, val = labels[order], values[order]
        starts = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
        out[lab[starts]] = np.maximum.reduceat(val, starts)
    return out


def peak_keep(
    occupied: np.ndarray, ref: np.ndarray, peak: np.ndarray, ref_peak: np.ndarray, ratio: float
) -> np.ndarray:
    """Cells of the 8-connected blobs of ``occupied`` whose peak reaches ``ratio`` x the highest
    peak of the ``ref`` blobs they touch; a blob that touches no ``ref`` cell stays."""
    n, lab = label_cell_grid(occupied)
    out = np.zeros(occupied.size, bool)
    if n <= 1:
        return out.reshape(occupied.shape)
    m, rlab = label_cell_grid(ref)
    sel = occupied.ravel()
    ls = lab.ravel()[sel].astype(np.int64)
    rls = rlab.ravel()[sel].astype(np.int64)
    blob_peak = segment_max(ls, peak.ravel()[sel].astype(np.float64), n)
    rsel = ref.ravel()
    ref_blob_peak = segment_max(
        rlab.ravel()[rsel].astype(np.int64), ref_peak.ravel()[rsel].astype(np.float64), max(m, 1)
    )
    inside = rls > 0
    compare = segment_max(ls[inside], ref_blob_peak[rls[inside]], n)
    keep = np.isneginf(compare) | (blob_peak >= ratio * compare)
    out[sel] = keep[ls]
    return out.reshape(occupied.shape)


def expand_cells(keep: Tensor, x: Tensor, cell: int) -> Tensor:
    """x [N, H, W, C] (bool) where its cell of ``keep`` [N, h, w, C] is set."""
    if cell == 1:
        return x & keep
    n, h, w, c = x.shape
    if h % cell == 0 and w % cell == 0:  # a broadcast over the cells: one kernel, no copy of keep
        cells = x.reshape(n, h // cell, cell, w // cell, cell, c)
        return (cells & keep[:, :, None, :, None, :]).reshape(n, h, w, c)
    keep = keep.repeat_interleave(cell, dim=1).repeat_interleave(cell, dim=2)
    return x & keep[:, :h, :w]


def filter_blobs(x: Tensor, weights: Tensor | None, threshold: float, cell: int) -> Tensor:
    """The blobs of x [N, C, H, W] (bool) whose summed weights [N, C, H, W] (None: x's own pixel
    counts) reach the threshold, labelled on the cell grid; one copy to the host."""
    counts = cell_sums(x.to(torch.float32), cell)
    if weights is None:
        occ_np = w_np = counts.cpu().numpy()
    else:
        host = torch.stack([counts, cell_sums(weights, cell)]).cpu().numpy()
        occ_np, w_np = host[0], host[1]
    occ_np = occ_np > 0
    keep = np.zeros(occ_np.shape, bool)
    for i in range(occ_np.shape[0]):
        for c in range(occ_np.shape[1]):
            if occ_np[i, c].any():
                keep[i, c] = keep_blobs(occ_np[i, c], [(w_np[i, c], threshold)])
    k = torch.from_numpy(keep).to(device=x.device).permute(0, 2, 3, 1)
    return expand_cells(k, x.permute(0, 2, 3, 1), cell).permute(0, 3, 1, 2)
