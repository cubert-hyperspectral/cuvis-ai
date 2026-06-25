"""Shared connected-components labeling helper.

cuvis.ai has no native torch connected-component labeling (CCL) op, so nodes
that need instance labels round-trip a single 2-D frame through OpenCV's
``cv2.connectedComponentsWithStats``.  Centralizing that here keeps the
CPU round-trip in one place and lets ``MaskRobustifier`` and ``ShapeMorphology``
share an identical labeling path.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch


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
        Pixel connectivity for ``cv2.connectedComponentsWithStats``; either
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
