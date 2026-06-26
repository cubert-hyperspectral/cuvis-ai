"""Blob detection: localize bright objects on a dark background.

Reduces a hyperspectral cube to a 2-D brightness image, thresholds it (Otsu,
quantile, or fixed), cleans it with morphological opening/closing, labels the
connected components, drops components outside an area range, optionally pins
the blob count to the largest ``keep_largest`` components, and emits a dense
label map plus per-blob bounding boxes, centroids, and a blob count.

Connected components are produced by the shared
:func:`cuvis_ai.utils.connected_components.label_connected_components` helper
(OpenCV), the same CPU round-trip ``MaskRobustifier`` and ``ShapeMorphology``
use; no separate hand-rolled labeling is introduced here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.utils.connected_components import label_connected_components
from cuvis_ai_core.node import Node


def _dilate2d(binary: torch.Tensor, kernel: int) -> torch.Tensor:
    """Binary dilation of a 2-D ``[H, W]`` bool image via max-pool."""
    if kernel < 2:
        return binary
    k = kernel | 1
    x = binary.to(torch.float32)[None, None]
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x[0, 0] > 0


def _erode2d(binary: torch.Tensor, kernel: int) -> torch.Tensor:
    """Binary erosion of a 2-D ``[H, W]`` bool image (max-pool on the complement)."""
    if kernel < 2:
        return binary
    k = kernel | 1
    x = (~binary).to(torch.float32)[None, None]
    x = F.max_pool2d(x, kernel_size=k, stride=1, padding=k // 2)
    return x[0, 0] == 0


class BlobDetector(Node):
    """Localize bright blobs (e.g. pills, granules, tray compartments) in a cube.

    The cube's first frame is reduced to a 2-D brightness image, thresholded
    into a foreground mask, morphologically cleaned, and labeled into connected
    components. Components are filtered by area and, optionally, capped to the
    ``keep_largest`` biggest so a fixed-layout scene yields a stable blob count.

    Parameters
    ----------
    brightness : str, optional
        How to reduce the cube ``[H, W, C]`` to a 2-D image: ``"band_mean"``
        (mean over channels, default), ``"max"`` (per-pixel channel max), or
        ``"index"`` (normalized difference ``(a - b) / (a + b)`` of the two
        bands nearest ``index_wavelengths``; falls back to the band mean when
        wavelengths are unavailable).
    threshold_method : str, optional
        ``"otsu"`` (between-class variance, default), ``"quantile"`` (keep
        pixels above the ``threshold`` quantile), or ``"fixed"`` (keep pixels
        with min-max-scaled brightness ``>= threshold``).
    threshold : float, optional
        Quantile in ``[0, 1]`` for ``"quantile"`` or scaled-brightness cutoff in
        ``[0, 1]`` for ``"fixed"``; ignored for ``"otsu"``. Default ``0.5``.
    index_wavelengths : tuple[float, float] or None, optional
        Two wavelengths (nm) for ``brightness="index"``.
    opening_kernel : int, optional
        Square structuring-element side for morphological opening (speckle
        removal); ``0`` or ``1`` disables. Default ``3``.
    closing_kernel : int, optional
        Square side for morphological closing (hole fill); ``0`` or ``1``
        disables. Default ``3``.
    min_area : int, optional
        Drop connected components with fewer than this many pixels. Default ``5``.
    max_area : int or None, optional
        Drop components larger than this many pixels (``None`` disables).
    keep_largest : int or None, optional
        After area filtering, keep only the ``keep_largest`` biggest components
        (``None`` disables). Stabilizes the blob count across re-scans: a fixed
        tray of ``N`` compartments yields exactly ``N`` groups even when a scan
        sprouts a spurious bright fragment. No effect if fewer than
        ``keep_largest`` components survive the area filter.
    connectivity : int, optional
        ``8`` (default) or ``4`` neighborhood for connected components.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.MASK, NodeTag.SEGMENTATION})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelengths [C] in nanometers (used by brightness='index').",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Dense blob label map [B, H, W]; ids 1..N, 0 = background.",
        ),
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, 4),
            description="Per-blob bounding boxes [B, N, 4] in xyxy pixel coords.",
        ),
        "centroids": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, 2),
            description="Per-blob centroids [B, N, 2] as (x, y).",
        ),
        "count": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Number of blobs per frame [B].",
        ),
    }

    def __init__(
        self,
        brightness: str = "band_mean",
        threshold_method: str = "otsu",
        threshold: float = 0.5,
        index_wavelengths: tuple[float, float] | None = None,
        opening_kernel: int = 3,
        closing_kernel: int = 3,
        min_area: int = 5,
        max_area: int | None = None,
        keep_largest: int | None = None,
        connectivity: int = 8,
        **kwargs: Any,
    ) -> None:
        """Validate and store the detection hyperparameters."""
        if brightness not in ("band_mean", "max", "index"):
            raise ValueError("brightness must be 'band_mean', 'max', or 'index'.")
        if threshold_method not in ("otsu", "quantile", "fixed"):
            raise ValueError("threshold_method must be 'otsu', 'quantile', or 'fixed'.")
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1].")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8.")
        if min_area < 1:
            raise ValueError("min_area must be >= 1.")
        if max_area is not None and max_area < min_area:
            raise ValueError("max_area must be >= min_area.")
        if keep_largest is not None and int(keep_largest) < 1:
            raise ValueError("keep_largest must be >= 1 or None.")

        self.brightness = str(brightness)
        self.threshold_method = str(threshold_method)
        self.threshold = float(threshold)
        self.index_wavelengths = (
            None if index_wavelengths is None else tuple(float(w) for w in index_wavelengths)
        )
        self.opening_kernel = int(opening_kernel)
        self.closing_kernel = int(closing_kernel)
        self.min_area = int(min_area)
        self.max_area = None if max_area is None else int(max_area)
        self.keep_largest = None if keep_largest is None else int(keep_largest)
        self.connectivity = int(connectivity)

        super().__init__(
            brightness=self.brightness,
            threshold_method=self.threshold_method,
            threshold=self.threshold,
            index_wavelengths=self.index_wavelengths,
            opening_kernel=self.opening_kernel,
            closing_kernel=self.closing_kernel,
            min_area=self.min_area,
            max_area=self.max_area,
            keep_largest=self.keep_largest,
            connectivity=self.connectivity,
            **kwargs,
        )

    def _brightness_image(
        self, cube0: torch.Tensor, wavelengths: np.ndarray | torch.Tensor | None
    ) -> torch.Tensor:
        """Reduce a cube ``[H, W, C]`` to a min-max scaled brightness image ``[H, W]``."""
        if self.brightness == "max":
            bright = cube0.amax(dim=-1)
        elif (
            self.brightness == "index"
            and self.index_wavelengths is not None
            and wavelengths is not None
        ):
            wl = np.asarray(wavelengths).astype(np.float32).ravel()
            ia = int(np.argmin(np.abs(wl - self.index_wavelengths[0])))
            ib = int(np.argmin(np.abs(wl - self.index_wavelengths[1])))
            band_a = cube0[..., ia]
            band_b = cube0[..., ib]
            bright = (band_a - band_b) / (band_a + band_b + 1e-6)
        else:
            bright = cube0.mean(dim=-1)

        bright = bright.to(torch.float32)
        bmin = bright.min()
        bmax = bright.max()
        return (bright - bmin) / (bmax - bmin + 1e-9)

    @staticmethod
    def _otsu_threshold(bright: torch.Tensor) -> float:
        """Otsu between-class-variance threshold on a ``[0, 1]`` image."""
        hist = torch.histc(bright, bins=256, min=0.0, max=1.0)
        total = hist.sum()
        if total <= 0:
            return 0.5
        centers = (torch.arange(256, device=bright.device, dtype=torch.float32) + 0.5) / 256.0
        w0 = torch.cumsum(hist, dim=0)
        w1 = total - w0
        cum_mean = torch.cumsum(hist * centers, dim=0)
        global_mean = cum_mean[-1]
        eps = 1e-9
        mu0 = cum_mean / (w0 + eps)
        mu1 = (global_mean - cum_mean) / (w1 + eps)
        between = w0 * w1 * (mu0 - mu1) ** 2
        idx = int(torch.argmax(between))
        return float(centers[idx])

    def _foreground(
        self, cube0: torch.Tensor, wavelengths: np.ndarray | torch.Tensor | None
    ) -> torch.Tensor:
        """Threshold and morphologically clean the brightness image into a bool mask ``[H, W]``."""
        bright = self._brightness_image(cube0, wavelengths)
        if self.threshold_method == "otsu":
            thr = self._otsu_threshold(bright)
        elif self.threshold_method == "quantile":
            thr = float(torch.quantile(bright.flatten(), self.threshold))
        else:
            thr = self.threshold
        fg = bright >= thr
        fg = _dilate2d(_erode2d(fg, self.opening_kernel), self.opening_kernel)
        fg = _erode2d(_dilate2d(fg, self.closing_kernel), self.closing_kernel)
        return fg

    def _finalize(
        self, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """Area-filter, optionally count-pin, densely relabel, and measure bbox + centroid."""
        height, width = labels.shape
        device = labels.device
        rows = torch.arange(height, device=device).view(height, 1).expand(height, width)
        cols = torch.arange(width, device=device).view(1, width).expand(height, width)

        out = torch.zeros((height, width), dtype=torch.int32, device=device)
        boxes: list[list[float]] = []
        centers: list[list[float]] = []

        uniq = torch.unique(labels)
        uniq = uniq[uniq != 0]

        # candidates passing the area filter, in label order (preserves numbering)
        cands: list[tuple[int, int]] = []
        for seed in uniq.tolist():
            area = int((labels == seed).sum())
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            cands.append((seed, area))

        if self.keep_largest is not None and len(cands) > self.keep_largest:
            keep = {
                s for s, _ in sorted(cands, key=lambda t: t[1], reverse=True)[: self.keep_largest]
            }
            cands = [(s, a) for (s, a) in cands if s in keep]

        new_id = 0
        for seed, _area in cands:
            comp = labels == seed
            new_id += 1
            out[comp] = new_id
            ys = rows[comp].to(torch.float32)
            xs = cols[comp].to(torch.float32)
            boxes.append(
                [float(xs.min()), float(ys.min()), float(xs.max()) + 1.0, float(ys.max()) + 1.0]
            )
            centers.append([float(xs.mean()), float(ys.mean())])

        if new_id == 0:
            boxes_t = torch.zeros((1, 0, 4), dtype=torch.float32, device=device)
            centers_t = torch.zeros((1, 0, 2), dtype=torch.float32, device=device)
        else:
            boxes_t = torch.tensor(boxes, dtype=torch.float32, device=device).unsqueeze(0)
            centers_t = torch.tensor(centers, dtype=torch.float32, device=device).unsqueeze(0)
        return out[None], boxes_t, centers_t, new_id

    @torch.no_grad()
    def forward(
        self,
        cube: torch.Tensor,
        wavelengths: np.ndarray | torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Detect blobs in the first frame of ``cube``.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube ``[B, H, W, C]``; only ``cube[0]`` is processed.
        wavelengths : numpy.ndarray or torch.Tensor or None, optional
            Wavelengths ``[C]`` in nanometers, used only by ``brightness="index"``.
        **_ : Any
            Additional unused keyword arguments (e.g. the pipeline ``context``).

        Returns
        -------
        dict[str, torch.Tensor]
            ``mask`` int32 ``[1, H, W]`` (blob ids 1..N, 0 background),
            ``bboxes`` float32 ``[1, N, 4]`` (xyxy), ``centroids`` float32
            ``[1, N, 2]`` (x, y), and ``count`` int32 ``[1]``.
        """
        cube0 = cube[0]
        fg = self._foreground(cube0, wavelengths)
        if not bool(fg.any()):
            height, width = cube0.shape[0], cube0.shape[1]
            return {
                "mask": torch.zeros((1, height, width), dtype=torch.int32, device=cube.device),
                "bboxes": torch.zeros((1, 0, 4), dtype=torch.float32, device=cube.device),
                "centroids": torch.zeros((1, 0, 2), dtype=torch.float32, device=cube.device),
                "count": torch.zeros((1,), dtype=torch.int32, device=cube.device),
            }

        labels = label_connected_components(fg, connectivity=self.connectivity).to(torch.int64)
        mask, bboxes, centroids, count = self._finalize(labels)
        return {
            "mask": mask.to(device=cube.device, dtype=torch.int32),
            "bboxes": bboxes.to(cube.device),
            "centroids": centroids.to(cube.device),
            "count": torch.tensor([count], dtype=torch.int32, device=cube.device),
        }


__all__ = ["BlobDetector"]
