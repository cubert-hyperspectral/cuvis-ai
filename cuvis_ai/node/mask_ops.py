"""Mask cleanup and mask-to-bbox tracking nodes.

Provides two nodes used by the SPAM invisible-ink pipeline:

- ``MaskRobustifier``: morphological open/close + largest-component filter to
  suppress false-positive speckle in per-frame binary masks.

- ``MaskToBBoxKalman``: derive a bounding box from a robust mask and smooth /
  predict it across frames with a constant-velocity Kalman filter
  (``cv2.KalmanFilter``), so brief empty-mask frames do not make downstream
  zoom insets jitter or disappear.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


def _dilate(binary: torch.Tensor, kernel: int) -> torch.Tensor:
    """Binary dilation via max-pool. Input/output [B, H, W] bool."""
    x = binary.to(torch.float32).unsqueeze(1)
    x = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x.squeeze(1) > 0


def _erode(binary: torch.Tensor, kernel: int) -> torch.Tensor:
    """Binary erosion via min (= -max on negated). Input/output [B, H, W] bool."""
    x = (~binary).to(torch.float32).unsqueeze(1)
    x = F.max_pool2d(x, kernel_size=kernel, stride=1, padding=kernel // 2)
    return x.squeeze(1) == 0


class MaskRobustifier(Node):
    """Clean a binary/labelled mask with morphology + largest-component filter.

    Applies morphological opening (remove speckle), then closing (fill small
    holes), optionally drops connected components below ``min_area`` pixels,
    and optionally keeps only the single largest component.

    Output is an int32 mask with the same spatial shape as the input; non-zero
    values are preserved where the original mask was non-zero and survives the
    cleanup.

    Parameters
    ----------
    opening_kernel : int
        Side length of the square structuring element used for ``cv2.MORPH_OPEN``.
        ``0`` or ``1`` disables opening.  Default ``0`` (disabled); opening is
        aggressive enough to erase narrow real detections, so the default is
        off and callers enable it explicitly when needed.
    closing_kernel : int
        Side length for ``cv2.MORPH_CLOSE``.  ``0``/``1`` disables closing.
        Default ``3``.
    min_area : int
        Drop connected components with fewer than this many pixels.  ``0``
        disables the filter.  Default ``10`` (kills singleton/doubleton
        speckle while preserving small compact detections).
    keep_largest : bool
        If True, keep only the single largest surviving component.  Default
        ``True``.
    """

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Input mask [B, H, W]; >0 is foreground.",
        ),
    }

    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Cleaned mask [B, H, W]; preserves input label values.",
        ),
    }

    def __init__(
        self,
        opening_kernel: int = 0,
        closing_kernel: int = 3,
        min_area: int = 10,
        keep_largest: bool = True,
        **kwargs: Any,
    ) -> None:
        if opening_kernel < 0:
            raise ValueError("opening_kernel must be >= 0")
        if closing_kernel < 0:
            raise ValueError("closing_kernel must be >= 0")
        if min_area < 0:
            raise ValueError("min_area must be >= 0")

        self.opening_kernel = int(opening_kernel)
        self.closing_kernel = int(closing_kernel)
        self.min_area = int(min_area)
        self.keep_largest = bool(keep_largest)

        super().__init__(
            opening_kernel=self.opening_kernel,
            closing_kernel=self.closing_kernel,
            min_area=self.min_area,
            keep_largest=self.keep_largest,
            **kwargs,
        )

    @torch.no_grad()
    def forward(self, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        # Pure-torch morphology on the device the mask arrives on (GPU-friendly).
        binary = mask > 0  # [B, H, W] bool
        if self.opening_kernel >= 2:
            k = self.opening_kernel | 1  # force odd kernel for symmetric padding
            binary = _dilate(_erode(binary, k), k)
        if self.closing_kernel >= 2:
            k = self.closing_kernel | 1
            binary = _erode(_dilate(binary, k), k)

        if self.min_area <= 0 and not self.keep_largest:
            return {"mask": (mask * binary.to(mask.dtype))}

        # Connected-components requires a CPU round-trip; no native torch CCL.
        binary_np = binary.to(torch.uint8).cpu().numpy()
        surviving = np.zeros_like(binary_np, dtype=bool)
        for i in range(binary_np.shape[0]):
            frame = binary_np[i]
            if not frame.any():
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(frame, connectivity=8)
            if num_labels <= 1:
                continue
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep_ids = np.arange(1, num_labels, dtype=np.int32)
            if self.min_area > 0:
                m = areas >= self.min_area
                keep_ids = keep_ids[m]
                areas = areas[m]
            if keep_ids.size == 0:
                continue
            if self.keep_largest:
                keep_ids = keep_ids[int(np.argmax(areas)) : int(np.argmax(areas)) + 1]
            surviving[i] = np.isin(labels, keep_ids)

        surviving_t = torch.from_numpy(surviving).to(device=mask.device)
        return {"mask": (mask * surviving_t.to(mask.dtype))}


class MaskToBBoxKalman(Node):
    """Mask -> bounding box with constant-velocity Kalman smoothing.

    Each frame the bbox tight to the non-zero extent of the mask (with padding)
    is used as a measurement to update an 8-state Kalman filter (cx, cy, w, h,
    vx, vy, vw, vh).  When the mask is empty the filter is stepped in
    prediction-only mode, so the downstream ROI stays pinned to a plausible
    location for a few frames rather than vanishing.

    A warm-up of ``min_hits`` consecutive measurement frames is required
    before the track is confirmed; hits during the warm-up never leak to
    downstream consumers (``valid=0``), and a single missed frame during
    warm-up resets the hit counter.  This suppresses isolated false-positive
    detections that would otherwise briefly pop the inset into view.

    Output ``valid`` encodes track state per frame:

    * ``1`` - measurement used this frame on a confirmed track.
    * ``2`` - predicted only (mask empty on a confirmed track, within budget).
    * ``0`` - unconfirmed warm-up, no track, or post-drop.

    Parameters
    ----------
    padding_fraction : float
        Fractional padding applied to the measurement bbox before it is fed
        to the filter.  ``0.2`` adds 10% on each side.  Default ``0.2``.
    min_size_px : int
        Lower bound on the output bbox edge length (post-Kalman).  Small
        measurements are expanded around the centre.  Default ``96``.
    min_hits : int
        Number of consecutive measurement frames required to confirm a new
        track.  Missed frames during warm-up reset the hit counter back to
        zero, so transient false positives never graduate.  Default ``3``;
        ``1`` disables the warm-up.
    max_predict_frames : int
        After this many consecutive empty frames (on a confirmed track) the
        track is dropped and subsequent empty frames emit ``valid=0`` until
        a new measurement.  Default ``20``.
    process_noise : float
        Scalar multiplier for the Kalman process-noise covariance.
    measurement_noise : float
        Scalar multiplier for the Kalman measurement-noise covariance.
    """

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Robust mask [B, H, W]; >0 is foreground.",
        ),
    }

    OUTPUT_SPECS = {
        "bbox": PortSpec(
            dtype=torch.float32,
            shape=(-1, 4),
            description="Bounding box per frame [B, 4] in xyxy pixel coordinates.",
        ),
        "valid": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Track status per frame [B]: 0=none, 1=measured, 2=predicted.",
        ),
    }

    def __init__(
        self,
        padding_fraction: float = 0.2,
        min_size_px: int = 96,
        min_hits: int = 3,
        max_predict_frames: int = 20,
        process_noise: float = 1e-2,
        measurement_noise: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if padding_fraction < 0:
            raise ValueError("padding_fraction must be >= 0")
        if min_size_px < 1:
            raise ValueError("min_size_px must be >= 1")
        if min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        if max_predict_frames < 0:
            raise ValueError("max_predict_frames must be >= 0")

        self.padding_fraction = float(padding_fraction)
        self.min_size_px = int(min_size_px)
        self.min_hits = int(min_hits)
        self.max_predict_frames = int(max_predict_frames)
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)

        super().__init__(
            padding_fraction=self.padding_fraction,
            min_size_px=self.min_size_px,
            min_hits=self.min_hits,
            max_predict_frames=self.max_predict_frames,
            process_noise=self.process_noise,
            measurement_noise=self.measurement_noise,
            **kwargs,
        )

        self._kf: cv2.KalmanFilter | None = None
        self._missed = 0
        self._hits = 0
        self._has_track = False
        self._confirmed = False

    def _new_filter(self) -> cv2.KalmanFilter:
        """Build a fresh 8-state constant-velocity Kalman filter."""
        kf = cv2.KalmanFilter(8, 4)
        # Measurement: [cx, cy, w, h]
        kf.measurementMatrix = np.eye(4, 8, dtype=np.float32)
        # Transition: constant velocity, dt=1 frame
        transition = np.eye(8, dtype=np.float32)
        for i in range(4):
            transition[i, i + 4] = 1.0
        kf.transitionMatrix = transition
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * self.process_noise
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * self.measurement_noise
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        return kf

    @staticmethod
    def _bbox_from_mask_torch(
        mask_2d: torch.Tensor,
    ) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, x1, y1) in pixels, or None if mask is empty.

        Pure-torch: projects the 2-D mask onto row and column axes and reads
        the nonzero extent via ``torch.where`` — no numpy round-trip.
        """
        fg = mask_2d > 0
        if not torch.any(fg):
            return None
        rows = torch.any(fg, dim=1).nonzero(as_tuple=False).flatten()
        cols = torch.any(fg, dim=0).nonzero(as_tuple=False).flatten()
        return (
            int(cols[0].item()),
            int(rows[0].item()),
            int(cols[-1].item()) + 1,
            int(rows[-1].item()) + 1,
        )

    def _apply_padding(
        self, bbox: tuple[int, int, int, int], h: int, w: int
    ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bbox
        bw = x1 - x0
        bh = y1 - y0
        dx = int(round(bw * self.padding_fraction * 0.5))
        dy = int(round(bh * self.padding_fraction * 0.5))
        return (
            max(0, x0 - dx),
            max(0, y0 - dy),
            min(w, x1 + dx),
            min(h, y1 + dy),
        )

    def _enforce_min_size(
        self, cx: float, cy: float, bw: float, bh: float, h: int, w: int
    ) -> tuple[float, float, float, float]:
        bw = max(bw, float(self.min_size_px))
        bh = max(bh, float(self.min_size_px))
        bw = min(bw, float(w))
        bh = min(bh, float(h))
        cx = min(max(cx, bw / 2.0), w - bw / 2.0)
        cy = min(max(cy, bh / 2.0), h - bh / 2.0)
        return cx, cy, bw, bh

    def _clamp_xyxy(
        self, cx: float, cy: float, bw: float, bh: float, h: int, w: int
    ) -> tuple[float, float, float, float]:
        cx, cy, bw, bh = self._enforce_min_size(cx, cy, bw, bh, h, w)
        x0 = max(0.0, cx - bw / 2.0)
        y0 = max(0.0, cy - bh / 2.0)
        x1 = min(float(w), cx + bw / 2.0)
        y1 = min(float(h), cy + bh / 2.0)
        return x0, y0, x1, y1

    @torch.no_grad()
    def forward(self, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        device = mask.device
        b, h, w = mask.shape
        bboxes = np.zeros((b, 4), dtype=np.float32)
        valids = np.zeros((b,), dtype=np.int32)

        for i in range(b):
            raw = self._bbox_from_mask_torch(mask[i])

            if raw is not None:
                x0, y0, x1, y1 = self._apply_padding(raw, h=h, w=w)
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5
                bw = float(x1 - x0)
                bh = float(y1 - y0)
                meas = np.array([[cx], [cy], [bw], [bh]], dtype=np.float32)

                if not self._has_track:
                    # First hit of a new (still-unconfirmed) track.
                    self._kf = self._new_filter()
                    self._kf.statePost = np.array(
                        [cx, cy, bw, bh, 0, 0, 0, 0], dtype=np.float32
                    ).reshape(8, 1)
                    self._has_track = True
                    self._missed = 0
                    self._hits = 1
                    self._confirmed = self._hits >= self.min_hits
                    state = self._kf.statePost
                else:
                    assert self._kf is not None
                    self._kf.predict()
                    state = self._kf.correct(meas)
                    self._missed = 0
                    self._hits += 1
                    if not self._confirmed and self._hits >= self.min_hits:
                        self._confirmed = True

                if self._confirmed:
                    cx_k, cy_k, bw_k, bh_k = (float(state[j, 0]) for j in range(4))
                    x0, y0, x1, y1 = self._clamp_xyxy(cx_k, cy_k, bw_k, bh_k, h=h, w=w)
                    bboxes[i] = (x0, y0, x1, y1)
                    valids[i] = 1
                # else: warm-up hit, suppress downstream (valid=0, zero bbox).
                continue

            # No measurement this frame.
            if not self._confirmed:
                # Warm-up broken by a missed frame: drop everything and restart.
                self._has_track = False
                self._kf = None
                self._missed = 0
                self._hits = 0
                self._confirmed = False
                continue

            if self._has_track and self._missed < self.max_predict_frames:
                assert self._kf is not None
                state = self._kf.predict()
                self._missed += 1
                cx_k, cy_k, bw_k, bh_k = (float(state[j, 0]) for j in range(4))
                x0, y0, x1, y1 = self._clamp_xyxy(cx_k, cy_k, bw_k, bh_k, h=h, w=w)
                bboxes[i] = (x0, y0, x1, y1)
                valids[i] = 2
            else:
                # Drop the track after a long miss streak.
                self._has_track = False
                self._kf = None
                self._missed = 0
                self._hits = 0
                self._confirmed = False

        return {
            "bbox": torch.from_numpy(bboxes).to(device=device, dtype=torch.float32),
            "valid": torch.from_numpy(valids).to(device=device, dtype=torch.int32),
        }
