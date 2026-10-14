"""Mask cleanup, mask-to-bbox tracking, and per-blob label-voting nodes.

- ``MaskRobustifier``: morphological open/close + largest-component filter to
  suppress false-positive speckle in per-frame binary masks.

- ``MaskToBBoxKalman``: derive a bounding box from a robust mask and smooth /
  predict it across frames with a constant-velocity Kalman filter
  (``cv2.KalmanFilter``), so brief empty-mask frames do not make downstream
  zoom insets jitter or disappear.

- ``MajorityVoteByBlob``: collapse a noisy per-pixel label map to one label per
  detected blob via majority vote, turning soft per-pixel classifications into a
  clean one-label-per-object map.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai.utils.connected_components import (
    cell_max,
    cell_sums,
    expand_cells,
    filter_blobs,
    keep_blobs,
    label_connected_components,
    peak_keep,
)
from cuvis_ai_core.node import Node


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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

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
        """Open, close and filter the mask; returns ``{"mask": ...}`` with the surviving pixels."""
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
        surviving = np.zeros(binary.shape, dtype=bool)
        for i in range(binary.shape[0]):
            frame = binary[i]
            if not bool(frame.any()):
                continue
            labels = label_connected_components(frame, connectivity=8).cpu().numpy()
            num_labels = int(labels.max())
            if num_labels < 1:
                continue
            areas = np.bincount(labels.reshape(-1), minlength=num_labels + 1)[1:]
            keep_ids = np.arange(1, num_labels + 1, dtype=np.int32)
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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.MASK,
            NodeTag.BBOX,
            NodeTag.TRACKING,
            NodeTag.POSTPROCESSING,
            NodeTag.STATEFUL,
            NodeTag.NUMPY,
        }
    )

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
        """Grow ``bbox`` by ``padding_fraction`` of its size, clamped to the frame."""
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
        """Raise the box to ``min_size_px``, cap it at the frame and keep its centre inside."""
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
        """Centre/size box to ``(x0, y0, x1, y1)`` inside the frame after the minimum-size rule."""
        cx, cy, bw, bh = self._enforce_min_size(cx, cy, bw, bh, h, w)
        x0 = max(0.0, cx - bw / 2.0)
        y0 = max(0.0, cy - bh / 2.0)
        x1 = min(float(w), cx + bw / 2.0)
        y1 = min(float(h), cy + bh / 2.0)
        return x0, y0, x1, y1

    @torch.no_grad()
    def forward(self, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Track one box per frame through the Kalman filter.

        Returns ``bbox`` ``[B, 4]`` (xyxy) and ``valid`` ``[B]`` (0=none, 1=measured, 2=predicted).
        """
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
            if self._confirmed and self._missed < self.max_predict_frames:
                assert self._kf is not None
                state = self._kf.predict()
                self._missed += 1
                cx_k, cy_k, bw_k, bh_k = (float(state[j, 0]) for j in range(4))
                x0, y0, x1, y1 = self._clamp_xyxy(cx_k, cy_k, bw_k, bh_k, h=h, w=w)
                bboxes[i] = (x0, y0, x1, y1)
                valids[i] = 2
            else:
                # Warm-up broken by a missed frame, or a confirmed track missed
                # too long: drop everything and restart.
                self._has_track = False
                self._kf = None
                self._missed = 0
                self._hits = 0
                self._confirmed = False

        return {
            "bbox": torch.from_numpy(bboxes).to(device=device, dtype=torch.float32),
            "valid": torch.from_numpy(valids).to(device=device, dtype=torch.int32),
        }


class MajorityVoteByBlob(Node):
    """Assign each blob the majority per-pixel label found inside it.

    Per-pixel classifiers (e.g. a Spectral Angle Mapper) produce noisy labels
    when reference spectra are close together. Voting within each detected blob
    denoises that into a single robust label per object: for every blob id in
    ``blob_mask`` (1..N), the most frequent nonzero ``identity_mask`` value over
    that blob's pixels becomes the blob's label; blobs with no labelled pixels
    stay ``0``, as does the background.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.CLASSIFICATION})

    INPUT_SPECS = {
        "identity_mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Per-pixel labels [1, H, W]; 0 = unassigned.",
        ),
        "blob_mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Blob label map [1, H, W]; ids 1..N, 0 = background.",
        ),
    }

    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Per-blob majority-label map [1, H, W]; 0 = background.",
        ),
    }

    @torch.no_grad()
    def forward(
        self, identity_mask: torch.Tensor, blob_mask: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Paint each blob with the majority label of its pixels.

        Parameters
        ----------
        identity_mask : torch.Tensor
            Per-pixel labels ``[1, H, W]`` (int32); ``0`` is unassigned.
        blob_mask : torch.Tensor
            Blob label map ``[1, H, W]`` (int32); ids ``1..N``, ``0`` background.
        **_ : Any
            Additional unused keyword arguments (e.g. the pipeline ``context``).

        Returns
        -------
        dict[str, torch.Tensor]
            ``mask`` int32 ``[1, H, W]`` with each blob painted its majority
            label and background left at ``0``.
        """
        ident = identity_mask[0].to(torch.int64)
        blobs = blob_mask[0].to(torch.int64)
        out = torch.zeros_like(blobs, dtype=torch.int32)

        for blob_id in torch.unique(blobs).tolist():
            if blob_id == 0:
                continue
            region = blobs == blob_id
            votes = ident[region]
            votes = votes[votes > 0]
            if votes.numel() == 0:
                continue
            majority = int(torch.bincount(votes).argmax())
            out[region] = majority
        return {"mask": out.unsqueeze(0)}


class ClassMapRobustifier(Node):
    """Per-class morphological cleanup of an integer label map.

    Runs :class:`MaskRobustifier` independently on the binary mask of each class
    present in ``class_map`` (despeckle + close + min-area / largest-component
    filter), then repaints the survivors into one label map. Classes are painted
    in ascending surviving-area order, so a larger class wins any pixel an
    overlapping smaller class also kept. Pixels removed as speckle become
    ``background_value`` (holes); :class:`NearestLabelFill` is the companion node
    that fills them. The input map is echoed verbatim on the ``source`` port so the
    fill node has both the foreground extent and the fallback labels.

    Parameters
    ----------
    opening_kernel : int
        Morphological opening kernel for the internal ``MaskRobustifier``. ``0``/``1``
        disables opening. Default ``0``.
    closing_kernel : int
        Morphological closing kernel. ``0``/``1`` disables. Default ``3``.
    min_area : int
        Drop per-class connected components smaller than this. ``0`` disables.
        Default ``10``.
    keep_largest : bool
        Keep only the largest surviving component of each class. Default ``True``.
    background_value : int
        Label value for unassigned pixels in the output. Default ``-1``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.MASK,
            NodeTag.SEGMENTATION,
            NodeTag.CLASSIFICATION,
            NodeTag.POSTPROCESSING,
            NodeTag.NUMPY,
        }
    )

    INPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Integer label map [B, H, W]; background_value elsewhere.",
        ),
    }

    OUTPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Cleaned label map [B, H, W]; removed speckle -> background_value.",
        ),
        "source": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Verbatim passthrough of the input label map [B, H, W].",
        ),
    }

    def __init__(
        self,
        opening_kernel: int = 0,
        closing_kernel: int = 3,
        min_area: int = 10,
        keep_largest: bool = True,
        background_value: int = -1,
        **kwargs: Any,
    ) -> None:
        self.opening_kernel = int(opening_kernel)
        self.closing_kernel = int(closing_kernel)
        self.min_area = int(min_area)
        self.keep_largest = bool(keep_largest)
        self.background_value = int(background_value)

        super().__init__(
            opening_kernel=self.opening_kernel,
            closing_kernel=self.closing_kernel,
            min_area=self.min_area,
            keep_largest=self.keep_largest,
            background_value=self.background_value,
            **kwargs,
        )

        # Assigned after super().__init__ so nn.Module is initialised before this
        # submodule is registered. MaskRobustifier validates its own kwargs (so a
        # negative kernel/area raises here at construction).
        self._robust = MaskRobustifier(
            opening_kernel=self.opening_kernel,
            closing_kernel=self.closing_kernel,
            min_area=self.min_area,
            keep_largest=self.keep_largest,
        )

    @torch.no_grad()
    def forward(self, class_map: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Clean each present class with morphology, then repaint area-sorted into one map."""
        bg = self.background_value
        out = torch.full_like(class_map, bg)
        for b in range(class_map.shape[0]):
            pred = class_map[b]
            present = [int(c) for c in torch.unique(pred).tolist() if int(c) != bg]
            surv: dict[int, torch.Tensor] = {}
            for c in present:
                binary = (pred == c).to(torch.int32).unsqueeze(0)
                surv[c] = self._robust.forward(mask=binary)["mask"][0] > 0
            # Larger classes painted last -> they win pixels a smaller class also kept.
            for c in sorted(present, key=lambda cc: int(surv[cc].sum())):
                out[b][surv[c]] = c
        return {"class_map": out, "source": class_map.clone()}


class NearestLabelFill(Node):
    """Fill morphology-removed gaps in a label map with the nearest surviving label.

    After per-class morphology (:class:`ClassMapRobustifier`) some pixels that were
    labelled in the original map are dropped to ``background_value``. This node
    repaints every such gap with the label of its nearest surviving pixel, found by
    iterative single-pixel dilation (8-connected / Chebyshev nearest; ties resolved
    toward the larger class id). Gaps no label can reach -- e.g. a class wiped out
    entirely by an area filter -- fall back to the original label on the ``source``
    port. The foreground to fill is ``source != background_value``.

    Parameters
    ----------
    background_value : int
        Label value treated as "unassigned" in both inputs and the output.
        Default ``-1``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING}
    )

    INPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Cleaned label map [B, H, W] with background_value gaps to fill.",
        ),
        "source": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Original label map [B, H, W]; defines foreground + fallback labels.",
        ),
    }

    OUTPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int64,
            shape=(-1, -1, -1),
            description="Gap-filled label map [B, H, W].",
        ),
    }

    def __init__(self, background_value: int = -1, **kwargs: Any) -> None:
        self.background_value = int(background_value)
        super().__init__(background_value=self.background_value, **kwargs)

    @torch.no_grad()
    def forward(
        self, class_map: torch.Tensor, source: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Grow surviving labels into the gaps; fall back to ``source`` where unreachable."""
        bg = self.background_value
        neg = -1.0e9  # any class id (>= 0) beats this in the max-pool, so unknown never wins
        out = class_map.clone()
        target = source != bg
        known = out != bg
        remaining = target & ~known
        while bool(remaining.any()):
            filled = out.to(torch.float32).masked_fill(~known, neg)
            cand = F.max_pool2d(filled.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            known_f = known.to(torch.float32).unsqueeze(1)
            reach = F.max_pool2d(known_f, kernel_size=3, stride=1, padding=1).squeeze(1) > 0
            newly = remaining & reach
            if not bool(newly.any()):
                break
            out[newly] = cand[newly].round().to(out.dtype)
            known = out != bg
            remaining = target & ~known
        still = target & (out == bg)
        out[still] = source[still]
        return {"class_map": out}


class LabelOffset(Node):
    """Add a constant offset to every label in an integer label map.

    Mainly used to lift a 0-based dense label map (e.g. a ``KMeansClusterer`` /
    ``GaussianMixtureClusterer`` ``class_mask``, where cluster ids run ``0..k-1``)
    to 1-based ids before :class:`MajorityVoteByBlob`, which treats ``0`` as
    background in both its vote and its output. Without the shift a cluster-``0``
    region would be dropped as background and collide with the unassigned label.

    Parameters
    ----------
    offset : int
        Value added to every label. Default ``1``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.CLASSIFICATION})

    INPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Integer label map [B, H, W].",
        ),
    }

    OUTPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Label map [B, H, W] with `offset` added to every label.",
        ),
    }

    def __init__(self, offset: int = 1, **kwargs: Any) -> None:
        self.offset = int(offset)
        super().__init__(offset=self.offset, **kwargs)

    @torch.no_grad()
    def forward(self, class_map: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Return the label map with `offset` added to every element."""
        return {"class_map": class_map.to(torch.int32) + self.offset}


class MaskMinArea(Node):
    """Drop the 8-connected blobs of a boolean mask [B, H, W, C] below ``min_area`` pixels."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C], e.g. a decider's decisions; each frame "
            "and channel is filtered on its own.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape: the blobs of `decisions` with at least `min_area` pixels "
            "(8-connected).",
        ),
    }

    def __init__(self, min_area: int = 250, cell: int = 4, **kwargs: Any) -> None:
        """Create a minimum-area mask filter.

        Parameters
        ----------
        min_area : smallest blob that is kept, in pixels (8-connected, default 250); ``0`` or ``1``
            keep every blob. Choose it below the smallest real object's blob (the object plus the
            score map's halo) and above the specks of the background.
        cell : side of the cells the blobs are labelled on (default 4; ``1`` labels every pixel).
        """
        if isinstance(min_area, bool) or not isinstance(min_area, int) or min_area < 0:
            raise ValueError(f"MaskMinArea: min_area must be an integer >= 0, got {min_area!r}.")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 1:
            raise ValueError(f"MaskMinArea: cell must be an integer >= 1, got {cell!r}.")
        self.min_area = int(min_area)
        self.cell = int(cell)
        super().__init__(min_area=self.min_area, cell=self.cell, **kwargs)

    def forward(self, decisions: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the mask without its blobs smaller than ``min_area`` pixels."""
        if self.min_area <= 1 or decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        x = decisions.permute(0, 3, 1, 2)
        out = filter_blobs(x, None, float(self.min_area), self.cell)
        return {"decisions": out.permute(0, 2, 3, 1)}


class MaskBlobGate(Node):
    """Keep the blobs of a boolean mask that hold enough pixels of a second (gating) mask."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] whose blobs are gated, e.g. an anomaly mask.",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Gating mask [B, H, W, C'] of the same batch, height and width (a pixel "
            "counts where any channel is set), e.g. a SpectralObjectMask: where objects are.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape as `decisions`: its 8-connected blobs that hold at least "
            "`min_px` pixels of `mask` (with `invert`: fewer than `min_px`).",
        ),
    }

    def __init__(
        self, min_px: int = 16, cell: int = 4, invert: bool = False, **kwargs: Any
    ) -> None:
        """Create a blob gate.

        Parameters
        ----------
        min_px : pixels of the gating mask a blob needs to stay (default 16; ``0`` keeps every
            blob). With a SpectralObjectMask as the gate, a blob on the empty background goes and a
            blob around an object (its halo included) stays.
        cell : side of the cells the blobs are labelled on (default 4; ``1`` labels every pixel).
        invert : keep the other blobs instead, those with fewer than ``min_px`` pixels of the
            gating mask (default False). With ``min_px=1``, a mask before a cut as ``decisions`` and
            the mask after it as ``mask``, these are the marks the cut removed entirely; fused with
            the cut mask (``DecisionFusion("any")``) the cut trims marks but never deletes one.
        """
        if isinstance(min_px, bool) or not isinstance(min_px, int) or min_px < 0:
            raise ValueError(f"MaskBlobGate: min_px must be an integer >= 0, got {min_px!r}.")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 1:
            raise ValueError(f"MaskBlobGate: cell must be an integer >= 1, got {cell!r}.")
        if not isinstance(invert, bool):
            raise ValueError(f"MaskBlobGate: invert must be a bool, got {invert!r}.")
        self.min_px = int(min_px)
        self.cell = int(cell)
        self.invert = invert
        super().__init__(min_px=self.min_px, cell=self.cell, invert=self.invert, **kwargs)

    def forward(self, decisions: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the blobs of ``decisions`` that overlap ``mask`` by at least ``min_px`` pixels
        (with ``invert``: by fewer)."""
        gate = mask.any(dim=-1)
        if gate.shape != decisions.shape[:3]:
            raise ValueError(
                f"MaskBlobGate: mask is [B, H, W] = {tuple(gate.shape)}, decisions "
                f"{tuple(decisions.shape[:3])}."
            )
        if decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        if self.min_px == 0:  # every blob passes
            return {"decisions": torch.zeros_like(decisions) if self.invert else decisions.clone()}
        x = decisions.permute(0, 3, 1, 2)
        w = (x & gate[:, None]).to(torch.float32)
        out = filter_blobs(x, w, float(self.min_px), self.cell)
        if self.invert:  # each set pixel's cell lies in one blob, kept or not
            out = x & ~out
        return {"decisions": out.permute(0, 2, 3, 1)}


class MaskPeakGate(Node):
    """Keep the blobs of a mask whose peak score reaches a share of their reference blob's peak."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] whose blobs are kept or dropped, e.g. the "
            "pieces of a mask after a cut.",
        ),
        "reference": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C'] of the same batch, height and width (a pixel "
            "counts where any channel is set) whose blobs set the peak to reach, e.g. the mask "
            "before the cut.",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Score map [B, H, W, C''] the peaks are read from (the first channel), "
            "e.g. the anomaly map the mask was thresholded from; finite and positive where marked.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape as `decisions`: the blobs whose highest score reaches `ratio` "
            "times the highest score of the reference blob they lie in; blobs outside every "
            "reference blob stay.",
        ),
    }

    def __init__(self, ratio: float = 0.8, cell: int = 4, **kwargs: Any) -> None:
        """Create a peak gate.

        Parameters
        ----------
        ratio : share of the reference blob's peak a blob must reach to stay (default 0.8; ``0``
            keeps every blob). After a cut of an anomaly mask to the objects, the piece holding the
            mark's peak stays and halo pieces on neighbouring objects, far below it, go.
        cell : side of the cells the blobs are labelled on (default 4; ``1`` labels every pixel).
        """
        if (
            isinstance(ratio, bool)
            or not isinstance(ratio, (int, float))
            or not 0.0 <= ratio <= 1.0
        ):
            raise ValueError(f"MaskPeakGate: ratio must be in [0, 1], got {ratio!r}.")
        if isinstance(cell, bool) or not isinstance(cell, int) or cell < 1:
            raise ValueError(f"MaskPeakGate: cell must be an integer >= 1, got {cell!r}.")
        self.ratio = float(ratio)
        self.cell = int(cell)
        super().__init__(ratio=self.ratio, cell=self.cell, **kwargs)

    def forward(
        self, decisions: Tensor, reference: Tensor, scores: Tensor, **_: Any
    ) -> dict[str, Tensor]:
        """Return the blobs of ``decisions`` whose peak reaches ``ratio`` x their reference peak."""
        ref = reference.any(dim=-1)
        if ref.shape != decisions.shape[:3] or tuple(scores.shape[:3]) != tuple(
            decisions.shape[:3]
        ):
            raise ValueError(
                f"MaskPeakGate: reference {tuple(ref.shape)} / scores {tuple(scores.shape[:3])} do "
                f"not match decisions [B, H, W] = {tuple(decisions.shape[:3])}."
            )
        if self.ratio == 0.0 or decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        x = decisions.permute(0, 3, 1, 2)
        s = scores[..., :1].permute(0, 3, 1, 2).to(torch.float32)
        r = ref[:, None]
        low = torch.full_like(s, float("-inf"))
        peak = cell_max(torch.where(x, s, low), self.cell)
        ref_peak = cell_max(torch.where(r, s, low), self.cell)
        host = torch.cat([peak, ref_peak], dim=1).cpu().numpy()  # one copy to the host
        keep = np.zeros(peak.shape, bool)
        for i in range(keep.shape[0]):
            rp = host[i, -1]
            for c in range(keep.shape[1]):
                occ = host[i, c] > -np.inf  # a cell holds a mark iff its maximum is finite
                if occ.any():
                    keep[i, c] = peak_keep(occ, rp > -np.inf, host[i, c], rp, self.ratio)
        kk = torch.from_numpy(keep).to(device=decisions.device).permute(0, 2, 3, 1)
        return {"decisions": expand_cells(kk, decisions, self.cell)}


class MaskBlobFilter(Node):
    """Keep the blobs of a boolean mask that are large enough and lie on non-background material."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.HYPERSPECTRAL, NodeTag.NUMPY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] whose blobs are filtered, e.g. a "
            "decider's decisions; each frame and channel on its own.",
        ),
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            optional=True,
            description="Hyperspectral cube [B, H, W, K] of the same frames, for the object test; "
            "without it only the size test runs.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Same shape: the blobs of `decisions` with at least `min_area` pixels that "
            "hold at least `min_object_px` pixels of non-background material.",
        ),
    }

    def __init__(
        self,
        min_area: int = 250,
        min_object_px: int = 16,
        min_angle_deg: float = 6.0,
        cell: int = 4,
        median_stride: int = 8,
        **kwargs: Any,
    ) -> None:
        """Create a blob filter: MaskMinArea, SpectralObjectMask and MaskBlobGate in one pass.

        Parameters
        ----------
        min_area : smallest blob that is kept, in pixels (default 250; 0 or 1 keep every size).
        min_object_px : pixels of the blob that must be objects (default 16; 0 skips the test).
        min_angle_deg : spectral angle to the frame's median spectrum above which a pixel is an
            object (default 6.0).
        cell : the blobs are labelled on cells of cell x cell pixels (default 4) and the object test
            reads each cell's first pixel (the stride-``cell`` grid of SpectralObjectMask).
        median_stride : grid step of the pixels whose per-band median is the background spectrum
            (default 8).
        """
        for name, val, lo in (
            ("min_area", min_area, 0),
            ("min_object_px", min_object_px, 0),
            ("cell", cell, 1),
            ("median_stride", median_stride, 1),
        ):
            if isinstance(val, bool) or not isinstance(val, int) or val < lo:
                raise ValueError(f"MaskBlobFilter: {name} must be an integer >= {lo}, got {val!r}.")
        if (
            isinstance(min_angle_deg, bool)
            or not isinstance(min_angle_deg, (int, float))
            or not 0.0 <= float(min_angle_deg) <= 180.0
        ):
            raise ValueError(
                f"MaskBlobFilter: min_angle_deg must be in [0, 180], got {min_angle_deg!r}."
            )
        self.min_area = int(min_area)
        self.min_object_px = int(min_object_px)
        self.min_angle_deg = float(min_angle_deg)
        self.cell = int(cell)
        self.median_stride = int(median_stride)
        super().__init__(
            min_area=self.min_area,
            min_object_px=self.min_object_px,
            min_angle_deg=self.min_angle_deg,
            cell=self.cell,
            median_stride=self.median_stride,
            **kwargs,
        )

    def _object_cells(self, cube: Tensor) -> Tensor:
        """[B, h, w] bool on the cell grid: the cells whose first pixel is not background, i.e. has
        a spectral angle above ``min_angle_deg`` to the frame's median spectrum."""
        b, k = cube.shape[0], cube.shape[-1]
        m = self.median_stride
        sample = cube[:, ::m, ::m, :].to(torch.float32).permute(0, 3, 1, 2).reshape(b, k, -1)
        ref = sample.median(dim=2).values  # [B, K]: along the contiguous axis, 4x faster on a GPU
        px = cube[:, :: self.cell, :: self.cell, :].to(torch.float32)  # [B, h, w, K]
        dot = torch.einsum("bhwk,bk->bhw", px, ref)
        norm = torch.linalg.vector_norm(px, dim=-1) * ref.norm(dim=-1)[:, None, None]
        cos = (dot / (norm + 1e-6)).clamp(-1.0, 1.0)
        return cos < math.cos(math.radians(self.min_angle_deg))  # angle > a <=> cos < cos(a)

    def forward(self, decisions: Tensor, cube: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Return the blobs that pass the size test and, with a cube, the object test."""
        if decisions.numel() == 0 or not bool(decisions.any()):
            return {"decisions": decisions.clone()}
        counts = cell_sums(decisions.permute(0, 3, 1, 2).to(torch.float32), self.cell)
        small = torch.uint8 if self.cell * self.cell < 256 else torch.int32
        planes = [counts.to(small)]  # marked pixels per cell, [B, C, h, w]
        if cube is not None and self.min_object_px > 0:
            obj = self._object_cells(cube)
            planes.append((counts * obj[:, None]).to(small))  # marked object pixels per cell
        host = torch.stack(planes).cpu().numpy()  # the one copy to the host
        keep = np.zeros(host.shape[1:], bool)
        for i in range(keep.shape[0]):
            for c in range(keep.shape[1]):
                occ = host[0, i, c] > 0
                if not occ.any():
                    continue
                tests = [(host[0, i, c], max(self.min_area, 1))]
                if len(planes) > 1:
                    tests.append((host[1, i, c], self.min_object_px))
                keep[i, c] = keep_blobs(occ, tests)
        kk = torch.from_numpy(keep).to(device=decisions.device).permute(0, 2, 3, 1)
        return {"decisions": expand_cells(kk, decisions, self.cell)}


class MaskComposite(Node):
    """Merge N boolean masks [B, H, W, C] into one label map and one level map for display."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.TORCH})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            variadic=True,
            description="Boolean masks [B, H, W, C] of one batch, height and width, one per "
            "inbound connection (fan-in), in the order of `labels` / `levels`; a pixel of a mask "
            "counts as set where any of its channels is.",
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Label map [B, H, W]: 0 where no mask is set, else the largest `labels` "
            "entry among the masks set there.",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Level map [B, H, W, 1]: 0 where no mask is set, else the largest `levels` "
            "entry among the masks set there.",
        ),
    }

    def __init__(
        self, labels: list[int] | None = None, levels: list[float] | None = None, **kwargs: Any
    ) -> None:
        """Create a mask composite.

        Parameters
        ----------
        labels : one positive integer label per inbound mask, in connection order (default
            ``[1, 2]``). Where masks overlap the largest label wins, so give the mask that should
            stay visible on top (e.g. a defect on an object) the largest one.
        levels : one non-negative level per inbound mask, in connection order (default
            ``[0.5, 1.0]``); where masks overlap the largest level wins.
        """
        labels = [1, 2] if labels is None else list(labels)
        levels = [0.5, 1.0] if levels is None else list(levels)
        if not labels or len(labels) != len(levels):
            raise ValueError(
                "MaskComposite: labels and levels need one entry per mask, "
                f"got {labels!r} / {levels!r}."
            )
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in labels):
            raise ValueError(f"MaskComposite: labels must be integers >= 1, got {labels!r}.")
        if any(
            isinstance(x, bool)
            or not isinstance(x, (int, float))
            or not 0.0 <= float(x) < float("inf")
            for x in levels
        ):
            raise ValueError(f"MaskComposite: levels must be finite numbers >= 0, got {levels!r}.")
        self.labels = [int(x) for x in labels]
        self.levels = [float(x) for x in levels]
        super().__init__(labels=self.labels, levels=self.levels, **kwargs)

    def forward(self, decisions: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the label map and the level map of the inbound masks."""
        masks = list(decisions) if isinstance(decisions, (list, tuple)) else [decisions]
        if len(masks) != len(self.labels):
            raise ValueError(
                f"MaskComposite: {len(masks)} masks connected, "
                f"{len(self.labels)} labels configured."
            )
        bhw = masks[0].shape[:3]
        for i, m in enumerate(masks[1:], start=1):
            if m.shape[:3] != bhw:
                raise ValueError(
                    f"MaskComposite: mask {i} is [B, H, W] = {tuple(m.shape[:3])}, "
                    f"expected {tuple(bhw)}."
                )
        stack = torch.stack([m.any(dim=-1) for m in masks], dim=0)  # [N, B, H, W]
        dev = stack.device
        labels = torch.tensor(self.labels, dtype=torch.int32, device=dev).view(-1, 1, 1, 1)
        levels = torch.tensor(self.levels, dtype=torch.float32, device=dev).view(-1, 1, 1, 1)
        label_map = (stack.to(torch.int32) * labels).amax(dim=0)
        level_map = (stack.to(torch.float32) * levels).amax(dim=0).unsqueeze(-1)
        return {"mask": label_map, "scores": level_map}


class ScoreMapSuppression(Node):
    """Down-weight a score map [B, H, W, 1] inside a boolean mask shrunk by ``erode_px``."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.MASK, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Score map [B, H, W, 1], e.g. a fused anomaly map.",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Boolean mask [B, H, W, C] of the same batch, height and width: where the "
            "score map is suppressed (a pixel counts as set where any channel is), e.g. a "
            "segmenter's decisions for an object class that cannot be anomalous.",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="scores x (1 - weight x the eroded mask): unchanged outside the mask and "
            "within `erode_px` of its edge, scaled by 1 - weight inside.",
        ),
    }

    def __init__(self, weight: float = 1.0, erode_px: int = 4, **kwargs: Any) -> None:
        """Create a masked score suppression.

        Parameters
        ----------
        weight : how much of the score is removed inside the eroded mask, in ``[0, 1]`` (default
            1.0: set to zero; 0.0 leaves the map unchanged).
        erode_px : margin in pixels by which the mask is shrunk first (square erosion, default 4),
            so that an object touching a masked one keeps its score near the shared edge. The image
            border does not shrink the mask.
        """
        if (
            isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not 0.0 <= weight <= 1.0
        ):
            raise ValueError(
                f"ScoreMapSuppression: weight must be a number in [0, 1], got {weight!r}."
            )
        if isinstance(erode_px, bool) or not isinstance(erode_px, int) or erode_px < 0:
            raise ValueError(
                f"ScoreMapSuppression: erode_px must be an integer >= 0, got {erode_px!r}."
            )
        self.weight = float(weight)
        self.erode_px = int(erode_px)
        super().__init__(weight=self.weight, erode_px=self.erode_px, **kwargs)

    def forward(self, scores: Tensor, mask: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the score map with the eroded mask's pixels scaled by ``1 - weight``."""
        inside = mask.any(dim=-1)  # [B, H, W]
        if inside.shape != scores.shape[:3]:
            raise ValueError(
                f"ScoreMapSuppression: mask is [B, H, W] = {tuple(inside.shape)}, scores "
                f"{tuple(scores.shape[:3])}."
            )
        if self.erode_px:
            # a pixel stays inside if no outside pixel lies within erode_px (square window);
            # max-pool pads with zeros, so beyond the image border counts as inside
            outside = (~inside).to(scores.dtype).unsqueeze(1)
            near_outside = torch.nn.functional.max_pool2d(
                outside, kernel_size=2 * self.erode_px + 1, stride=1, padding=self.erode_px
            )
            inside = near_outside.squeeze(1) == 0
        keep = 1.0 - self.weight * inside.to(scores.dtype)
        return {"scores": scores * keep.unsqueeze(-1)}
