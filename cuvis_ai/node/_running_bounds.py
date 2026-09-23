"""Per-frame running percentile bounds shared by the running normalization modes.

``FixedWavelengthSelector`` (through ``ChannelSelectorBase``) and ``PercentileNormalizer`` keep
per-channel ``running_min`` / ``running_max`` buffers and a frame counter. This module owns the
step that moves those buffers along, so both nodes follow one timeline and the counts they are
configured with mean frames whatever the batch size::

    frame    1 ............. warmup | warmup+1 ......... freeze | freeze+1 ...........
    output   per-frame percentiles  | accumulated bounds        | accumulated bounds
    bounds   accumulating           | accumulating              | frozen

One step per frame, never per batch: a forward call with B frames takes B steps, so a warmup or
freeze boundary can fall inside a batch, and each frame is normalized with the bounds as they
stood right after that frame was counted, exactly as a one-frame-at-a-time run produces.

Cost: a frame's own percentiles are needed only while it is inside the warmup (they scale it) or
before the freeze (they fold into the bounds). Past both, the frame only counts, so no quantile is
computed for it; a batch computes both quantiles in one ``torch.quantile`` call over its leading
frames, reads the frame counter once and checks the bounds for the unfitted state once.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class RunningPhase:
    """Where one frame fell on the running-bounds timeline."""

    count: int
    """Frames counted so far, this frame included."""

    in_warmup: bool
    """The frame is normalized with its own percentiles (``count <= warmup_frames``)."""

    should_update: bool
    """The frame's percentiles were folded into the bounds (the bounds were not yet frozen)."""


class RunningBounds:
    """Per-frame accumulation of percentile bounds held in a node's own buffers.

    The helper binds to the node's ``running_min``, ``running_max`` and frame-count buffers
    instead of owning tensors, so the node's ``state_dict`` keys and every checkpoint on disk
    stay as they are.

    Parameters
    ----------
    running_min, running_max : Tensor
        Per-channel ``[C]`` float buffers, NaN while unfitted; updated in place.
    frame_count : Tensor
        0-dim integer buffer counting frames; advanced in place.
    warmup_frames : int
        Frames normalized with their own percentiles while the bounds accumulate.
    freeze_after_frames : int | None
        Stop folding frames into the bounds after this many; ``None`` never freezes.
    """

    def __init__(
        self,
        running_min: Tensor,
        running_max: Tensor,
        frame_count: Tensor,
        *,
        warmup_frames: int,
        freeze_after_frames: int | None,
    ) -> None:
        self.running_min = running_min
        self.running_max = running_max
        self.frame_count = frame_count
        self.warmup_frames = int(warmup_frames)
        self.freeze_after_frames = freeze_after_frames
        # Whether the buffers hold values, read from ``running_min`` on first use and then
        # remembered, so a batch pays that host sync once rather than per frame.
        self._fitted: bool | None = None

    @torch.no_grad()
    def step(
        self, frame_lo: Tensor | None, frame_hi: Tensor | None, *, count: int | None = None
    ) -> RunningPhase:
        """Count one frame and fold its per-channel percentile bounds into the buffers.

        The first counted frame initialises NaN bounds; later frames accumulate with
        ``torch.minimum`` / ``torch.maximum``, which is order-free, so the result does not
        depend on how frames are grouped into batches.

        ``count`` is this frame's ordinal once counted. A caller that derived it from one read
        of ``frame_count`` passes it in, and the step syncs nothing to the host; without it the
        counter is read back. ``frame_lo`` / ``frame_hi`` may be ``None`` for a frame that is
        past the freeze, which only counts.
        """
        self.frame_count.add_(1)
        if count is None:
            count = int(self.frame_count.item())
        should_update = self.freeze_after_frames is None or count <= self.freeze_after_frames
        if should_update:
            if frame_lo is None or frame_hi is None:
                raise ValueError("a frame that folds into the bounds needs its quantiles")
            if not self._is_fitted():
                self.running_min.copy_(frame_lo)
                self.running_max.copy_(frame_hi)
                self._fitted = True
            else:
                torch.minimum(self.running_min, frame_lo, out=self.running_min)
                torch.maximum(self.running_max, frame_hi, out=self.running_max)
        return RunningPhase(
            count=count, in_warmup=count <= self.warmup_frames, should_update=should_update
        )

    def frames_needing_quantiles(self, count0: int, batch: int) -> int:
        """How many leading frames of a ``batch`` that starts after ``count0`` frames need quantiles.

        A frame needs its own percentiles while it is inside the warmup or before the freeze;
        once past both they would be computed and discarded. Frames are counted in order, so the
        frames that still need them are the leading ones of the batch.
        """
        if self.freeze_after_frames is None:
            return batch
        limit = max(self.warmup_frames, self.freeze_after_frames)
        return max(0, min(batch, limit - count0))

    def _is_fitted(self) -> bool:
        """Whether the bounds hold values: one read of the buffer, then remembered."""
        if self._fitted is None:
            self._fitted = not bool(torch.isnan(self.running_min).any())
        return self._fitted


def per_frame_quantiles(data: Tensor, low: float, high: float) -> tuple[Tensor, Tensor]:
    """Per-frame, per-channel quantiles of a BHWC tensor as two ``[B, C]`` float tensors.

    Both quantiles come out of one ``torch.quantile`` call, so every frame is sorted once.
    """
    batch, channels = data.shape[0], data.shape[-1]
    flat = data.reshape(batch, -1, channels).float()  # quantile() requires float/double
    q = torch.tensor([low, high], dtype=flat.dtype, device=flat.device)
    lo, hi = torch.quantile(flat, q, dim=1)  # [2, B, C]
    return lo, hi


def normalize_with_bounds(data: Tensor, lo: Tensor, hi: Tensor, eps: float) -> Tensor:
    """Scale a BHWC tensor to ``[0, 1]`` with per-frame ``[B, C]`` or shared ``[C]`` bounds."""
    channels = data.shape[-1]
    lo = lo.reshape(-1, 1, 1, channels)
    hi = hi.reshape(-1, 1, 1, channels)
    denom = (hi - lo).clamp_min(eps)
    return ((data - lo) / denom).clamp_(0.0, 1.0)


@torch.no_grad()
def running_normalize(
    data: Tensor,
    bounds: RunningBounds,
    *,
    quantile_low: float,
    quantile_high: float,
    eps: float,
) -> Tensor:
    """Normalize a BHWC batch frame by frame through the running bounds.

    Every frame takes one :meth:`RunningBounds.step`. Warmup frames are scaled with their own
    percentiles, later frames with the accumulated bounds as they stood right after their own
    step. At batch size 1 this is the classic one-quantile, one-count, one-decision call.

    The frame counter is read once; quantiles are computed only for the leading frames that are
    still inside the warmup or before the freeze (see
    :meth:`RunningBounds.frames_needing_quantiles`). The frames after them are scaled with the
    bounds as they stand, which no longer move.
    """
    batch, channels = data.shape[0], data.shape[-1]
    count0 = int(bounds.frame_count.item())
    leading = bounds.frames_needing_quantiles(count0, batch)
    lows: list[Tensor] = []
    highs: list[Tensor] = []
    if leading:
        frame_lo, frame_hi = per_frame_quantiles(data[:leading], quantile_low, quantile_high)
        for b in range(leading):
            phase = bounds.step(frame_lo[b], frame_hi[b], count=count0 + b + 1)
            if phase.in_warmup:
                lows.append(frame_lo[b].unsqueeze(0))
                highs.append(frame_hi[b].unsqueeze(0))
            else:
                lows.append(bounds.running_min.clone().unsqueeze(0))
                highs.append(bounds.running_max.clone().unsqueeze(0))
    for b in range(leading, batch):
        bounds.step(None, None, count=count0 + b + 1)
    tail = batch - leading
    if tail:
        lows.append(bounds.running_min.expand(tail, channels))
        highs.append(bounds.running_max.expand(tail, channels))
    if not lows:
        return data.clone()
    return normalize_with_bounds(data, torch.cat(lows), torch.cat(highs), eps)


__all__ = [
    "RunningBounds",
    "RunningPhase",
    "normalize_with_bounds",
    "per_frame_quantiles",
    "running_normalize",
]
