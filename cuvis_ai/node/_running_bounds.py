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

    @torch.no_grad()
    def step(self, frame_lo: Tensor, frame_hi: Tensor) -> RunningPhase:
        """Count one frame and fold its per-channel percentile bounds into the buffers.

        The first counted frame initialises NaN bounds; later frames accumulate with
        ``torch.minimum`` / ``torch.maximum``, which is order-free, so the result does not
        depend on how frames are grouped into batches.
        """
        self.frame_count.add_(1)
        count = int(self.frame_count.item())
        should_update = self.freeze_after_frames is None or count <= self.freeze_after_frames
        if should_update:
            if torch.isnan(self.running_min).any():
                self.running_min.copy_(frame_lo)
                self.running_max.copy_(frame_hi)
            else:
                torch.minimum(self.running_min, frame_lo, out=self.running_min)
                torch.maximum(self.running_max, frame_hi, out=self.running_max)
        return RunningPhase(
            count=count, in_warmup=count <= self.warmup_frames, should_update=should_update
        )


def per_frame_quantiles(data: Tensor, low: float, high: float) -> tuple[Tensor, Tensor]:
    """Per-frame, per-channel quantiles of a BHWC tensor as two ``[B, C]`` float tensors."""
    batch, channels = data.shape[0], data.shape[-1]
    flat = data.reshape(batch, -1, channels).float()  # quantile() requires float/double
    return torch.quantile(flat, low, dim=1), torch.quantile(flat, high, dim=1)


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
    """
    frame_lo, frame_hi = per_frame_quantiles(data, quantile_low, quantile_high)
    lows: list[Tensor] = []
    highs: list[Tensor] = []
    for b in range(data.shape[0]):
        phase = bounds.step(frame_lo[b], frame_hi[b])
        if phase.in_warmup:
            lows.append(frame_lo[b])
            highs.append(frame_hi[b])
        else:
            lows.append(bounds.running_min.clone())
            highs.append(bounds.running_max.clone())
    return normalize_with_bounds(data, torch.stack(lows), torch.stack(highs), eps)


__all__ = [
    "RunningBounds",
    "RunningPhase",
    "normalize_with_bounds",
    "per_frame_quantiles",
    "running_normalize",
]
