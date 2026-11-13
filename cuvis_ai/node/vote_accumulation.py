"""Turn a per-pixel field of relative 2D vectors into a vote map at the targets it points to."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node

_ORDERS = ("xy", "yx")


# Pixel coordinates are float32, like the vectors; integers are exact there up to 2**24.
_MAX_AXIS = 2**24


class VectorFieldVoteAccumulator(Node):
    """Scatter each pixel's vote at the position its 2D vector points to.

    Every pixel of ``vectors`` holds an offset relative to its own row and column; the
    pixel's vote (its ``weights`` entry, or one when ``weights`` is not connected) is
    added to the score map at the rounded target position. Collisions sum; a vote whose
    target falls outside the frame, or whose vector or weight is not finite, is dropped.

    The target is ``coordinate + round(offset)``: the offset is rounded first
    (``torch.round``, halves to the nearest even integer), then added to the integer
    source coordinate, not the other way around. The op is a single ``scatter_add`` on
    the score map flattened to ``[B, H * W]``: every pixel keeps its slot in that
    computation (a dropped vote's index is clamped into range and its weight is zeroed),
    so no boolean-mask indexing and no data-dependent shape enters the graph.

    The votes are summed in float64 and the map is cast back to float32. A running
    float32 sum drifts when many fractional votes meet on one target (about ``1e-4`` of
    the total for 100 000 votes of 0.1) and loses small weights beside large ones that
    cancel; the wider sum keeps both exact to float32 rounding. A float32 score holds a
    count of unit votes exactly up to ``2**24`` per target.

    The output has no gradient with respect to ``vectors`` (the rounding step has none),
    and a gradient with respect to ``weights`` that matches a plain weighted sum at each
    kept target.

    Parameters
    ----------
    order : str, optional
        How the two channels of ``vectors`` map to column and row offsets (default
        ``"xy"``): ``"xy"`` reads channel 0 as the column offset ``dx`` and channel 1 as
        the row offset ``dy``; ``"yx"`` reverses that.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ONNX, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "vectors": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 2),
            description="Per-pixel relative vector field [B, H, W, 2].",
        ),
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            optional=True,
            description="Per-pixel vote weight [B, H, W, 1]; one for every pixel when not "
            "connected.",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Vote map [B, H, W, 1]: the sum of weights of every vote landing on "
            "each pixel.",
        ),
    }

    def __init__(self, order: str = "xy", **kwargs: Any) -> None:
        if order not in _ORDERS:
            raise ValueError(
                f"VectorFieldVoteAccumulator: order must be one of {_ORDERS}, got {order!r}."
            )
        self.order = order
        super().__init__(order=self.order, **kwargs)

    def forward(
        self, vectors: Tensor, weights: Tensor | None = None, **_: Any
    ) -> dict[str, Tensor]:
        """Return the vote map; see the class docstring for the scatter rule.

        Parameters
        ----------
        vectors : Tensor
            Per-pixel vector field ``[B, H, W, 2]``. Raises ``ValueError`` when the
            height or the width exceeds ``2**24``: the coordinates are float32, like the
            vectors, and count exactly only up to there.
        weights : Tensor, optional
            Per-pixel vote weight ``[B, H, W, 1]``; defaults to one everywhere. Raises
            ``ValueError`` if its shape differs from ``vectors``' first three dims plus
            a trailing size of one.

        Returns
        -------
        dict[str, Tensor]
            ``{"scores": ...}`` ``[B, H, W, 1]``.
        """
        b, h, w = vectors.shape[:3]
        expected_shape = (b, h, w, 1)
        if weights is not None and tuple(weights.shape) != expected_shape:
            raise ValueError(
                f"VectorFieldVoteAccumulator: weights is {tuple(weights.shape)}, expected "
                f"{expected_shape} (vectors' batch, height and width plus a trailing 1)."
            )

        if max(h, w) > _MAX_AXIS:
            raise ValueError(
                f"VectorFieldVoteAccumulator: height and width must not exceed {_MAX_AXIS}, "
                f"the number of pixel coordinates float32 counts exactly; got {h} x {w}."
            )

        channel0 = vectors[..., 0]
        channel1 = vectors[..., 1]
        dx, dy = (channel0, channel1) if self.order == "xy" else (channel1, channel0)

        rows = torch.arange(h, dtype=vectors.dtype, device=vectors.device).view(1, h, 1)
        cols = torch.arange(w, dtype=vectors.dtype, device=vectors.device).view(1, 1, w)
        target_row = rows + torch.round(dy)
        target_col = cols + torch.round(dx)

        wgt = torch.ones_like(channel0) if weights is None else weights[..., 0]
        valid = (
            torch.isfinite(target_row)
            & torch.isfinite(target_col)
            & (target_row >= 0)
            & (target_row < h)
            & (target_col >= 0)
            & (target_col < w)
            & torch.isfinite(wgt)
        )

        safe_row = torch.where(valid, target_row, torch.zeros_like(target_row))
        safe_col = torch.where(valid, target_col, torch.zeros_like(target_col))
        index = (safe_row.to(torch.int64) * w + safe_col.to(torch.int64)).reshape(b, h * w)
        scattered_weight = torch.where(valid, wgt, torch.zeros_like(wgt)).reshape(b, h * w)

        # Summed in float64, then cast back; the class docstring says why.
        flat_scores = torch.zeros(b, h * w, dtype=torch.float64, device=vectors.device)
        flat_scores = flat_scores.scatter_add(1, index, scattered_weight.to(torch.float64))
        return {"scores": flat_scores.to(vectors.dtype).reshape(b, h, w, 1)}


__all__ = ["VectorFieldVoteAccumulator"]
