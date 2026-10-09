"""Score and decision fusion: one map from several, by a fixed elementwise rule.

:class:`ScoreMapFusion` combines the score maps of several detectors or segmenters into one.
Averaging maps on a common scale (a fitted normalizer per detector) cancels each detector's
private noise while keeping what they agree on; ``min`` is the hard AND (a pixel stays high only
where every map is high, so a threshold on the result is "all maps above it"), ``max`` the OR,
``wmean`` a weighted average, ``gmean`` the geometric mean (it penalises disagreement more than
the mean), ``softmin`` a soft AND between the minimum and the mean, and ``first`` a priority rule
for gated maps: per frame the first inbound map that is not all zero, e.g. one detector's map
whenever its gate opens and a second detector's only on the frames the first one misses.

:class:`DecisionFusion` is the counterpart for boolean masks: ``any`` and ``all`` combine them
pixel by pixel, ``first`` takes per frame the first inbound mask with a set pixel, the mask of the
map a ``first`` score fusion shows.

Both nodes are stateless, torch-native and take their inputs through one variadic port, one
tensor per inbound connection in connection order; ``ScoreMapFusion`` is differentiable.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node.node import Node

_SCORE_MODES = ("mean", "min", "max", "wmean", "gmean", "first", "softmin")
_DECISION_MODES = ("any", "all", "first")


def _stack(values: Sequence[Tensor] | Tensor, node: str, what: str) -> Tensor:
    """Stack the inbound tensors of a variadic port into ``[N, ...]``; one shape for all."""
    items = list(values) if isinstance(values, (list, tuple)) else [values]
    if not items:
        raise ValueError(f"{node}: no {what}s connected.")
    shape = items[0].shape
    for i, item in enumerate(items[1:], start=1):
        if item.shape != shape:
            raise ValueError(
                f"{node}: {what} {i} has shape {tuple(item.shape)}, expected {tuple(shape)}."
            )
    return torch.stack(items, dim=0)


def _first_live(stack: Tensor, live: Tensor) -> Tensor:
    """Per frame, the first of the ``N`` stacked tensors flagged in ``live`` ``[N, B]``.

    A frame where none is flagged takes the first tensor.
    """
    pick = live.to(torch.int64).argmax(dim=0)  # [B]: the first live index, 0 when none
    return stack[pick, torch.arange(stack.shape[1], device=stack.device)]


class ScoreMapFusion(Node):
    """Fuse N score maps of one shape into one map by a fixed elementwise rule (``mode``)."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.ANOMALY, NodeTag.POSTPROCESSING, NodeTag.DIFFERENTIABLE, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            variadic=True,
            description="Score maps [B, H, W, C] of one shape, one per inbound connection "
            "(fan-in), in connection order; anomaly and segmentation maps have C = 1.",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Fused score map [B, H, W, C] per ``mode``.",
        ),
    }

    def __init__(
        self,
        mode: str = "mean",
        weights: list[float] | None = None,
        beta: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Create a fusion node.

        Parameters
        ----------
        mode : str
            ``"mean"`` (arithmetic mean), ``"min"`` (AND), ``"max"`` (OR), ``"wmean"`` (weighted
            mean with ``weights``, normalised to sum to one), ``"gmean"`` (geometric mean of the
            maps floored at 0), ``"first"`` (per frame, the first inbound map in connection order
            that is not all zero; the first map when all are) or ``"softmin"`` (the soft minimum
            ``-(1/beta) log(sum_i w_i exp(-beta x_i))`` with the weights normalised to sum to one,
            equal by default: at most ``log(N) / beta`` above the minimum for equal weights, and
            the weighted mean as ``beta`` goes to 0).
        weights : list[float] | None
            One non-negative weight per inbound map with a positive sum: required for
            ``"wmean"``, optional for ``"softmin"``, rejected for the other modes. The count is
            checked against the inbound maps when the node runs.
        beta : float | None
            Sharpness of the soft minimum (> 0), in the units of the maps; required for
            ``"softmin"``, rejected for the other modes.
        """
        if mode not in _SCORE_MODES:
            raise ValueError(f"ScoreMapFusion: mode must be one of {_SCORE_MODES}, got {mode!r}.")
        if mode == "wmean" and not weights:
            raise ValueError("ScoreMapFusion: mode 'wmean' requires weights.")
        if mode in ("wmean", "softmin") and weights is not None:
            if any(float(x) < 0.0 for x in weights) or float(sum(weights)) <= 0.0:
                raise ValueError(
                    "ScoreMapFusion: weights must be non-negative with a positive sum."
                )
        elif weights is not None:
            raise ValueError(
                f"ScoreMapFusion: weights are only used with mode 'wmean' or 'softmin', "
                f"not {mode!r}."
            )
        if mode == "softmin":
            if beta is None or not 0.0 < float(beta) < float("inf"):
                raise ValueError(f"ScoreMapFusion: mode 'softmin' requires beta > 0, got {beta!r}.")
        elif beta is not None:
            raise ValueError(
                f"ScoreMapFusion: beta is only used with mode 'softmin', not {mode!r}."
            )
        self.mode = mode
        self.weights = [float(x) for x in weights] if weights is not None else None
        self.beta = float(beta) if beta is not None else None
        super().__init__(mode=self.mode, weights=self.weights, beta=self.beta, **kwargs)

    def _weights(self, n: int, like: Tensor) -> Tensor:
        """The normalised weights ``[N, 1, ..., 1]``, equal ones when none were given."""
        if self.weights is not None and len(self.weights) != n:
            raise ValueError(f"ScoreMapFusion: {len(self.weights)} weights for {n} score maps.")
        w = torch.tensor(self.weights or [1.0] * n, dtype=like.dtype, device=like.device)
        return (w / w.sum()).view(-1, *([1] * (like.dim() - 1)))

    def forward(self, scores: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the fused map.

        Parameters
        ----------
        scores : list[Tensor] | Tensor
            The inbound score maps, one shape for all.

        Returns
        -------
        dict[str, Tensor]
            ``scores``: the fused map, the shape of one inbound map.
        """
        stack = _stack(scores, "ScoreMapFusion", "score map")  # [N, B, H, W, C]
        if self.mode == "mean":
            out = stack.mean(dim=0)
        elif self.mode == "min":
            out = stack.amin(dim=0)
        elif self.mode == "max":
            out = stack.amax(dim=0)
        elif self.mode == "first":
            out = _first_live(stack, stack.flatten(start_dim=2).ne(0).any(dim=2))
        elif self.mode == "gmean":
            product = stack.clamp_min(0.0).prod(dim=0)
            n = stack.shape[0]
            out = product.sqrt() if n == 2 else product.pow(1.0 / n)
        elif self.mode == "softmin":  # lo - log(sum_i w_i exp(-beta (x_i - lo))) / beta, stable
            w = self._weights(stack.shape[0], stack)
            lo = stack.amin(dim=0)
            out = lo - torch.log((w * torch.exp(-self.beta * (stack - lo))).sum(dim=0)) / self.beta
        else:  # wmean
            out = (stack * self._weights(stack.shape[0], stack)).sum(dim=0)
        return {"scores": out}


class DecisionFusion(Node):
    """Fuse N boolean masks of one shape into one mask by ``mode`` (any, all or first)."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            variadic=True,
            description="Boolean masks [B, H, W, C] of one shape, one per inbound connection "
            "(fan-in), in connection order, e.g. the decisions of several deciders.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, -1),
            description="Fused mask [B, H, W, C] per ``mode``.",
        ),
    }

    def __init__(self, mode: str = "any", **kwargs: Any) -> None:
        """Create a mask fusion node.

        Parameters
        ----------
        mode : str
            ``"any"`` (pixel-wise OR), ``"all"`` (pixel-wise AND) or ``"first"`` (per frame, the
            first inbound mask in connection order with a set pixel; the first mask when none has
            one).
        """
        if mode not in _DECISION_MODES:
            raise ValueError(
                f"DecisionFusion: mode must be one of {_DECISION_MODES}, got {mode!r}."
            )
        self.mode = mode
        super().__init__(mode=self.mode, **kwargs)

    def forward(self, decisions: list[Tensor] | Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the fused mask.

        Parameters
        ----------
        decisions : list[Tensor] | Tensor
            The inbound boolean masks, one shape for all.

        Returns
        -------
        dict[str, Tensor]
            ``decisions``: the fused mask, the shape of one inbound mask.
        """
        stack = _stack(decisions, "DecisionFusion", "mask")  # [N, B, H, W, C]
        if self.mode == "any":
            out = stack.any(dim=0)
        elif self.mode == "all":
            out = stack.all(dim=0)
        else:
            out = _first_live(stack, stack.flatten(start_dim=2).any(dim=2))
        return {"decisions": out}


__all__ = ["DecisionFusion", "ScoreMapFusion"]
