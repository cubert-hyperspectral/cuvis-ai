"""Binary decision nodes for thresholding anomaly scores and logits.

This module provides threshold-based decision nodes that convert continuous anomaly
scores or logits into binary decisions (anomaly/normal). Two strategies are available:

- **BinaryDecider**: Fixed threshold applied globally to sigmoid-transformed logits
- **QuantileBinaryDecider**: Adaptive per-batch thresholding using quantile statistics
- **ScoreThreshold**: Fixed threshold on a raw score, keeping the values below or above it

Decision nodes are typically placed at the end of anomaly detection pipelines to
convert detector outputs into actionable binary masks for visualization or evaluation.
"""

import math
import numbers
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.deciders.base_decider import BinaryDecider as BaseDecider
from cuvis_ai_core.node import Node

from . import _calibration
from ._port_inputs import call_threshold


def resolve_reduce_dims(reduce_dims: tuple[int, ...] | None, tensor_ndim: int) -> tuple[int, ...]:
    """Resolve reduction dimensions, handling negative indices.

    Parameters
    ----------
    reduce_dims : tuple[int, ...] | None
        Dimension indices to reduce over (may contain negatives).
        When ``None``, returns all non-batch dimensions ``(1, ..., ndim-1)``.
    tensor_ndim : int
        Number of dimensions in the tensor.

    Returns
    -------
    tuple[int, ...]
        Sorted, deduplicated positive dimension indices.
    """
    if reduce_dims is None:
        return tuple(range(1, tensor_ndim))

    resolved: list[int] = []
    for dim in reduce_dims:
        adjusted = dim if dim >= 0 else tensor_ndim + dim
        resolved.append(adjusted)
    return tuple(sorted(set(resolved)))


class BinaryDecider(BaseDecider):
    """Simple decider node using a static threshold to classify data.

    Accepts logits as input, applies sigmoid transformation to convert to
    probabilities [0, 1], then applies threshold to produce binary decisions.

    Parameters
    ----------
    threshold : float
        The threshold to use for classification after sigmoid.
        Values >= threshold are classified as anomalies (True).
        Default: 0.5

    The threshold is also an optional input of shape ``[1]``: a value on the port replaces
    the hparam for that call only, so a GUI slider needs no pipeline reload and the yaml keeps
    the hparam.

    :meth:`calibrate` fits the F1 optimum unless the ``calibration_objective`` hparam names a
    target: ``"precision"`` or ``"recall"`` with ``calibration_target`` in [0, 1], measured at
    ``calibration_level`` ``"pixels"`` or ``"frames"``. The three persist in the pipeline
    yaml, so a retrain calibrates to the same target.

    Examples
    --------
    >>> from cuvis_ai.node.deciders.binary_decider import BinaryDecider
    >>> import torch
    >>>
    >>> # Create decider with default threshold
    >>> decider = BinaryDecider(threshold=0.5)
    >>>
    >>> # Apply to RX anomaly logits
    >>> logits = torch.randn(4, 256, 256, 1)  # [B, H, W, C]
    >>> output = decider.forward(logits=logits)
    >>> decisions = output["decisions"]  # [4, 256, 256, 1] boolean mask
    >>>
    >>> # Use in pipeline
    >>> pipeline.connect(
    ...     (logit_head.logits, decider.logits),
    ...     (decider.decisions, visualizer.mask),
    ... )

    See Also
    --------
    QuantileBinaryDecider : Adaptive per-batch thresholding
    ScoreToLogit : Convert scores to logits before decisioning
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY, NodeTag.CALIBRATION}
    )

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits to threshold (BHWC format)",
        ),
        "threshold": PortSpec(
            dtype=torch.float32,
            shape=(1,),
            description="Threshold on sigmoid(logits) for this call [1]; replaces the "
            "`threshold` hparam.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(
        self,
        threshold: float = 0.5,
        calibration_objective: str = "f1",
        calibration_target: float | None = None,
        calibration_level: str = "pixels",
        **kwargs,
    ) -> None:
        self.threshold = threshold
        self._calibration_objective = _calibration.calibration_objective_of(
            calibration_objective, calibration_target, calibration_level
        )
        # Forward the hparams to BaseDecider so Serializable captures them
        super().__init__(
            threshold=threshold,
            calibration_objective=self._calibration_objective.objective,
            calibration_target=self._calibration_objective.target,
            calibration_level=self._calibration_objective.level,
            **kwargs,
        )

    def forward(
        self,
        logits: Tensor,
        *,
        threshold: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply sigmoid and threshold-based decisioning on channels-last data.

        Args:
            logits: Tensor shaped (B, H, W, C) containing logits.
            threshold: Threshold for this call, shape [1]; ``None`` uses the hparam.

        Returns:
            Dictionary with "decisions" key containing (B, H, W, 1) decision mask.

        Raises:
            ValueError: A ``threshold`` input of another shape than [1] or a non-finite value.
        """
        cut = call_threshold("threshold", threshold, self.threshold)
        return {"decisions": torch.sigmoid(logits) >= cut}

    def calibrate(
        self,
        scores: Tensor,
        targets: Tensor,
        *,
        num_candidates: int = 256,
    ) -> dict[str, Any]:
        """Refit ``threshold`` on labelled validation scores.

        ``scores`` is the decider-input tensor stacked over the split (``[N, H, W, C]``);
        ``targets`` the matching ground-truth mask. The sweep runs on ``torch.sigmoid(scores)``
        in float32, exactly what ``forward`` compares against ``threshold``, so the fitted
        value is reachable at runtime even when raw logits saturate the sigmoid. With the
        default ``f1`` objective the optimum is the F1 maximum; with a precision or recall
        objective (the ``calibration_objective``, ``calibration_target`` and
        ``calibration_level`` hparams) it is the cutoff that reaches the target with the best
        other metric (:func:`_calibration.select_calibration_row`), found on every distinct
        score of a small split and by a refinement around the best sampled cutoff on a large
        one; a target no cutoff reaches lands on the closest one and the report says so.
        Either value is moved to the midpoint of its plateau (``_calibration.margin_below``)
        and written to the live attribute and to ``hparams``; ``pipeline.save_to_file`` then
        carries it in the pipeline yaml. The ``.pt`` weights are unchanged, so load the saved
        yaml rather than the preset plus weights. Multi-channel input is reduced to the
        per-pixel max, which matches ``forward`` exactly for single-channel scores.

        The report carries, at the applied point and counted as ``forward`` decides, the
        pixel-level ``f1``, ``precision`` and ``recall`` (also as ``pixel_*``), the frame-level
        ``frame_*`` numbers, ``frames`` and ``anomalous_frames`` and, for a target objective,
        ``target`` and ``reached`` (1 or 0).

        Raises:
            CalibrationError: shape mismatch, non-finite scores, or a single-class split.
        """
        probabilities = torch.sigmoid(scores.detach().to("cpu", torch.float32))
        _, pixel, gt, frame_labels = _calibration.reduce_scores_targets(probabilities, targets)
        objective = self._calibration_objective
        old = self.threshold
        report: dict[str, Any] = {"class": type(self).__name__}
        if objective.targeted:
            best, reached = _calibration.sweep_absolute_target(
                pixel, gt, frame_labels, objective, num_candidates
            )
            report["target"] = objective.target
            report["reached"] = 1.0 if reached else 0.0
        else:
            best = _calibration.sweep_absolute(pixel, gt, num_candidates)
        new = float(best["margin_threshold"])
        counts = _calibration.counts_at_point(
            pixel,
            gt,
            frame_labels,
            np.ones(pixel.shape[0], dtype=bool),
            np.full(pixel.shape[0], np.float32(new)),
        )
        self.threshold = new
        self.hparams["threshold"] = new
        report.update(
            {
                "threshold": {"old": old, "new": new},
                "on_point_threshold": best["threshold"],
                "objective": objective.objective,
                "level": objective.level,
                "f1": counts["f1"],
                "precision": counts["precision"],
                "recall": counts["recall"],
                "pixel_f1": counts["f1"],
                "pixel_precision": counts["precision"],
                "pixel_recall": counts["recall"],
                "pixel_iou": counts["iou"],
                "frame_f1": counts["frame_f1"],
                "frame_precision": counts["frame_precision"],
                "frame_recall": counts["frame_recall"],
                "frame_iou": counts["frame_iou"],
                "frames": int(frame_labels.size),
                "anomalous_frames": int(frame_labels.sum()),
            }
        )
        return report


class QuantileBinaryDecider(BaseDecider):
    """Quantile-based thresholding node operating on BHWC logits or scores.

    This decider computes a tensor-valued threshold per batch item using the
    requested quantile over one or more non-batch dimensions, then produces a
    binary mask where values greater than or equal to that threshold are marked
    as anomalies. Useful for adaptive thresholding when score distributions vary
    across batches.

    Parameters
    ----------
    quantile : float, optional
        Quantile in the closed interval [0, 1] used for the threshold
        computation (default: 0.995). Higher values (e.g., 0.99, 0.995) are
        typical for anomaly detection to capture rare events.
    reduce_dims : Sequence[int] | None, optional
        Axes (relative to the input tensor) over which to compute the quantile.
        When ``None`` (default), all non-batch dimensions (H, W, C) are reduced.
        For per-channel thresholds, use reduce_dims=[1, 2] (reduce H, W only).

    The quantile is also an optional input of shape ``[1]``: a value on the port replaces
    the hparam for that call only, so a GUI slider needs no pipeline reload and the yaml keeps
    the hparam.

    :meth:`calibrate` fits the F1 optimum unless the ``calibration_objective`` hparam names a
    target: ``"precision"`` or ``"recall"`` with ``calibration_target`` in [0, 1]. The node
    flags pixels in every frame, so ``calibration_level`` accepts ``"pixels"`` only. The
    three persist in the pipeline yaml, so a retrain calibrates to the same target.

    Examples
    --------
    >>> from cuvis_ai.node.deciders.binary_decider import QuantileBinaryDecider
    >>> import torch
    >>>
    >>> # Create quantile-based decider (99.5th percentile)
    >>> decider = QuantileBinaryDecider(quantile=0.995)
    >>>
    >>> # Apply to anomaly scores
    >>> scores = torch.randn(4, 256, 256, 1)  # [B, H, W, C]
    >>> output = decider.forward(logits=scores)
    >>> decisions = output["decisions"]  # [4, 256, 256, 1] boolean mask
    >>>
    >>> # Per-channel thresholding (reduce H, W only)
    >>> decider_perchannel = QuantileBinaryDecider(
    ...     quantile=0.99,
    ...     reduce_dims=[1, 2],  # Compute threshold per channel
    ... )

    See Also
    --------
    BinaryDecider : Fixed threshold decisioning
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY, NodeTag.CALIBRATION}
    )

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits or anomaly scores (BHWC format)",
        ),
        "quantile": PortSpec(
            dtype=torch.float32,
            shape=(1,),
            description="Quantile for this call [1], within [0, 1]; replaces the `quantile` "
            "hparam.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(
        self,
        quantile: float = 0.995,
        reduce_dims: Sequence[int] | None = None,
        calibration_objective: str = "f1",
        calibration_target: float | None = None,
        calibration_level: str = "pixels",
        **kwargs,
    ) -> None:
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be within [0.0, 1.0]; received quantile={quantile}")
        self.quantile = float(quantile)
        self.reduce_dims = (
            tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
        )
        self._calibration_objective = _calibration.calibration_objective_of(
            calibration_objective, calibration_target, calibration_level, levels=("pixels",)
        )
        # Forward init params so Serializable records them for config serialization
        super().__init__(
            quantile=self.quantile,
            reduce_dims=self.reduce_dims,
            calibration_objective=self._calibration_objective.objective,
            calibration_target=self._calibration_objective.target,
            calibration_level=self._calibration_objective.level,
            **kwargs,
        )

    def forward(
        self, logits: Tensor, *, quantile: Tensor | None = None, **_: Any
    ) -> dict[str, Tensor]:
        """Apply quantile-based thresholding to produce binary decisions.

        Computes per-batch thresholds using the specified quantile over reduce_dims,
        then classifies values >= threshold as anomalies.

        Parameters
        ----------
        logits : Tensor
            Input logits or anomaly scores, shape (B, H, W, C)
        quantile : Tensor | None
            Quantile for this call, shape [1], within [0, 1]; ``None`` uses the hparam.

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing:

            - "decisions" : Tensor
                Binary decision mask, shape (B, H, W, 1)

        Raises
        ------
        ValueError
            A ``quantile`` input of another shape than [1], or a value that is not finite or
            lies outside [0, 1].
        """
        q = call_threshold("quantile", quantile, self.quantile)
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile input must be within [0, 1], got {q!r}")
        tensor = logits
        dims = resolve_reduce_dims(self.reduce_dims, tensor.dim())

        if len(dims) == 1:
            threshold = torch.quantile(
                tensor,
                q,
                dim=dims[0],
                keepdim=True,
            )
        else:
            dims_to_keep = tuple(i for i in range(tensor.dim()) if i not in dims)
            sizes_keep = [tensor.size(i) for i in dims_to_keep]
            flattened = tensor.permute(*dims_to_keep, *dims).reshape(*sizes_keep, -1)
            broadcast_shape = [1 if i in dims else tensor.size(i) for i in range(tensor.dim())]
            threshold = torch.quantile(flattened, q, dim=-1).reshape(broadcast_shape)

        decisions = (tensor >= threshold).to(torch.bool)
        return {"decisions": decisions}

    def calibrate(
        self,
        scores: Tensor,
        targets: Tensor,
        *,
        num_candidates: int = 256,
    ) -> dict[str, Any]:
        """Refit ``quantile`` on labelled validation scores.

        The node recomputes its cutoff from each frame's own scores, so there is no absolute
        threshold to fit - the sweep finds the best flagged-pixel fraction instead (per frame,
        over the full ``[H, W, C]`` tensor, matching the runtime ``torch.quantile`` with the
        default ``reduce_dims``): the F1-best one by default, or, with a precision or recall
        objective (the ``calibration_objective`` and ``calibration_target`` hparams), the one
        that reaches the target with the best other metric on a grid that reaches both ends
        (:func:`_calibration.select_calibration_row`); a target no quantile on the grid reaches
        lands on the closest one and the report says so. The result is written to the live
        attribute and to ``hparams``; ``pipeline.save_to_file`` then carries it in the pipeline
        yaml (the ``.pt`` weights are unchanged). Decisions are scored on the per-pixel max over
        channels, which matches ``forward`` exactly for single-channel scores.

        The report carries, at the applied quantile and counted as ``forward`` decides, the
        pixel-level ``f1``, ``precision`` and ``recall`` (also as ``pixel_*``), the frame-level
        ``frame_*`` numbers (the same at every quantile: every frame flags at least its
        maximum), ``frames`` and ``anomalous_frames`` and, for a target objective, ``target``
        and ``reached`` (1 or 0).

        Raises:
            CalibrationError: shape mismatch, non-finite scores, a single-class split, or a
                ``reduce_dims`` that keeps a real (size > 1) non-batch axis out of the quantile,
                a decision rule the per-frame sweep cannot reproduce.
        """
        kept = [
            dim
            for dim in range(1, scores.dim())
            if dim not in resolve_reduce_dims(self.reduce_dims, scores.dim())
            and scores.shape[dim] > 1
        ]
        if kept:
            raise _calibration.CalibrationError(
                f"{type(self).__name__} reduce_dims={self.reduce_dims!r} keeps axes {kept} of "
                f"shape {tuple(scores.shape)} out of the quantile; calibration sweeps one "
                "quantile per frame over all non-batch dims and cannot reproduce that rule"
            )
        full, pixel, gt, frame_labels = _calibration.reduce_scores_targets(scores, targets)
        flat = full.reshape(full.shape[0], -1)  # [N, H*W*C] - the runtime quantile space
        objective = self._calibration_objective
        old = self.quantile
        report: dict[str, Any] = {"class": type(self).__name__}
        if objective.targeted:
            grid = _calibration.quantile_target_grid(flat.shape[1], self.quantile, num_candidates)
            per_frame = np.quantile(flat, grid, axis=1)  # [len(grid), N]
            frame_quantiles = {float(q): per_frame[i] for i, q in enumerate(grid)}
            rows = _calibration.quantile_rows(pixel, gt, frame_labels, frame_quantiles)
            best, reached = _calibration.select_calibration_row(rows, objective)
            report["target"] = objective.target
            report["reached"] = 1.0 if reached else 0.0
        else:
            grid = _calibration.quantile_grid_builder(self.quantile, num_candidates)(flat.shape[1])
            per_frame = np.quantile(flat, grid, axis=1)  # [len(grid), N]
            frame_quantiles = {float(q): per_frame[i] for i, q in enumerate(grid)}
            best = _calibration.sweep_quantile(pixel, gt, frame_quantiles)
        new = float(best["quantile"])
        # The cutoffs forward will apply: torch's quantile on each frame's own values.
        cutoffs = torch.quantile(torch.from_numpy(flat), new, dim=1).numpy().astype(np.float32)
        counts = _calibration.counts_at_point(
            pixel, gt, frame_labels, np.ones(pixel.shape[0], dtype=bool), cutoffs
        )
        self.quantile = new
        self.hparams["quantile"] = new
        report.update(
            {
                "quantile": {"old": old, "new": new},
                "objective": objective.objective,
                "level": objective.level,
                "f1": counts["f1"],
                "precision": counts["precision"],
                "recall": counts["recall"],
                "pixel_f1": counts["f1"],
                "pixel_precision": counts["precision"],
                "pixel_recall": counts["recall"],
                "pixel_iou": counts["iou"],
                "frame_f1": counts["frame_f1"],
                "frame_precision": counts["frame_precision"],
                "frame_recall": counts["frame_recall"],
                "frame_iou": counts["frame_iou"],
                "frames": int(frame_labels.size),
                "anomalous_frames": int(frame_labels.sum()),
            }
        )
        return report


_KEEP = ("below", "above")


def _valid_threshold(value: Any) -> float:
    """``value`` as a float when it is a finite, non-negative number; ``ValueError`` otherwise."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"ScoreThreshold: threshold must be a number, got {value!r}.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"ScoreThreshold: threshold must be finite and >= 0, got {value!r}.")
    return float(value)


class ScoreThreshold(Node):
    """Mark the pixels whose raw score lies below or above a threshold (no sigmoid).

    ``keep: "below"`` marks ``scores <= threshold``, for a distance such as a spectral angle to a
    reference; ``keep: "above"`` marks ``scores > threshold``. NaN scores are never marked. The
    optional ``threshold`` input replaces the hparam for one call, so a slider in a GUI needs no
    pipeline reload.

    Parameters
    ----------
    threshold : float
        Finite, non-negative threshold in the unit of ``scores`` (default 0.5).
    keep : str
        ``"below"`` (default) or ``"above"``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel score map [B, H, W, 1], e.g. a spectral angle in radians.",
        ),
        "threshold": PortSpec(
            dtype=torch.float32,
            shape=(1,),
            description="Threshold for this call [1]; replaces the `threshold` hparam.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Pixels with a score at or below (`keep: below`) or above (`keep: above`) "
            "the threshold [B, H, W, 1].",
        ),
    }

    def __init__(self, threshold: float = 0.5, keep: str = "below", **kwargs: Any) -> None:
        if keep not in _KEEP:
            raise ValueError(f"ScoreThreshold: keep must be one of {_KEEP}, got {keep!r}.")
        self.threshold = _valid_threshold(threshold)
        self.keep = keep
        super().__init__(threshold=self.threshold, keep=self.keep, **kwargs)

    @torch.no_grad()
    def forward(
        self, scores: Tensor, threshold: Tensor | None = None, **_: Any
    ) -> dict[str, Tensor]:
        """Return the pixels on the kept side of the threshold."""
        value = self.threshold
        if threshold is not None:
            if tuple(threshold.shape) != (1,):
                raise ValueError(
                    f"ScoreThreshold: the threshold input must have shape [1], got "
                    f"{list(threshold.shape)}."
                )
            value = _valid_threshold(float(threshold.item()))
        if self.keep == "below":
            return {"decisions": scores <= value}
        return {"decisions": scores > value}
