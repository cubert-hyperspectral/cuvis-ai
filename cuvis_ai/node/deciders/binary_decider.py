"""Binary decision nodes for thresholding anomaly scores and logits.

This module provides threshold-based decision nodes that convert continuous anomaly
scores or logits into binary decisions (anomaly/normal). Two strategies are available:

- **BinaryDecider**: Fixed threshold applied globally to sigmoid-transformed logits
- **QuantileBinaryDecider**: Adaptive per-batch thresholding using quantile statistics

Decision nodes are typically placed at the end of anomaly detection pipelines to
convert detector outputs into actionable binary masks for visualization or evaluation.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.deciders.base_decider import BinaryDecider as BaseDecider

from . import _calibration


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
    _tags = frozenset({NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits to threshold (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(self, threshold: float = 0.5, **kwargs) -> None:
        self.threshold = threshold
        # Forward threshold to BaseDecider so Serializable captures it in hparams
        super().__init__(threshold=threshold, **kwargs)

    def forward(
        self,
        logits: Tensor,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply sigmoid and threshold-based decisioning on channels-last data.

        Args:
            logits: Tensor shaped (B, H, W, C) containing logits.

        Returns:
            Dictionary with "decisions" key containing (B, H, W, 1) decision mask.
        """

        # Apply sigmoid if needed to convert logits to probabilities
        tensor = torch.sigmoid(logits)

        # Apply threshold to get binary decisions
        decisions = tensor >= self.threshold
        return {"decisions": decisions}

    def calibrate(
        self, scores: Tensor, targets: Tensor, *, num_candidates: int = 256
    ) -> dict[str, Any]:
        """Refit ``threshold`` to F1-max on labelled validation scores.

        ``scores`` is the decider-input tensor stacked over the split (``[N, H, W, C]``);
        ``targets`` the matching ground-truth mask. The sweep runs on ``torch.sigmoid(scores)``
        in float32, exactly what ``forward`` compares against ``threshold``, so the fitted
        value is reachable at runtime even when raw logits saturate the sigmoid. The optimum
        is moved to the midpoint of its F1 plateau (``_calibration.margin_below``) and written
        to the live attribute and to ``hparams``; ``pipeline.save_to_file`` then carries it in
        the pipeline yaml. The ``.pt`` weights are unchanged, so load the saved yaml rather
        than the preset plus weights. Multi-channel input is reduced to the per-pixel max, which
        matches ``forward`` exactly for single-channel scores. Returns a report.

        Raises:
            CalibrationError: shape mismatch, non-finite scores, or a single-class split.
        """
        probabilities = torch.sigmoid(scores.detach().to("cpu", torch.float32))
        _, pixel, gt, _ = _calibration.reduce_scores_targets(probabilities, targets)
        best = _calibration.sweep_absolute(pixel, gt, num_candidates)
        old = self.threshold
        new = float(best["margin_threshold"])
        self.threshold = new
        self.hparams["threshold"] = new
        return {
            "class": type(self).__name__,
            "threshold": {"old": old, "new": new},
            "on_point_threshold": best["threshold"],
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
        }


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
    _tags = frozenset({NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits or anomaly scores (BHWC format)",
        )
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
        **kwargs,
    ) -> None:
        self._validate_quantile(quantile)
        self.quantile = float(quantile)
        self.reduce_dims = (
            tuple(int(dim) for dim in reduce_dims) if reduce_dims is not None else None
        )
        # Forward init params so Serializable records them for config serialization
        super().__init__(quantile=self.quantile, reduce_dims=self.reduce_dims, **kwargs)

    def forward(self, logits: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply quantile-based thresholding to produce binary decisions.

        Computes per-batch thresholds using the specified quantile over reduce_dims,
        then classifies values >= threshold as anomalies.

        Parameters
        ----------
        logits : Tensor
            Input logits or anomaly scores, shape (B, H, W, C)

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing:

            - "decisions" : Tensor
                Binary decision mask, shape (B, H, W, 1)
        """
        tensor = logits
        dims = resolve_reduce_dims(self.reduce_dims, tensor.dim())

        if len(dims) == 1:
            threshold = torch.quantile(
                tensor,
                self.quantile,
                dim=dims[0],
                keepdim=True,
            )
        else:
            tensor_ndim = tensor.dim()
            dims_to_keep = tuple(i for i in range(tensor_ndim) if i not in dims)
            new_order = (*dims_to_keep, *dims)
            permuted = tensor.permute(new_order)
            sizes_keep = [permuted.size(i) for i in range(len(dims_to_keep))]
            flattened = permuted.reshape(*sizes_keep, -1)
            threshold_flat = torch.quantile(
                flattened,
                self.quantile,
                dim=len(dims_to_keep),
                keepdim=True,
            )
            threshold_permuted = threshold_flat.reshape(
                *sizes_keep,
                *([1] * len(dims)),
            )
            inverse_order = [0] * tensor_ndim
            for original_idx, permuted_idx in enumerate(new_order):
                inverse_order[permuted_idx] = original_idx
            threshold = threshold_permuted.permute(*inverse_order)

        decisions = (tensor >= threshold).to(torch.bool)
        return {"decisions": decisions}

    def calibrate(
        self, scores: Tensor, targets: Tensor, *, num_candidates: int = 256
    ) -> dict[str, Any]:
        """Refit ``quantile`` to F1-max on labelled validation scores.

        The node recomputes its cutoff from each frame's own scores, so there is no absolute
        threshold to fit - the sweep finds the F1-best flagged-pixel fraction instead (per
        frame, over the full ``[H, W, C]`` tensor, matching the runtime ``torch.quantile``
        with the default ``reduce_dims``). The result is written to the live attribute and to
        ``hparams``; ``pipeline.save_to_file`` then carries it in the pipeline yaml (the ``.pt``
        weights are unchanged). Decisions are scored on the per-pixel max over channels, which
        matches ``forward`` exactly for single-channel scores.

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
        full, pixel, gt, _ = _calibration.reduce_scores_targets(scores, targets)
        flat = full.reshape(full.shape[0], -1)  # [N, H*W*C] - the runtime quantile space
        grid = _calibration.quantile_grid_builder(self.quantile, num_candidates)(flat.shape[1])
        per_frame = np.quantile(flat, grid, axis=1)  # [len(grid), N]
        frame_quantiles = {float(q): per_frame[i] for i, q in enumerate(grid)}
        best = _calibration.sweep_quantile(pixel, gt, frame_quantiles)
        old = self.quantile
        new = float(best["quantile"])
        self.quantile = new
        self.hparams["quantile"] = new
        return {
            "class": type(self).__name__,
            "quantile": {"old": old, "new": new},
            "f1": best["f1"],
            "precision": best["precision"],
            "recall": best["recall"],
        }

    @staticmethod
    def _validate_quantile(quantile: float) -> None:
        """Validate that quantile is in the valid range [0, 1].

        Parameters
        ----------
        quantile : float
            Quantile value to validate

        Raises
        ------
        ValueError
            If quantile is outside the valid range [0, 1]
        """
        if not 0.0 <= quantile <= 1.0:
            raise ValueError(f"Quantile must be within [0.0, 1.0]; received quantile={quantile}")
