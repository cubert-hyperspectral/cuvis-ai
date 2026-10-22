"""Metric nodes for training pipeline (port-based architecture)."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context, Metric
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor
from torchmetrics import Metric as TorchMetric
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from cuvis_ai_core.node.metric_utils import subsample_hw, warn_below_vectorized_cutoff
from cuvis_ai_core.node.node import Node


class ExplainedVarianceMetric(Node):
    """Track explained variance ratio for PCA components.

    Executes only during validation and test stages.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.DIM_REDUCTION})

    INPUT_SPECS = {
        "explained_variance_ratio": PortSpec(
            dtype=torch.float32, shape=(-1,), description="Explained variance ratio from PCA node"
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, explained_variance_ratio: Tensor, context: Context) -> dict[str, Any]:
        """Compute explained variance metrics.

        Parameters
        ----------
        explained_variance_ratio : Tensor
            Explained variance ratios from PCA node
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        named_values = []

        # Per-component variance
        for i, ratio in enumerate(explained_variance_ratio):
            named_values.append((f"explained_variance_pc{i + 1}", ratio.item()))

        # Total variance explained
        named_values.append(("total_explained_variance", explained_variance_ratio.sum().item()))

        # Cumulative variance
        cumulative = torch.cumsum(explained_variance_ratio, dim=0)
        for i, cum_var in enumerate(cumulative):
            named_values.append((f"cumulative_variance_pc{i + 1}", cum_var.item()))

        metrics = [
            Metric(
                name=name,
                value=value,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for name, value in named_values
        ]

        return {"metrics": metrics}


class AnomalyDetectionMetrics(Node):
    """Compute anomaly detection metrics (precision, recall, F1, etc.).

    Uses torchmetrics for GPU-optimized, robust metric computation.
    Expects binary decisions and targets to be binary masks.
    Executes only during validation and test stages.

    ``precision``, ``recall``, ``f1_score`` and ``iou`` are dataset-level pixel metrics:
    stateful torchmetrics objects that accumulate with ``update()`` over every batch of
    the split and are reduced once with a single ``compute()``, reset only at the
    ``(stage, epoch)`` boundary. The value on the ``metrics`` port each batch is therefore
    a *running* value over the batches seen so far in the current epoch, not that batch's
    own value; the last batch of the epoch carries the true epoch-level number. This
    matches ``average_precision``, which was already pooled the same way.

    ``image_precision``, ``image_recall`` and ``image_f1`` are frame-level: a frame counts
    as predicted anomalous when any of its decision pixels is true, and as anomalous when
    any of its target pixels is true, pooled the same way over every frame of the epoch.
    They are computed on the full-resolution ``decisions``/``targets`` before
    ``pixel_stride`` subsampling runs, so a small anomaly that falls between the retained
    pixels still marks its frame anomalous.

    ``decisions`` is optional: without it (an uncalibrated decider upstream, outside
    inference) only ``average_precision`` is emitted, from ``logits`` alone, and the node
    does not fail.

    ``pixel_stride`` subsamples the pixel grid on H and W before the pixel-level metrics
    see it, so a full-frame validation step no longer allocates flattened copies of
    every pixel. It changes only how many pixels those metrics score, never the
    ``scores`` grid a decider, overlay or heatmap reads, and never the frame-level
    image_* reduction above.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
            optional=True,
        ),
        "targets": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground truth binary masks [B, H, W, 1]",
        ),
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Optional anomaly logits/probabilities for AP",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    # Every name here accumulates across the epoch and must be reduced by a single
    # pooled compute() at epoch end, not by averaging the per-batch running values.
    # The trainer skips these names in per-batch logging and instead logs the live
    # torchmetrics object from pooled_metrics() with on_epoch=True, so Lightning does
    # the pooled compute()+reset() natively.
    POOLED_METRIC_NAMES: ClassVar[frozenset[str]] = frozenset(
        {
            "precision",
            "recall",
            "f1_score",
            "iou",
            "average_precision",
            "image_precision",
            "image_recall",
            "image_f1",
        }
    )

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, ap_thresholds: int = 200, pixel_stride: int = 1, **kwargs) -> None:
        if isinstance(pixel_stride, bool) or not isinstance(pixel_stride, int):
            raise ValueError(f"pixel_stride must be an int >= 1, got {type(pixel_stride).__name__}")
        if pixel_stride < 1:
            raise ValueError(f"pixel_stride must be an int >= 1, got {pixel_stride}")

        self.ap_thresholds = ap_thresholds
        self.pixel_stride = pixel_stride
        super().__init__(ap_thresholds=ap_thresholds, pixel_stride=pixel_stride, **kwargs)

        # Every metric below keeps O(1) running confmat state (full_state_update=False
        # on the pixel ones) except BinaryAveragePrecision, which uses histogram-based
        # AP (thresholds=N) for O(N) state instead of O(n_pixels). All of them accumulate
        # via update() across batches within a (stage, epoch) and reset only at the
        # boundary, so the value emitted each batch is a *running* value across batches
        # seen so far in the current epoch — the last batch's value is the true
        # epoch-level number. validate_args=False skips torchmetrics' per-call input
        # check, which sorts the whole input to prove it is binary, pure overhead for
        # ports the pipeline already types as bool/float32.
        self.precision_metric = BinaryPrecision(validate_args=False)
        self.recall_metric = BinaryRecall(validate_args=False)
        self.f1_metric = BinaryF1Score(validate_args=False)
        self.iou_metric = BinaryJaccardIndex(validate_args=False)
        self.average_precision_metric = BinaryAveragePrecision(
            thresholds=ap_thresholds, validate_args=False
        )
        self.image_precision_metric = BinaryPrecision(validate_args=False)
        self.image_recall_metric = BinaryRecall(validate_args=False)
        self.image_f1_metric = BinaryF1Score(validate_args=False)
        self._ap_last_key: tuple[ExecutionStage, int] | None = None
        self._pixel_last_key: tuple[ExecutionStage, int] | None = None
        self._image_last_key: tuple[ExecutionStage, int] | None = None
        self._stride_warn_state: dict[str, bool] = {}

    def forward(
        self,
        targets: Tensor,
        context: Context,
        decisions: Tensor | None = None,
        logits: Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute anomaly detection metrics using torchmetrics.

        With ``pixel_stride`` above 1 the pixel-level grids are subsampled to every
        s-th row and column first, so those metrics are computed on that subsample; the
        frame-level ``image_*`` metrics always see the full-resolution tensors.

        The two optional ports default to ``None`` because the pipeline passes no
        argument at all for an optional port nothing feeds.

        Parameters
        ----------
        targets : Tensor
            Ground truth binary masks [B, H, W, 1]
        context : Context
            Execution context with stage, epoch, batch_idx
        decisions : Tensor | None
            Binary anomaly decisions [B, H, W, 1]; absent while an uncalibrated
            decider upstream withholds them, in which case only ``average_precision``
            (when ``logits`` is given) is emitted.
        logits : Tensor | None
            Optional anomaly logits/probabilities for ``average_precision``.

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        values: dict[str, float] = {}
        epoch_key = (context.stage, context.epoch)

        if decisions is not None:
            # Frame-level reduction first, on the full-resolution masks: a stride
            # subsample must never decide whether a frame counts as anomalous.
            frame_pred = decisions.flatten(1).any(dim=1).to(torch.int64)
            frame_gt = targets.flatten(1).any(dim=1).to(torch.int64)
            if self._image_last_key != epoch_key:
                self.image_precision_metric.reset()
                self.image_recall_metric.reset()
                self.image_f1_metric.reset()
                self._image_last_key = epoch_key
            values["image_precision"] = self.image_precision_metric(frame_pred, frame_gt).item()
            values["image_recall"] = self.image_recall_metric(frame_pred, frame_gt).item()
            values["image_f1"] = self.image_f1_metric(frame_pred, frame_gt).item()

            # Ensure consistent shapes and flatten spatial dimensions
            pixel_decisions = decisions.squeeze(-1)  # [B, H, W]
            pixel_targets = targets.squeeze(-1)  # [B, H, W]

            # Subsample the pixel grid before anything is flattened: every copy below
            # (and the AP sigmoid) then costs ceil(H/s)*ceil(W/s) per frame instead of H*W.
            full_elements = pixel_decisions.numel()
            pixel_decisions = subsample_hw(pixel_decisions, self.pixel_stride)
            pixel_targets = subsample_hw(pixel_targets, self.pixel_stride)
            warn_below_vectorized_cutoff(
                self.name, pixel_decisions.numel(), full_elements, self._stride_warn_state
            )

            # Flatten to [N] where N = B*ceil(H/s)*ceil(W/s) for torchmetrics
            preds_flat = pixel_decisions.flatten()
            targets_flat = pixel_targets.flatten()

            if self._pixel_last_key != epoch_key:
                self.precision_metric.reset()
                self.recall_metric.reset()
                self.f1_metric.reset()
                self.iou_metric.reset()
                self._pixel_last_key = epoch_key

            values["precision"] = self.precision_metric(preds_flat, targets_flat).item()
            values["recall"] = self.recall_metric(preds_flat, targets_flat).item()
            values["f1_score"] = self.f1_metric(preds_flat, targets_flat).item()
            values["iou"] = self.iou_metric(preds_flat, targets_flat).item()
        else:
            targets_flat = subsample_hw(targets.squeeze(-1), self.pixel_stride).flatten()

        if logits is not None:
            raw_scores = subsample_hw(logits.squeeze(-1), self.pixel_stride).flatten().float()
            probs_for_ap = torch.sigmoid(raw_scores)

            if self._ap_last_key != epoch_key:
                self.average_precision_metric.reset()
                self._ap_last_key = epoch_key

            self.average_precision_metric.update(probs_for_ap, targets_flat)
            values["average_precision"] = self.average_precision_metric.compute().item()

        metrics = [
            Metric(
                name=name,
                value=value,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for name, value in values.items()
        ]

        return {"metrics": metrics}

    def pooled_metrics(self) -> dict[str, TorchMetric]:
        """Live torchmetrics objects for the epoch-pooled metrics, keyed by name.

        Every metric named in :data:`POOLED_METRIC_NAMES` accumulates across the epoch
        (reset only at the ``(stage, epoch)`` boundary), so the trainer logs each live
        object with ``on_epoch=True`` and Lightning computes a single pooled value and
        resets it at epoch end, exact and batch-size-invariant. A metric is present
        only once it has seen at least one batch: the pixel and frame-level ones once a
        batch with ``decisions`` has been seen, ``average_precision`` once a batch with
        ``logits`` has been seen. Returns an empty mapping until then, so nothing is
        logged for a run that never produced the matching input.
        """
        pooled: dict[str, TorchMetric] = {}
        if self._pixel_last_key is not None:
            pooled["precision"] = self.precision_metric
            pooled["recall"] = self.recall_metric
            pooled["f1_score"] = self.f1_metric
            pooled["iou"] = self.iou_metric
        if self._image_last_key is not None:
            pooled["image_precision"] = self.image_precision_metric
            pooled["image_recall"] = self.image_recall_metric
            pooled["image_f1"] = self.image_f1_metric
        if self._ap_last_key is not None:
            pooled["average_precision"] = self.average_precision_metric
        return pooled


class ScoreStatisticsMetric(Node):
    """Compute statistical properties of score distributions.

    Tracks mean, std, min, max, median, and quantiles of scores.
    Executes only during validation and test stages.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1), description="Score values [B, H, W]"
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, scores: Tensor, context: Context) -> dict[str, Any]:
        """Compute score statistics.

        Parameters
        ----------
        scores : Tensor
            Score values [B, H, W]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Flatten scores
        scores_flat = scores.reshape(-1)

        values = {
            "scores/mean": scores_flat.mean().item(),
            "scores/std": scores_flat.std().item(),
            "scores/min": scores_flat.min().item(),
            "scores/max": scores_flat.max().item(),
            "scores/median": scores_flat.median().item(),
            "scores/q25": torch.quantile(scores_flat, 0.25).item(),
            "scores/q75": torch.quantile(scores_flat, 0.75).item(),
            "scores/q95": torch.quantile(scores_flat, 0.95).item(),
            "scores/q99": torch.quantile(scores_flat, 0.99).item(),
        }
        metrics = [
            Metric(
                name=name,
                value=value,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for name, value in values.items()
        ]

        return {"metrics": metrics}


class ComponentOrthogonalityMetric(Node):
    """Track orthogonality of PCA components during training.

    Measures how close the component matrix is to being orthonormal.
    Executes only during validation and test stages.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.DIM_REDUCTION})

    INPUT_SPECS = {
        "components": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="PCA components matrix [n_components, n_features]",
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, components: Tensor, context: Context) -> dict[str, Any]:
        """Compute component orthogonality metrics.

        Parameters
        ----------
        components : Tensor
            PCA components matrix [n_components, n_features]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Compute gram matrix: W @ W.T
        gram = components @ components.T
        n = components.shape[0]

        # Target: identity matrix
        eye = torch.eye(n, device=components.device, dtype=components.dtype)

        # Frobenius norm of difference
        orth_error = torch.norm(gram - eye, p="fro").item()

        # Average absolute deviation from identity
        avg_off_diagonal = (gram - eye).abs().mean().item()

        # Diagonal elements (should be close to 1)
        diagonal = torch.diagonal(gram)
        diagonal_mean = diagonal.mean().item()
        diagonal_std = diagonal.std().item()

        values = {
            "orthogonality_error": orth_error,
            "avg_off_diagonal": avg_off_diagonal,
            "diagonal_mean": diagonal_mean,
            "diagonal_std": diagonal_std,
        }
        metrics = [
            Metric(
                name=name,
                value=value,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for name, value in values.items()
        ]

        return {"metrics": metrics}


class SelectorEntropyMetric(Node):
    """Track entropy of channel selection distribution.

    Measures the uncertainty/diversity in channel selection weights.
    Higher entropy indicates more uniform selection (less confident).
    Lower entropy indicates more peaked selection (more confident).

    Executes only during validation and test stages.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.DIM_REDUCTION})

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights from selector node",
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, eps: float = 1e-6, **kwargs) -> None:
        self.eps = eps
        super().__init__(eps=eps, **kwargs)

    def forward(self, weights: Tensor, context: Context) -> dict[str, Any]:
        """Compute entropy of selection weights.

        Parameters
        ----------
        weights : Tensor
            Channel selection weights [n_channels]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Normalize weights to probabilities
        probs = weights / (weights.sum() + self.eps)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + self.eps)).sum()

        metrics = [
            Metric(
                name="selector/entropy",
                value=entropy.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
        ]

        return {"metrics": metrics}


class SelectorDiversityMetric(Node):
    """Track diversity of channel selection.

    Measures how spread out the selection weights are across channels.
    Uses Gini coefficient - lower values indicate more diverse selection.

    Executes only during validation and test stages.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.DIM_REDUCTION})

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights from selector node",
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, weights: Tensor, context: Context) -> dict[str, Any]:
        """Compute diversity metrics for selection weights.

        Parameters
        ----------
        weights : Tensor
            Channel selection weights [n_channels]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Compute variance (measure of spread)
        mean_weight = weights.mean()
        variance = ((weights - mean_weight) ** 2).mean()

        # Compute Gini coefficient (0 = perfect equality, 1 = perfect inequality)
        # Lower Gini = more diverse selection
        sorted_weights, _ = torch.sort(weights)
        n = len(sorted_weights)
        index = torch.arange(1, n + 1, device=weights.device, dtype=weights.dtype)
        gini = (2 * (sorted_weights * index).sum()) / (n * sorted_weights.sum()) - (n + 1) / n

        metrics = [
            Metric(
                name="weight_variance",
                value=variance.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="gini_coefficient",
                value=gini.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
        ]

        return {"metrics": metrics}


class AnomalyPixelStatisticsMetric(Node):
    """Compute anomaly pixel statistics from binary decisions.

    Calculates total pixels, anomalous pixels count, and anomaly percentage.
    Useful for monitoring the proportion of detected anomalies in batches.
    Executes only during validation and test stages. ``decisions`` is optional: an
    uncalibrated decider upstream emits none outside inference, and this node then
    emits no metrics at all rather than failing.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, context: Context, decisions: Tensor | None = None) -> dict[str, Any]:
        """Compute anomaly pixel statistics.

        Parameters
        ----------
        context : Context
            Execution context with stage, epoch, batch_idx
        decisions : Tensor | None
            Binary anomaly decisions [B, H, W, 1]; ``None`` while an uncalibrated decider
            upstream withholds them, in which case no metrics are emitted.

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects, empty
            without ``decisions``.
        """
        if decisions is None:
            return {"metrics": []}

        # Calculate statistics
        total_pixels = decisions.numel()
        anomalous_pixels = int(decisions.sum().item())
        anomaly_percentage = (anomalous_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        values = {
            "anomaly/total_pixels": float(total_pixels),
            "anomaly/anomalous_pixels": float(anomalous_pixels),
            "anomaly/anomaly_percentage": anomaly_percentage,
        }
        metrics = [
            Metric(
                name=name,
                value=value,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for name, value in values.items()
        ]

        return {"metrics": metrics}


class DistinctLabelCount(Node):
    """Count the distinct non-zero labels per frame in an integer label map.

    Reports how many separate segments a label map contains, e.g. how many compartments survived
    a per-blob majority vote or how many clusters a frame holds. Emits the per-frame count both as
    a ``count`` tensor (for pipeline reads / notebook printing) and as ``Metric`` objects for
    training-time logging. Declares ``ExecutionStage.ALWAYS`` (and the ``INFERENCE`` tag) so it
    also runs under ``Predictor`` inference, not only validation / test.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.INFERENCE, NodeTag.MASK})
    EXECUTION_STAGES = {ExecutionStage.ALWAYS}

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Integer label map [B, H, W]; 0 = background.",
        ),
    }
    OUTPUT_SPECS = {
        "count": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Distinct non-zero label count per frame [B].",
        ),
        "metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects"),
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, mask: Tensor, context: Context) -> dict[str, Any]:
        """Count distinct non-zero labels in each frame of *mask*.

        Parameters
        ----------
        mask : Tensor
            Integer label map [B, H, W]; 0 is background.
        context : Context
            Execution context with stage, epoch, batch_idx.

        Returns
        -------
        dict[str, Any]
            ``count`` [B] int64 and a ``metrics`` list with one ``num_distinct_labels`` per frame.
        """
        counts = [int((torch.unique(mask[b]) != 0).sum().item()) for b in range(mask.shape[0])]
        count = torch.tensor(counts, dtype=torch.int64, device=mask.device)
        metrics = [
            Metric(
                name="num_distinct_labels",
                value=float(c),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
            for c in counts
        ]
        return {"count": count, "metrics": metrics}


__all__ = [
    "ExplainedVarianceMetric",
    "AnomalyDetectionMetrics",
    "ScoreStatisticsMetric",
    "ComponentOrthogonalityMetric",
    "SelectorEntropyMetric",
    "SelectorDiversityMetric",
    "AnomalyPixelStatisticsMetric",
    "DistinctLabelCount",
]
