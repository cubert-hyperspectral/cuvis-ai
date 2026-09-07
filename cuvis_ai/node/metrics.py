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
    BinaryAUROC,
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
        metrics = []

        # Per-component variance
        for i, ratio in enumerate(explained_variance_ratio):
            metrics.append(
                Metric(
                    name=f"explained_variance_pc{i + 1}",
                    value=ratio.item(),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                )
            )

        # Total variance explained
        metrics.append(
            Metric(
                name="total_explained_variance",
                value=explained_variance_ratio.sum().item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            )
        )

        # Cumulative variance
        cumulative = torch.cumsum(explained_variance_ratio, dim=0)
        for i, cum_var in enumerate(cumulative):
            metrics.append(
                Metric(
                    name=f"cumulative_variance_pc{i + 1}",
                    value=cum_var.item(),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                )
            )

        return {"metrics": metrics}


class AnomalyDetectionMetrics(Node):
    """Compute anomaly detection metrics (precision, recall, F1, etc.).

    Uses torchmetrics for GPU-optimized, robust metric computation.
    Expects binary decisions and targets to be binary masks.
    Executes only during validation and test stages.

    ``pixel_stride`` subsamples the pixel grid on H and W before the metrics see
    it, so a full-frame validation step no longer allocates flattened copies of
    every pixel. It changes only how many pixels the metrics score, never the
    ``scores`` grid a decider, overlay or heatmap reads.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
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

    # average_precision accumulates across the epoch and must be reduced by a
    # single pooled compute() at epoch end, not by averaging the per-batch
    # running values. The trainer skips these names in per-batch logging and
    # instead logs the live torchmetrics object from pooled_metrics() with
    # on_epoch=True, so Lightning does the pooled compute()+reset() natively.
    # Precision/recall/F1/IoU stay per-batch (mean reduction is their intended
    # epoch value).
    POOLED_METRIC_NAMES: ClassVar[frozenset[str]] = frozenset({"average_precision"})

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, ap_thresholds: int = 200, pixel_stride: int = 1, **kwargs) -> None:
        if isinstance(pixel_stride, bool) or not isinstance(pixel_stride, int):
            raise ValueError(f"pixel_stride must be an int >= 1, got {type(pixel_stride).__name__}")
        if pixel_stride < 1:
            raise ValueError(f"pixel_stride must be an int >= 1, got {pixel_stride}")

        self.ap_thresholds = ap_thresholds
        self.pixel_stride = pixel_stride
        super().__init__(ap_thresholds=ap_thresholds, pixel_stride=pixel_stride, **kwargs)

        # Precision/Recall/F1/IoU keep O(1) running confmat state and are stateless
        # under torchmetrics __call__ (full_state_update=False) — per-batch values.
        # BinaryAveragePrecision uses histogram-based AP (thresholds=N) so state is
        # O(N) instead of O(n_pixels). We accumulate via update() across batches
        # within a (stage, epoch) and reset only at the boundary, so the value
        # emitted each batch is a *running* AP across batches seen so far in the
        # current epoch — the last batch's value is true epoch-level AP.
        # validate_args=False skips torchmetrics' per-call input check, which sorts
        # the whole input to prove it is binary, pure overhead for ports the
        # pipeline already types as bool/float32.
        self.precision_metric = BinaryPrecision(validate_args=False)
        self.recall_metric = BinaryRecall(validate_args=False)
        self.f1_metric = BinaryF1Score(validate_args=False)
        self.iou_metric = BinaryJaccardIndex(validate_args=False)
        self.average_precision_metric = BinaryAveragePrecision(
            thresholds=ap_thresholds, validate_args=False
        )
        self._ap_last_key: tuple[ExecutionStage, int] | None = None
        self._stride_warn_state: dict[str, bool] = {}

    def forward(
        self,
        decisions: Tensor,
        targets: Tensor,
        context: Context,
        logits: Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute anomaly detection metrics using torchmetrics.

        With ``pixel_stride`` above 1 the three pixel grids are subsampled to
        every s-th row and column first, so the reported values are computed on
        that subsample.

        Parameters
        ----------
        decisions : Tensor
            Binary anomaly decisions [B, H, W, 1]
        targets : Tensor
            Ground truth binary masks [B, H, W, 1]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Ensure consistent shapes and flatten spatial dimensions
        decisions = decisions.squeeze(-1)  # [B, H, W]
        targets = targets.squeeze(-1)  # [B, H, W]

        # Subsample the pixel grid before anything is flattened: every copy below
        # (and the AP sigmoid) then costs ceil(H/s)*ceil(W/s) per frame instead of H*W.
        full_elements = decisions.numel()
        decisions = subsample_hw(decisions, self.pixel_stride)
        targets = subsample_hw(targets, self.pixel_stride)
        warn_below_vectorized_cutoff(
            self.name, decisions.numel(), full_elements, self._stride_warn_state
        )

        # Flatten to [N] where N = B*ceil(H/s)*ceil(W/s) for torchmetrics
        preds_flat = decisions.flatten()
        targets_flat = targets.flatten()

        # Compute metrics using torchmetrics (they handle edge cases robustly)
        precision = self.precision_metric(preds_flat, targets_flat)
        recall = self.recall_metric(preds_flat, targets_flat)
        f1 = self.f1_metric(preds_flat, targets_flat)
        iou = self.iou_metric(preds_flat, targets_flat)

        metrics = [
            Metric(
                name="precision",
                value=precision.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="recall",
                value=recall.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="f1_score",
                value=f1.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="iou",
                value=iou.item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
        ]

        if logits is not None:
            raw_scores = subsample_hw(logits.squeeze(-1), self.pixel_stride).flatten().float()
            probs_for_ap = torch.sigmoid(raw_scores)

            current_key = (context.stage, context.epoch)
            if self._ap_last_key != current_key:
                self.average_precision_metric.reset()
                self._ap_last_key = current_key

            self.average_precision_metric.update(probs_for_ap, targets_flat)
            average_precision = self.average_precision_metric.compute()

            metrics.append(
                Metric(
                    name="average_precision",
                    value=average_precision.item(),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                )
            )

        return {"metrics": metrics}

    def pooled_metrics(self) -> dict[str, TorchMetric]:
        """Live torchmetrics objects for the epoch-pooled metrics, keyed by name.

        ``average_precision`` accumulates across the epoch (reset only at the
        ``(stage, epoch)`` boundary), so the trainer logs this object with
        ``on_epoch=True`` and Lightning computes the single pooled AP and resets
        it at epoch end, exact and batch-size-invariant. Returns an empty mapping
        until at least one batch with ``logits`` has been seen, so nothing is
        logged for a run that never produced scores.
        """
        if self._ap_last_key is None:
            return {}
        return {"average_precision": self.average_precision_metric}


class AnomalyAUROCMetrics(Node):
    """Streaming pixel-AUROC + image-AUROC for anomaly detection (validation / test).

    Mirrors :class:`AnomalyDetectionMetrics`: a ``torchmetrics`` ``BinaryAUROC`` with
    histogram ``thresholds`` (so per-epoch state is O(thresholds), not the couple-GB-per-epoch
    CPU concat of every pixel) is accumulated via ``update()`` across batches and reset on the
    ``(stage, epoch)`` boundary. Each forward emits the *running* AUROC as a
    :class:`~cuvis_ai_schemas.execution.Metric`, so no bespoke Lightning callback is needed.

    The per-batch ``Metric.value`` emitted by ``forward`` is a running AUROC — a
    batch-size-sensitive approximation if mean-reduced over the epoch. The authoritative epoch
    value comes from :meth:`pooled_metrics`: the node lists ``auroc_pixel`` / ``auroc_image`` in
    ``POOLED_METRIC_NAMES``, so the trainer skips their per-batch float logging and instead logs
    the live ``BinaryAUROC`` objects with ``on_epoch=True``, and Lightning does one pooled
    ``compute()`` + ``reset()`` at epoch end — exact and batch-size-invariant. The per-batch
    values remain on the ``metrics`` port for live monitoring (e.g. the TensorBoard node).

    Scores pass through ``sigmoid`` before the binned metric so the thresholds span ``[0, 1]``;
    AUROC is rank-invariant under a monotonic transform, so the value is unchanged.

    ``pixel_stride`` trades a little pixel-AUROC resolution for the transient memory the per-step
    update costs: the score map and the mask are subsampled on H and W before they are flattened,
    so torchmetrics sees ``ceil(H / s) * ceil(W / s) * B`` elements instead of the full frame. It
    defaults to 1 (no subsampling, byte-identical to before). The image-level pair is never
    subsampled — the per-image label is read off the full-resolution mask, so a single anomalous
    pixel the stride skips still marks the frame anomalous.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    # auroc_pixel / auroc_image accumulate across the whole epoch and must be reduced by a single
    # pooled compute() at epoch end, not by averaging the per-batch running values (badly biased
    # at batch_size=1). The trainer skips these names in per-batch logging and instead logs the
    # live torchmetrics objects from pooled_metrics() with on_epoch=True, so Lightning does the
    # pooled compute()+reset() natively. Mirrors AnomalyDetectionMetrics.average_precision.
    POOLED_METRIC_NAMES: ClassVar[frozenset[str]] = frozenset({"auroc_pixel", "auroc_image"})

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Raw anomaly map [B, H, W, 1]",
        ),
        "targets": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground-truth pixel masks [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Per-image anomaly score [B]",
        ),
    }
    OUTPUT_SPECS = {
        "metrics": PortSpec(
            dtype=list, shape=(), description="List of Metric objects (running AUROC)"
        ),
    }

    def __init__(self, thresholds: int = 200, pixel_stride: int = 1, **kwargs: Any) -> None:
        if isinstance(thresholds, bool) or not isinstance(thresholds, int) or thresholds < 2:
            raise ValueError(
                "AnomalyAUROCMetrics: thresholds must be an int >= 2 (histogram bins of the "
                f"binned AUROC), got {thresholds!r}."
            )
        if isinstance(pixel_stride, bool) or not isinstance(pixel_stride, int) or pixel_stride < 1:
            raise ValueError(
                "AnomalyAUROCMetrics: pixel_stride must be an int >= 1 (1 = no subsampling), "
                f"got {pixel_stride!r}."
            )
        self.thresholds = thresholds
        # thresholds / pixel_stride ride into hparams so they survive a pipeline save/restore.
        super().__init__(thresholds=thresholds, pixel_stride=pixel_stride, **kwargs)
        self.pixel_stride = pixel_stride
        self._stride_warn_state: dict[str, Any] = {}
        # Histogram-based AUROC: O(thresholds) state, accumulated across batches and reset only at
        # the (stage, epoch) boundary. forward() emits the running value per batch; the pooled
        # epoch value is logged via pooled_metrics() (see POOLED_METRIC_NAMES). validate_args=False
        # drops torchmetrics' per-update unique()/sort over every pixel — the ports already
        # guarantee float scores and a bool mask.
        self.pixel_auroc = BinaryAUROC(thresholds=thresholds, validate_args=False)
        self.image_auroc = BinaryAUROC(thresholds=thresholds, validate_args=False)
        self._last_key: tuple[ExecutionStage, int] | None = None

    @staticmethod
    def _binned_preds(scores: Tensor) -> Tensor:
        """Flatten a raw anomaly map and sigmoid it onto ``[0, 1]`` for the binned metric."""
        return torch.sigmoid(scores.flatten().float())

    def reset(self) -> None:
        """Reset both AUROC accumulators (called by the Predictor before a run, and by tests)."""
        self.pixel_auroc.reset()
        self.image_auroc.reset()
        self._last_key = None

    def forward(
        self,
        scores: Tensor,
        targets: Tensor,
        anomaly_score: Tensor,
        context: Context,
    ) -> dict[str, Any]:
        """Accumulate this batch into the pixel/image AUROC and emit the running values.

        Parameters
        ----------
        scores : Tensor
            Raw anomaly map [B, H, W, 1] (not thresholded).
        targets : Tensor
            Ground-truth pixel masks [B, H, W, 1] (bool).
        anomaly_score : Tensor
            Per-image anomaly score [B].
        context : Context
            Execution context with stage, epoch, batch_idx.

        Returns
        -------
        dict[str, Any]
            ``metrics`` — running ``auroc_pixel`` / ``auroc_image`` for this batch.
        """
        # Reset on the (stage, epoch) boundary so each epoch accumulates fresh.
        key = (context.stage, context.epoch)
        if self._last_key != key:
            self.pixel_auroc.reset()
            self.image_auroc.reset()
            self._last_key = key

        # Image-level label first, off the FULL-resolution mask: any positive pixel makes the
        # frame anomalous, including one that pixel_stride skips.
        img_tgts = targets.squeeze(-1).flatten(1).any(dim=1)

        # Pixel-level: subsample H/W before the flatten and the sigmoid, so both the copy and
        # torchmetrics' per-update confusion matrix scale with the stride. Targets stay bool
        # (no int64 promotion); the binned update handles bool directly.
        pixel_scores = subsample_hw(scores, self.pixel_stride)
        pixel_targets = subsample_hw(targets, self.pixel_stride)
        warn_below_vectorized_cutoff(
            self.name, pixel_targets.numel(), targets.numel(), self._stride_warn_state
        )
        self.pixel_auroc.update(
            self._binned_preds(pixel_scores), pixel_targets.squeeze(-1).flatten().bool()
        )
        # Image-level: per-image score vs "any GT pixel positive" label (never subsampled).
        self.image_auroc.update(self._binned_preds(anomaly_score), img_tgts)

        return {
            "metrics": [
                Metric(
                    name="auroc_pixel",
                    value=float(self.pixel_auroc.compute()),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                ),
                Metric(
                    name="auroc_image",
                    value=float(self.image_auroc.compute()),
                    stage=context.stage,
                    epoch=context.epoch,
                    batch_idx=context.batch_idx,
                ),
            ]
        }

    def pooled_metrics(self) -> dict[str, TorchMetric]:
        """Live torchmetrics objects for the epoch-pooled AUROCs, keyed by metric name.

        ``auroc_pixel`` / ``auroc_image`` accumulate across the epoch (reset only at the
        ``(stage, epoch)`` boundary), so the trainer logs these objects with ``on_epoch=True``
        and Lightning computes the single pooled AUROC and resets at epoch end — exact and
        batch-size-invariant, unlike the per-batch running values emitted in ``forward``. Returns
        an empty mapping until the first batch has been seen, so nothing is logged for a run that
        never produced scores.
        """
        if self._last_key is None:
            return {}
        return {"auroc_pixel": self.pixel_auroc, "auroc_image": self.image_auroc}


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

        metrics = [
            Metric(
                name="scores/mean",
                value=scores_flat.mean().item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/std",
                value=scores_flat.std().item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/min",
                value=scores_flat.min().item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/max",
                value=scores_flat.max().item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/median",
                value=scores_flat.median().item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/q25",
                value=torch.quantile(scores_flat, 0.25).item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/q75",
                value=torch.quantile(scores_flat, 0.75).item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/q95",
                value=torch.quantile(scores_flat, 0.95).item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="scores/q99",
                value=torch.quantile(scores_flat, 0.99).item(),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
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

        metrics = [
            Metric(
                name="orthogonality_error",
                value=orth_error,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="avg_off_diagonal",
                value=avg_off_diagonal,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="diagonal_mean",
                value=diagonal_mean,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="diagonal_std",
                value=diagonal_std,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
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
    Executes only during validation and test stages.
    """

    _category = NodeCategory.METRIC
    _tags = frozenset({NodeTag.EVALUATION, NodeTag.ANOMALY})

    INPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly decisions [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    EXECUTION_STAGES = {ExecutionStage.VAL, ExecutionStage.TEST}

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, decisions: Tensor, context: Context) -> dict[str, Any]:
        """Compute anomaly pixel statistics.

        Parameters
        ----------
        decisions : Tensor
            Binary anomaly decisions [B, H, W, 1]
        context : Context
            Execution context with stage, epoch, batch_idx

        Returns
        -------
        dict[str, Any]
            Dictionary with "metrics" key containing list of Metric objects
        """
        # Calculate statistics
        total_pixels = decisions.numel()
        anomalous_pixels = int(decisions.sum().item())
        anomaly_percentage = (anomalous_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0

        metrics = [
            Metric(
                name="anomaly/total_pixels",
                value=float(total_pixels),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="anomaly/anomalous_pixels",
                value=float(anomalous_pixels),
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
            Metric(
                name="anomaly/anomaly_percentage",
                value=anomaly_percentage,
                stage=context.stage,
                epoch=context.epoch,
                batch_idx=context.batch_idx,
            ),
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
    "AnomalyAUROCMetrics",
    "ScoreStatisticsMetric",
    "ComponentOrthogonalityMetric",
    "SelectorEntropyMetric",
    "SelectorDiversityMetric",
    "AnomalyPixelStatisticsMetric",
    "DistinctLabelCount",
]
