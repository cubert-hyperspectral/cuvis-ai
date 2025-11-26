"""Metric nodes for training pipeline (port-based architecture)."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torchmetrics.classification import (
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
)

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context, ExecutionStage, Metric


class ExplainedVarianceMetric(Node):
    """Track explained variance ratio for PCA components.

    Executes only during validation and test stages.
    """

    INPUT_SPECS = {
        "explained_variance_ratio": PortSpec(
            dtype=torch.float32, shape=(-1,), description="Explained variance ratio from PCA node"
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(
        self,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            **kwargs,
        )

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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class AnomalyDetectionMetrics(Node):
    """Compute anomaly detection metrics (precision, recall, F1, etc.).

    Uses torchmetrics for GPU-optimized, robust metric computation.
    Expects binary decisions and targets to be binary masks.
    Executes only during validation and test stages.
    """

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

    def __init__(
        self,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            **kwargs,
        )

        # Initialize torchmetrics for binary classification
        # These are stateless (compute per-batch) since we don't call update()
        self.precision_metric = BinaryPrecision()
        self.recall_metric = BinaryRecall()
        self.f1_metric = BinaryF1Score()
        self.iou_metric = BinaryJaccardIndex()
        self.average_precision_metric = BinaryAveragePrecision()

    def forward(
        self,
        decisions: Tensor,
        targets: Tensor,
        context: Context,
        logits: Tensor | None = None,
    ) -> dict[str, Any]:
        """Compute anomaly detection metrics using torchmetrics.

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

        # Flatten to [N] where N = B*H*W for torchmetrics
        preds_flat = decisions.flatten()  # [B*H*W]
        targets_flat = targets.flatten()  # [B*H*W]

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
            raw_scores = logits.squeeze(-1).flatten().float()
            probs_for_ap = torch.sigmoid(raw_scores)
            average_precision = self.average_precision_metric(probs_for_ap, targets_flat)

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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class ScoreStatisticsMetric(Node):
    """Compute statistical properties of score distributions.

    Tracks mean, std, min, max, median, and quantiles of scores.
    Executes only during validation and test stages.
    """

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1), description="Score values [B, H, W]"
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(
        self,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            **kwargs,
        )

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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class ComponentOrthogonalityMetric(Node):
    """Track orthogonality of PCA components during training.

    Measures how close the component matrix is to being orthonormal.
    Executes only during validation and test stages.
    """

    INPUT_SPECS = {
        "components": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="PCA components matrix [n_components, n_features]",
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(
        self,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            **kwargs,
        )

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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class SelectorEntropyMetric(Node):
    """Track entropy of channel selection distribution.

    Measures the uncertainty/diversity in channel selection weights.
    Higher entropy indicates more uniform selection (less confident).
    Lower entropy indicates more peaked selection (more confident).

    Executes only during validation and test stages.
    """

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights from selector node",
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(
        self,
        eps: float = 1e-6,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        self.eps = eps
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            eps=eps,
            **kwargs,
        )

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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


class SelectorDiversityMetric(Node):
    """Track diversity of channel selection.

    Measures how spread out the selection weights are across channels.
    Uses Gini coefficient - lower values indicate more diverse selection.

    Executes only during validation and test stages.
    """

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights from selector node",
        )
    }

    OUTPUT_SPECS = {"metrics": PortSpec(dtype=list, shape=(), description="List of Metric objects")}

    def __init__(
        self,
        execution_stages: set[ExecutionStage] | None = None,
        **kwargs,
    ) -> None:
        name, execution_stages = Node.consume_base_kwargs(
            kwargs, execution_stages or {ExecutionStage.VAL, ExecutionStage.TEST}
        )
        super().__init__(
            name=name,
            execution_stages=execution_stages,
            **kwargs,
        )

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

    def serialize(self, serial_dir: str) -> dict:
        return {**self.hparams}

    def load(self, params: dict, serial_dir: str) -> None:
        pass


__all__ = [
    "ExplainedVarianceMetric",
    "AnomalyDetectionMetrics",
    "ScoreStatisticsMetric",
    "ComponentOrthogonalityMetric",
    "SelectorEntropyMetric",
    "SelectorDiversityMetric",
]
