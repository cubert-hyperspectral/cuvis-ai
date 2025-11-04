"""Metric leaf nodes for training pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from cuvis_ai.node import LabelLike, MetaLike, NodeOutput
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.training.leaf_nodes import MetricNode


class ExplainedVarianceMetric(MetricNode):
    """Track explained variance ratio for PCA components.
    
    Compatible Parents
    ------------------
    TrainablePCA : PCA node with explained variance tracking
    """

    compatible_parent_types = (TrainablePCA,)
    required_parent_attributes = ("get_explained_variance_ratio",)

    def __init__(self) -> None:
        super().__init__()

    def compute_metric(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute explained variance metrics.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Output from parent PCA node
        labels : LabelLike, optional
            Labels (not used)
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        metrics : dict
            Dictionary with explained variance ratios per component and total
        """
        parent_node = self.parent
        variance_ratios = parent_node.get_explained_variance_ratio()

        if variance_ratios is None:
            return {}

        metrics = {}

        # Per-component variance
        for i, ratio in enumerate(variance_ratios):
            metrics[f"pca/explained_variance_pc{i+1}"] = ratio.item()

        # Total variance explained
        metrics["pca/total_explained_variance"] = variance_ratios.sum().item()

        # Cumulative variance
        cumulative = torch.cumsum(variance_ratios, dim=0)
        for i, cum_var in enumerate(cumulative):
            metrics[f"pca/cumulative_variance_pc{i+1}"] = cum_var.item()

        return metrics


class AnomalyDetectionMetrics(MetricNode):
    """Compute anomaly detection metrics (precision, recall, F1, etc.).
    
    Expects parent output to be anomaly scores and labels to be binary masks.
    
    Parameters
    ----------
    threshold : float, optional
        Threshold for binary classification (default: 0.5)
        For logits, this is applied after sigmoid
    
    Compatible Parents
    ------------------
    Any node that outputs anomaly scores
    """

    compatible_parent_types = (nn.Module,)  # Any node
    required_parent_attributes = ()

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def compute_metric(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute anomaly detection metrics.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Tuple of (scores, labels, metadata) where scores is [B, H, W, 1]
        labels : LabelLike, optional
            Binary anomaly masks [B, H, W, 1] or [B, H, W]
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        metrics : dict
            Dictionary with precision, recall, F1, accuracy, etc.
        """
        # Extract scores from parent output
        scores, labels_from_output, _ = parent_output

        # Use labels from parent output if not provided
        if labels is None:
            labels = labels_from_output

        if labels is None:
            return {"warning": "no_labels_available"}

        # Ensure consistent shapes
        if scores.dim() == 4 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)  # [B, H, W]

        if labels.dim() == 4 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)  # [B, H, W]

        # Apply sigmoid if scores look like logits (values outside [0,1])
        if scores.min() < 0 or scores.max() > 1:
            scores = torch.sigmoid(scores)

        # Binary predictions
        predictions = (scores > self.threshold).float()
        labels = labels.float()

        # Compute confusion matrix components
        tp = ((predictions == 1) & (labels == 1)).sum().float()
        fp = ((predictions == 1) & (labels == 0)).sum().float()
        tn = ((predictions == 0) & (labels == 0)).sum().float()
        fn = ((predictions == 0) & (labels == 1)).sum().float()

        # Compute metrics with epsilon to avoid division by zero
        eps = 1e-8

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        # Specificity (true negative rate)
        specificity = tn / (tn + fp + eps)

        # Balanced accuracy
        balanced_accuracy = (recall + specificity) / 2

        # IoU (Intersection over Union)
        intersection = tp
        union = tp + fp + fn
        iou = intersection / (union + eps)

        metrics = {
            "anomaly/precision": precision.item(),
            "anomaly/recall": recall.item(),
            "anomaly/f1_score": f1.item(),
            "anomaly/accuracy": accuracy.item(),
            "anomaly/specificity": specificity.item(),
            "anomaly/balanced_accuracy": balanced_accuracy.item(),
            "anomaly/iou": iou.item(),
            "anomaly/true_positives": tp.item(),
            "anomaly/false_positives": fp.item(),
            "anomaly/true_negatives": tn.item(),
            "anomaly/false_negatives": fn.item(),
            "anomaly/positive_rate": labels.mean().item(),
            "anomaly/pred_positive_rate": predictions.mean().item(),
        }

        return metrics


class ScoreStatisticsMetric(MetricNode):
    """Compute statistical properties of score distributions.
    
    Tracks mean, std, min, max, median, and quantiles of scores.
    
    Compatible Parents
    ------------------
    Any node that outputs scores
    """

    compatible_parent_types = (nn.Module,)  # Any node
    required_parent_attributes = ()

    def __init__(self) -> None:
        super().__init__()

    def compute_metric(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute score statistics.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Tuple of (scores, labels, metadata)
        labels : LabelLike, optional
            Labels (not used)
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        metrics : dict
            Dictionary with statistical metrics
        """
        scores, _, _ = parent_output

        # Flatten scores
        scores_flat = scores.reshape(-1)

        metrics = {
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

        return metrics


class ComponentOrthogonalityMetric(MetricNode):
    """Track orthogonality of PCA components during training.
    
    Measures how close the component matrix is to being orthonormal.
    
    Compatible Parents
    ------------------
    TrainablePCA : PCA node with trainable components
    """

    compatible_parent_types = (TrainablePCA,)
    required_parent_attributes = ("components",)

    def __init__(self) -> None:
        super().__init__()

    def compute_metric(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """Compute component orthogonality metrics.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Output from parent PCA node (not directly used)
        labels : LabelLike, optional
            Labels (not used)
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        metrics : dict
            Dictionary with orthogonality metrics
        """
        parent_node = self.parent
        components = parent_node.components

        if components is None:
            return {}

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

        metrics = {
            "pca/orthogonality_error": orth_error,
            "pca/avg_off_diagonal": avg_off_diagonal,
            "pca/diagonal_mean": diagonal_mean,
            "pca/diagonal_std": diagonal_std,
        }

        return metrics


__all__ = [
    "ExplainedVarianceMetric",
    "AnomalyDetectionMetrics",
    "ScoreStatisticsMetric",
    "ComponentOrthogonalityMetric",
]
