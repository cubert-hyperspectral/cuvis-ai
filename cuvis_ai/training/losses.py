"""Loss leaf nodes for training pipeline."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cuvis_ai.node import LabelLike, MetaLike, NodeOutput
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.training.leaf_nodes import LossNode


class OrthogonalityLoss(LossNode):
    """Orthogonality regularization loss for TrainablePCA.
    
    Encourages PCA components to remain orthonormal during training.
    Loss = weight * ||W @ W.T - I||^2_F where W is components matrix
    
    Parameters
    ----------
    weight : float, optional
        Weight for orthogonality loss (default: 1.0)
    
    Compatible Parents
    ------------------
    TrainablePCA : PCA node with trainable components
    """

    compatible_parent_types = (TrainablePCA,)
    required_parent_attributes = ("compute_orthogonality_loss",)

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(weight=weight)

    def compute_loss(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute orthogonality loss from parent PCA node.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Output from parent TrainablePCA node (not directly used)
        labels : LabelLike, optional
            Labels (not used)
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        loss : Tensor
            Weighted orthogonality loss (scalar)
        info : dict
            Additional information (loss value, weight)
        """
        # Get parent PCA node
        parent_node = self.parent

        # Compute orthogonality loss
        orth_loss = parent_node.compute_orthogonality_loss()

        # Apply weight
        weighted_loss = self.weight * orth_loss

        info = {
            "orthogonality_loss": orth_loss.item(),
            "weight": self.weight,
        }

        return weighted_loss, info


class AnomalyBCEWithLogits(LossNode):
    """Binary cross-entropy loss for anomaly detection with logits.
    
    Expects parent output to be anomaly scores (logits) and labels to be binary masks.
    Uses BCEWithLogitsLoss for numerical stability.
    
    Parameters
    ----------
    weight : float, optional
        Overall weight for this loss component (default: 1.0)
    pos_weight : float, optional
        Weight for positive class (anomaly) to handle class imbalance (default: None)
    reduction : str, optional
        Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
    
    Compatible Parents
    ------------------
    Any node that outputs anomaly scores in spatial format [B, H, W, 1]
    """

    compatible_parent_types = (nn.Module,)  # Any node
    required_parent_attributes = ()

    def __init__(
        self,
        weight: float = 1.0,
        pos_weight: float = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__(weight=weight)
        self.pos_weight = pos_weight
        self.reduction = reduction

        # Create loss function
        if pos_weight is not None:
            pos_weight_tensor = torch.tensor([pos_weight])
            self.register_buffer("_pos_weight", pos_weight_tensor)
            self.loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=self._pos_weight,
                reduction=reduction,
            )
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction=reduction)

    def compute_loss(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute binary cross-entropy loss for anomaly detection.
        
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
        loss : Tensor
            BCE loss (scalar or per-sample depending on reduction)
        info : dict
            Additional information (loss value, positive rate, etc.)
        """
        # Extract scores from parent output
        scores, labels_from_output, _ = parent_output

        # Use labels from parent output if not provided
        if labels is None:
            labels = labels_from_output

        if labels is None:
            raise ValueError("Labels are required for AnomalyBCEWithLogits")

        # Ensure consistent shapes
        if scores.dim() == 4 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)  # [B, H, W]

        if labels.dim() == 4 and labels.shape[-1] == 1:
            labels = labels.squeeze(-1)  # [B, H, W]

        # Convert labels to float
        labels = labels.float()

        # Compute loss
        loss = self.loss_fn(scores, labels)

        # Apply weight
        weighted_loss = self.weight * loss

        # Compute additional statistics
        with torch.no_grad():
            predictions = torch.sigmoid(scores) > 0.5
            positive_rate = labels.mean().item()
            pred_positive_rate = predictions.float().mean().item()
            accuracy = (predictions == labels.bool()).float().mean().item()

        info = {
            "bce_loss": loss.item() if loss.numel() == 1 else loss.mean().item(),
            "weighted_bce_loss": weighted_loss.item() if weighted_loss.numel() == 1 else weighted_loss.mean().item(),
            "weight": self.weight,
            "positive_rate": positive_rate,
            "pred_positive_rate": pred_positive_rate,
            "accuracy": accuracy,
        }

        return weighted_loss, info


class MSEReconstructionLoss(LossNode):
    """Mean squared error reconstruction loss.
    
    Computes MSE between parent output and original input or target.
    Useful for autoencoder-style architectures.
    
    Parameters
    ----------
    reduction : str, optional
        Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
    
    Compatible Parents
    ------------------
    Any node that outputs reconstructed data
    """

    compatible_parent_types = (nn.Module,)  # Any node
    required_parent_attributes = ()

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def compute_loss(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute MSE reconstruction loss.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Tuple of (reconstruction, labels, metadata)
        labels : LabelLike, optional
            Target for reconstruction (if None, uses metadata['original'])
        metadata : MetaLike, optional
            Metadata that may contain original input
        
        Returns
        -------
        loss : Tensor
            MSE loss (scalar or per-sample depending on reduction)
        info : dict
            Additional information (loss value, SNR, etc.)
        """
        reconstruction, _, _ = parent_output

        # Get target from labels or metadata
        if labels is not None:
            target = labels
        elif metadata is not None and isinstance(metadata, dict):
            target = metadata.get("original")
        else:
            target = None

        if target is None:
            raise ValueError(
                "Target required for MSEReconstructionLoss. "
                "Provide via labels or metadata['original']"
            )

        # Ensure consistent shapes
        if target.shape != reconstruction.shape:
            raise ValueError(
                f"Shape mismatch: reconstruction {reconstruction.shape} "
                f"vs target {target.shape}"
            )

        # Compute loss
        loss = self.loss_fn(reconstruction, target)

        # Compute additional statistics
        with torch.no_grad():
            # Signal-to-noise ratio
            signal_power = (target ** 2).mean()
            noise_power = ((reconstruction - target) ** 2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8))

        info = {
            "mse_loss": loss.item() if loss.numel() == 1 else loss.mean().item(),
            "snr_db": snr.item(),
        }

        return loss, info


class WeightedMultiLoss(LossNode):
    """Weighted combination of multiple losses.
    
    Aggregates losses from multiple child loss nodes with configurable weights.
    
    Parameters
    ----------
    loss_weights : dict[str, float], optional
        Mapping of loss names to weights (default: equal weights)
    
    Compatible Parents
    ------------------
    Any node (delegates to child loss nodes)
    """

    compatible_parent_types = (nn.Module,)  # Any node
    required_parent_attributes = ()

    def __init__(self, loss_weights: dict[str, float] = None) -> None:
        super().__init__()
        self.loss_weights = loss_weights or {}
        self.loss_nodes = nn.ModuleDict()

    def add_loss(self, name: str, loss_node: LossNode, weight: float = 1.0) -> None:
        """Add a child loss node.
        
        Parameters
        ----------
        name : str
            Name for this loss component
        loss_node : LossNode
            Loss node to add
        weight : float, optional
            Weight for this loss (default: 1.0)
        """
        self.loss_nodes[name] = loss_node
        self.loss_weights[name] = weight

    def compute_loss(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute weighted sum of all child losses.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Output from parent node
        labels : LabelLike, optional
            Labels
        metadata : MetaLike, optional
            Metadata
        
        Returns
        -------
        loss : Tensor
            Weighted sum of losses (scalar)
        info : dict
            Information from all child losses
        """
        total_loss = torch.tensor(0.0, device=parent_output[0].device)
        all_info = {}

        for name, loss_node in self.loss_nodes.items():
            weight = self.loss_weights.get(name, 1.0)
            loss, info = loss_node.compute_loss(
                parent_output, labels, metadata, **kwargs
            )

            weighted_loss = weight * loss
            total_loss = total_loss + weighted_loss

            # Store info with prefixed keys
            for key, value in info.items():
                all_info[f"{name}/{key}"] = value
            all_info[f"{name}/weight"] = weight

        all_info["total_loss"] = total_loss.item()

        return total_loss, all_info


class SelectorEntropyRegularizer(LossNode):
    """Entropy regularization for SoftChannelSelector.
    
    Encourages exploration by penalizing low-entropy (over-confident) selections.
    Loss = -weight * entropy where entropy = -sum(p * log(p))
    
    Higher entropy = more uniform selection (encouraged early in training)
    Lower entropy = more peaked selection (emerges naturally as training progresses)
    
    Parameters
    ----------
    weight : float, optional
        Weight for entropy regularization (default: 0.01)
        Positive weight encourages exploration (maximizes entropy)
        Negative weight encourages exploitation (minimizes entropy)
    target_entropy : float, optional
        Target entropy for regularization (default: None, no target)
        If set, uses squared error: (entropy - target)^2
    
    Compatible Parents
    ------------------
    SoftChannelSelector : Channel selector with entropy computation
    """

    compatible_parent_types = (nn.Module,)  # Will check for compute_entropy method
    required_parent_attributes = ("compute_entropy",)

    def __init__(
        self,
        weight: float = 0.01,
        target_entropy: float = None,
    ) -> None:
        super().__init__(weight=weight)
        self.target_entropy = target_entropy

    def compute_loss(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute entropy regularization loss.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Output from parent SoftChannelSelector (not directly used)
        labels : LabelLike, optional
            Labels (not used)
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        loss : Tensor
            Entropy regularization loss (scalar)
        info : dict
            Additional information (entropy value, weight)
        """
        # Get parent selector node
        parent_node = self.parent

        # Compute entropy
        entropy = parent_node.compute_entropy()

        # Compute loss
        if self.target_entropy is not None:
            # Target-based regularization: minimize distance to target
            loss = ((entropy - self.target_entropy) ** 2)
        else:
            # Simple regularization: maximize (positive weight) or minimize (negative weight) entropy
            loss = -entropy

        # Apply weight
        weighted_loss = self.weight * loss

        info = {
            "selector_entropy": entropy.item(),
            "entropy_loss": weighted_loss.item(),
            "weight": self.weight,
        }

        if self.target_entropy is not None:
            info["target_entropy"] = self.target_entropy

        return weighted_loss, info


class SelectorDiversityRegularizer(LossNode):
    """Diversity regularization for SoftChannelSelector.
    
    Encourages diverse channel selection by penalizing concentration on few channels.
    Uses variance of selection weights as a diversity measure.
    
    Loss = weight * diversity_loss where diversity_loss favors spread-out selection
    
    Parameters
    ----------
    weight : float, optional
        Weight for diversity regularization (default: 0.01)
    
    Compatible Parents
    ------------------
    SoftChannelSelector : Channel selector with diversity computation
    """

    compatible_parent_types = (nn.Module,)  # Will check for compute_diversity_loss method
    required_parent_attributes = ("compute_diversity_loss",)

    def __init__(self, weight: float = 0.01) -> None:
        super().__init__(weight=weight)

    def compute_loss(
        self,
        parent_output: NodeOutput,
        labels: LabelLike = None,
        metadata: MetaLike = None,
        **kwargs: Any,
    ) -> tuple[Tensor, dict[str, Any]]:
        """Compute diversity regularization loss.
        
        Parameters
        ----------
        parent_output : NodeOutput
            Output from parent SoftChannelSelector (not directly used)
        labels : LabelLike, optional
            Labels (not used)
        metadata : MetaLike, optional
            Metadata (not used)
        
        Returns
        -------
        loss : Tensor
            Diversity regularization loss (scalar)
        info : dict
            Additional information (diversity value, weight)
        """
        # Get parent selector node
        parent_node = self.parent

        # Compute diversity loss (negative variance)
        diversity_loss = parent_node.compute_diversity_loss()

        # Apply weight
        weighted_loss = self.weight * diversity_loss

        # Get selection weights for additional statistics
        with torch.no_grad():
            weights = parent_node.get_selection_weights(hard=False)
            weight_std = weights.std().item()
            weight_max = weights.max().item()
            weight_min = weights.min().item()

        info = {
            "diversity_loss": diversity_loss.item(),
            "weighted_diversity_loss": weighted_loss.item(),
            "weight": self.weight,
            "weight_std": weight_std,
            "weight_max": weight_max,
            "weight_min": weight_min,
        }

        return weighted_loss, info


__all__ = [
    "OrthogonalityLoss",
    "AnomalyBCEWithLogits",
    "MSEReconstructionLoss",
    "WeightedMultiLoss",
    "SelectorEntropyRegularizer",
    "SelectorDiversityRegularizer",
]
