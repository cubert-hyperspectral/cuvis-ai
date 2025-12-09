"""Loss nodes for training pipeline (port-based architecture)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import ExecutionStage


class LossNode(Node):
    """Base class for loss nodes that restricts execution to training stages.

    Loss nodes should not execute during inference - only during train, val, and test.
    """

    def __init__(self, **kwargs) -> None:
        # Default to train/val/test stages, but allow override
        assert "execution_stages" not in kwargs, (
            "Loss nodes can only execute in train, val, and test stages."
        )

        super().__init__(
            execution_stages={
                ExecutionStage.TRAIN,
                ExecutionStage.VAL,
                ExecutionStage.TEST,
            },
            **kwargs,
        )


class OrthogonalityLoss(LossNode):
    """Orthogonality regularization loss for TrainablePCA.

    Encourages PCA components to remain orthonormal during training.
    Loss = weight * ||W @ W.T - I||^2_F

    Parameters
    ----------
    weight : float, optional
        Weight for orthogonality loss (default: 1.0)
    """

    INPUT_SPECS = {
        "components": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="PCA components matrix [n_components, n_features]",
        )
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Weighted orthogonality loss")
    }

    def __init__(self, weight: float = 1.0, **kwargs) -> None:
        self.weight = weight

        super().__init__(
            weight=weight,
            **kwargs,
        )

    def forward(self, components: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute weighted orthogonality loss from PCA components.

        Parameters
        ----------
        components : Tensor
            PCA components matrix [n_components, n_features]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "loss" key containing weighted loss
        """
        # Compute gram matrix: W @ W.T
        gram = components @ components.T

        # Target: identity matrix
        n_components = components.shape[0]
        eye = torch.eye(
            n_components,
            device=components.device,
            dtype=components.dtype,
        )

        # Frobenius norm of difference
        orth_loss = torch.sum((gram - eye) ** 2)

        return {"loss": self.weight * orth_loss}


class AnomalyBCEWithLogits(LossNode):
    """Binary cross-entropy loss for anomaly detection with logits.

    Computes BCE loss between predicted anomaly scores and ground truth masks.
    Uses BCEWithLogitsLoss for numerical stability.

    Parameters
    ----------
    weight : float, optional
        Overall weight for this loss component (default: 1.0)
    pos_weight : float, optional
        Weight for positive class (anomaly) to handle class imbalance (default: None)
    reduction : str, optional
        Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
    """

    INPUT_SPECS = {
        "predictions": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Predicted anomaly scores (logits) [B, H, W, 1]",
        ),
        "targets": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Ground truth binary masks [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Scalar BCE loss value")
    }

    def __init__(
        self,
        weight: float = 1.0,
        pos_weight: float | None = None,
        reduction: str = "mean",
        **kwargs,
    ) -> None:
        self.weight = weight
        self.pos_weight = pos_weight
        self.reduction = reduction

        super().__init__(
            weight=weight,
            pos_weight=pos_weight,
            reduction=reduction,
            **kwargs,
        )

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

    def forward(self, predictions: Tensor, targets: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute weighted BCE loss.

        Parameters
        ----------
        predictions : Tensor
            Predicted scores [B, H, W, 1]
        targets : Tensor
            Ground truth masks [B, H, W, 1]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "loss" key containing scalar loss
        """
        # Squeeze channel dimension to [B, H, W] for BCEWithLogitsLoss
        if predictions.dim() == 4 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze(-1)

        if targets.dim() == 4 and targets.shape[-1] == 1:
            targets = targets.squeeze(-1)

        # Convert labels to float
        targets = targets.float()

        # Compute loss
        loss = self.loss_fn(predictions, targets)

        # Apply weight
        weighted_loss = self.weight * loss

        return {"loss": weighted_loss}


class MSEReconstructionLoss(LossNode):
    """Mean squared error reconstruction loss.

    Computes MSE between reconstruction and target.
    Useful for autoencoder-style architectures.

    Parameters
    ----------
    weight : float, optional
        Weight for this loss component (default: 1.0)
    reduction : str, optional
        Reduction method: 'mean', 'sum', or 'none' (default: 'mean')
    """

    INPUT_SPECS = {
        "reconstruction": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="Reconstructed data"
        ),
        "target": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Target data for reconstruction",
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Scalar MSE loss value")
    }

    def __init__(self, weight: float = 1.0, reduction: str = "mean", **kwargs) -> None:
        self.weight = weight
        self.reduction = reduction
        # Extract Node base parameters from kwargs to avoid duplication
        super().__init__(
            weight=weight,
            reduction=reduction,
            **kwargs,
        )
        self.loss_fn = nn.MSELoss(reduction=reduction)

    def forward(self, reconstruction: Tensor, target: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute MSE reconstruction loss.

        Parameters
        ----------
        reconstruction : Tensor
            Reconstructed data
        target : Tensor
            Target for reconstruction
        **kwargs
            Additional arguments (e.g., context) - ignored but accepted for compatibility

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "loss" key containing scalar loss
        """
        # Ensure consistent shapes
        if target.shape != reconstruction.shape:
            raise ValueError(
                f"Shape mismatch: reconstruction {reconstruction.shape} vs target {target.shape}"
            )

        # Compute loss
        loss = self.loss_fn(reconstruction, target)

        # Apply weight
        return {"loss": self.weight * loss}


class SelectorEntropyRegularizer(LossNode):
    """Entropy regularization for SoftChannelSelector.

    Encourages exploration by penalizing low-entropy (over-confident) selections.
    Computes entropy from selection weights and applies regularization.

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
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)
    """

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights from selector node",
        )
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Entropy regularization loss")
    }

    def __init__(
        self,
        weight: float = 0.01,
        target_entropy: float | None = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.weight = weight
        self.target_entropy = target_entropy
        self.eps = eps

        super().__init__(
            weight=weight,
            target_entropy=target_entropy,
            eps=eps,
            **kwargs,
        )

    def forward(self, weights: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute entropy regularization loss from selection weights.

        Parameters
        ----------
        weights : Tensor
            Channel selection weights [n_channels]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "loss" key containing regularization loss
        """
        # Normalize weights to probabilities
        probs = weights / (weights.sum() + self.eps)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + self.eps)).sum()

        # Compute loss
        if self.target_entropy is not None:
            # Target-based regularization: minimize distance to target
            loss = (entropy - self.target_entropy) ** 2
        else:
            # Simple regularization:
            # maximize (positive weight) or minimize (negative weight) entropy
            loss = -entropy

        # Apply weight
        return {"loss": self.weight * loss}


class SelectorDiversityRegularizer(LossNode):
    """Diversity regularization for SoftChannelSelector.

    Encourages diverse channel selection by penalizing concentration on few channels.
    Uses negative variance to encourage spread (higher variance = more diverse).

    Parameters
    ----------
    weight : float, optional
        Weight for diversity regularization (default: 0.01)
    """

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights from selector node",
        )
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Weighted diversity loss")
    }

    def __init__(self, weight: float = 0.01, **kwargs) -> None:
        self.weight = weight
        super().__init__(
            weight=weight,
            **kwargs,
        )

    def forward(self, weights: Tensor, **_: Any) -> dict[str, Tensor]:
        """Compute weighted diversity loss from selection weights.

        Parameters
        ----------
        weights : Tensor
            Channel selection weights [n_channels]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "loss" key containing weighted loss
        """
        # Compute variance of weights (high variance = diverse selection)
        mean_weight = weights.mean()
        variance = ((weights - mean_weight) ** 2).mean()

        # Return negative variance (minimizing loss = maximizing variance = maximizing diversity)
        diversity_loss = -variance

        return {"loss": self.weight * diversity_loss}


class DeepSVDDSoftBoundaryLoss(LossNode):
    """Soft-boundary Deep SVDD objective operating on BHWD embeddings."""

    INPUT_SPECS = {
        "embeddings": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Deep SVDD embeddings [B, H, W, D]",
        ),
        "center": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Deep SVDD center vector",
        ),
    }

    OUTPUT_SPECS = {
        "loss": PortSpec(dtype=torch.float32, shape=(), description="Deep SVDD soft-boundary loss"),
    }

    def __init__(self, nu: float = 0.05, weight: float = 1.0, **kwargs) -> None:
        if not (0.0 < nu < 1.0):
            raise ValueError("nu must be in (0, 1)")
        self.nu = float(nu)
        self.weight = float(weight)

        super().__init__(nu=self.nu, weight=self.weight, **kwargs)

        self.r_unconstrained = nn.Parameter(torch.tensor(0.0))

    def forward(self, embeddings: Tensor, center: Tensor, **_: Any) -> dict[str, Tensor]:
        B, H, W, D = embeddings.shape
        z = embeddings.reshape(B * H * W, D)
        R = torch.nn.functional.softplus(self.r_unconstrained, beta=10.0)
        dist = torch.sum((z - center.view(1, -1)) ** 2, dim=1)
        slack = torch.relu(dist - R**2)
        base_loss = R**2 + (1.0 / self.nu) * slack.mean()
        loss = self.weight * base_loss

        return {"loss": loss}


__all__ = [
    "LossNode",
    "OrthogonalityLoss",
    "AnomalyBCEWithLogits",
    "MSEReconstructionLoss",
    "SelectorEntropyRegularizer",
    "SelectorDiversityRegularizer",
    "DeepSVDDSoftBoundaryLoss",
]
