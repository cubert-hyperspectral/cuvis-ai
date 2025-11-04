"""Trainable PCA node for dimensionality reduction with gradient-based optimization."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
from torch import Tensor

from cuvis_ai.node import LabelLike, MetaLike, Node, NodeOutput


class TrainablePCA(Node):
    """Trainable PCA node with orthogonality regularization.
    
    This node performs Principal Component Analysis (PCA) for dimensionality reduction
    and can be trained end-to-end with gradient descent. It supports:
    - Statistical initialization from data
    - Gradient-based fine-tuning with orthogonality constraints
    - Explained variance tracking
    
    Parameters
    ----------
    n_components : int
        Number of principal components to retain
    whiten : bool, optional
        If True, scale components by explained variance (default: False)
    trainable : bool, optional
        If True, components become trainable parameters (default: True)
    init_method : {"svd", "random"}, optional
        Initialization method for components (default: "svd")
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)
    
    Attributes
    ----------
    components : nn.Parameter or Tensor
        Principal components matrix [n_components, n_features]
    mean : Tensor
        Feature-wise mean [n_features]
    explained_variance : Tensor
        Variance explained by each component [n_components]
    """

    def __init__(
        self,
        n_components: int,
        whiten: bool = False,
        trainable: bool = True,
        init_method: Literal["svd", "random"] = "svd",
        eps: float = 1e-6,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.trainable = trainable
        self.init_method = init_method
        self.eps = eps

        super().__init__(
            n_components=n_components,
            whiten=whiten,
            trainable=trainable,
            init_method=init_method,
            eps=eps,
        )

        # Buffers for statistical initialization
        self.register_buffer("mean", None)
        self.register_buffer("explained_variance", None)

        # Components will be initialized during initialize_from_data
        # or randomly if not using statistical initialization
        self.components = None
        self._initialized = False

    @property
    def requires_initial_fit(self) -> bool:
        """PCA requires statistical initialization."""
        return True

    @property
    def is_trainable(self) -> bool:
        """Whether PCA components can be trained with gradients."""
        return self.trainable

    def initialize_from_data(self, iterator) -> None:
        """Initialize PCA components from data using SVD.
        
        Parameters
        ----------
        iterator : Iterator
            Iterator yielding (x, y, m) tuples where x has shape [B, H, W, C]
        """
        # Collect all data
        all_data = []
        for batch_data in iterator:
            x, _, _ = batch_data
            if x is not None:
                # Flatten spatial dimensions: [B, H, W, C] -> [B*H*W, C]
                flat = x.reshape(-1, x.shape[-1])
                all_data.append(flat)

        if not all_data:
            raise ValueError("No data provided for PCA initialization")

        # Concatenate all samples
        X = torch.cat(all_data, dim=0)  # [N, C]

        # Compute mean and center data
        self.mean = X.mean(dim=0)  # [C]
        X_centered = X - self.mean  # [N, C]

        # Compute SVD
        # X_centered = U @ S @ V.T where V contains principal components
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

        # Extract top n_components
        self.components = Vt[:self.n_components, :]  # [n_components, C]

        # Compute explained variance
        # Variance = (S^2) / (N - 1)
        variance = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance = variance[:self.n_components]  # [n_components]

        self._initialized = True
        self._initialized = True

    def prepare_for_train(self) -> None:
        """Convert components to trainable parameters if trainable=True."""
        if self.trainable and self.components is not None:
            # Convert buffer to parameter
            self.components = nn.Parameter(self.components.clone())

    def forward(
        self,
        x: Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
        **_: Any,
    ) -> NodeOutput:
        """Project data onto principal components.
        
        Parameters
        ----------
        x : Tensor
            Input tensor [B, H, W, C]
        y : LabelLike, optional
            Labels
        m : MetaLike, optional
            Metadata
        
        Returns
        -------
        NodeOutput
            Projected data [B, H, W, n_components], labels, metadata
        """
        if not self._initialized:
            raise RuntimeError(
                "PCA not initialized. Call initialize_from_data() first."
            )

        B, H, W, C = x.shape

        # Flatten spatial dimensions
        x_flat = x.reshape(-1, C)  # [B*H*W, C]

        # Ensure mean is on the same device as input
        mean = self.mean.to(x.device)

        # Center data
        x_centered = x_flat - mean  # [B*H*W, C]

        # Ensure components are on the same device as input
        components = self.components.to(x.device) if isinstance(self.components, Tensor) else self.components

        # Project onto components: X_proj = X @ components.T
        x_proj = x_centered @ components.T  # [B*H*W, n_components]

        # Whiten if requested
        if self.whiten:
            # Ensure explained_variance is on the same device
            explained_variance = self.explained_variance.to(x.device)
            # Scale by 1/sqrt(explained_variance)
            scale = 1.0 / torch.sqrt(explained_variance + self.eps)
            x_proj = x_proj * scale

        # Reshape back to spatial dimensions
        x_proj = x_proj.reshape(B, H, W, self.n_components)

        return x_proj, y, m

    @property
    def input_dim(self) -> tuple[int, int, int, int]:
        """Expected input shape (flexible spatial dims)."""
        return (-1, -1, -1, -1)

    @property
    def output_dim(self) -> tuple[int, int, int, int]:
        """Output shape with reduced channels."""
        return (-1, -1, -1, self.n_components)

    def compute_orthogonality_loss(self) -> Tensor:
        """Compute orthogonality loss for components.
        
        Encourages components to remain orthonormal during training.
        Loss = ||W @ W.T - I||^2_F where W is components matrix
        
        Returns
        -------
        Tensor
            Orthogonality loss (scalar)
        """
        if self.components is None:
            return torch.tensor(0.0)

        # Compute gram matrix: W @ W.T
        gram = self.components @ self.components.T  # [n_components, n_components]

        # Target: identity matrix
        eye = torch.eye(
            self.n_components,
            device=self.components.device,
            dtype=self.components.dtype,
        )

        # Frobenius norm of difference
        loss = torch.sum((gram - eye) ** 2)

        return loss

    def get_explained_variance_ratio(self) -> Tensor:
        """Get proportion of variance explained by each component.
        
        Returns
        -------
        Tensor
            Explained variance ratios [n_components]
        """
        if self.explained_variance is None:
            return None

        total_variance = self.explained_variance.sum()
        return self.explained_variance / (total_variance + self.eps)

    @classmethod
    def load(cls, state: dict) -> TrainablePCA:
        """Load TrainablePCA from serialized state.
        
        Parameters
        ----------
        state : dict
            Serialized state dictionary
            
        Returns
        -------
        TrainablePCA
            Loaded PCA instance
        """
        # Extract hparams
        hparams = state.get("hparams", {})

        # Create instance
        instance = cls(**hparams)

        # Load state dict if present
        if "state_dict" in state:
            instance.load_state_dict(state["state_dict"])

        # Mark as initialized
        instance._initialized = True

        return instance


__all__ = ["TrainablePCA"]
