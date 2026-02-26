"""Trainable PCA node for dimensionality reduction with gradient-based optimization."""

from __future__ import annotations

from typing import Any, Literal

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai.utils.welford import WelfordAccumulator

## This node is not approved
# missing tests against standard implementations
# missing tutorial examples and approved documentation


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

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "projected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "n_components"),
            description="PCA-projected data with reduced dimensions",
        ),
        "explained_variance_ratio": PortSpec(
            dtype=torch.float32,
            shape=("n_components",),
            description="Proportion of variance explained by each component",
            optional=True,
        ),
        "components": PortSpec(
            dtype=torch.float32,
            shape=("n_components", -1),
            description="Principal components matrix",
            optional=True,
        ),
    }

    TRAINABLE_BUFFERS = ("_components",)

    def __init__(
        self,
        num_channels: int,
        n_components: int,
        whiten: bool = False,
        init_method: Literal["svd", "random"] = "svd",
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self.init_method = init_method
        self.eps = eps

        super().__init__(
            num_channels=num_channels,
            n_components=n_components,
            whiten=whiten,
            init_method=init_method,
            eps=eps,
            **kwargs,
        )

        # Buffers for statistical initialization (private to avoid conflicts with output ports)
        self.register_buffer("_mean", torch.empty(num_channels))
        self.register_buffer("_explained_variance", torch.empty(n_components))
        self.register_buffer("_components", torch.empty(n_components, num_channels))

        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize PCA components from data using SVD.

        Parameters
        ----------
        input_stream : InputStream
            Input stream yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"data": tensor} where tensor is BHWC
        """
        acc = None
        for batch_data in input_stream:
            x = batch_data["data"]
            if x is not None:
                flat = x.reshape(-1, x.shape[-1])  # [B*H*W, C]
                if acc is None:
                    acc = WelfordAccumulator(flat.shape[1], track_covariance=True)
                acc.update(flat)

        if acc is None or acc.count == 0:
            raise ValueError("No data provided for PCA initialization")

        self._mean = acc.mean  # [C]
        cov = acc.cov.to(torch.float64)  # [C, C]

        # Eigen decomposition on covariance (equivalent to SVD on centered data)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        # eigh returns ascending order; flip to descending
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Extract top n_components (rows = principal components)
        self._components = eigenvectors[:, : self.n_components].T.float()  # [n_components, C]
        self._explained_variance = eigenvalues[: self.n_components].float()  # [n_components]

        self._statistically_initialized = True

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Project data onto principal components.

        Parameters
        ----------
        data : Tensor
            Input tensor [B, H, W, C]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "projected" key containing PCA-projected data
        """
        if not self._statistically_initialized:
            raise RuntimeError("PCA not initialized. Call statistical_initialization() first.")

        B, H, W, C = data.shape

        # Flatten spatial dimensions
        x_flat = data.reshape(-1, C)  # [B*H*W, C]

        # Ensure mean is on the same device as input
        mean = self._mean.to(data.device)

        # Center data
        x_centered = x_flat - mean  # [B*H*W, C]

        # Ensure components are on the same device as input
        components = (
            self._components.to(data.device)
            if isinstance(self._components, Tensor)
            else self._components
        )

        # Project onto components: X_proj = X @ components.T
        x_proj = x_centered @ components.T  # [B*H*W, n_components]

        # Whiten if requested
        if self.whiten:
            # Ensure explained_variance is on the same device
            explained_variance = self._explained_variance.to(data.device)
            # Scale by 1/sqrt(explained_variance)
            scale = 1.0 / torch.sqrt(explained_variance + self.eps)
            x_proj = x_proj * scale

        # Reshape back to spatial dimensions
        x_proj = x_proj.reshape(B, H, W, self.n_components)

        # Prepare output dictionary
        outputs = {"projected": x_proj}

        # Add optional outputs for loss/metric nodes
        # Expose explained variance ratio
        if self._explained_variance.numel() > 0:
            total_variance = self._explained_variance.sum()
            variance_ratio = self._explained_variance / (total_variance + self.eps)
            outputs["explained_variance_ratio"] = variance_ratio.to(data.device)

        # Expose components for loss/metric nodes
        if self._components.numel() > 0:
            outputs["components"] = self._components

        return outputs


__all__ = ["TrainablePCA"]
