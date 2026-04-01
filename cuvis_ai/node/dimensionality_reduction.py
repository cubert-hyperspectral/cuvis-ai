"""PCA nodes for dimensionality reduction."""

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


class PCA(Node):
    """Project each frame independently onto its principal components."""

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

    def __init__(
        self,
        n_components: int,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.n_components = int(n_components)
        self.eps = float(eps)

        super().__init__(n_components=self.n_components, eps=self.eps, **kwargs)

    def _fit_frame(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Fit PCA on one HWC frame and return components, mean, and eigenvalues."""
        if frame.ndim != 3:
            raise ValueError(f"Expected one frame with shape [H, W, C], got {tuple(frame.shape)}")

        _, _, channel_count = frame.shape
        if channel_count < self.n_components:
            raise ValueError(
                f"Expected at least {self.n_components} channels for PCA, got {channel_count}"
            )

        flat = frame.reshape(-1, channel_count).to(dtype=torch.float64)
        if flat.shape[0] < 2:
            raise ValueError("Per-frame PCA requires at least 2 pixels.")

        mean = flat.mean(dim=0)
        centered = flat - mean
        covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        components = eigenvectors[:, : self.n_components].T.to(dtype=torch.float32)
        return (
            components,
            mean.to(dtype=torch.float32),
            eigenvalues[: self.n_components].to(dtype=torch.float32),
        )

    def _project(self, flat: Tensor, mean: Tensor, components: Tensor) -> Tensor:
        """Center and project flattened pixels onto PCA components."""
        mean = mean.to(device=flat.device, dtype=flat.dtype)
        components = components.to(device=flat.device, dtype=flat.dtype)
        return (flat - mean) @ components.T

    def _variance_ratio(self, eigenvalues: Tensor) -> Tensor:
        """Normalize retained eigenvalues to explained-variance ratios."""
        eigenvalues = eigenvalues.to(dtype=torch.float32)
        return eigenvalues / (eigenvalues.sum() + self.eps)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Fit PCA independently on each frame and return the per-frame projection."""
        if data.ndim != 4:
            raise ValueError(f"Expected data with shape [B, H, W, C], got {tuple(data.shape)}")
        if data.shape[0] == 0:
            raise ValueError("PCA requires a non-empty batch.")

        projected_frames: list[Tensor] = []
        explained_variance_ratio: Tensor | None = None
        components: Tensor | None = None

        for frame in data:
            frame_components, mean, eigenvalues = self._fit_frame(frame)
            flat = frame.reshape(-1, frame.shape[-1]).to(dtype=torch.float32)
            projected = self._project(flat, mean, frame_components).reshape(
                frame.shape[0],
                frame.shape[1],
                self.n_components,
            )

            projected_frames.append(projected.to(dtype=torch.float32))
            explained_variance_ratio = self._variance_ratio(eigenvalues).to(device=data.device)
            components = frame_components.to(device=data.device)

        assert explained_variance_ratio is not None
        assert components is not None

        return {
            "projected": torch.stack(projected_frames, dim=0),
            "explained_variance_ratio": explained_variance_ratio,
            "components": components,
        }


class TrainablePCA(PCA):
    """Trainable PCA node with orthogonality regularization."""

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
        self.whiten = whiten
        self.init_method = init_method

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
        """Initialize PCA components from data using covariance eigen decomposition."""
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

        self._mean = acc.mean.to(dtype=torch.float32)  # [C]
        cov = acc.cov.to(torch.float64)  # [C, C]

        # Eigen decomposition on covariance (equivalent to SVD on centered data)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Extract top n_components (rows = principal components)
        self._components = eigenvectors[:, : self.n_components].T.float()  # [n_components, C]
        self._explained_variance = eigenvalues[: self.n_components].float()  # [n_components]

        self._statistically_initialized = True

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Project data onto statistically initialized global components."""
        if not self._statistically_initialized:
            raise RuntimeError("PCA not initialized. Call statistical_initialization() first.")

        if data.ndim != 4:
            raise ValueError(f"Expected data with shape [B, H, W, C], got {tuple(data.shape)}")

        batch_size, height, width, channels = data.shape
        flat = data.reshape(-1, channels)

        components = (
            self._components.to(data.device)
            if isinstance(self._components, Tensor)
            else self._components
        )
        projected = self._project(flat, self._mean, components)

        if self.whiten:
            explained_variance = self._explained_variance.to(
                device=data.device, dtype=projected.dtype
            )
            scale = 1.0 / torch.sqrt(explained_variance + self.eps)
            projected = projected * scale

        outputs = {
            "projected": projected.reshape(batch_size, height, width, self.n_components),
        }

        if self._explained_variance.numel() > 0:
            outputs["explained_variance_ratio"] = self._variance_ratio(self._explained_variance).to(
                data.device
            )

        if self._components.numel() > 0:
            outputs["components"] = self._components

        return outputs


__all__ = ["PCA", "TrainablePCA"]
