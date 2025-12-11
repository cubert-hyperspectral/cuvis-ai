from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec


class _ScoreNormalizerBase(Node):
    """Base class for differentiable score normalizers operating on BHWC tensors.

    Notes
    -----
    All normalization nodes in this module expect inputs in BHWC format
    ([batch, height, width, channels]). Callers are responsible for adding
    a batch dimension when working with HWC tensors (use `x.unsqueeze(0)`).
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input data tensor to normalize (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32, shape=(-1, -1, -1, -1), description="Normalized output tensor"
        )
    }

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Normalize input data (BHWC only).

        Parameters
        ----------
        data : Tensor
            Input tensor in BHWC format [B, H, W, C]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "normalized" key containing normalized tensor
        """
        normalized = self._normalize(data)
        return {"normalized": normalized}

    def _normalize(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError


class IdentityNormalizer(_ScoreNormalizerBase):
    """No-op normalizer; preserves incoming scores."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _normalize(self, tensor: Tensor) -> Tensor:
        return tensor


class MinMaxNormalizer(_ScoreNormalizerBase):
    """Min-max normalization per sample and channel (keeps gradients).

    Can operate in two modes:
    1. Per-sample normalization (default): min/max computed per batch
    2. Global normalization: uses running statistics from initialization
    """

    def __init__(self, eps: float = 1e-6, use_running_stats: bool = True, **kwargs) -> None:
        self.eps = float(eps)
        self.use_running_stats = use_running_stats
        super().__init__(eps=eps, use_running_stats=use_running_stats, **kwargs)

        # Running statistics for global normalization
        self.register_buffer("running_min", torch.tensor(float("nan")))
        self.register_buffer("running_max", torch.tensor(float("nan")))

        # Only require initialization when running stats are requested
        self._requires_initial_fit_override = self.use_running_stats

    def statistical_initialization(self, input_stream) -> None:
        """Compute global min/max from data iterator.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"data": tensor} where tensor is the scores/data
        """
        all_mins = []
        all_maxs = []

        for batch_data in input_stream:
            # Extract data from port-based dict
            x = batch_data.get("data")
            if x is not None:
                # Flatten spatial dimensions
                flat = x.reshape(x.shape[0], -1)
                batch_min = flat.min()
                batch_max = flat.max()
                all_mins.append(batch_min)
                all_maxs.append(batch_max)

        if all_mins:
            self.running_min = torch.stack(all_mins).min()
            self.running_max = torch.stack(all_maxs).max()
            self._statistically_initialized = True

    def _is_initialized(self) -> bool:
        """Check if running statistics have been initialized."""
        return not torch.isnan(self.running_min).item()

    def _normalize(self, tensor: Tensor) -> Tensor:
        B, H, W, C = tensor.shape
        flat = tensor.view(B, -1, C)

        # Use running stats if available and initialized
        if self.use_running_stats and self._is_initialized():
            mins = self.running_min
            maxs = self.running_max
            ranges = torch.clamp(maxs - mins, min=self.eps)
            scaled = (flat - mins) / ranges
        else:
            # Per-sample normalization
            mins = flat.min(dim=1, keepdim=True).values
            maxs = flat.max(dim=1, keepdim=True).values
            ranges = torch.clamp(maxs - mins, min=self.eps)
            scaled = (flat - mins) / ranges

        return scaled.view(B, H, W, C)


class SigmoidNormalizer(_ScoreNormalizerBase):
    """Median-centered sigmoid squashing per sample and channel."""

    def __init__(self, std_floor: float = 1e-6, **kwargs) -> None:
        self.std_floor = float(std_floor)
        super().__init__(std_floor=std_floor, **kwargs)

    def _normalize(self, tensor: Tensor) -> Tensor:
        B, H, W, C = tensor.shape
        flat = tensor.view(B, -1, C)
        medians = flat.median(dim=1, keepdim=True).values
        stds = flat.std(dim=1, unbiased=False, keepdim=True)
        stds = torch.clamp(stds, min=self.std_floor)
        normalized = torch.sigmoid((flat - medians) / stds)
        return normalized.view(B, H, W, C)


class ZScoreNormalizer(_ScoreNormalizerBase):
    """Z-score (standardization) normalization along specified dimensions.

    Computes: (x - mean) / (std + eps) along specified dims.
    Per-sample normalization with no statistical initialization required.

    Parameters
    ----------
    dims : list[int], optional
        Dimensions to compute statistics over (default: [1,2] for H,W in BHWC format)
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)
    keepdim : bool, optional
        Whether to keep reduced dimensions (default: True)

    Examples
    --------
    >>> # Normalize over spatial dimensions (H, W)
    >>> zscore = ZScoreNormalizer(dims=[1, 2])
    >>>
    >>> # Normalize over all spatial and channel dimensions
    >>> zscore_all = ZScoreNormalizer(dims=[1, 2, 3])
    """

    def __init__(
        self, dims: list[int] | None = None, eps: float = 1e-6, keepdim: bool = True, **kwargs
    ) -> None:
        self.dims = dims if dims is not None else [1, 2]
        self.eps = float(eps)
        self.keepdim = keepdim
        super().__init__(dims=self.dims, eps=eps, keepdim=keepdim, **kwargs)

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Apply z-score normalization.

        Parameters
        ----------
        tensor : Tensor
            Input tensor in BHWC format

        Returns
        -------
        Tensor
            Z-score normalized tensor
        """
        # Compute mean and std along specified dimensions
        mean = tensor.mean(dim=self.dims, keepdim=self.keepdim)
        std = tensor.std(dim=self.dims, keepdim=self.keepdim, unbiased=False)

        # Apply z-score normalization
        normalized = (tensor - mean) / (std + self.eps)

        return normalized


class SigmoidTransform(Node):
    """Applies sigmoid transformation to convert logits to probabilities [0,1].

    General-purpose sigmoid node for converting raw scores/logits to probability space.
    Useful for visualization or downstream nodes that expect bounded [0,1] values.

    Examples
    --------
    >>> sigmoid = SigmoidTransform()
    >>> # Route logits to both loss (raw) and visualization (sigmoid)
    >>> graph.connect(
    ...     (rx.scores, loss_node.predictions),  # Raw logits to loss
    ...     (rx.scores, sigmoid.data),           # Logits to sigmoid
    ...     (sigmoid.transformed, viz.scores),   # Probabilities to viz
    ... )
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input tensor (any shape)",
        )
    }

    OUTPUT_SPECS = {
        "transformed": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Sigmoid-transformed tensor",
        )
    }

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply sigmoid transformation.

        Parameters
        ----------
        data : Tensor
            Input tensor

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "transformed" key containing sigmoid output
        """
        return {"transformed": torch.sigmoid(data)}


class PerPixelUnitNorm(_ScoreNormalizerBase):
    """Per-pixel mean-centering and L2 normalization across channels."""

    def __init__(self, eps: float = 1e-8, **kwargs) -> None:
        self.eps = float(eps)
        super().__init__(eps=self.eps, **kwargs)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Normalize BHWC tensors per pixel."""
        normalized = self._normalize(data)
        return {"normalized": normalized}

    def _normalize(self, tensor: Tensor) -> Tensor:
        B, H, W, C = tensor.shape
        flat = tensor.view(B, -1, C)
        mu = flat.mean(dim=2, keepdim=True)
        centered = flat - mu
        l2 = centered.norm(p=2, dim=2, keepdim=True).clamp_min(self.eps)
        normalized = (centered / l2).view(B, H, W, C)
        return normalized


__all__ = [
    "IdentityNormalizer",
    "MinMaxNormalizer",
    "SigmoidNormalizer",
    "ZScoreNormalizer",
    "SigmoidTransform",
    "PerPixelUnitNorm",
]
