"""Differentiable normalization nodes for BHWC hyperspectral data.

This module provides a collection of normalization nodes designed for hyperspectral
imaging pipelines. All normalizers operate on BHWC format ([batch, height, width, channels])
and maintain gradient flow for end-to-end training.

Normalization strategies:

- **MinMaxNormalizer**: Scales data to [0, 1] range using min-max statistics
- **ZScoreNormalizer**: Standardizes data to zero mean and unit variance
- **SigmoidNormalizer**: Applies sigmoid transformation with median centering
- **PerPixelUnitNorm**: L2 normalization per pixel across channels
- **IdentityNormalizer**: No-op passthrough for testing or baseline comparisons
- **SigmoidTransform**: General-purpose sigmoid for logits→probabilities

**Why Normalize?**

Normalization is critical for stable anomaly detection and deep learning:

1. **Stable covariance estimation**: RX detectors require well-conditioned covariance matrices
2. **Gradient stability**: Prevents exploding/vanishing gradients during training
3. **Comparable scales**: Ensures different spectral ranges contribute equally
4. **Faster convergence**: Accelerates gradient-based optimization

**BHWC Format Requirement**

All normalizers expect BHWC input format. For HWC tensors, add batch dimension:

>>> hwc_tensor = torch.randn(256, 256, 61)  # [H, W, C]
>>> bhwc_tensor = hwc_tensor.unsqueeze(0)   # [1, H, W, C]
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


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
        """Abstract normalization method to be implemented by subclasses.

        Parameters
        ----------
        tensor : Tensor
            Input tensor in BHWC format

        Returns
        -------
        Tensor
            Normalized tensor in BHWC format
        """
        raise NotImplementedError


class IdentityNormalizer(_ScoreNormalizerBase):
    """No-op normalizer; preserves incoming scores."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Return input tensor unchanged (identity transformation).

        Parameters
        ----------
        tensor : Tensor
            Input tensor

        Returns
        -------
        Tensor
            Same tensor as input
        """
        return tensor


class MinMaxNormalizer(_ScoreNormalizerBase):
    """Min-max normalization per sample and channel (keeps gradients).

    Scales data to [0, 1] range using (x - min) / (max - min) transformation.
    Can operate in two modes:

    1. **Per-sample normalization** (use_running_stats=False): min/max computed per batch
    2. **Global normalization** (use_running_stats=True): uses running statistics from
       statistical initialization

    Parameters
    ----------
    eps : float, optional
        Small constant for numerical stability, prevents division by zero (default: 1e-6)
    use_running_stats : bool, optional
        If True, use global min/max from statistical_initialization(). If False, compute
        min/max per batch during forward pass (default: True)
    **kwargs : dict
        Additional arguments passed to Node base class

    Attributes
    ----------
    running_min : Tensor
        Global minimum value computed during statistical initialization
    running_max : Tensor
        Global maximum value computed during statistical initialization

    Examples
    --------
    >>> from cuvis_ai.node.normalization import MinMaxNormalizer
    >>> from cuvis_ai_core.training import StatisticalTrainer
    >>> import torch
    >>>
    >>> # Mode 1: Global normalization with statistical initialization
    >>> normalizer = MinMaxNormalizer(eps=1.0e-6, use_running_stats=True)
    >>> stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    >>> stat_trainer.fit()  # Computes global min/max from training data
    >>>
    >>> # Inference uses global statistics
    >>> output = normalizer.forward(data=hyperspectral_cube)
    >>> normalized = output["normalized"]  # [B, H, W, C], values in [0, 1]
    >>>
    >>> # Mode 2: Per-sample normalization (no initialization required)
    >>> normalizer_local = MinMaxNormalizer(use_running_stats=False)
    >>> output = normalizer_local.forward(data=hyperspectral_cube)
    >>> # Each sample normalized independently using its own min/max

    See Also
    --------
    ZScoreNormalizer : Z-score standardization
    SigmoidNormalizer : Sigmoid-based normalization
    docs/tutorials/rx-statistical.md : RX pipeline with MinMaxNormalizer

    Notes
    -----
    Global normalization (use_running_stats=True) is recommended for RX detectors to
    ensure consistent scaling between training and inference. Per-sample normalization
    can be useful for real-time processing when training data is unavailable.
    """

    TRAINABLE_BUFFERS = ("running_min", "running_max")

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
        # Reset previous running statistics before recomputing.
        self.running_min.fill_(float("nan"))
        self.running_max.fill_(float("nan"))
        self._statistically_initialized = False

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

        if not all_mins:
            raise RuntimeError(
                "MinMaxNormalizer.statistical_initialization() did not receive any data."
            )

        self.running_min.copy_(torch.stack(all_mins).min())
        self.running_max.copy_(torch.stack(all_maxs).max())
        self._statistically_initialized = True

    def _is_initialized(self) -> bool:
        """Check if running statistics have been initialized."""
        return not torch.isnan(self.running_min).item()

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Apply min-max normalization to input tensor.

        Parameters
        ----------
        tensor : Tensor
            Input tensor in BHWC format

        Returns
        -------
        Tensor
            Normalized tensor with values in [0, 1] range
        """
        B, H, W, C = tensor.shape
        flat = tensor.view(B, -1, C)

        # Running-stats mode is strict: statistical initialization is required.
        if self.use_running_stats:
            if not self._is_initialized() or not self._statistically_initialized:
                raise RuntimeError(
                    "MinMaxNormalizer requires statistical_initialization() before forward() "
                    "when use_running_stats=True."
                )
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
    """Median-centered sigmoid squashing per sample and channel.

    Applies sigmoid transformation centered at the median with standard deviation scaling:

        sigmoid((x - median) / std)

    Produces values in [0, 1] range with median mapped to 0.5.

    Parameters
    ----------
    std_floor : float, optional
        Minimum standard deviation threshold to prevent division by zero (default: 1e-6)
    **kwargs : dict
        Additional arguments passed to Node base class

    Examples
    --------
    >>> from cuvis_ai.node.normalization import SigmoidNormalizer
    >>> import torch
    >>>
    >>> # Create sigmoid normalizer
    >>> normalizer = SigmoidNormalizer(std_floor=1.0e-6)
    >>>
    >>> # Apply to hyperspectral data
    >>> data = torch.randn(4, 256, 256, 61)  # [B, H, W, C]
    >>> output = normalizer.forward(data=data)
    >>> normalized = output["normalized"]  # [4, 256, 256, 61], values in [0, 1]

    See Also
    --------
    MinMaxNormalizer : Min-max scaling to [0, 1]
    ZScoreNormalizer : Z-score standardization

    Notes
    -----
    Sigmoid normalization is robust to outliers because extreme values are squashed
    asymptotically to 0 or 1. This makes it suitable for data with heavy-tailed
    distributions or sporadic anomalies.
    """

    def __init__(self, std_floor: float = 1e-6, **kwargs) -> None:
        self.std_floor = float(std_floor)
        super().__init__(std_floor=std_floor, **kwargs)

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Apply median-centered sigmoid normalization.

        Parameters
        ----------
        tensor : Tensor
            Input tensor in BHWC format

        Returns
        -------
        Tensor
            Sigmoid-normalized tensor with values in [0, 1]
        """
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
        """Apply per-pixel L2 normalization across channels.

        Parameters
        ----------
        tensor : Tensor
            Input tensor in BHWC format

        Returns
        -------
        Tensor
            L2-normalized tensor with unit norm per pixel
        """
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
