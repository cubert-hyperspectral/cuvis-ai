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

from enum import StrEnum
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node


class NormMode(StrEnum):
    """Normalization mode for percentile-based normalizers.

    Shared by :class:`PercentileNormalizer` and the channel selectors in
    :mod:`cuvis_ai.node.channel_selector` (which re-exports this enum).
    """

    PER_FRAME = "per_frame"
    RUNNING = "running"
    STATISTICAL = "statistical"


class _ScoreNormalizerBase(Node):
    """Base class for BHWC normalization nodes.

    Notes
    -----
    All normalization nodes in this module expect inputs in BHWC format
    ([batch, height, width, channels]). Callers are responsible for adding
    a batch dimension when working with HWC tensors (use `x.unsqueeze(0)`).

    Most subclasses are differentiable, but some keep fitted statistical state
    (e.g. :class:`MinMaxNormalizer` with running stats, :class:`PercentileNormalizer`)
    and update it under ``no_grad`` during ``forward``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

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
    docs/usecases/rx-statistical.md : RX pipeline with MinMaxNormalizer

    Notes
    -----
    Global normalization (use_running_stats=True) is recommended for RX detectors to
    ensure consistent scaling between training and inference. Per-sample normalization
    can be useful for real-time processing when training data is unavailable.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.POSTPROCESSING, NodeTag.TORCH})

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

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

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


class PercentileNormalizer(_ScoreNormalizerBase):
    """Per-channel normalization to ``[0, 1]`` for BHWC data of any channel count.

    Extracted from ``ChannelSelectorBase`` so band selection and display
    normalization are separate, composable steps. Operates on any channel count
    ``C`` (fixed at construction via ``n_channels``). Does **not** apply sRGB
    gamma; chain :class:`DisplayNormalizer` after it for the false-RGB display
    path. ML / n-channel callers use this node alone.

    Modes (``norm_mode``):

    - ``per_frame``: per-batch, per-channel absolute min/max; no inter-frame state.
    - ``statistical``: global percentile bounds precomputed via ``StatisticalTrainer``.
    - ``running`` (default): the first ``running_warmup_frames`` frames use per-frame
      percentile normalization while accumulating global percentile bounds (min-of-lows,
      max-of-highs); afterwards those bounds are used, frozen after
      ``freeze_running_bounds_after_frames`` calls. Bounds update on every call including
      inference, which live false-RGB video relies on; the freeze guards late drift.

    Parameters
    ----------
    n_channels : int
        Channel count ``C`` of the input. Sizes the per-channel bound buffers.
    norm_mode : str | NormMode
        Normalization mode. Default ``running``.
    freeze_running_bounds_after_frames : int | None
        Stop updating ``running`` bounds after this many calls. Default ``20``;
        ``None`` keeps unbounded accumulation.
    running_warmup_frames : int
        Frames to normalize per-frame while accumulating bounds. Default ``10``.
    quantile_low, quantile_high : float
        Percentile bounds (fractions) for ``running`` / ``statistical`` modes.
        Default ``0.005`` / ``0.995``.
    eps : float
        Floor for the ``(max - min)`` denominator. Default ``1e-8``.

    Ports
    -----
    INPUT_SPECS
        ``data`` : float32, shape (-1, -1, -1, -1), BHWC tensor with ``C == n_channels``.
    OUTPUT_SPECS
        ``normalized`` : float32, shape (-1, -1, -1, -1), same shape, values in ``[0, 1]``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

    def __init__(
        self,
        n_channels: int,
        norm_mode: str | NormMode = NormMode.RUNNING,
        freeze_running_bounds_after_frames: int | None = 20,
        running_warmup_frames: int = 10,
        quantile_low: float = 0.005,
        quantile_high: float = 0.995,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        if isinstance(n_channels, bool) or not isinstance(n_channels, int) or n_channels < 1:
            raise ValueError("PercentileNormalizer: n_channels must be an integer >= 1")
        norm_mode = NormMode(str(norm_mode) if isinstance(norm_mode, NormMode) else norm_mode)
        if freeze_running_bounds_after_frames is not None and (
            isinstance(freeze_running_bounds_after_frames, bool)
            or not isinstance(freeze_running_bounds_after_frames, int)
            or freeze_running_bounds_after_frames < 1
        ):
            raise ValueError("freeze_running_bounds_after_frames must be an integer >= 1 or None")
        if (
            isinstance(running_warmup_frames, bool)
            or not isinstance(running_warmup_frames, int)
            or running_warmup_frames < 0
        ):
            raise ValueError("running_warmup_frames must be an integer >= 0")
        if not 0.0 <= float(quantile_low) < float(quantile_high) <= 1.0:
            raise ValueError("PercentileNormalizer: require 0 <= quantile_low < quantile_high <= 1")

        self.n_channels = int(n_channels)
        self.norm_mode = norm_mode
        self.freeze_running_bounds_after_frames = freeze_running_bounds_after_frames
        self.running_warmup_frames = int(running_warmup_frames)
        self.quantile_low = float(quantile_low)
        self.quantile_high = float(quantile_high)
        self.eps = float(eps)

        super().__init__(
            n_channels=self.n_channels,
            norm_mode=str(norm_mode),
            freeze_running_bounds_after_frames=freeze_running_bounds_after_frames,
            running_warmup_frames=self.running_warmup_frames,
            quantile_low=self.quantile_low,
            quantile_high=self.quantile_high,
            eps=self.eps,
            **kwargs,
        )

        # Per-channel bounds + frame counter persist in state_dict (so warmup /
        # freeze survive a reload) but are deliberately NOT in TRAINABLE_BUFFERS:
        # they are fitted display statistics, and a gradient-learned bound could
        # violate lo < hi and turn normalization into an unconstrained transform.
        self.register_buffer("running_min", torch.full((self.n_channels,), float("nan")))
        self.register_buffer("running_max", torch.full((self.n_channels,), float("nan")))
        self.register_buffer("_norm_frame_count", torch.zeros((), dtype=torch.long))

        # Only the statistical path needs a fit pass; running / per_frame do not.
        self._requires_initial_fit_override = self.norm_mode == NormMode.STATISTICAL

    def _fitted(self) -> bool:
        """Whether per-channel bounds have been populated (non-NaN)."""
        return not bool(torch.isnan(self.running_min).any())

    def _per_frame_minmax(self, data: Tensor) -> Tensor:
        """Per-batch, per-channel absolute min/max to ``[0, 1]``."""
        lo = data.amin(dim=(1, 2), keepdim=True)
        hi = data.amax(dim=(1, 2), keepdim=True)
        denom = (hi - lo).clamp_min(self.eps)
        return ((data - lo) / denom).clamp_(0.0, 1.0)

    def _per_frame_percentile(self, data: Tensor) -> Tensor:
        """Per-frame percentile normalization (matches the running quantiles)."""
        flat = data.reshape(-1, self.n_channels).float()
        lo = torch.quantile(flat, self.quantile_low, dim=0).view(1, 1, 1, self.n_channels)
        hi = torch.quantile(flat, self.quantile_high, dim=0).view(1, 1, 1, self.n_channels)
        denom = (hi - lo).clamp_min(self.eps)
        return ((data - lo) / denom).clamp_(0.0, 1.0)

    def _apply_bounds(self, data: Tensor) -> Tensor:
        """Normalize using the accumulated per-channel bounds."""
        lo = self.running_min.view(1, 1, 1, self.n_channels)
        hi = self.running_max.view(1, 1, 1, self.n_channels)
        denom = (hi - lo).clamp_min(self.eps)
        return ((data - lo) / denom).clamp_(0.0, 1.0)

    @torch.no_grad()
    def _running_normalize(self, data: Tensor) -> Tensor:
        """Warmup + min/max percentile accumulation hybrid normalization."""
        flat = data.reshape(-1, self.n_channels).float()
        frame_lo = torch.quantile(flat, self.quantile_low, dim=0)
        frame_hi = torch.quantile(flat, self.quantile_high, dim=0)

        self._norm_frame_count.add_(1)
        count = int(self._norm_frame_count.item())
        should_update = (
            self.freeze_running_bounds_after_frames is None
            or count <= self.freeze_running_bounds_after_frames
        )
        if should_update:
            if torch.isnan(self.running_min).any():
                self.running_min.copy_(frame_lo)
                self.running_max.copy_(frame_hi)
            else:
                torch.minimum(self.running_min, frame_lo, out=self.running_min)
                torch.maximum(self.running_max, frame_hi, out=self.running_max)

        if count <= self.running_warmup_frames:
            return self._per_frame_percentile(data)
        return self._apply_bounds(data)

    def _normalize(self, tensor: Tensor) -> Tensor:
        """Dispatch on ``norm_mode`` (raises on channel mismatch / unfitted statistical)."""
        if tensor.shape[-1] != self.n_channels:
            raise ValueError(
                f"PercentileNormalizer expected {self.n_channels} channels, got {tensor.shape[-1]}"
            )
        if self.norm_mode == NormMode.STATISTICAL:
            if not self._fitted():
                raise RuntimeError(
                    "PercentileNormalizer: statistical mode requires "
                    "statistical_initialization() before forward()"
                )
            return self._apply_bounds(tensor)
        if self.norm_mode == NormMode.RUNNING:
            return self._running_normalize(tensor)
        return self._per_frame_minmax(tensor)

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Accumulate global per-channel percentile bounds across the dataset.

        Preserves the established min-of-batch-lows / max-of-batch-highs
        accumulation (batch-order sensitive); a true streaming percentile is a
        deliberate follow-up, not changed here.
        """
        for batch_data in input_stream:
            data = batch_data["data"]
            flat = data.reshape(-1, self.n_channels).float()
            frame_lo = torch.quantile(flat, self.quantile_low, dim=0)
            frame_hi = torch.quantile(flat, self.quantile_high, dim=0)
            if torch.isnan(self.running_min).any():
                self.running_min.copy_(frame_lo)
                self.running_max.copy_(frame_hi)
            else:
                torch.minimum(self.running_min, frame_lo, out=self.running_min)
                torch.maximum(self.running_max, frame_hi, out=self.running_max)
        if torch.isnan(self.running_min).any():
            raise RuntimeError("PercentileNormalizer.statistical_initialization received no data")


class DisplayNormalizer(_ScoreNormalizerBase):
    """Apply sRGB gamma companding (IEC 61966-2-1) to a ``[0, 1]`` BHWC tensor.

    The stateless display-encoding companion to :class:`PercentileNormalizer`:
    chain it after the normalizer on the false-RGB display path
    (``selector -> PercentileNormalizer -> DisplayNormalizer``) to lift midtones
    so images look natural on standard displays. ML / n-channel paths skip it.

    Ports
    -----
    INPUT_SPECS
        ``data`` : float32, shape (-1, -1, -1, -1), BHWC tensor, values in ``[0, 1]``.
    OUTPUT_SPECS
        ``normalized`` : float32, shape (-1, -1, -1, -1), sRGB gamma-encoded, ``[0, 1]``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.NORMALIZATION, NodeTag.PREPROCESSING, NodeTag.NUMPY})

    def _normalize(self, tensor: Tensor) -> Tensor:
        """sRGB companding: linear ``[0, 1]`` -> gamma-encoded ``[0, 1]``."""
        low = 12.92 * tensor
        high = 1.055 * tensor.clamp_min(1e-10).pow(1.0 / 2.4) - 0.055
        return torch.where(tensor <= 0.0031308, low, high)


__all__ = [
    "DisplayNormalizer",
    "IdentityNormalizer",
    "MinMaxNormalizer",
    "NormMode",
    "PercentileNormalizer",
    "SigmoidNormalizer",
    "ZScoreNormalizer",
    "SigmoidTransform",
    "PerPixelUnitNorm",
]
