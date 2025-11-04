from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai.node import LabelLike, MetaLike, Node, NodeOutput


class _ScoreNormalizerBase(Node):
    """Base class for differentiable score normalizers operating on BHWC tensors."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        x: Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
        **_: Any,
    ) -> NodeOutput:
        normalized = self._normalize(x)
        return normalized, y, m

    def _normalize(self, tensor: Tensor) -> Tensor:
        raise NotImplementedError

    @property
    def input_dim(self) -> tuple[int, int, int, int]:
        return (-1, -1, -1, -1)

    @property
    def output_dim(self) -> tuple[int, int, int, int]:
        return (-1, -1, -1, 1)

    def load(self, params: dict, serial_dir: str) -> None:
        config = params.get("config", {})
        for key, value in config.items():
            setattr(self, key, value)
        self._config = dict(config)


class IdentityNormalizer(_ScoreNormalizerBase):
    """No-op normalizer; preserves incoming scores."""

    def __init__(self) -> None:
        super().__init__()

    def _normalize(self, tensor: Tensor) -> Tensor:
        return tensor


class MinMaxNormalizer(_ScoreNormalizerBase):
    """Min-max normalization per sample and channel (keeps gradients).
    
    Can operate in two modes:
    1. Per-sample normalization (default): min/max computed per batch
    2. Global normalization: uses running statistics from initialization
    """

    def __init__(self, eps: float = 1e-6, use_running_stats: bool = True) -> None:
        self.eps = float(eps)
        self.use_running_stats = use_running_stats
        super().__init__(eps=eps, use_running_stats=use_running_stats)

        # Running statistics for global normalization
        self.register_buffer("running_min", None)
        self.register_buffer("running_max", None)
        self._stats_initialized = False

    @property
    def requires_initial_fit(self) -> bool:
        """MinMaxNormalizer can optionally use global statistics."""
        return self.use_running_stats

    def initialize_from_data(self, iterator) -> None:
        """Compute global min/max from data iterator.
        
        Parameters
        ----------
        iterator : Iterator
            Iterator yielding (x, y, m) tuples where x is the scores tensor
        """
        all_mins = []
        all_maxs = []

        for batch_data in iterator:
            x, _, _ = batch_data
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
            self._stats_initialized = True
            self._initialized = True

    def _normalize(self, tensor: Tensor) -> Tensor:
        B, H, W, C = tensor.shape
        flat = tensor.view(B, -1, C)

        # Use running stats if available and initialized
        if self.use_running_stats and self._stats_initialized:
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

    def load(self, params: dict, serial_dir: str) -> None:
        super().load(params, serial_dir)
        cfg_eps = self._config.get("eps", self.eps)
        self.eps = float(cfg_eps)
        self.use_running_stats = self._config.get("use_running_stats", True)
        self._config["eps"] = self.eps
        self._config["use_running_stats"] = self.use_running_stats


class SigmoidNormalizer(_ScoreNormalizerBase):
    """Median-centered sigmoid squashing per sample and channel."""

    def __init__(self, std_floor: float = 1e-6) -> None:
        super().__init__()
        self.std_floor = float(std_floor)
        self._config.update({"std_floor": self.std_floor})

    def _normalize(self, tensor: Tensor) -> Tensor:
        B, H, W, C = tensor.shape
        flat = tensor.view(B, -1, C)
        medians = flat.median(dim=1, keepdim=True).values
        stds = flat.std(dim=1, unbiased=False, keepdim=True)
        stds = torch.clamp(stds, min=self.std_floor)
        normalized = torch.sigmoid((flat - medians) / stds)
        return normalized.view(B, H, W, C)

    def load(self, params: dict, serial_dir: str) -> None:
        super().load(params, serial_dir)
        cfg_std = self._config.get("std_floor", self.std_floor)
        self.std_floor = float(cfg_std)
        self._config["std_floor"] = self.std_floor


__all__ = [
    "IdentityNormalizer",
    "MinMaxNormalizer",
    "SigmoidNormalizer",
    # "resolve_score_normalizer",
]
