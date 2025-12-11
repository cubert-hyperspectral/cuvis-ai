"""Soft channel selector node for learnable channel selection."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import InputStream


class SoftChannelSelector(Node):
    """Soft channel selector with temperature-based Gumbel-Softmax selection.

    This node learns to select a subset of input channels using differentiable
    channel selection with temperature annealing. Supports:
    - Statistical initialization (uniform or importance-based)
    - Gradient-based optimization with temperature scheduling
    - Entropy and diversity regularization
    - Hard selection at inference time

    Parameters
    ----------
    n_select : int
        Number of channels to select
    input_channels : int
        Number of input channels
    init_method : {"uniform", "variance"}, optional
        Initialization method for channel weights (default: "uniform")
    temperature_init : float, optional
        Initial temperature for Gumbel-Softmax (default: 5.0)
    temperature_min : float, optional
        Minimum temperature (default: 0.1)
    temperature_decay : float, optional
        Temperature decay factor per epoch (default: 0.9)
    hard : bool, optional
        If True, use hard selection at inference (default: False)
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)

    Attributes
    ----------
    channel_logits : nn.Parameter or Tensor
        Unnormalized channel importance scores [n_channels]
    temperature : float
        Current temperature for Gumbel-Softmax
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "selected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Channel-weighted output (same shape as input)",
        ),
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Current channel selection weights",
        ),
    }

    def __init__(
        self,
        n_select: int,
        input_channels: int,
        init_method: Literal["uniform", "variance"] = "uniform",
        temperature_init: float = 5.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.9,
        hard: bool = False,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.n_select = n_select
        self.input_channels = input_channels
        self.init_method = init_method
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.hard = hard
        self.eps = eps

        super().__init__(
            n_select=n_select,
            input_channels=input_channels,
            init_method=init_method,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            hard=hard,
            eps=eps,
            **kwargs,
        )

        # Temperature tracking (not a parameter, managed externally)
        self.temperature = temperature_init
        self._n_channels = input_channels

        # Validate selection size
        if self.n_select > self._n_channels:
            raise ValueError(
                f"Cannot select {self.n_select} channels from {self._n_channels} available channels"
            )

        # Initialize channel logits based on method - always as buffer
        if self.init_method == "uniform":
            # Uniform initialization
            logits = torch.zeros(self._n_channels)
        elif self.init_method == "variance":
            # Random initialization - will be refined with fit if called
            logits = torch.randn(self._n_channels) * 0.01
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Store as buffer initially
        self.register_buffer("channel_logits", logits)

        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize channel selection weights from data.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"data": tensor} where tensor is BHWC
        """
        # Collect statistics from first batch to determine n_channels
        first_batch = next(iter(input_stream))
        x = first_batch["data"]

        if x is None:
            raise ValueError("No data provided for selector initialization")

        self._n_channels = x.shape[-1]

        if self.n_select > self._n_channels:
            raise ValueError(
                f"Cannot select {self.n_select} channels from {self._n_channels} available channels"
            )

        # Initialize channel logits based on method
        if self.init_method == "uniform":
            # Uniform initialization
            logits = torch.zeros(self._n_channels)
        elif self.init_method == "variance":
            # Importance-based initialization using channel variance
            all_data = []
            all_data.append(x)

            # Collect more data for better statistics
            for batch_data in input_stream:
                x_batch = batch_data["data"]
                if x_batch is not None:
                    all_data.append(x_batch)

            # Concatenate and compute variance per channel
            X = torch.cat(all_data, dim=0)  # [B, H, W, C]
            X_flat = X.reshape(-1, X.shape[-1])  # [B*H*W, C]

            # Compute variance for each channel
            variance = X_flat.var(dim=0)  # [C]

            # Use log variance as initial logits (high variance = high importance)
            logits = torch.log(variance + self.eps)
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        # Store as buffer
        self.channel_logits.data[:] = logits.clone()
        self._statistically_initialized = True

    def unfreeze(self) -> None:
        """Convert channel logits buffer to trainable nn.Parameter.

        Call this method to enable gradient-based optimization of channel
        selection weights. The logits will be converted from a buffer to an
        nn.Parameter, allowing gradient updates during training.

        Example
        -------
        >>> selector = SoftChannelSelector(n_select=10, input_channels=100)
        >>> selector.unfreeze()  # Enable gradient training
        >>> # Now channel selection weights can be optimized
        """
        if self.channel_logits is not None:
            # Convert buffer to parameter
            self.channel_logits = nn.Parameter(self.channel_logits.clone())
        # Call parent to enable requires_grad
        super().unfreeze()

    def update_temperature(self, epoch: int | None = None, step: int | None = None) -> None:
        """Update temperature with decay schedule.

        Parameters
        ----------
        epoch : int, optional
            Current epoch number (used for per-epoch decay)
        step : int, optional
            Current training step (for more granular control)
        """
        if epoch is not None:
            # Exponential decay per epoch
            self.temperature = max(
                self.temperature_min, self.temperature_init * (self.temperature_decay**epoch)
            )

    def get_selection_weights(self, hard: bool | None = None) -> Tensor:
        """Get current channel selection weights.

        Parameters
        ----------
        hard : bool, optional
            If True, use hard selection (top-k). If None, uses self.hard.

        Returns
        -------
        Tensor
            Selection weights [n_channels] summing to n_select
        """
        if hard is None:
            hard = self.hard and not self.training

        if hard:
            # Hard selection: top-k channels
            _, top_indices = torch.topk(self.channel_logits, self.n_select)
            weights = torch.zeros_like(self.channel_logits)
            weights[top_indices] = 1.0
        else:
            # Soft selection with Gumbel-Softmax
            # First, compute selection probabilities
            probs = F.softmax(self.channel_logits / self.temperature, dim=-1)

            # Scale to sum to n_select instead of 1
            weights = probs * self.n_select

        return weights

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Apply soft channel selection to input.

        Parameters
        ----------
        data : Tensor
            Input tensor [B, H, W, C]

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "selected" key containing reweighted channels
            and optional "weights" key containing selection weights
        """
        # Get selection weights
        weights = self.get_selection_weights()

        # Ensure weights are on same device
        # weights = weights.to(data.device) # no need

        # Apply channel-wise weighting: [B, H, W, C] * [C]
        selected = data * weights.view(1, 1, 1, -1)

        # Prepare output dictionary - weights always exposed for loss/metric nodes
        outputs = {"selected": selected, "weights": weights}

        return outputs


class TopKIndices(Node):
    """Utility node that surfaces the top-k channel indices from selector weights.

    This node extracts the indices of the top-k weighted channels from a selector's
    weight vector. Useful for introspection and reporting which channels were selected.

    Parameters
    ----------
    k : int
        Number of top indices to return

    Attributes
    ----------
    k : int
        Number of top indices to return
    """

    INPUT_SPECS = {
        "weights": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Channel selection weights",
        )
    }
    OUTPUT_SPECS = {
        "indices": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Top-k channel indices",
        )
    }

    def __init__(self, k: int, **kwargs: Any) -> None:
        self.k = int(k)

        # Extract Node base parameters from kwargs to avoid duplication
        name = kwargs.pop("name", None)
        execution_stages = kwargs.pop("execution_stages", None)

        super().__init__(
            name=name,
            execution_stages=execution_stages,
            k=self.k,
            **kwargs,
        )

    def forward(self, weights: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Return the indices of the top-k weighted channels.

        Parameters
        ----------
        weights : torch.Tensor
            Channel selection weights [n_channels]

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "indices" key containing top-k indices
        """
        top_k = min(self.k, weights.shape[-1]) if weights.numel() else 0
        if top_k == 0:
            return {"indices": torch.zeros(0, dtype=torch.int64, device=weights.device)}

        _, indices = torch.topk(weights, top_k)
        return {"indices": indices}


__all__ = ["SoftChannelSelector", "TopKIndices"]
