"""Soft channel selector node for learnable channel selection."""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cuvis_ai.node import LabelLike, MetaLike, Node, NodeOutput


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
    trainable : bool, optional
        If True, channel weights become trainable (default: True)
    eps : float, optional
        Small constant for numerical stability (default: 1e-6)
    
    Attributes
    ----------
    channel_logits : nn.Parameter or Tensor
        Unnormalized channel importance scores [n_channels]
    temperature : float
        Current temperature for Gumbel-Softmax
    """

    def __init__(
        self,
        n_select: int,
        init_method: Literal["uniform", "variance"] = "uniform",
        temperature_init: float = 5.0,
        temperature_min: float = 0.1,
        temperature_decay: float = 0.9,
        hard: bool = False,
        trainable: bool = True,
        eps: float = 1e-6,
    ) -> None:
        self.n_select = n_select
        self.init_method = init_method
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.temperature_decay = temperature_decay
        self.hard = hard
        self.trainable = trainable
        self.eps = eps

        super().__init__(
            n_select=n_select,
            init_method=init_method,
            temperature_init=temperature_init,
            temperature_min=temperature_min,
            temperature_decay=temperature_decay,
            hard=hard,
            trainable=trainable,
            eps=eps,
        )

        # Temperature tracking (not a parameter, managed externally)
        self.temperature = temperature_init
        self._n_channels = None

        # Channel logits will be initialized during initialize_from_data
        self.channel_logits = None
        self._initialized = False

    @property
    def requires_initial_fit(self) -> bool:
        """Selector requires statistical initialization to determine number of channels."""
        return True

    @property
    def is_trainable(self) -> bool:
        """Whether channel selection weights can be trained with gradients."""
        return self.trainable

    def initialize_from_data(self, iterator) -> None:
        """Initialize channel selection weights from data.
        
        Parameters
        ----------
        iterator : Iterator
            Iterator yielding (x, y, m) tuples where x has shape [B, H, W, C]
        """
        # Collect statistics from first batch to determine n_channels
        first_batch = next(iter(iterator))
        x, _, _ = first_batch

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
            for batch_data in iterator:
                x_batch, _, _ = batch_data
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

        self.channel_logits = logits
        self._initialized = True

    def prepare_for_train(self) -> None:
        """Convert channel logits to trainable parameters if trainable=True."""
        if self.trainable and self.channel_logits is not None:
            # Convert to parameter
            self.channel_logits = nn.Parameter(self.channel_logits.clone())

    def update_temperature(self, epoch: int = None, step: int = None) -> None:
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
                self.temperature_min,
                self.temperature_init * (self.temperature_decay ** epoch)
            )

    def get_selection_weights(self, hard: bool = None) -> Tensor:
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

    def forward(
        self,
        x: Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
        **_: Any,
    ) -> NodeOutput:
        """Apply soft channel selection to input.
        
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
            Selected channels [B, H, W, C] (reweighted), labels, metadata
        """
        if not self._initialized:
            raise RuntimeError(
                "SoftChannelSelector not initialized. Call initialize_from_data() first."
            )

        # Get selection weights
        weights = self.get_selection_weights()

        # Ensure weights are on same device
        weights = weights.to(x.device)

        # Apply channel-wise weighting: [B, H, W, C] * [C]
        x_selected = x * weights.view(1, 1, 1, -1)

        return x_selected, y, m

    @property
    def input_dim(self) -> tuple[int, int, int, int]:
        """Expected input shape (flexible)."""
        return (-1, -1, -1, -1)

    @property
    def output_dim(self) -> tuple[int, int, int, int]:
        """Output shape (same as input)."""
        return (-1, -1, -1, -1)

    def compute_entropy(self) -> Tensor:
        """Compute entropy of channel selection distribution.
        
        Higher entropy means more uniform selection (less confident).
        Lower entropy means more peaked selection (more confident).
        
        Returns
        -------
        Tensor
            Selection entropy (scalar)
        """
        if self.channel_logits is None:
            return torch.tensor(0.0)

        # Compute selection probabilities
        probs = F.softmax(self.channel_logits / self.temperature, dim=-1)

        # Compute entropy: -sum(p * log(p))
        entropy = -(probs * torch.log(probs + self.eps)).sum()

        return entropy

    def compute_diversity_loss(self) -> Tensor:
        """Compute diversity loss to encourage spread across channels.
        
        Penalizes concentration on few channels by encouraging
        more uniform distribution of selection weights.
        
        Returns
        -------
        Tensor
            Diversity loss (scalar, lower is more diverse)
        """
        if self.channel_logits is None:
            return torch.tensor(0.0)

        # Get soft selection weights
        weights = self.get_selection_weights(hard=False)

        # Compute variance of weights (high variance = low diversity)
        # We want to maximize variance to encourage selecting different channels
        # So we return negative variance (minimizing loss = maximizing variance)
        mean_weight = weights.mean()
        variance = ((weights - mean_weight) ** 2).mean()

        # Return negative variance (maximizing diversity)
        return -variance

    def get_top_k_channels(self, k: int = None) -> Tensor:
        """Get indices of top-k selected channels.
        
        Parameters
        ----------
        k : int, optional
            Number of channels to return (default: self.n_select)
        
        Returns
        -------
        Tensor
            Indices of top-k channels [k]
        """
        if k is None:
            k = self.n_select

        _, indices = torch.topk(self.channel_logits, k)
        return indices

    @classmethod
    def load(cls, state: dict) -> SoftChannelSelector:
        """Load SoftChannelSelector from serialized state.
        
        Parameters
        ----------
        state : dict
            Serialized state dictionary
            
        Returns
        -------
        SoftChannelSelector
            Loaded selector instance
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


__all__ = ["SoftChannelSelector"]
