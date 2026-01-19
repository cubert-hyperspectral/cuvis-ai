"""Concrete/Gumbel-Softmax band selector node for hyperspectral data.

This module implements a learnable band selector using the Concrete /
Gumbel-Softmax relaxation, suitable for end-to-end training with AdaClip.

The selector learns ``K`` categorical distributions over ``T`` input bands,
and during training uses the Gumbel-Softmax trick to produce differentiable
approximate one-hot selection weights that become increasingly peaked as the
temperature :math:`\\tau` is annealed.

For each output channel :math:`c \\in {1, \\dots, K}`, we learn logits
``L_c in R^T`` and sample:

.. math::

    w_c = \\text{softmax}\\left( \\frac{L_c + g}{\\tau} \\right), \\quad
    g \\sim \\text{Gumbel}(0, 1)

The resulting weights are used to form K-channel RGB-like images:

.. math::

    Y[:, :, c] = \\sum_{t=1}^T w_c[t] \\cdot X[:, :, t]

where ``X`` is the input hyperspectral cube in ``[0, 1]``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
from cuvis_ai_core.utils.types import Context, ExecutionStage


def _sample_gumbel(shape: tuple[int, ...], device: torch.device, eps: float = 1e-10) -> Tensor:
    """Sample Gumbel(0, 1) noise."""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


class ConcreteBandSelector(Node):
    """Concrete/Gumbel-Softmax band selector for hyperspectral cubes.

    Parameters
    ----------
    input_channels : int
        Number of input spectral channels (e.g., 61 for hyperspectral cube).
    output_channels : int, optional
        Number of output channels (default: 3 for RGB/AdaClip compatibility).
    tau_start : float, optional
        Initial temperature for Gumbel-Softmax (default: 10.0).
    tau_end : float, optional
        Final temperature for Gumbel-Softmax (default: 0.1).
    max_epochs : int, optional
        Number of epochs over which to exponentially anneal :math:`\\tau`
        from ``tau_start`` to ``tau_end`` (default: 20).
    use_hard_inference : bool, optional
        If True, uses hard argmax selection at inference/validation time
        (one-hot weights). If False, uses softmax over logits (default: True).
    eps : float, optional
        Small constant for numerical stability (default: 1e-6).

    Notes
    -----
    - During training (``context.stage == 'train'``), the node samples
      Gumbel noise and uses the Concrete relaxation with the current
      temperature :math:`\\tau(\\text{epoch})`.
    - During validation/test/inference, it uses deterministic weights
      without Gumbel noise.
    - The node exposes ``selection_weights`` so that repulsion penalties
      (e.g., DistinctnessLoss) can be attached in the pipeline.
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C_in] in BHWC format.",
        )
    }

    OUTPUT_SPECS = {
        "rgb": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "output_channels"),
            description="Selected-band RGB-like image [B, H, W, C_out].",
        ),
        "selection_weights": PortSpec(
            dtype=torch.float32,
            shape=("output_channels", -1),
            description="Current selection weights [C_out, C_in].",
        ),
    }

    def __init__(
        self,
        input_channels: int,
        output_channels: int = 3,
        tau_start: float = 10.0,
        tau_end: float = 0.1,
        max_epochs: int = 20,
        use_hard_inference: bool = True,
        eps: float = 1e-6,
        **kwargs: Any,
    ) -> None:
        self.input_channels = int(input_channels)
        self.output_channels = int(output_channels)
        self.tau_start = float(tau_start)
        self.tau_end = float(tau_end)
        self.max_epochs = int(max_epochs)
        self.use_hard_inference = bool(use_hard_inference)
        self.eps = float(eps)

        if self.output_channels <= 0:
            raise ValueError(f"output_channels must be positive, got {output_channels}")
        if self.input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}")
        if self.tau_start <= 0.0 or self.tau_end <= 0.0:
            raise ValueError("tau_start and tau_end must be positive.")

        super().__init__(
            input_channels=self.input_channels,
            output_channels=self.output_channels,
            tau_start=self.tau_start,
            tau_end=self.tau_end,
            max_epochs=self.max_epochs,
            use_hard_inference=self.use_hard_inference,
            eps=self.eps,
            **kwargs,
        )

        # Learnable logits for Categorical over input channels: [C_out, C_in]
        self.logits = nn.Parameter(torch.zeros(self.output_channels, self.input_channels))

    def _current_tau(self, context: Context | None) -> float:
        """Compute current temperature based on epoch (exponential schedule)."""
        if context is None or context.stage != ExecutionStage.TRAIN:
            return self.tau_end

        if self.max_epochs <= 1:
            return self.tau_end

        epoch = max(0, min(context.epoch, self.max_epochs - 1))
        return self._get_tau(epoch)

    def _get_tau(self, epoch: int) -> float:
        """Get temperature for a specific epoch (exponential schedule).

        Parameters
        ----------
        epoch : int
            Epoch number (0-indexed).

        Returns
        -------
        float
            Temperature value for the given epoch.
        """
        if self.max_epochs <= 1:
            return self.tau_end

        epoch = max(0, min(epoch, self.max_epochs - 1))
        frac = epoch / float(self.max_epochs - 1)

        # Exponential interpolation in log-space between tau_start and tau_end
        log_tau_start = math.log(self.tau_start)
        log_tau_end = math.log(self.tau_end)
        log_tau = (1.0 - frac) * log_tau_start + frac * log_tau_end
        return float(math.exp(log_tau))

    def get_selection_weights(self, deterministic: bool = True) -> Tensor:
        """Return current selection weights without data dependency.

        Parameters
        ----------
        deterministic : bool, optional
            If True, uses softmax over logits (no Gumbel noise) at a
            "midpoint" temperature (geometric mean of start/end). If False,
            uses current logits with ``tau_end``.
        """
        if deterministic:
            tau = math.sqrt(self.tau_start * self.tau_end)
        else:
            tau = self.tau_end

        return F.softmax(self.logits / tau, dim=-1)

    def get_selected_bands(self) -> Tensor:
        """Return argmax band indices per output channel."""
        with torch.no_grad():
            return torch.argmax(self.logits, dim=-1)

    def forward(
        self,
        data: Tensor,
        context: Context | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply Concrete/Gumbel-Softmax band selection.

        Parameters
        ----------
        data : Tensor
            Input tensor [B, H, W, C_in] in BHWC format.
        context : Context, optional
            Execution context with stage and epoch information.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with:
            - ``"rgb"``: [B, H, W, C_out] RGB-like image.
            - ``"selection_weights"``: [C_out, C_in] current weights.
        """
        B, H, W, C_in = data.shape

        tau = self._current_tau(context)
        device = data.device

        if self.training and context is not None and context.stage == ExecutionStage.TRAIN:
            # Gumbel-Softmax sampling during training
            g = _sample_gumbel(self.logits.shape, device=device, eps=self.eps)
            weights = F.softmax((self.logits + g) / tau, dim=-1)  # [C_out, C_in]
        else:
            # Deterministic selection for val/test/inference
            if self.use_hard_inference:
                # Hard argmax → one-hot
                indices = torch.argmax(self.logits, dim=-1)  # [C_out]
                weights = torch.zeros_like(self.logits)
                weights.scatter_(1, indices.unsqueeze(-1), 1.0)
            else:
                # Softmax over logits at low temperature
                weights = F.softmax(self.logits / self.tau_end, dim=-1)

        # Weighted sum over spectral dimension: [B, H, W, C_in] x [C_out, C_in] -> [B, H, W, C_out]
        rgb = torch.einsum("bhwc,kc->bhwk", data, weights)

        return {
            "rgb": rgb,
            "selection_weights": weights,
        }


__all__ = ["ConcreteBandSelector"]
