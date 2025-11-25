"""RX Logit Head for Anomaly Detection

This module provides a trainable head that converts RX anomaly scores into
logits for binary anomaly classification. It can be trained end-to-end with
binary cross-entropy loss.
"""

import torch
import torch.nn as nn

from cuvis_ai.node import Node
from cuvis_ai.pipeline.ports import PortSpec


class RXLogitHead(Node):
    """Trainable head that converts RX scores to anomaly logits.

    This node takes RX anomaly scores (typically Mahalanobis distances) and
    applies a learned affine transformation to produce logits suitable for
    binary classification with BCEWithLogitsLoss.

    The transformation is: logit = scale * (score - bias)

    Parameters
    ----------
    init_scale : float, default=1.0
        Initial value for the scale parameter
    init_bias : float, default=0.0
        Initial value for the bias parameter (threshold)

    Attributes
    ----------
    scale : nn.Parameter or torch.Tensor
        Scale factor applied to scores
    bias : nn.Parameter or torch.Tensor
        Bias (threshold) subtracted from scores before scaling

    Examples
    --------
    >>> # After RX detector
    >>> rx = RXGlobal(eps=1e-6)
    >>> logit_head = RXLogitHead(init_scale=1.0, init_bias=5.0)
    >>> logit_head.unfreeze()  # Enable gradient training
    >>> graph.connect(rx.scores, logit_head.scores)
    """

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="RX anomaly scores (Mahalanobis distances)",
        )
    }

    OUTPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly logits for BCEWithLogitsLoss",
        )
    }

    def __init__(
        self,
        init_scale: float = 1.0,
        init_bias: float = 0.0,
        **kwargs,
    ) -> None:
        self.init_scale = init_scale
        self.init_bias = init_bias

        super().__init__(
            init_scale=init_scale,
            init_bias=init_bias,
            **kwargs,
        )

        # Initialize as buffers (frozen by default)
        self.register_buffer("scale", torch.tensor(init_scale, dtype=torch.float32))
        self.register_buffer("bias", torch.tensor(init_bias, dtype=torch.float32))

        # Streaming accumulators for statistics (similar to RXGlobal)
        self.register_buffer("_mean", None)
        self.register_buffer("_M2", None)
        self._n = 0
        self._fitted = False

    @property
    def requires_initial_fit(self) -> bool:
        """RXLogitHead requires statistical initialization from data."""
        return True

    def unfreeze(self) -> None:
        """Convert scale and bias buffers to trainable nn.Parameters.

        Call this method to enable gradient-based optimization of the
        scale and bias parameters. They will be converted from buffers to
        nn.Parameters, allowing gradient updates during training.

        Example
        -------
        >>> logit_head = RXLogitHead(init_scale=1.0, init_bias=5.0)
        >>> logit_head.unfreeze()  # Enable gradient training
        >>> # Now scale and bias can be optimized
        """
        if self.scale is not None and self.bias is not None:
            # Convert buffers to parameters
            self.scale = nn.Parameter(self.scale.clone(), requires_grad=True)
            self.bias = nn.Parameter(self.bias.clone(), requires_grad=True)
        # Call parent to enable requires_grad
        super().unfreeze()

    def fit(self, input_stream) -> None:
        """Initialize bias from statistics of RX scores using streaming approach.

        Uses Welford's algorithm for numerically stable online computation of
        mean and standard deviation, similar to RXGlobal.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"scores": tensor} where tensor is the RX scores
        """
        self.reset()
        for batch_data in input_stream:
            # Extract scores from port-based dict
            scores = batch_data.get("scores")
            if scores is not None:
                self.update(scores)

        if self._n > 0:
            self.finalize()
        self._initialized = True

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        """Update running statistics with a batch of scores.

        Uses Welford's online algorithm for numerically stable computation.

        Parameters
        ----------
        scores : torch.Tensor
            Batch of RX scores in BHWC format
        """
        # Flatten to 1D
        X = scores.flatten()
        m = X.shape[0]
        if m <= 1:
            return

        mean_b = X.mean()
        M2_b = ((X - mean_b) ** 2).sum()

        if self._n == 0:
            self._n = m
            self._mean = mean_b
            self._M2 = M2_b
        else:
            # Type guard: these should not be None if _n > 0
            assert self._mean is not None and self._M2 is not None
            n = self._n
            tot = self._n + m
            delta = mean_b - self._mean
            new_mean = self._mean + delta * (m / tot)
            self._M2 = self._M2 + M2_b + (delta**2) * (n * m / tot)
            self._n = tot
            self._mean = new_mean
        self._fitted = False

    @torch.no_grad()
    def finalize(self) -> None:
        """Finalize statistics and set bias to mean + 2*std.

        This threshold (mean + 2*std) is a common heuristic for anomaly detection,
        capturing ~95% of normal data under Gaussian assumption.
        """
        if self._n <= 1:
            raise ValueError("Not enough samples to finalize RXLogitHead statistics.")

        # Type guard: these should not be None if _n > 1
        assert self._mean is not None and self._M2 is not None

        mean = self._mean.clone()
        variance = self._M2 / (self._n - 1)
        std = torch.sqrt(variance)

        # Set bias to mean + 2*std (threshold for anomalies)
        self.bias = mean + 2.0 * std
        self._fitted = True

    def reset(self) -> None:
        """Reset all statistics and accumulators."""
        self._n = 0
        self._mean = None
        self._M2 = None
        self._fitted = False

    def forward(self, scores: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Transform RX scores to logits.

        Parameters
        ----------
        scores : torch.Tensor
            Input RX scores with shape (B, H, W, 1)

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "logits" key containing transformed scores
        """

        # Apply affine transformation: logit = scale * (score - bias)
        logits = self.scale * (scores - self.bias)

        return {"logits": logits}

    def serialize(self, serial_dir: str) -> dict:
        """Serialize RXLogitHead state."""
        return {
            "params": {
                "init_scale": self.init_scale,
                "init_bias": self.init_bias,
            },
            "state_dict": self.state_dict(),
        }

    def load(self, params: dict, serial_dir: str) -> None:
        """Load RXLogitHead state from serialized data."""
        config = params.get("params", {})
        self.init_scale = config.get("init_scale", self.init_scale)
        self.init_bias = config.get("init_bias", self.init_bias)

        state = params.get("state_dict", {})
        if state:
            self.load_state_dict(state, strict=False)

    def get_threshold(self) -> float:
        """Get the current anomaly threshold (bias value).

        Returns
        -------
        float
            Current threshold value
        """
        return self.bias.item()

    def set_threshold(self, threshold: float) -> None:
        """Set the anomaly threshold (bias value).

        Parameters
        ----------
        threshold : float
            New threshold value
        """
        with torch.no_grad():
            self.bias.fill_(threshold)

    def predict_anomalies(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to binary anomaly predictions.

        Parameters
        ----------
        logits : torch.Tensor
            Logits from forward pass, shape (B, H, W, 1)

        Returns
        -------
        torch.Tensor
            Binary predictions (0=normal, 1=anomaly), shape (B, H, W, 1)
        """
        return (logits > 0).float()
