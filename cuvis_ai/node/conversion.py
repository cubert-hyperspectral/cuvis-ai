"""RX Logit Head for Anomaly Detection

This module provides a trainable head that converts RX anomaly scores into
logits for binary anomaly classification. It can be trained end-to-end with
binary cross-entropy loss.
"""

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.utils.welford import WelfordAccumulator


class ScoreToLogit(Node):
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
    >>> logit_head = ScoreToLogit(init_scale=1.0, init_bias=5.0)
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

    TRAINABLE_BUFFERS = ("scale", "bias")

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

        self._welford = WelfordAccumulator(1)
        # Allow using the head with the provided init_scale/init_bias without forcing a fit()
        self._statistically_initialized = True

    def statistical_initialization(self, input_stream) -> None:
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

        if self._welford.count > 0:
            self.finalize()
        self._statistically_initialized = True

    @torch.no_grad()
    def update(self, scores: torch.Tensor) -> None:
        """Update running statistics with a batch of scores.

        Parameters
        ----------
        scores : torch.Tensor
            Batch of RX scores in BHWC format
        """
        X = scores.flatten()
        if X.shape[0] <= 1:
            return
        self._welford.update(X)
        self._statistically_initialized = False

    @torch.no_grad()
    def finalize(self) -> None:
        """Finalize statistics and set bias to mean + 2*std.

        This threshold (mean + 2*std) is a common heuristic for anomaly detection,
        capturing ~95% of normal data under Gaussian assumption.
        """
        if self._welford.count <= 1:
            raise ValueError("Not enough samples to finalize ScoreToLogit statistics.")

        mean = self._welford.mean.squeeze()
        std = self._welford.std.squeeze()

        # Set bias to mean + 2*std (threshold for anomalies)
        self.bias = mean + 2.0 * std
        self._statistically_initialized = True

    def reset(self) -> None:
        """Reset all statistics and accumulators."""
        self._welford.reset()
        self._statistically_initialized = False

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

        if not self._statistically_initialized:
            raise RuntimeError(
                "ScoreToLogit not initialized. Call statistical_initialization() before forward()."
            )
        # Apply affine transformation: logit = scale * (score - bias)
        logits = self.scale * (scores - self.bias)

        return {"logits": logits}

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
