"""RX anomaly detection nodes for hyperspectral imaging.

This module implements the Reed-Xiaoli (RX) anomaly detection algorithm, a widely used
statistical method for detecting anomalies in hyperspectral images. The RX algorithm
computes squared Mahalanobis distance from the background distribution, treating
pixels with large distances as potential anomalies.

The module provides two variants:

- **RXGlobal**: Uses global statistics (mean, covariance) estimated from training data.
  Supports two-phase training: statistical initialization followed by optional gradient-based
  fine-tuning via unfreeze().

- **RXPerBatch**: Computes statistics independently for each batch on-the-fly without
  requiring initialization. Useful for real-time processing or when training data is unavailable.

Reference:
    Reed, I. S., & Yu, X. (1990). "Adaptive multiple-band CFAR detection of an optical
    pattern with unknown spectral distribution." IEEE Transactions on Acoustics, Speech,
    and Signal Processing, 38(10), 1760-1770.
"""

import torch
import torch.nn as nn
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec


def _flatten_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Flatten spatial dimensions of BHWC tensor to BNC format.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor with shape (B, H, W, C)

    Returns
    -------
    torch.Tensor
        Flattened tensor with shape (B, H*W, C)
    """
    B, H, W, C = x.shape
    return x.view(B, H * W, C)


## This node is not approved
# missing unfreeze/freeze
# missing approved documentation and alignment with current API


# ---------- Shared base ----------
class RXBase(Node):
    """Base class for RX anomaly detectors."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Anomaly scores per pixel (BHW1 format)",
        )
    }

    def __init__(self, eps: float = 1e-6, **kwargs) -> None:
        self.eps = eps
        super().__init__(eps=eps, **kwargs)

    @staticmethod
    def _quad_form_solve(Xc: torch.Tensor, cov: torch.Tensor) -> torch.Tensor:
        """
        Xc: (B,N,C) centered; cov: (C,C) or (B,C,C)
        returns md2: (B,N)
        """
        B, N, C = Xc.shape
        rhs = Xc.transpose(1, 2)  # (B,C,N)
        covB = cov if cov.dim() == 3 else cov.unsqueeze(0).expand(B, C, C)
        y = torch.linalg.solve(covB, rhs)  # (B,C,N)
        md2 = (rhs * y).sum(dim=1)  # (B,N)
        return md2


# ---------- Trained/global variant ----------
class RXGlobal(RXBase):
    """RX anomaly detector with global background statistics.

    Uses global mean (μ) and covariance (Σ) estimated from training data to compute
    Mahalanobis distance scores. Supports two-phase training: statistical initialization
    followed by optional gradient-based fine-tuning.

    The detector computes anomaly scores as:

        RX(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)

    where x is a pixel spectrum, μ is the background mean, and Σ is the covariance matrix.

    Parameters
    ----------
    num_channels : int
        Number of spectral channels in input data
    eps : float, optional
        Small constant added to covariance diagonal for numerical stability (default: 1e-6)
    cache_inverse : bool, optional
        If True, precompute and cache Σ⁻¹ for faster inference (default: True)
    **kwargs : dict
        Additional arguments passed to Node base class

    Attributes
    ----------
    mu : Tensor or nn.Parameter
        Background mean spectrum, shape (C,). Initially a buffer, becomes Parameter after unfreeze()
    cov : Tensor or nn.Parameter
        Background covariance matrix, shape (C, C)
    cov_inv : Tensor or nn.Parameter
        Cached pseudo-inverse of covariance (if cache_inverse=True)
    _statistically_initialized : bool
        Flag indicating whether statistical_initialization() has been called

    Examples
    --------
    >>> from cuvis_ai.anomaly.rx_detector import RXGlobal
    >>> from cuvis_ai_core.training import StatisticalTrainer
    >>>
    >>> # Create RX detector
    >>> rx = RXGlobal(num_channels=61, eps=1.0e-6)
    >>>
    >>> # Phase 1: Statistical initialization
    >>> stat_trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
    >>> stat_trainer.fit()  # Computes μ and Σ from training data
    >>>
    >>> # Inference with frozen statistics
    >>> output = rx.forward(data=hyperspectral_cube)
    >>> scores = output["scores"]  # [B, H, W, 1]
    >>>
    >>> # Phase 2: Optional gradient-based fine-tuning
    >>> rx.unfreeze()  # Convert buffers to nn.Parameters
    >>> # Now μ and Σ can be updated with gradient descent

    See Also
    --------
    RXPerBatch : Per-batch RX variant without training
    MinMaxNormalizer : Recommended preprocessing before RX
    ScoreToLogit : Convert scores to logits for classification
    docs/tutorials/rx-statistical.md : Complete RX pipeline tutorial

    Notes
    -----
    After statistical_initialization(), mu and cov are stored as buffers (frozen by default).
    Call unfreeze() to convert them to trainable nn.Parameters for gradient-based optimization.
    """

    def __init__(
        self, num_channels: int, eps: float = 1e-6, cache_inverse: bool = True, **kwargs
    ) -> None:
        self.num_channels = int(num_channels)
        self.eps = eps
        self.cache_inverse = cache_inverse
        # Call Node.__init__ directly with all parameters for proper serialization
        # We bypass RXBase.__init__ since it only accepts eps
        # Node.__init__(self, num_channels=self.num_channels, eps=self.eps, cache_inverse=self.cache_inverse)

        super().__init__(
            num_channels=self.num_channels, eps=self.eps, cache_inverse=self.cache_inverse, **kwargs
        )

        # global stats - all stored as buffers initially
        self.register_buffer("mu", torch.zeros(self.num_channels, dtype=torch.float32))  # (C,)
        self.register_buffer(
            "cov", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float32)
        )  # (C,C)
        self.register_buffer(
            "cov_inv", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float32)
        )  # (C,C)
        # Streaming accumulators (float64 for numerical stability)
        self.register_buffer("_mean", torch.zeros(self.num_channels, dtype=torch.float64))
        self.register_buffer(
            "_M2", torch.zeros(self.num_channels, self.num_channels, dtype=torch.float64)
        )
        self._n = 0
        self._statistically_initialized = False

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize mu and Sigma from data iterator.

        Parameters
        ----------
        input_stream : InputStream
            Iterator yielding dicts matching INPUT_SPECS (port-based format)
            Expected format: {"data": tensor} where tensor is BHWC
        """
        self.reset()
        for batch_data in input_stream:
            # Extract data from port-based dict
            x = batch_data["data"]
            if x is not None:
                self.update(x)

        if self._n > 0:
            self.finalize()
        self._statistically_initialized = True

    def unfreeze(self) -> None:
        """Convert mu and cov buffers to trainable nn.Parameters.

        Call this method after fit() to enable gradient-based optimization of
        the mean and covariance statistics. They will be converted from buffers
        to nn.Parameters, allowing gradient updates during training.

        Example
        -------
        >>> rx.fit(input_stream)  # Statistical initialization
        >>> rx.unfreeze()  # Enable gradient training
        >>> # Now RX statistics can be fine-tuned with gradient descent
        """
        if self.mu.numel() > 0 and self.cov.numel() > 0:
            # Convert buffers to parameters
            self.mu = nn.Parameter(self.mu.clone(), requires_grad=True)
            self.cov = nn.Parameter(self.cov.clone(), requires_grad=True)
            if self.cov_inv.numel() > 0:
                self.cov_inv = nn.Parameter(self.cov_inv.clone(), requires_grad=True)
        # Call parent to enable requires_grad
        super().unfreeze()

    @torch.no_grad()
    def update(self, batch_bhwc: torch.Tensor) -> None:
        """Update streaming statistics with a new batch using Welford's algorithm.

        This method incrementally computes mean and covariance from batches of data
        using Welford's numerically stable online algorithm. Multiple batches can be
        processed sequentially before calling finalize() to compute final statistics.

        Parameters
        ----------
        batch_bhwc : torch.Tensor
            Input batch in BHWC format, shape (B, H, W, C)

        Notes
        -----
        The algorithm maintains running accumulators (_mean, _M2, _n) in float64
        for numerical stability. Batches with ≤1 samples are ignored. After update(),
        call finalize() to compute mu and cov from the accumulated statistics.
        """
        X = _flatten_bhwc(batch_bhwc).reshape(-1, batch_bhwc.shape[-1])  # (M,C)
        m = X.shape[0]
        if m <= 1:
            return
        mean_b = X.mean(0)
        M2_b = (X - mean_b).T @ (X - mean_b)
        if self._n == 0:
            self._n, self._mean, self._M2 = m, mean_b, M2_b
        else:
            n, tot = self._n, self._n + m
            delta = mean_b - self._mean
            new_mean = self._mean + delta * (m / tot)
            outer = torch.outer(delta, delta) * (n * m / tot)
            self._n, self._mean, self._M2 = tot, new_mean, self._M2 + M2_b + outer
        self._statistically_initialized = False

    @torch.no_grad()
    def finalize(self) -> "RXGlobal":
        """Compute final mean and covariance from accumulated streaming statistics.

        This method converts the running accumulators (_mean, _M2) into the final
        mean (mu) and covariance (cov) matrices. The covariance is regularized with
        eps * I for numerical stability, and optionally caches the pseudo-inverse.

        Returns
        -------
        RXGlobal
            Returns self for method chaining

        Raises
        ------
        ValueError
            If fewer than 2 samples were accumulated (insufficient for covariance estimation)

        Notes
        -----
        After finalization, mu and cov are stored as buffers (frozen by default).
        Call unfreeze() to convert them to nn.Parameters for gradient-based training.
        """
        if self._n <= 1:
            raise ValueError("Not enough samples to finalize.")
        mu = self._mean.clone()
        cov = self._M2 / (self._n - 1)
        if self.eps > 0:
            cov = cov + self.eps * torch.eye(cov.shape[0], device=cov.device, dtype=cov.dtype)
        # Always store as buffers initially (frozen by default)
        self.mu = mu
        self.cov = cov
        if self.cache_inverse:
            self.cov_inv = torch.linalg.pinv(cov)
        else:
            self.cov_inv = torch.empty(0, 0)
        self._statistically_initialized = True
        return self

    def reset(self) -> None:
        """Reset all statistics and accumulators to empty state.

        Clears mu, cov, cov_inv, and all streaming accumulators (_mean, _M2, _n).
        After reset, the detector must be re-initialized via statistical_initialization()
        before it can be used for inference.

        Notes
        -----
        Use this method when you need to re-initialize the detector with different
        training data or when switching between different dataset distributions.
        """
        self.mu = torch.empty(0)
        self.cov = torch.empty(0, 0)
        self.cov_inv = torch.empty(0, 0)
        self._n = 0
        self._mean = torch.empty(0)
        self._M2 = torch.empty(0, 0)
        self._statistically_initialized = False

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Forward pass computing anomaly scores.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor in BHWC format

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "scores" key containing BHW1 anomaly scores
        """
        if not self._statistically_initialized or self.mu.numel() == 0:
            raise RuntimeError(
                "RXGlobal not initialized. Call statistical_initialization() before inference."
            )
        B, H, W, C = data.shape
        N = H * W
        X = data.view(B, N, C)
        # Convert dtype if needed, but don't change device (assumes everything on same device)
        Xc = X - self.mu.to(X.dtype)
        if self.cov_inv.numel() > 0:
            cov_inv = self.cov_inv.to(X.dtype)
            md2 = torch.einsum("bnc,cd,bnd->bn", Xc, cov_inv, Xc)  # (B,N)
        else:
            md2 = self._quad_form_solve(Xc, self.cov.to(X.dtype))
        scores = md2.view(B, H, W).unsqueeze(-1)  # Add channel dimension (B,H,W,1)
        return {"scores": scores}


# ---------- Per-batch/stateless variant ----------
class RXPerBatch(RXBase):
    """
    Computes μ, Σ per image in the batch on the fly; no fit/finalize.
    """

    def forward(self, data: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Forward pass computing per-batch anomaly scores.

        Parameters
        ----------
        data : torch.Tensor
            Input tensor in BHWC format

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "scores" key containing BHW1 anomaly scores
        """
        B, H, W, C = data.shape
        N = H * W
        X_flat = _flatten_bhwc(data)  # (B,N,C)
        mu = X_flat.mean(1, keepdim=True)  # (B,1,C)
        Xc = X_flat - mu
        cov = torch.matmul(Xc.transpose(1, 2), Xc) / max(N - 1, 1)  # (B,C,C)
        eye = torch.eye(C, device=data.device, dtype=data.dtype).expand(B, C, C)
        cov = cov + self.eps * eye
        md2 = self._quad_form_solve(Xc, cov)  # (B,N)
        scores = md2.view(B, H, W)
        return {"scores": scores.unsqueeze(-1)}
