import torch
import torch.nn as nn

from cuvis_ai.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.torch import _flatten_bhwc
from cuvis_ai.utils.types import InputStream


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

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        super().__init__(eps=eps)

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
    """
    Uses global μ, Σ (estimated from train).

    After statistical initialization with fit(), all parameters are stored as
    buffers (frozen). Call unfreeze() to convert them to trainable nn.Parameters
    for gradient-based optimization.
    """

    def __init__(self, eps: float = 1e-6, cache_inverse: bool = True) -> None:
        super().__init__(eps)
        self.cache_inverse = cache_inverse
        # global stats - all stored as buffers initially
        self.register_buffer("mu", None)  # (C,)

        self.register_buffer("cov", None)  # (C,C)
        self.register_buffer("cov_inv", None)  # (C,C)
        # streaming accumulators
        self.register_buffer("_mean", None)
        self.register_buffer("_M2", None)
        self._n = 0
        self._fitted = False

    @property
    def requires_initial_fit(self) -> bool:
        """RXGlobal requires statistical initialization from data."""
        return True

    def fit(self, input_stream: InputStream) -> None:
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
        self._initialized = True

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
        if self.mu is not None and self.cov is not None:
            # Convert buffers to parameters
            self.mu = nn.Parameter(self.mu.clone(), requires_grad=True)
            self.cov = nn.Parameter(self.cov.clone(), requires_grad=True)
            if self.cov_inv is not None:
                self.cov_inv = nn.Parameter(self.cov_inv.clone(), requires_grad=True)
        # Call parent to enable requires_grad
        super().unfreeze()

    @torch.no_grad()
    def update(self, batch_bhwc: torch.Tensor) -> None:
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
        self._fitted = False

    @torch.no_grad()
    def finalize(self) -> "RXGlobal":
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
            self.cov_inv = None
        self._fitted = True
        return self

    def reset(self) -> None:
        self.mu = None
        self.cov = None
        self.cov_inv = None
        self._n = 0
        self._mean = None
        self._M2 = None
        self._fitted = False

    def serialize(self, serial_dir: str) -> dict:
        """Serialize RXGlobal state for saving."""
        return {
            "params": {
                "eps": self.eps,
                "cache_inverse": self.cache_inverse,
            },
            "state_dict": self.state_dict(),
        }

    def load(self, params: dict, serial_dir: str) -> None:
        """Load RXGlobal state from serialized data."""
        config = params.get("params", {})
        self.eps = config.get("eps", self.eps)
        self.cache_inverse = config.get("cache_inverse", self.cache_inverse)

        state = params.get("state_dict", {})
        if state:
            self.load_state_dict(state, strict=False)

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
        if not self._fitted or self.mu is None:
            raise RuntimeError("RXGlobal not finalized. Call update()/finalize() or fit().")
        B, H, W, C = data.shape
        N = H * W
        X = data.view(B, N, C)
        # Convert dtype if needed, but don't change device (assumes everything on same device)
        Xc = X - self.mu.to(X.dtype)
        if self.cov_inv is not None:
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

    def serialize(self, serial_dir: str) -> dict:
        return {
            "config": self._config,
            "state_dict": self.rx.state_dict(),
        }

    def load(self, params: dict, serial_dir: str) -> None:
        config = params.get("config", {})
        if config and config != self._config:
            self.rx = RXPerBatch(**config)
            self._config = dict(config)
        state = params.get("state_dict", {})
        if state:
            self.rx.load_state_dict(state, strict=False)
