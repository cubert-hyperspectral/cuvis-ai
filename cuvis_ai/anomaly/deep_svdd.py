"""Deep SVDD encoder for the port-based cuvis.ai stack."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context, InputStream, Metric
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.utils.welford import WelfordAccumulator


class SpectralNet(nn.Module):
    """Simple 2-layer MLP used by DeepSVDD to produce latent embeddings."""

    def __init__(self, in_dim: int, rep_dim: int = 32, hidden: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, rep_dim, bias=False)

        nn.init.kaiming_uniform_(self.fc1.weight, a=math.sqrt(5))
        if self.fc1.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.fc1.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.fc1.bias, -bound, bound)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through two-layer spectral network.

        Parameters
        ----------
        x : torch.Tensor
            Input features [B, C].

        Returns
        -------
        torch.Tensor
            Projected features [B, rep_dim].
        """
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class RFFLayer(nn.Module):
    """Random Fourier feature encoder for RBF kernels."""

    def __init__(self, input_dim: int, n_features: int = 2048, gamma: float = 0.1) -> None:
        super().__init__()
        scale = math.sqrt(2.0 * float(gamma))
        W = torch.randn(input_dim, n_features, dtype=torch.get_default_dtype()) * scale
        b = torch.rand(n_features, dtype=torch.get_default_dtype()) * (2.0 * math.pi)
        self.register_buffer("W", W)
        self.register_buffer("b", b)
        self.register_buffer(
            "z_scale",
            torch.tensor(math.sqrt(2.0 / float(n_features)), dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute random Fourier features for RBF kernel approximation.

        Parameters
        ----------
        x : torch.Tensor
            Input features [B, input_dim].

        Returns
        -------
        torch.Tensor
            Random Fourier features [B, n_features].

        Notes
        -----
        Approximates RBF kernel via random Fourier features using:
        z(x) = sqrt(2/D) * cos(Wx + b) where W ~ N(0, 2*gamma*I).
        """
        W: torch.Tensor = self.W  # type: ignore[assignment]
        b: torch.Tensor = self.b  # type: ignore[assignment]
        z_scale: torch.Tensor = self.z_scale  # type: ignore[assignment]
        proj = x @ W + b
        return z_scale * torch.cos(proj)


class DeepSVDDProjection(Node):
    """Projection head that maps per-pixel features to Deep SVDD embeddings."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel feature tensor [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "embeddings": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Deep SVDD embeddings [B, H, W, rep_dim]",
        )
    }

    def __init__(
        self,
        *,
        in_channels: int,
        rep_dim: int = 32,
        hidden: int = 128,
        kernel: str = "linear",
        n_rff: int = 2048,
        gamma: float | None = None,
        mlp_forward_batch_size: int = 65_536,
        **kwargs: Any,
    ) -> None:
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        self.in_channels = int(in_channels)
        self.rep_dim = int(rep_dim)
        self.hidden = int(hidden)
        self.kernel = str(kernel)
        self.n_rff = int(n_rff)
        self.gamma = None if gamma is None else float(gamma)
        self.mlp_forward_batch_size = max(1, int(mlp_forward_batch_size))

        super().__init__(
            in_channels=self.in_channels,
            rep_dim=self.rep_dim,
            hidden=self.hidden,
            kernel=self.kernel,
            n_rff=self.n_rff,
            gamma=self.gamma,
            mlp_forward_batch_size=self.mlp_forward_batch_size,
            **kwargs,
        )

        # Build projection network eagerly with known in_channels
        self._build_network()

    def _build_network(self) -> None:
        """Build the projection network."""
        if self.kernel == "linear":
            self.net = SpectralNet(
                in_dim=self.in_channels,
                rep_dim=self.rep_dim,
                hidden=self.hidden,
            )
        elif self.kernel == "rbf":
            eff_gamma = self.gamma if self.gamma is not None else (1.0 / float(self.in_channels))
            rff = RFFLayer(input_dim=self.in_channels, n_features=self.n_rff, gamma=eff_gamma)
            head = SpectralNet(in_dim=self.n_rff, rep_dim=self.rep_dim, hidden=self.hidden)
            self.net = nn.Sequential(rff, head)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'. Expected 'linear' or 'rbf'.")

    def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Project BHWC features into a latent embedding space."""
        B, H, W, C = data.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        flat = data.contiguous().reshape(B * H * W, C)

        batch_size = self.mlp_forward_batch_size
        embeddings = []
        for start in range(0, flat.shape[0], batch_size):
            chunk = flat[start : start + batch_size]
            embeddings.append(self.net(chunk))
        z = torch.cat(embeddings, dim=0).reshape(B, H, W, self.rep_dim)

        return {"embeddings": z}


class ZScoreNormalizerGlobal(Node):
    """Port-based Deep SVDD z-score normalizer for BHWC cubes."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel z-score normalized cube [B, H, W, C]",
        )
    }

    def __init__(
        self,
        *,
        num_channels: int,
        sample_n: int = 500_000,
        seed: int = 0,
        eps: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        self.num_channels = int(num_channels)
        self.sample_n = int(sample_n)
        self.seed = int(seed)
        self.eps = float(eps)

        super().__init__(
            num_channels=self.num_channels,
            sample_n=self.sample_n,
            seed=self.seed,
            eps=self.eps,
            **kwargs,
        )

        # Pre-allocate buffers with known dimensions
        self.register_buffer(
            "zscore_mean", torch.zeros(num_channels, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "zscore_std", torch.ones(num_channels, dtype=torch.get_default_dtype())
        )

    @property
    def requires_initial_fit(self) -> bool:
        """Whether this node requires statistical initialization from training data.

        Returns
        -------
        bool
            Always True for Z-score normalization.
        """
        return True

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Estimate per-band z-score statistics from the provided stream."""
        acc = WelfordAccumulator(self.num_channels)
        for batch in input_stream:
            data = batch.get("data")
            if data is None:
                continue
            data = data.contiguous()

            B, H, W, C = data.shape
            if C != self.num_channels:
                raise ValueError(f"Channel mismatch: expected {self.num_channels}, got {C}")
            acc.update(data.reshape(B * H * W, C))

        if acc.count == 0:
            raise RuntimeError(
                "DeepSVDDEncoder.statistical_initialization() did not receive any data"
            )

        self.zscore_mean.copy_(acc.mean)
        self.zscore_std.copy_(acc.std + self.eps)
        self._statistically_initialized = True

    def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Apply per-channel Z-score normalization.

        Parameters
        ----------
        data : torch.Tensor
            Input feature tensor [B, H, W, C].
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "normalized" key containing Z-score normalized data [B, H, W, C].

        Raises
        ------
        RuntimeError
            If statistical_initialization() has not been called.
        ValueError
            If input channel count doesn't match initialized num_channels.
        """
        if not self._statistically_initialized:
            raise RuntimeError(
                "DeepSVDDEncoder requires statistical_initialization() before forward()"
            )

        B, H, W, C = data.shape
        if C != self.num_channels:
            raise ValueError(f"Channel mismatch: expected {self.num_channels}, got {C}")

        flat = data.contiguous().reshape(B * H * W, C)
        mean = self.zscore_mean.to(dtype=flat.dtype, copy=False).unsqueeze(0)
        std = self.zscore_std.to(dtype=flat.dtype, copy=False).unsqueeze(0)
        flat = (flat - mean) / std

        normalized = flat.reshape(B, H, W, C)
        return {"normalized": normalized}


class DeepSVDDScores(Node):
    """Convert Deep SVDD embeddings + center vector into anomaly scores."""

    INPUT_SPECS = {
        "embeddings": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Deep SVDD embeddings [B, H, W, D]",
        ),
        "center": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Center vector from DeepSVDDCenterTracker",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Squared distance scores [B, H, W, 1]",
        )
    }

    def forward(
        self, embeddings: torch.Tensor, center: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Compute anomaly scores as squared distance from center.

        Parameters
        ----------
        embeddings : torch.Tensor
            Deep SVDD embeddings [B, H, W, D] from projection network.
        center : torch.Tensor
            Center vector [D] from DeepSVDDCenterTracker.
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary with "scores" key containing squared distances [B, H, W, 1].
        """
        scores = ((embeddings - center.view(1, 1, 1, -1)) ** 2).sum(dim=-1, keepdim=True)
        return {"scores": scores}


class DeepSVDDCenterTracker(Node):
    """Track and expose Deep SVDD center statistics with optional logging."""

    INPUT_SPECS = {
        "embeddings": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Deep SVDD embeddings [B, H, W, D]",
        )
    }

    OUTPUT_SPECS = {
        "center": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Tracked Deep SVDD center vector",
        ),
        "metrics": PortSpec(
            dtype=list,
            shape=(),
            description="Optional scalar summaries of the center",
            optional=True,
        ),
    }

    def __init__(
        self, *, rep_dim: int, alpha: float = 0.1, update_in_eval: bool = False, **kwargs
    ) -> None:
        if rep_dim <= 0:
            raise ValueError(f"rep_dim must be positive, got {rep_dim}")
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.rep_dim = int(rep_dim)
        self.alpha = float(alpha)
        self.update_in_eval = bool(update_in_eval)

        super().__init__(
            rep_dim=self.rep_dim, alpha=self.alpha, update_in_eval=self.update_in_eval, **kwargs
        )

        # Pre-allocate buffer with known dimensions
        self.register_buffer(
            "_tracked_center", torch.zeros(rep_dim, dtype=torch.get_default_dtype())
        )

    @property
    def requires_initial_fit(self) -> bool:
        """Whether this node requires statistical initialization from training data.

        Returns
        -------
        bool
            Always True for center tracking initialization.
        """
        return True

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize the Deep SVDD center from training embeddings.

        Computes the mean embedding across all training samples to initialize
        the hypersphere center.

        Parameters
        ----------
        input_stream : InputStream
            Training data stream with embeddings [B, H, W, D].

        Raises
        ------
        RuntimeError
            If no embeddings are received from the input stream.
        ValueError
            If embedding dimensions don't match initialized rep_dim.
        """
        total = None
        count = 0
        for batch in input_stream:
            embeddings = batch.get("embeddings")
            if embeddings is None:
                embeddings = batch.get("data")
            if embeddings is None:
                continue

            # Validate dimensions
            if embeddings.shape[-1] != self.rep_dim:
                raise ValueError(
                    f"Embedding dimension mismatch: expected {self.rep_dim}, got {embeddings.shape[-1]}"
                )

            flat = embeddings.reshape(-1, embeddings.shape[-1])
            batch_sum = flat.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += flat.shape[0]

        if total is None or count == 0:
            raise RuntimeError(
                "DeepSVDDCenterTracker.statistical_initialization() received no embeddings"
            )

        self._tracked_center.copy_((total / count).detach())
        self._statistically_initialized = True

    def forward(
        self, embeddings: torch.Tensor, context: Context | None = None, **_: Any
    ) -> dict[str, Any]:
        """Track and output the Deep SVDD center with exponential moving average.

        Updates the center using EMA during training (and optionally during eval),
        then outputs the current center and center norm metric.

        Parameters
        ----------
        embeddings : torch.Tensor
            Deep SVDD embeddings [B, H, W, D].
        context : Context, optional
            Execution context determining whether to update center.
        **_ : Any
            Additional unused keyword arguments.

        Returns
        -------
        dict[str, Any]
            Dictionary with:
            - "center" : torch.Tensor [D] - Current tracked center
            - "metrics" : list[Metric] - Center norm metric

        Raises
        ------
        RuntimeError
            If statistical_initialization() has not been called.
        ValueError
            If embedding dimensions don't match initialized rep_dim.
        """
        if not self._statistically_initialized:
            raise RuntimeError(
                "DeepSVDDCenterTracker requires statistical_initialization() before forward()"
            )

        # Validate dimensions
        if embeddings.shape[-1] != self.rep_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.rep_dim}, got {embeddings.shape[-1]}"
            )

        batch_mean = embeddings.mean(dim=(0, 1, 2)).detach()
        should_update = context is None or context.stage is ExecutionStage.TRAIN
        if not should_update and self.update_in_eval and context is not None:
            should_update = context.stage in {ExecutionStage.VAL, ExecutionStage.TEST}

        if should_update:
            self._tracked_center.copy_(
                (1.0 - self.alpha) * self._tracked_center + self.alpha * batch_mean
            )

        metrics = []
        center_cpu = self._tracked_center.detach().cpu()
        metrics.append(
            Metric(
                name="deepsvdd_center/norm",
                value=float(center_cpu.norm().item()),
                stage=context.stage if context else ExecutionStage.INFERENCE,
                epoch=context.epoch if context else 0,
                batch_idx=context.batch_idx if context else 0,
            )
        )

        center_value = self._tracked_center.detach().clone()
        return {"center": center_value, "metrics": metrics}


__all__ = [
    "ZScoreNormalizerGlobal",
    "DeepSVDDProjection",
    "DeepSVDDScores",
    "DeepSVDDCenterTracker",
]
