"""Deep SVDD encoder for the port-based cuvis.ai stack."""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cuvis_ai.node import Node
from cuvis_ai.pipeline.ports import PortSpec
from cuvis_ai.utils.types import Context, ExecutionStage, InputStream, Metric

logger = logging.getLogger(__name__)


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
        proj = x @ self.W + self.b
        return self.z_scale * torch.cos(proj)


class DeepSVDDEncoder(Node):
    """Port-based Deep SVDD feature extractor producing BHWD embeddings."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
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
        rep_dim: int = 32,
        hidden: int = 128,
        sample_n: int = 500_000,
        seed: int = 0,
        eps: float = 1e-8,
        kernel: str = "linear",
        n_rff: int = 2048,
        gamma: float | None = None,
        use_mlp_after_rff: bool = True,
        mlp_forward_batch_size: int = 65_536,
        **kwargs: Any,
    ) -> None:
        self.rep_dim = int(rep_dim)
        self.hidden = int(hidden)
        self.sample_n = int(sample_n)
        self.seed = int(seed)
        self.eps = float(eps)
        self.kernel = str(kernel)
        self.n_rff = int(n_rff)
        self.gamma = None if gamma is None else float(gamma)
        self.use_mlp_after_rff = bool(use_mlp_after_rff)
        self.mlp_forward_batch_size = max(1, int(mlp_forward_batch_size))

        super().__init__(
            rep_dim=self.rep_dim,
            hidden=self.hidden,
            sample_n=self.sample_n,
            seed=self.seed,
            eps=self.eps,
            kernel=self.kernel,
            n_rff=self.n_rff,
            gamma=self.gamma,
            use_mlp_after_rff=self.use_mlp_after_rff,
            mlp_forward_batch_size=self.mlp_forward_batch_size,
            **kwargs,
        )

        self.net: nn.Module | None = None
        self.register_buffer("zscore_mean", torch.empty(0))  # (C,)
        self.register_buffer("zscore_std", torch.empty(0))  # (C,)
        self.register_buffer("_num_channels", torch.tensor(0, dtype=torch.long))
        self._fitted: bool = False
        self._statistically_initialized = False

        # WARNING: This node has a custom load_state_dict() method that won't be called
        # automatically during standard PyTorch deserialization. The custom method is necessary
        # because the network architecture (self.net) is created lazily based on the number of
        # channels learned during statistical_initialization(). When loading state, the network must be reconstructed
        # before loading weights. This is handled properly by the pipeline serialization system.
        logger.warning(
            "DeepSVDDEncoder has a custom load_state_dict() method due to lazy network "
            "initialization. This method is required and properly integrated with pipeline "
            "serialization but developers should be aware of this non-standard pattern."
        )

    # ------------------------------------------------------------------
    # Statistical initialization hooks
    # ------------------------------------------------------------------
    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Estimate per-band z-score statistics from the provided stream."""
        pixels: list[np.ndarray] = []
        for batch in input_stream:
            data = batch.get("data")
            if data is None:
                continue
            data = data.contiguous()

            B, H, W, C = data.shape
            num_channels = self._get_num_channels()
            if num_channels is None:
                self._num_channels = torch.tensor(C, dtype=torch.long)
            elif num_channels != C:
                raise ValueError(f"Channel mismatch: expected {num_channels}, got {C}")
            pixels.append(data.reshape(B * H * W, C).detach().cpu().numpy().astype(np.float32))

        if not pixels:
            raise RuntimeError(
                "DeepSVDDEncoder.statistical_initialization() did not receive any data"
            )

        X_all = np.concatenate(pixels, axis=0)
        n_stats = min(self.sample_n, X_all.shape[0])
        if n_stats < X_all.shape[0]:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(X_all.shape[0], size=n_stats, replace=False)
            X_stats = X_all[idx]
        else:
            X_stats = X_all

        mean, std = self._zscore_fit(X_stats, eps=self.eps)
        self.zscore_mean = torch.from_numpy(mean).squeeze(0)
        self.zscore_std = torch.from_numpy(std).squeeze(0)
        self._statistically_initialized = True

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(self, data: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        num_channels = self._get_num_channels()
        if (
            not self._statistically_initialized
            or num_channels is None
            or self.zscore_mean is None
            or self.zscore_std is None
        ):
            raise RuntimeError(
                "DeepSVDDEncoder requires statistical_initialization() before forward()"
            )
        if data.shape[-1] != num_channels:
            raise ValueError(f"Channel mismatch: expected {num_channels}, got {data.shape[-1]}")

        self._ensure_network(device=data.device, dtype=data.dtype)
        assert self.net is not None

        B, H, W, C = data.shape
        flat = data.contiguous().reshape(B * H * W, C)

        mean = self.zscore_mean.to(device=flat.device, dtype=flat.dtype).unsqueeze(0)
        std = self.zscore_std.to(device=flat.device, dtype=flat.dtype).unsqueeze(0)
        flat = (flat - mean) / std

        batch_size = self.mlp_forward_batch_size
        embeddings = []
        for start in range(0, flat.shape[0], batch_size):
            chunk = flat[start : start + batch_size]
            embeddings.append(self.net(chunk))
        z = torch.cat(embeddings, dim=0).reshape(B, H, W, self.rep_dim)
        return {"embeddings": z}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_num_channels(self) -> int | None:
        if isinstance(self._num_channels, torch.Tensor):
            val = int(self._num_channels.item())
            return None if val == 0 else val
        return int(self._num_channels) if self._num_channels != 0 else None

    def _ensure_network(self, device: torch.device, dtype: torch.dtype) -> None:
        if self.net is not None:
            return
        # TODO[ALL-4973]: Replace this manual module instantiation with a dedicated node so we don't
        # need to construct + .to() the network ourselves. See Jira task ALL-4973.
        num_channels = self._get_num_channels()
        if num_channels is None:
            raise RuntimeError(
                "Number of channels unknown. Call statistical_initialization() first."
            )

        if self.kernel == "linear":
            self.net = SpectralNet(
                in_dim=num_channels,
                rep_dim=self.rep_dim,
                hidden=self.hidden,
            )
        elif self.kernel == "rbf":
            eff_gamma = self.gamma if self.gamma is not None else (1.0 / float(num_channels))
            rff = RFFLayer(input_dim=num_channels, n_features=self.n_rff, gamma=eff_gamma)
            if self.use_mlp_after_rff:
                head = SpectralNet(in_dim=self.n_rff, rep_dim=self.rep_dim, hidden=self.hidden)
                self.net = nn.Sequential(rff, head)
            else:
                linear_head = nn.Linear(self.n_rff, self.rep_dim, bias=False)
                nn.init.xavier_uniform_(linear_head.weight)
                self.net = nn.Sequential(rff, linear_head)
        else:
            raise ValueError(f"Unknown kernel '{self.kernel}'. Expected 'linear' or 'rbf'.")

        self.net = self.net.to(device=device, dtype=dtype)
        self._fitted = True
        self._statistically_initialized = True

    @staticmethod
    def _zscore_fit(X: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
        mean = X.mean(axis=0, keepdims=True).astype(np.float32)
        std = X.std(axis=0, keepdims=True).astype(np.float32)
        std = std + eps
        return mean, std

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------
    def unfreeze(self) -> None:
        self._ensure_network(device=self.zscore_mean.device, dtype=self.zscore_mean.dtype)
        super().unfreeze()

    def load_state_dict(self, state_dict, strict: bool = True) -> Any:  # type: ignore[override]
        # Resize buffers to match incoming state_dict shapes before loading
        if "zscore_mean" in state_dict:
            self.zscore_mean = torch.empty_like(state_dict["zscore_mean"])
        if "zscore_std" in state_dict:
            self.zscore_std = torch.empty_like(state_dict["zscore_std"])
        if "_num_channels" in state_dict:
            self._num_channels = torch.empty_like(state_dict["_num_channels"])

        saved_channels = state_dict.get("_num_channels")
        if saved_channels is None and "zscore_mean" in state_dict:
            z_mean = state_dict.get("zscore_mean")
            if z_mean is not None:
                saved_channels = torch.tensor(z_mean.shape[0], dtype=torch.long)

        if saved_channels is not None:
            self._num_channels = saved_channels.clone()
            num_channels = self._get_num_channels()
            sample_tensor = next(
                (v for v in state_dict.values() if isinstance(v, torch.Tensor) and v.numel() > 0),
                None,
            )
            device = sample_tensor.device if sample_tensor is not None else torch.device("cpu")
            dtype = sample_tensor.dtype if sample_tensor is not None else torch.get_default_dtype()
            if num_channels is not None:
                self._ensure_network(device=device, dtype=dtype)

        _ = super().load_state_dict(state_dict, strict=strict)

        self._statistically_initialized = (
            self.zscore_mean is not None and self.zscore_std is not None and self.net is not None
        )


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

    def __init__(self, alpha: float = 0.1, update_in_eval: bool = False, **kwargs) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = float(alpha)
        self.update_in_eval = bool(update_in_eval)
        super().__init__(alpha=self.alpha, update_in_eval=self.update_in_eval, **kwargs)
        self.register_buffer("_tracked_center", torch.empty(0))

    def statistical_initialization(self, input_stream: InputStream) -> None:
        total = None
        count = 0
        for batch in input_stream:
            embeddings = batch.get("embeddings")
            if embeddings is None:
                embeddings = batch.get("data")
            if embeddings is None:
                continue
            flat = embeddings.reshape(-1, embeddings.shape[-1])
            batch_sum = flat.sum(dim=0)
            total = batch_sum if total is None else total + batch_sum
            count += flat.shape[0]

        if total is None or count == 0:
            raise RuntimeError(
                "DeepSVDDCenterTracker.statistical_initialization() received no embeddings"
            )

        self._tracked_center = (total / count).detach()
        self._statistically_initialized = True

    def forward(
        self, embeddings: torch.Tensor, context: Context | None = None, **_: Any
    ) -> dict[str, Any]:
        batch_mean = embeddings.mean(dim=(0, 1, 2)).detach()
        should_update = context is None or context.stage is ExecutionStage.TRAIN
        if not should_update and self.update_in_eval and context is not None:
            should_update = context.stage in {ExecutionStage.VAL, ExecutionStage.TEST}

        if self._tracked_center.numel() == 0:
            self._tracked_center = batch_mean.clone()
        elif should_update:
            device_batch = batch_mean.device
            self._tracked_center = self._tracked_center.to(device_batch)
            self._tracked_center = (
                1.0 - self.alpha
            ) * self._tracked_center + self.alpha * batch_mean

        metrics = []
        if self._tracked_center.numel() > 0:
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

        center_value = (
            self._tracked_center.detach().clone()
            if self._tracked_center.numel() > 0
            else batch_mean.detach().clone()
        )
        return {"center": center_value, "metrics": metrics}

    def load_state_dict(self, state_dict, strict: bool = True) -> Any:  # type: ignore[override]
        # Resize buffers to match incoming state_dict shapes before loading
        if "_tracked_center" in state_dict:
            self._tracked_center = torch.empty_like(state_dict["_tracked_center"])

        result = super().load_state_dict(state_dict, strict=strict)
        if self._tracked_center.numel() > 0:
            self._statistically_initialized = True
        return result


__all__ = ["DeepSVDDEncoder", "DeepSVDDScores", "DeepSVDDCenterTracker"]
