"""PCA nodes for dimensionality reduction."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai.utils.welford import WelfordAccumulator
from cuvis_ai_core.node import Node

## This node is not approved
# missing tests against standard implementations
# missing tutorial examples and approved documentation


class PCA(Node):
    """Project each frame independently onto its principal components."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.DIM_REDUCTION, NodeTag.PREPROCESSING, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "projected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "n_components"),
            description="PCA-projected data with reduced dimensions",
        ),
        "explained_variance_ratio": PortSpec(
            dtype=torch.float32,
            shape=("n_components",),
            description="Proportion of variance explained by each component",
            optional=True,
        ),
        "components": PortSpec(
            dtype=torch.float32,
            shape=("n_components", -1),
            description="Principal components matrix",
            optional=True,
        ),
    }

    def __init__(
        self,
        n_components: int,
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.n_components = int(n_components)
        self.eps = float(eps)

        super().__init__(n_components=self.n_components, eps=self.eps, **kwargs)

    def _fit_frame(self, frame: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Fit PCA on one HWC frame and return components, mean, and eigenvalues."""
        if frame.ndim != 3:
            raise ValueError(f"Expected one frame with shape [H, W, C], got {tuple(frame.shape)}")

        _, _, channel_count = frame.shape
        if channel_count < self.n_components:
            raise ValueError(
                f"Expected at least {self.n_components} channels for PCA, got {channel_count}"
            )

        flat = frame.reshape(-1, channel_count).to(dtype=torch.float64)
        if flat.shape[0] < 2:
            raise ValueError("Per-frame PCA requires at least 2 pixels.")

        mean = flat.mean(dim=0)
        centered = flat - mean
        covariance = centered.T @ centered / max(flat.shape[0] - 1, 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        components = eigenvectors[:, : self.n_components].T.to(dtype=torch.float32)
        return (
            components,
            mean.to(dtype=torch.float32),
            eigenvalues[: self.n_components].to(dtype=torch.float32),
        )

    def _project(self, flat: Tensor, mean: Tensor, components: Tensor) -> Tensor:
        """Center and project flattened pixels onto PCA components."""
        mean = mean.to(device=flat.device, dtype=flat.dtype)
        components = components.to(device=flat.device, dtype=flat.dtype)
        return (flat - mean) @ components.T

    def _variance_ratio(self, eigenvalues: Tensor) -> Tensor:
        """Normalize retained eigenvalues to explained-variance ratios."""
        eigenvalues = eigenvalues.to(dtype=torch.float32)
        return eigenvalues / (eigenvalues.sum() + self.eps)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Fit PCA independently on each frame and return the per-frame projection."""
        if data.ndim != 4:
            raise ValueError(f"Expected data with shape [B, H, W, C], got {tuple(data.shape)}")
        if data.shape[0] == 0:
            raise ValueError("PCA requires a non-empty batch.")

        projected_frames: list[Tensor] = []
        explained_variance_ratio: Tensor | None = None
        components: Tensor | None = None

        for frame in data:
            frame_components, mean, eigenvalues = self._fit_frame(frame)
            flat = frame.reshape(-1, frame.shape[-1]).to(dtype=torch.float32)
            projected = self._project(flat, mean, frame_components).reshape(
                frame.shape[0],
                frame.shape[1],
                self.n_components,
            )

            projected_frames.append(projected.to(dtype=torch.float32))
            explained_variance_ratio = self._variance_ratio(eigenvalues).to(device=data.device)
            components = frame_components.to(device=data.device)

        assert explained_variance_ratio is not None
        assert components is not None

        return {
            "projected": torch.stack(projected_frames, dim=0),
            "explained_variance_ratio": explained_variance_ratio,
            "components": components,
        }


class TrainablePCA(PCA):
    """Trainable PCA node with orthogonality regularization."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.HYPERSPECTRAL,
            NodeTag.DIM_REDUCTION,
            NodeTag.PREPROCESSING,
            NodeTag.LEARNABLE,
            NodeTag.STATEFUL,
            NodeTag.TORCH,
        }
    )

    TRAINABLE_BUFFERS = ("_components",)

    def __init__(
        self,
        num_channels: int,
        n_components: int,
        whiten: bool = False,
        init_method: Literal["svd", "random"] = "svd",
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        self.whiten = whiten

        super().__init__(
            num_channels=num_channels,
            n_components=n_components,
            whiten=whiten,
            init_method=init_method,
            eps=eps,
            **kwargs,
        )

        # Buffers for statistical initialization (private to avoid conflicts with output ports)
        self.register_buffer("_mean", torch.empty(num_channels))
        self.register_buffer("_explained_variance", torch.empty(n_components))
        self.register_buffer("_components", torch.empty(n_components, num_channels))

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Initialize PCA components from data using covariance eigen decomposition."""
        acc = None
        for batch_data in input_stream:
            x = batch_data["data"]
            if x is not None:
                flat = x.reshape(-1, x.shape[-1])  # [B*H*W, C]
                if acc is None:
                    acc = WelfordAccumulator(flat.shape[1], track_covariance=True)
                acc.update(flat)

        if acc is None or acc.count == 0:
            raise ValueError("No data provided for PCA initialization")

        self._mean = acc.mean.to(dtype=torch.float32)  # [C]
        cov = acc.cov.to(torch.float64)  # [C, C]

        # Eigen decomposition on covariance (equivalent to SVD on centered data)
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        eigenvalues = eigenvalues.flip(0)
        eigenvectors = eigenvectors.flip(1)

        # Extract top n_components (rows = principal components)
        self._components = eigenvectors[:, : self.n_components].T.float()  # [n_components, C]
        self._explained_variance = eigenvalues[: self.n_components].float()  # [n_components]

        self._statistically_initialized = True

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Project data onto statistically initialized global components."""
        if not self._statistically_initialized:
            raise RuntimeError("PCA not initialized. Call statistical_initialization() first.")

        if data.ndim != 4:
            raise ValueError(f"Expected data with shape [B, H, W, C], got {tuple(data.shape)}")

        batch_size, height, width, channels = data.shape
        flat = data.reshape(-1, channels)

        projected = self._project(flat, self._mean, self._components)

        if self.whiten:
            explained_variance = self._explained_variance.to(
                device=data.device, dtype=projected.dtype
            )
            scale = 1.0 / torch.sqrt(explained_variance + self.eps)
            projected = projected * scale

        outputs = {
            "projected": projected.reshape(batch_size, height, width, self.n_components),
        }

        if self._explained_variance.numel() > 0:
            outputs["explained_variance_ratio"] = self._variance_ratio(self._explained_variance).to(
                data.device
            )

        if self._components.numel() > 0:
            outputs["components"] = self._components

        return outputs


class FixedPCAProjection(TrainablePCA):
    """Project cubes with a fixed PCA loaded from an ``.npz`` file; the projection is never refit.

    The projection math is :class:`TrainablePCA`'s, but mean, components and the percentile range
    come from the file instead of ``statistical_initialization``: a projection that a downstream
    model was trained on must not be refit at inference, and the stateless :class:`PCA` refits per
    frame with arbitrary eigenvector signs. For the same reason the node opts out of the
    statistical fit of a trainrun.

    ``input_global_minmax`` (on by default) first min-maxes each cube to [0, 1] with one min and
    one max over all its H x W x C values, the normalisation the projection was fitted on. This
    makes the projection invariant to the caller's absolute reflectance scale (a raw-scale cube
    would otherwise push every pixel to the clamp and give the downstream model a flat image) and
    is idempotent on a [0, 1] cube; per-channel scaling would change the relative band magnitudes
    the projection depends on. After the parent's ``(x - mean) @ components.T`` the node applies
    the fixed scaling ``(p - lo) / (hi - lo)`` and clamps to [0, 1], which keeps specular outliers
    out of a downstream 8-bit conversion.

    npz keys: ``mean`` [C], ``comps`` [K, C], ``lo`` [K], ``hi`` [K], optional ``explained`` [K].
    """

    # A fixed transform like PCA, but with persistent state (the projection loaded from the file).
    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.HYPERSPECTRAL,
            NodeTag.DIM_REDUCTION,
            NodeTag.PREPROCESSING,
            NodeTag.STATEFUL,
            NodeTag.TORCH,
        }
    )

    # Only the projected image is exposed: the parent's ``components`` [K, C] and
    # ``explained_variance_ratio`` [K] are not images, a host that displays every output port
    # (cuvis.next) cannot show them, and nothing downstream consumes them.
    OUTPUT_SPECS = {
        "projected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Fixed-PCA projection, scaled to [0, 1] [B, H, W, K].",
        ),
    }

    def __init__(
        self,
        projection_path: str,
        scale_to_unit: bool = True,
        clamp01: bool = True,
        input_global_minmax: bool = True,
        **kwargs: Any,
    ) -> None:
        """Load the projection.

        Parameters
        ----------
        projection_path : path of the ``.npz`` file with the keys listed in the class docstring.
        scale_to_unit : apply the fixed scaling ``(p - lo) / (hi - lo)`` (default True).
        clamp01 : clamp the projection to [0, 1] (default True).
        input_global_minmax : min-max each cube to [0, 1] over all its values before projecting
            (default True).
        """
        # A restored pipeline passes back the parent's recorded num_channels and n_components
        # hparams; both come from the npz, so they are dropped before forwarding kwargs.
        kwargs.pop("num_channels", None)
        kwargs.pop("n_components", None)
        proj = np.load(projection_path)
        mean = torch.from_numpy(np.asarray(proj["mean"], dtype=np.float32))
        comps = torch.from_numpy(np.asarray(proj["comps"], dtype=np.float32))
        lo = torch.from_numpy(np.asarray(proj["lo"], dtype=np.float32))
        hi = torch.from_numpy(np.asarray(proj["hi"], dtype=np.float32))
        if comps.ndim != 2 or mean.ndim != 1 or comps.shape[1] != mean.shape[0]:
            raise ValueError(
                "FixedPCAProjection: expected comps [K, C] and mean [C], got "
                f"{tuple(comps.shape)} / {tuple(mean.shape)}."
            )
        self.projection_path = str(projection_path)
        self.scale_to_unit = bool(scale_to_unit)
        self.clamp01 = bool(clamp01)
        self.input_global_minmax = bool(input_global_minmax)
        super().__init__(
            num_channels=int(comps.shape[1]),
            n_components=int(comps.shape[0]),
            projection_path=self.projection_path,
            scale_to_unit=self.scale_to_unit,
            clamp01=self.clamp01,
            input_global_minmax=self.input_global_minmax,
            **kwargs,
        )
        self._mean.copy_(mean)
        self._components.copy_(comps)
        # The parent's buffer holds eigenvalues, the file holds explained-variance ratios; the
        # projection reads the buffer only with ``whiten=True``, which a fixed projection does
        # not use.
        if "explained" in proj.files and np.asarray(proj["explained"]).shape[0] == comps.shape[0]:
            self._explained_variance.copy_(
                torch.from_numpy(np.asarray(proj["explained"], dtype=np.float32))
            )
        else:
            self._explained_variance.zero_()
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)
        self._statistically_initialized = True
        # The file is the fit: a trainrun's statistical pass must not refit the projection.
        self._requires_initial_fit_override = False

    @staticmethod
    def _global_minmax(data: Tensor, eps: float = 1e-6) -> Tensor:
        """Min-max each frame to [0, 1] with one min and one max over all its H x W x C values.

        Global rather than per channel, the normalisation the projection was fitted on: the
        relative band magnitudes stay, so the fixed mean and components stay valid. Invariant to
        the caller's absolute reflectance scale and idempotent on a [0, 1] cube.
        """
        b = data.shape[0]
        flat = data.reshape(b, -1)
        mn = flat.min(dim=1, keepdim=True).values
        mx = flat.max(dim=1, keepdim=True).values
        return ((flat - mn) / (mx - mn).clamp(min=eps)).reshape(data.shape)

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Min-max the cube (optional), project it, apply the fixed scaling and the clamp."""
        if self.input_global_minmax:
            data = self._global_minmax(data)
        out = super().forward(data=data)
        p = out["projected"]
        if self.scale_to_unit:
            p = (p - self._lo) / (self._hi - self._lo)
        if self.clamp01:
            p = p.clamp(0.0, 1.0)
        return {"projected": p}


__all__ = ["PCA", "FixedPCAProjection", "TrainablePCA"]
