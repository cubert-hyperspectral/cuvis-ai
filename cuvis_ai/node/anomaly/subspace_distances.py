"""Distances of a pixel to a fitted subspace model: inside the subspace and off it."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import InputStream
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai.utils.welford import WelfordAccumulator
from cuvis_ai_core.node import Node


class SubspaceDistances(Node):
    """Score distance and orthogonal distance of every pixel to a fitted subspace model.

    A reduction node such as ``TrainablePCA`` describes normal data by a few scores per pixel
    and a reconstruction of the pixel from them. This node measures how far a pixel lies from
    that description in the two ways SIMCA uses:

    - ``score_distance``: the squared Mahalanobis distance of the scores to the scores of the
      fit data, ``(t - m)^T S^-1 (t - m)``. For PCA scores this is Hotelling's T2.

    - ``orthogonal_distance``: the squared residual between the pixel and its reconstruction,
      ``sum((x - x_hat)^2)`` over the channels, known as Q.

    The node reads scores and reconstruction from its inputs and knows nothing about the
    reduction that made them, so any node with a ``projected`` and a ``reconstructed`` output
    can feed it. The statistical fit learns the mean and the covariance of the scores and must
    see normal data only.

    The fit keeps its statistics in float64 and turns the covariance into a whitening matrix
    ``W`` with ``W^T W = S^-1``, through the eigen-decomposition of the correlation matrix; the
    score distance is the squared length of ``W (t - m)``. Working on the correlation matrix
    makes the result independent of the scale of each component (a whitened PCA gives the same
    value), however far apart the component variances lie, and the whitening form keeps the
    distance non-negative in float32. Components that are exactly collinear on the fit data
    span fewer directions than there are components; the direction they do not span is left
    out. A component without any variance on the fit data is an error: leaving it out would
    create a direction that neither distance sees.

    Parameters
    ----------
    n_components : int
        Number of score components K, the last dimension of ``projected``. It must equal the
        component count of the reduction node; port validation and the fit both check it.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.ONNX,
            NodeTag.ANOMALY,
            NodeTag.HYPERSPECTRAL,
            NodeTag.STATEFUL,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Cube the reduction node was given [B, H, W, C]",
        ),
        "projected": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, "n_components"),
            description="Scores of every pixel [B, H, W, K]",
        ),
        "reconstructed": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Cube rebuilt from the scores [B, H, W, C]",
        ),
    }

    OUTPUT_SPECS = {
        "score_distance": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Squared Mahalanobis distance of the scores, Hotelling T2 for PCA "
            "[B, H, W, 1]",
        ),
        "orthogonal_distance": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Squared residual between the cube and its reconstruction, Q [B, H, W, 1]",
        ),
    }

    def __init__(self, n_components: int, **kwargs: Any) -> None:
        self.n_components = int(n_components)
        if self.n_components < 1:
            raise ValueError(f"n_components must be at least 1, got {n_components}")

        super().__init__(n_components=self.n_components, **kwargs)

        self.register_buffer("score_mean", torch.zeros(self.n_components, dtype=torch.float32))
        self.register_buffer("score_whitening", torch.eye(self.n_components, dtype=torch.float32))

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Learn the mean and the whitening matrix of the scores from normal data.

        Parameters
        ----------
        input_stream : InputStream
            Batches keyed by input port name; only ``projected`` enters the fit.

        Raises
        ------
        ValueError
            The stream holds no pixels, ``projected`` does not have ``n_components``
            components, the scores are not finite, or a score component has no variance.
        """
        accumulator: WelfordAccumulator | None = None
        for batch in input_stream:
            scores = batch.get("projected")
            if scores is None:
                continue
            if scores.shape[-1] != self.n_components:
                raise ValueError(
                    f"SubspaceDistances was built with n_components={self.n_components}, but "
                    f"`projected` carries {scores.shape[-1]} components. Give both nodes the "
                    "same number."
                )
            flat = scores.reshape(-1, self.n_components)
            if accumulator is None:
                # On the stream's device: the accumulator merges each batch into its own
                # buffers. float64 getters: a covariance rounded to float32 can stop being
                # positive definite when score components are strongly correlated.
                accumulator = WelfordAccumulator(
                    self.n_components, track_covariance=True, dtype=torch.float64
                ).to(flat.device)
            accumulator.update(flat)

        if accumulator is None or accumulator.count < 2:
            raise ValueError(
                "SubspaceDistances received no pixels to fit on (it needs at least two)."
            )

        mean = accumulator.mean
        covariance = accumulator.cov
        if not (torch.isfinite(mean).all() and torch.isfinite(covariance).all()):
            raise ValueError(
                "SubspaceDistances found non-finite scores in the fit data; check the cube and "
                "the reduction node for NaN or inf."
            )

        variance = covariance.diagonal()
        flat_components = torch.nonzero(variance <= 0).flatten().tolist()
        if flat_components:
            raise ValueError(
                f"Score component(s) {flat_components} have no variance on the fit data, so "
                "their distance is undefined. Lower n_components of the reduction node and of "
                "SubspaceDistances."
            )

        # Decompose the correlation matrix, which does not depend on the scale of a component,
        # so a whitened and an unwhitened reduction give the same T2.
        inverse_std = variance.rsqrt()
        correlation = covariance * torch.outer(inverse_std, inverse_std)
        eigenvalues, eigenvectors = torch.linalg.eigh(correlation)
        # A direction the fit data does not span (exactly collinear components) has an
        # eigenvalue at round-off level, possibly below zero. It is left out, never inverted.
        tolerance = eigenvalues.max() * self.n_components * torch.finfo(eigenvalues.dtype).eps
        inverse_sqrt = torch.where(
            eigenvalues > tolerance,
            eigenvalues.clamp_min(tolerance).rsqrt(),
            torch.zeros_like(eigenvalues),
        )
        # Row i of the whitening matrix is eigenvector i, scaled to unit variance.
        whitening = (eigenvectors * inverse_sqrt).T * inverse_std

        self.score_mean.copy_(mean.to(self.score_mean.dtype))
        self.score_whitening.copy_(whitening.to(self.score_whitening.dtype))
        self._statistically_initialized = True

    def forward(
        self, data: Tensor, projected: Tensor, reconstructed: Tensor, **_: Any
    ) -> dict[str, Tensor]:
        """Return the score distance and the orthogonal distance of every pixel.

        Parameters
        ----------
        data : Tensor
            Cube the reduction node was given [B, H, W, C].
        projected : Tensor
            Scores [B, H, W, K].
        reconstructed : Tensor
            Cube rebuilt from the scores [B, H, W, C].

        Returns
        -------
        dict[str, Tensor]
            ``score_distance`` and ``orthogonal_distance``, each [B, H, W, 1].
        """
        if not self._statistically_initialized:
            raise RuntimeError(
                "SubspaceDistances is not fitted. Run statistical_initialization() on normal "
                "data first."
            )

        whitened = torch.einsum("bhwk,lk->bhwl", projected - self.score_mean, self.score_whitening)
        residual = data - reconstructed

        return {
            "score_distance": (whitened * whitened).sum(dim=-1, keepdim=True),
            "orthogonal_distance": (residual * residual).sum(dim=-1, keepdim=True),
        }


__all__ = ["SubspaceDistances"]
