"""Numerically stable streaming statistics using Welford's online algorithm.

Provides a reusable ``WelfordAccumulator`` that incrementally computes mean,
variance, covariance, and correlation from batches of data.  The accumulator
is an ``nn.Module`` so that ``.to(device)`` propagates to its internal
buffers automatically when the parent node is moved.

Reference:
    Welford, B. P. (1962). "Note on a method for calculating corrected sums
    of squares and products." Technometrics, 4(3), 419-420.

    Chan, T. F., Golub, G. H., & LeVeque, R. J. (1979). "Updating formulae
    and a pairwise algorithm for computing sample variances."
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class WelfordAccumulator(nn.Module):
    """Numerically stable streaming mean / variance / covariance.

    All internal accumulation happens in **float64** for numerical stability.
    Property getters return **float32** tensors by default; pass ``dtype=torch.float64``
    to read the statistics without that rounding, for example before inverting a
    covariance matrix.

    The buffers are registered with ``persistent=False`` so they are
    **excluded** from ``state_dict()`` — only the parent node's own
    buffers (``mu``, ``cov``, …) are serialised.  This is correct because
    the accumulator is transient training state that is consumed by
    ``finalize()`` inside ``statistical_initialization()``.

    Parameters
    ----------
    n_features : int
        Number of features (channels) per sample.
    track_covariance : bool, optional
        If ``True``, maintain the full (C, C) covariance matrix (O(C²)
        memory and compute per update).  If ``False`` (default), only
        per-feature variance is tracked (O(C)).
    dtype : torch.dtype, optional
        Floating-point type of the tensors the getters return (default
        ``torch.float32``).  The accumulation itself is always float64.
    """

    def __init__(
        self,
        n_features: int,
        *,
        track_covariance: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self._n_features = n_features
        self._track_cov = track_covariance
        self._dtype = dtype

        self.register_buffer("_n", torch.tensor(0, dtype=torch.long), persistent=False)
        self.register_buffer(
            "_mean", torch.zeros(n_features, dtype=torch.float64), persistent=False
        )
        m2_shape = (n_features, n_features) if track_covariance else (n_features,)
        self.register_buffer("_M2", torch.zeros(m2_shape, dtype=torch.float64), persistent=False)

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero all accumulators so the instance can be reused."""
        self._n.zero_()
        self._mean.zero_()
        self._M2.zero_()

    @torch.no_grad()
    def update(self, X: Tensor) -> None:
        """Incorporate a batch of samples.

        Parameters
        ----------
        X : Tensor
            Sample matrix of shape ``(N, C)`` where *N* is the number of
            samples and *C* equals ``n_features``.  A 1-D tensor of shape
            ``(N,)`` is accepted when ``n_features == 1`` and is reshaped
            to ``(N, 1)`` automatically.
        """
        if X.ndim == 1:
            X = X.unsqueeze(-1)

        X = X.to(dtype=torch.float64)
        m = X.shape[0]
        if m == 0:
            return

        mean_b = X.mean(dim=0)  # (C,)

        if self._track_cov:
            centered = X - mean_b
            M2_b = centered.T @ centered  # (C, C)
        else:
            M2_b = ((X - mean_b) ** 2).sum(dim=0)  # (C,)

        n = int(self._n.item())
        if n == 0:
            self._n.fill_(m)
            self._mean.copy_(mean_b)
            self._M2.copy_(M2_b)
        else:
            tot = n + m
            delta = mean_b - self._mean
            self._mean.add_(delta * (m / tot))
            if self._track_cov:
                self._M2.add_(M2_b + torch.outer(delta, delta) * (n * m / tot))
            else:
                self._M2.add_(M2_b + delta**2 * (n * m / tot))
            self._n.fill_(tot)

    # ------------------------------------------------------------------
    # Read-only property getters
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Total number of samples accumulated so far."""
        return int(self._n.item())

    @property
    def n_features(self) -> int:
        """Number of features (channels) per sample."""
        return self._n_features

    @property
    def mean(self) -> Tensor:
        """Per-feature mean, shape ``(C,)``, in the getter dtype (float32 by default)."""
        if self.count == 0:
            raise RuntimeError("No samples accumulated yet.")
        return self._mean.to(self._dtype)

    @property
    def var(self) -> Tensor:
        """Per-feature sample variance, shape ``(C,)``, in the getter dtype."""
        if self.count < 2:
            raise RuntimeError("Need at least 2 samples to compute variance.")
        if self._track_cov:
            return (self._M2.diagonal() / (self.count - 1)).to(self._dtype)
        return (self._M2 / (self.count - 1)).to(self._dtype)

    @property
    def std(self) -> Tensor:
        """Per-feature standard deviation, shape ``(C,)``, in the getter dtype."""
        return self.var.sqrt()

    @property
    def cov(self) -> Tensor:
        """Sample covariance matrix, shape ``(C, C)``, in the getter dtype.

        Raises
        ------
        RuntimeError
            If ``track_covariance`` was not enabled or fewer than 2
            samples have been accumulated.
        """
        if not self._track_cov:
            raise RuntimeError(
                "Covariance tracking was not enabled. Pass track_covariance=True to __init__."
            )
        if self.count < 2:
            raise RuntimeError("Need at least 2 samples to compute covariance.")
        return (self._M2 / (self.count - 1)).to(self._dtype)

    @property
    def corr(self) -> Tensor:
        """Absolute correlation matrix, shape ``(C, C)``, in the getter dtype.

        Raises
        ------
        RuntimeError
            If ``track_covariance`` was not enabled or fewer than 2
            samples have been accumulated.
        """
        c = self.cov  # (C, C) in the getter dtype; also validates the preconditions
        s = self.std.clamp(min=1e-12)
        denom = torch.outer(s, s)
        corr = c / denom
        return torch.abs(torch.nan_to_num(corr, nan=0.0))
