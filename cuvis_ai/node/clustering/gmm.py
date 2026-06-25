"""Gaussian-mixture clustering node for hyperspectral cubes.

``GaussianMixtureClusterer`` fits scikit-learn's ``GaussianMixture`` during
statistical initialization, copies the learned means, mixture weights, and
Cholesky factors of the precision matrices into torch buffers, then evaluates
the closed-form mixture log-probabilities in pure torch at inference. The
sklearn import is lazy and confined to the fit step.

The forward pass replicates scikit-learn's ``_estimate_log_gaussian_prob`` for
``covariance_type="full"`` exactly, so the emitted hard labels, responsibilities,
and per-pixel log-likelihood match ``predict`` / ``predict_proba`` /
``score_samples`` of the fitted model.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node._statistical_fit import _StatisticalFitNode


class GaussianMixtureClusterer(_StatisticalFitNode):
    """Cluster pixel spectra with a Gaussian mixture model.

    The node is fitted once via ``statistical_initialization`` (scikit-learn's
    ``GaussianMixture``). The means, mixture weights, and Cholesky factors of
    the precision matrices fully determine the Gaussian log-probabilities, so
    they are frozen as torch buffers and the forward pass needs no sklearn.

    Only ``covariance_type="full"`` is supported by the torch forward in this
    version; the fit accepts the parameter for API symmetry but other values
    are not evaluated at inference.

    Parameters
    ----------
    n_components : int, optional
        Number of mixture components (default: 3).
    covariance_type : str, optional
        scikit-learn covariance parametrization; only ``"full"`` is supported
        by the torch forward (default: "full").
    reg_covar : float, optional
        Non-negative regularization added to the covariance diagonals at fit
        for numerical stability (default: 1e-6).
    max_iter : int, optional
        Maximum EM iterations at fit (default: 100).
    n_init : int, optional
        Number of seeded EM re-initializations at fit (default: 1).
    random_state : int, optional
        Seed for the sklearn fit, for reproducible parameters (default: 0).
    **kwargs : Any
        Forwarded to ``_StatisticalFitNode`` (``max_fit_pixels``, ``fit_seed``)
        and the ``Node`` base.

    Attributes
    ----------
    means : torch.Tensor
        Component means, shape ``[K, C]`` after fit.
    precisions_chol : torch.Tensor
        Cholesky factors of the precision matrices, shape ``[K, C, C]`` after
        fit (full covariance).
    weights : torch.Tensor
        Mixture weights, shape ``[K]`` after fit; sum to 1.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.CLASSIFICATION, NodeTag.STATEFUL, NodeTag.TORCH}
    )

    OUTPUT_SPECS = {
        "class_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="0-based argmax component id per pixel [B, H, W]",
        ),
        "abundances": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel component responsibilities [B, H, W, K] (sum to 1 over K)",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel mixture log-likelihood [B, H, W, 1]",
        ),
    }

    def __init__(
        self,
        n_components: int = 3,
        covariance_type: str = "full",
        reg_covar: float = 1e-6,
        max_iter: int = 100,
        n_init: int = 1,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        """Store mixture hyperparameters and register the fitted-state buffers."""
        self.n_components = int(n_components)
        self.covariance_type = str(covariance_type)
        self.reg_covar = float(reg_covar)
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.random_state = int(random_state)
        super().__init__(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
            **kwargs,
        )
        self.register_buffer("means", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("precisions_chol", torch.zeros(0, dtype=torch.float32))
        self.register_buffer("weights", torch.zeros(0, dtype=torch.float32))

    def _fit(self, pixels: torch.Tensor) -> None:
        """Fit scikit-learn ``GaussianMixture`` and freeze the parameters.

        Parameters
        ----------
        pixels : torch.Tensor
            ``[N, C]`` float32 training-pixel matrix.
        """
        from sklearn.mixture import GaussianMixture

        model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            max_iter=self.max_iter,
            n_init=self.n_init,
            random_state=self.random_state,
        )
        model.fit(pixels.cpu().numpy())
        self.means = torch.tensor(model.means_, dtype=torch.float32)
        self.weights = torch.tensor(model.weights_, dtype=torch.float32)
        self.precisions_chol = torch.tensor(model.precisions_cholesky_, dtype=torch.float32)

    def _estimate_weighted_log_prob(self, flat: torch.Tensor) -> torch.Tensor:
        """Compute the weighted log Gaussian probabilities (full covariance).

        Replicates scikit-learn's ``_estimate_log_gaussian_prob`` plus the
        ``log(weights)`` term for ``covariance_type="full"``.

        Parameters
        ----------
        flat : torch.Tensor
            ``[P, C]`` float32 pixel matrix.

        Returns
        -------
        torch.Tensor
            ``[P, K]`` weighted log-probabilities ``log p(x | k) + log w_k``.
        """
        means = self.means.to(device=flat.device, dtype=flat.dtype)
        weights = self.weights.to(device=flat.device, dtype=flat.dtype)
        precisions_chol = self.precisions_chol.to(device=flat.device, dtype=flat.dtype)
        _, n_features = flat.shape

        # log-det of each Cholesky factor: sum of logs of its diagonal entries.
        diag = torch.diagonal(precisions_chol, dim1=-2, dim2=-1)  # [K, C]
        log_det = torch.log(diag).sum(dim=1)  # [K]

        # y_k = (x - mu_k) @ prec_chol_k ; quadratic term = sum(y_k^2, axis=1).
        # einsum gives [P, K, C]; subtract the per-component mean projection.
        proj = torch.einsum("pc,kcd->pkd", flat, precisions_chol)  # [P, K, C]
        mu_proj = torch.einsum("kc,kcd->kd", means, precisions_chol)  # [K, C]
        y = proj - mu_proj.unsqueeze(0)  # [P, K, C]
        quad = y.square().sum(dim=2)  # [P, K]

        log_gaussian = -0.5 * (n_features * math.log(2.0 * math.pi) + quad) + log_det
        return log_gaussian + torch.log(weights).unsqueeze(0)

    @torch.no_grad()
    def forward(self, cube: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Evaluate the mixture posterior for every pixel.

        Parameters
        ----------
        cube : torch.Tensor
            Input hyperspectral cube ``[B, H, W, C]``.
        **_ : Any
            Additional unused keyword arguments (e.g. the pipeline ``context``).

        Returns
        -------
        dict[str, torch.Tensor]
            ``class_mask`` int32 ``[B, H, W]`` (argmax component),
            ``abundances`` float32 ``[B, H, W, K]`` (responsibilities, sum to 1
            over K), and ``scores`` float32 ``[B, H, W, 1]`` (per-pixel mixture
            log-likelihood).
        """
        self._require_initialized()
        B, H, W, C = cube.shape
        K = self.weights.shape[0]
        flat = cube.reshape(-1, C).to(torch.float32)

        log_weighted = self._estimate_weighted_log_prob(flat)  # [P, K]
        log_norm = torch.logsumexp(log_weighted, dim=1)  # [P]
        log_resp = log_weighted - log_norm.unsqueeze(1)  # [P, K]

        class_mask = log_weighted.argmax(dim=1).reshape(B, H, W).to(torch.int32)
        abundances = torch.exp(log_resp).reshape(B, H, W, K).to(torch.float32)
        scores = log_norm.reshape(B, H, W, 1).to(torch.float32)
        return {"class_mask": class_mask, "abundances": abundances, "scores": scores}
