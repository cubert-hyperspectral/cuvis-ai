"""Blind spectral unmixing via non-negative matrix factorization.

:class:`NMFUnmixing` learns ``K`` endmember spectra from training pixels with
:class:`sklearn.decomposition.NMF` during ``statistical_initialization``, freezes
them as a buffer, and at inference solves per-pixel non-negative abundances
against those frozen endmembers using the same pure-torch projected-gradient NNLS
solver as :class:`~cuvis_ai.node.unmixing.nnls.NNLSUnmixing`. sklearn is imported
only at fit time; ``forward`` is torch-only.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node._statistical_fit import _StatisticalFitNode
from cuvis_ai.node.unmixing._solve import nnls_batch, reconstruction_residual

#: Projected-gradient iteration budget for the per-pixel abundance solve at
#: inference. Independent of the sklearn fit's ``max_iter`` so a small fit budget
#: never starves the abundance recovery against the frozen endmembers.
_FORWARD_NNLS_ITERS = 2000


class NMFUnmixing(_StatisticalFitNode):
    """Blind unmixing: learn endmembers by NMF, then solve per-pixel abundances.

    During ``statistical_initialization`` the node fits
    :class:`sklearn.decomposition.NMF` on the collected training pixels and stores
    the learned components ``[K, C]`` as a frozen buffer. At inference it solves
    ``min_{x >= 0} ||A x - b||`` per pixel against those frozen endmembers
    (``A = endmembers.T``) with batched projected-gradient descent, emitting
    abundances, the learned endmembers, the per-pixel reconstruction residual, and
    a 1-based argmax class mask.

    Parameters
    ----------
    n_components : int, optional
        Number of endmembers ``K`` to learn (default: 3).
    init : str, optional
        sklearn NMF initialization scheme (default: ``"nndsvda"``).
    beta_loss : str, optional
        sklearn NMF beta-divergence loss (default: ``"frobenius"``).
    max_iter : int, optional
        Maximum sklearn NMF solver iterations at fit time (default: 300).
    random_state : int, optional
        Seed for the sklearn NMF solver (default: 0).
    **kwargs : Any
        Forwarded to :class:`cuvis_ai.node._statistical_fit._StatisticalFitNode`,
        including ``max_fit_pixels`` and ``fit_seed``.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.STATEFUL, NodeTag.TORCH})

    TRAINABLE_BUFFERS = ("endmembers_buf",)

    OUTPUT_SPECS = {
        "abundances": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel non-negative endmember abundances [B, H, W, K]",
        ),
        "endmembers": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Learned endmember spectra [K, C]",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel reconstruction residual ||E.T a - x|| against the "
            "frozen endmembers E [B, H, W, 1]",
        ),
        "class_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="1-based argmax abundance per pixel [B, H, W]",
        ),
    }

    def __init__(
        self,
        n_components: int = 3,
        init: str = "nndsvda",
        beta_loss: str = "frobenius",
        max_iter: int = 300,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        self.n_components = int(n_components)
        self.init = str(init)
        self.beta_loss = str(beta_loss)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        super().__init__(
            n_components=self.n_components,
            init=self.init,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **kwargs,
        )
        # Placeholder; resized to [K, C] at fit time and on checkpoint reload.
        self.register_buffer("endmembers_buf", torch.zeros(0, dtype=torch.float32))

    def _fit(self, pixels: torch.Tensor) -> None:
        """Fit sklearn NMF on the training pixels and freeze the endmembers.

        Parameters
        ----------
        pixels : torch.Tensor
            Collected training pixel matrix ``[N, C]``.
        """
        import sklearn.decomposition

        model = sklearn.decomposition.NMF(
            n_components=self.n_components,
            init=self.init,
            beta_loss=self.beta_loss,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(pixels.cpu().numpy())
        self.endmembers_buf = torch.tensor(model.components_, dtype=torch.float32)

    def forward(self, cube: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Solve per-pixel abundances against the frozen learned endmembers.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube ``[B, H, W, C]``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``abundances`` ``[B, H, W, K]``, ``endmembers`` ``[K, C]``,
            ``scores`` (reconstruction residual) ``[B, H, W, 1]``, and
            ``class_mask`` ``[B, H, W]``.
        """
        self._require_initialized()
        if cube.ndim != 4:
            raise ValueError(f"Expected cube with shape [B, H, W, C], got {tuple(cube.shape)}")

        batch, height, width, channels = cube.shape
        endmembers = self.endmembers_buf.to(dtype=cube.dtype, device=cube.device)  # [K, C]
        components = endmembers.shape[0]
        if endmembers.shape[1] != channels:
            raise ValueError(
                f"Learned endmember channels {endmembers.shape[1]} do not match "
                f"cube channels {channels}."
            )

        a = endmembers.transpose(0, 1)  # [C, K]
        outputs: list[torch.Tensor] = []
        residuals: list[torch.Tensor] = []
        for frame in cube:
            b = frame.reshape(-1, channels)  # [P, C]
            x = nnls_batch(a, b, max_iter=_FORWARD_NNLS_ITERS, tol=1e-6)  # [P, K]
            outputs.append(x.reshape(height, width, components))
            residuals.append(reconstruction_residual(a, x, b).reshape(height, width, 1))

        abundances = torch.stack(outputs, dim=0)  # [B, H, W, K]
        scores = torch.stack(residuals, dim=0)  # [B, H, W, 1]
        class_mask = (abundances.argmax(dim=-1) + 1).to(torch.int32)  # [B, H, W]

        return {
            "abundances": abundances,
            "endmembers": endmembers,
            "scores": scores,
            "class_mask": class_mask,
        }


__all__ = ["NMFUnmixing"]
