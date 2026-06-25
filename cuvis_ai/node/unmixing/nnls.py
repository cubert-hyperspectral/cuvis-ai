"""Stateless non-negative least squares spectral unmixing.

:class:`NNLSUnmixing` decomposes each pixel spectrum into non-negative abundances
of a set of *known* endmember spectra supplied at runtime on the ``endmembers``
port. It solves ``min_{x >= 0} ||A x - b||`` per pixel, where ``A`` is the
endmember matrix and ``b`` the pixel spectrum, via batched projected-gradient
descent. The node is stateless: it fits nothing and holds no buffers.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.unmixing._solve import nnls_batch, reconstruction_residual
from cuvis_ai_core.node import Node


class NNLSUnmixing(Node):
    """Unmix each pixel into non-negative abundances of known endmembers.

    Given a hyperspectral cube ``[B, H, W, C]`` and a set of ``K`` endmember
    spectra ``[K, C]``, solve ``min_{x >= 0} ||A x - b||`` for every pixel, where
    ``A = endmembers.T`` has shape ``[C, K]`` and ``b`` is the pixel spectrum.
    The solve runs as batched projected-gradient descent, so the node is
    stateless and runs entirely in torch on the inputs' device.

    Parameters
    ----------
    max_iter : int, optional
        Maximum projected-gradient iterations per forward call (default: 200).
    tol : float, optional
        Early-stop threshold on the per-iteration update norm (default: 1e-6).
    min_total : float, optional
        Pixels whose summed abundance falls below this value are labelled
        background (class 0) in ``class_mask`` (default: 0.0).
    **kwargs : Any
        Forwarded to :class:`cuvis_ai_core.node.Node`.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "endmembers": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Known endmember spectra [K, C]",
        ),
    }

    OUTPUT_SPECS = {
        "abundances": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-pixel non-negative endmember abundances [B, H, W, K]",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel least-squares residual ||A x - b|| [B, H, W, 1]",
        ),
        "class_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="1-based argmax abundance per pixel, 0 below min_total [B, H, W]",
        ),
    }

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        min_total: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.min_total = float(min_total)
        super().__init__(
            max_iter=self.max_iter,
            tol=self.tol,
            min_total=self.min_total,
            **kwargs,
        )

    def forward(
        self, cube: torch.Tensor, endmembers: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Solve per-pixel non-negative least squares against the endmembers.

        Parameters
        ----------
        cube : torch.Tensor
            Hyperspectral cube ``[B, H, W, C]``.
        endmembers : torch.Tensor
            Endmember spectra ``[K, C]``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``abundances`` ``[B, H, W, K]``, ``scores`` (residual) ``[B, H, W, 1]``,
            and ``class_mask`` ``[B, H, W]``.
        """
        if cube.ndim != 4:
            raise ValueError(f"Expected cube with shape [B, H, W, C], got {tuple(cube.shape)}")
        if endmembers.ndim != 2:
            raise ValueError(
                f"Expected endmembers with shape [K, C], got {tuple(endmembers.shape)}"
            )

        batch, height, width, channels = cube.shape
        components = endmembers.shape[0]
        if endmembers.shape[1] != channels:
            raise ValueError(
                f"Endmember channels {endmembers.shape[1]} do not match cube channels {channels}."
            )

        endmembers = endmembers.to(dtype=cube.dtype, device=cube.device)
        a = endmembers.transpose(0, 1)  # [C, K]
        b = cube.reshape(-1, channels)  # [P, C]

        x = nnls_batch(a, b, max_iter=self.max_iter, tol=self.tol)  # [P, K]
        residual = reconstruction_residual(a, x, b)  # [P]

        abundances = x.reshape(batch, height, width, components)
        scores = residual.reshape(batch, height, width, 1)

        totals = x.sum(dim=1)  # [P]
        class_idx = x.argmax(dim=1) + 1  # [P], 1-based
        class_idx = torch.where(
            totals < self.min_total,
            torch.zeros_like(class_idx),
            class_idx,
        )
        class_mask = class_idx.reshape(batch, height, width).to(torch.int32)

        return {
            "abundances": abundances,
            "scores": scores,
            "class_mask": class_mask,
        }


__all__ = ["NNLSUnmixing"]
