"""Continuum-removal pretreatment node.

Normalizes each pixel spectrum by its upper convex hull (the continuum) so
that absorption features become comparable across spectra of differing
overall brightness. The output is ``cube / hull``, where the hull is the upper
convex envelope over the ``(wavelength, reflectance)`` points, linearly
interpolated back to every band. Pure ``torch``.
"""

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class ContinuumRemoval(Node):
    """Per-pixel continuum (upper convex hull) removal across the spectral axis.

    For every spectrum the upper convex hull over ``(wavelength, reflectance)``
    is computed, linearly interpolated to all bands, and the spectrum is
    divided by it. Continuum-free regions map to ``~1.0`` while absorption
    bands dip below ``1.0``, making feature depths comparable across spectra.

    The hull is built with a batched Andrew monotone-chain scan that runs
    entirely in ``torch`` (vectorized over pixels), so the node stays
    device-agnostic.

    Parameters
    ----------
    eps : float, optional
        Lower clamp on the hull before division, guarding against division by
        zero (default: 1e-8).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelength array [C] in nanometers",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Continuum-removed cube [B, H, W, C]",
        )
    }

    def __init__(self, eps: float = 1e-8, **kwargs) -> None:
        self.eps = float(eps)
        super().__init__(eps=self.eps, **kwargs)

    @staticmethod
    def _upper_hull(x: torch.Tensor, spectra: torch.Tensor) -> torch.Tensor:
        """Upper convex hull of a batch of spectra, interpolated to all bands.

        Uses a batched monotone-chain: a per-spectrum vertex stack is grown by
        sweeping bands left to right, popping any vertex that would make the
        chain non-concave. The hull vertices are then linearly interpolated
        back onto every band, fully vectorized over spectra.

        Parameters
        ----------
        x : torch.Tensor
            Strictly increasing band positions, shape ``(C,)``.
        spectra : torch.Tensor
            Spectra, shape ``(N, C)``.

        Returns
        -------
        torch.Tensor
            Hull values at every band, shape ``(N, C)``.
        """
        N, C = spectra.shape
        device = spectra.device
        if C <= 2:
            # With <=2 bands the hull is the data (or its straight line) itself.
            return spectra.clone()

        stack = torch.zeros(N, C, dtype=torch.long, device=device)
        size = torch.zeros(N, dtype=torch.long, device=device)
        rows = torch.arange(N, device=device)
        for j in range(C):
            while True:
                active = size >= 2
                if not bool(active.any()):
                    break
                top = stack[rows, (size - 1).clamp_min(0)]
                second = stack[rows, (size - 2).clamp_min(0)]
                ox, oy = x[second], spectra[rows, second]
                ax, ay = x[top], spectra[rows, top]
                bx, by = x[j], spectra[:, j]
                cross = (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)
                pop = active & (cross >= 0)
                if not bool(pop.any()):
                    break
                size = torch.where(pop, size - 1, size)
            stack[rows, size] = j
            size = size + 1

        col = torch.arange(C, device=device).unsqueeze(0).expand(N, C)
        is_vertex = torch.zeros(N, C, dtype=torch.bool, device=device)
        is_vertex.scatter_(1, stack, col < size.unsqueeze(1))

        vert_pos = torch.where(is_vertex, col, torch.full_like(col, -1))
        prev_idx = torch.cummax(vert_pos, dim=1).values.clamp_min(0)
        nxt_src = torch.where(is_vertex, col, torch.full_like(col, C))
        nxt_idx = torch.flip(torch.cummin(torch.flip(nxt_src, [1]), dim=1).values, [1]).clamp_max(
            C - 1
        )

        xp, xn = x[prev_idx], x[nxt_idx]
        yp = torch.gather(spectra, 1, prev_idx)
        yn = torch.gather(spectra, 1, nxt_idx)
        same = xn == xp
        t = torch.where(same, torch.zeros_like(xp), (x.unsqueeze(0) - xp) / (xn - xp))
        interp = yp + (yn - yp) * t
        return torch.where(is_vertex, spectra, interp)

    def forward(self, cube: torch.Tensor, wavelengths, **_) -> dict[str, torch.Tensor]:
        """Divide each spectrum by its upper convex hull.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.
        wavelengths : array-like
            Band wavelengths, length ``C``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": continuum_removed}`` with the same shape as the input.
        """
        B, H, W, C = cube.shape
        x = torch.as_tensor(np.asarray(wavelengths), device=cube.device).reshape(-1)
        x = x.to(cube.dtype)
        spectra = cube.reshape(-1, C)
        hull = self._upper_hull(x, spectra)
        removed = spectra / hull.clamp_min(self.eps)
        return {"cube": removed.reshape(B, H, W, C)}
