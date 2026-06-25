"""Spectral-derivative pretreatment node.

Computes the first or second derivative of each spectrum with respect to
wavelength using :func:`torch.gradient`, with the true (possibly non-uniform)
band spacing taken from the ``wavelengths`` port.
"""

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class SpectralDerivative(Node):
    """First- or second-order spectral derivative along the band axis.

    Derivatives suppress additive/multiplicative baseline effects and sharpen
    absorption features. The derivative is taken with respect to wavelength
    (nanometers), honouring non-uniform band spacing via the ``wavelengths``
    port.

    Parameters
    ----------
    order : int, optional
        Derivative order; ``1`` (default) or ``2`` (gradient applied twice).
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
            description="Spectral derivative cube [B, H, W, C]",
        )
    }

    def __init__(self, order: int = 1, **kwargs) -> None:
        self.order = int(order)
        super().__init__(order=self.order, **kwargs)

    def forward(self, cube: torch.Tensor, wavelengths, **_) -> dict[str, torch.Tensor]:
        """Differentiate each spectrum with respect to wavelength.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.
        wavelengths : array-like
            Band wavelengths, length ``C``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": derivative}`` with the same shape as the input.
        """
        x = torch.as_tensor(np.asarray(wavelengths), device=cube.device).reshape(-1)
        x = x.to(cube.dtype)
        out = cube
        for _step in range(self.order):
            out = torch.gradient(out, spacing=(x,), dim=-1)[0]
        return {"cube": out}
