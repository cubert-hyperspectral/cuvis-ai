"""Standard Normal Variate (SNV) pretreatment node.

Per-pixel scatter correction: each spectrum is centred by its own mean and
scaled by its own standard deviation across the band axis.
"""

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class SNVCorrection(Node):
    """Standard Normal Variate correction along the spectral axis.

    For every spectrum the per-band mean is subtracted and the result divided
    by the per-band standard deviation, removing multiplicative scatter and
    additive baseline effects on a pixel-by-pixel basis. Stateless: no fitting
    required.

    Parameters
    ----------
    eps : float, optional
        Lower clamp on the per-spectrum standard deviation, guarding against
        division by zero on flat spectra (default: 1e-8).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.NORMALIZATION, NodeTag.TORCH}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        )
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="SNV-corrected cube [B, H, W, C]",
        )
    }

    def __init__(self, eps: float = 1e-8, **kwargs) -> None:
        self.eps = float(eps)
        super().__init__(eps=self.eps, **kwargs)

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Apply per-spectrum mean centring and unit-variance scaling.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": corrected}`` with the same shape as the input.
        """
        mean = cube.mean(dim=-1, keepdim=True)
        std = cube.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return {"cube": (cube - mean) / std}
