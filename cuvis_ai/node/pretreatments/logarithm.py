"""Logarithm pretreatment node.

Applies a base-10 or natural logarithm to the cube, optionally negated. The
plain log (``negate=False``, default) returns ``+log10(x)`` / ``+ln(x)`` and is
useful for compressing dynamic range. Absorbance is ``A = -log10(R)``, so set
``negate=True`` to convert reflectance/transmittance to (pseudo-)absorbance.
"""

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class Logarithm(Node):
    """Element-wise logarithm of the cube.

    Computes ``log10(x)`` (default) or ``ln(x)`` after clamping the input to a
    small positive floor so non-positive values do not produce ``-inf`` or
    ``nan``. With ``negate=True`` the sign is flipped, yielding true absorbance
    ``-log10(R)`` from reflectance.

    Parameters
    ----------
    mode : str, optional
        ``"log10"`` (default) for base-10, or ``"ln"`` for the natural log.
    negate : bool, optional
        Negate the result so reflectance maps to absorbance (default: False).
    eps : float, optional
        Lower clamp applied before the logarithm (default: 1e-8).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

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
            description="Log-transformed cube [B, H, W, C]",
        )
    }

    _MODES = ("log10", "ln")

    def __init__(
        self, mode: str = "log10", negate: bool = False, eps: float = 1e-8, **kwargs
    ) -> None:
        self.mode = str(mode)
        if self.mode not in self._MODES:
            raise ValueError(f"mode must be one of {list(self._MODES)}, got {self.mode!r}")
        self.negate = bool(negate)
        self.eps = float(eps)
        super().__init__(mode=self.mode, negate=self.negate, eps=self.eps, **kwargs)

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Apply the configured logarithm to the cube.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": log_transformed}`` with the same shape as the input.
        """
        clamped = cube.clamp_min(self.eps)
        result = torch.log(clamped) if self.mode == "ln" else torch.log10(clamped)
        if self.negate:
            result = -result
        return {"cube": result}
