"""Scalar-to-RGB colormap nodes."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


def render_scalar_hsv_colormap(normalized: Tensor) -> Tensor:
    """Apply an HSV rainbow colormap to normalized scalar values in ``[0, 1]``.

    Parameters
    ----------
    normalized : Tensor
        Scalar image in BHWC format with a singleton channel dimension.

    Returns
    -------
    Tensor
        RGB image in BHWC format with values in ``[0, 1]``.
    """
    hue = normalized.clamp(0.0, 1.0)
    h6 = hue * 6.0
    sector = torch.floor(h6).to(torch.int64) % 6
    frac = h6 - torch.floor(h6)

    one = torch.ones_like(hue)
    zero = torch.zeros_like(hue)
    q = 1.0 - frac
    t = frac

    red = torch.zeros_like(hue)
    green = torch.zeros_like(hue)
    blue = torch.zeros_like(hue)

    mask0 = sector == 0
    red = torch.where(mask0, one, red)
    green = torch.where(mask0, t, green)
    blue = torch.where(mask0, zero, blue)

    mask1 = sector == 1
    red = torch.where(mask1, q, red)
    green = torch.where(mask1, one, green)
    blue = torch.where(mask1, zero, blue)

    mask2 = sector == 2
    red = torch.where(mask2, zero, red)
    green = torch.where(mask2, one, green)
    blue = torch.where(mask2, t, blue)

    mask3 = sector == 3
    red = torch.where(mask3, zero, red)
    green = torch.where(mask3, q, green)
    blue = torch.where(mask3, one, blue)

    mask4 = sector == 4
    red = torch.where(mask4, t, red)
    green = torch.where(mask4, zero, green)
    blue = torch.where(mask4, one, blue)

    mask5 = sector == 5
    red = torch.where(mask5, one, red)
    green = torch.where(mask5, zero, green)
    blue = torch.where(mask5, q, blue)

    return torch.cat([red, green, blue], dim=-1).clamp_(0.0, 1.0)


class ScalarHSVColormapNode(Node):
    """Map a scalar BHWC image to RGB using an HSV colormap."""

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Scalar image [B, H, W, 1].",
        )
    }
    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="HSV color-mapped RGB image [B, H, W, 3] in [0, 1].",
        )
    }

    def __init__(self, value_min: float = 0.0, value_max: float = 1.0, **kwargs: Any) -> None:
        if value_max <= value_min:
            raise ValueError("value_max must be greater than value_min")
        super().__init__(value_min=float(value_min), value_max=float(value_max), **kwargs)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self._value_range = self.value_max - self.value_min

    def forward(self, data: Tensor, **_: Any) -> dict[str, Tensor]:
        """Colorize a scalar image in BHWC format."""
        if data.ndim != 4 or data.shape[-1] != 1:
            raise ValueError(
                f"Expected scalar data with shape [B, H, W, 1], got {tuple(data.shape)}"
            )
        normalized = ((data - self.value_min) / self._value_range).clamp(0.0, 1.0)
        return {"rgb_image": render_scalar_hsv_colormap(normalized)}


__all__ = ["ScalarHSVColormapNode", "render_scalar_hsv_colormap"]
