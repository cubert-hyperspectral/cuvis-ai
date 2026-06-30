"""Scalar-to-RGB and class-index-to-RGB colormap nodes."""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node

#: Tableau-20 palette (RGB 0-255) used when ``ClassMapToRGB`` gets no explicit palette.
_TAB20: tuple[tuple[int, int, int], ...] = (
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
)


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

    _category = NodeCategory.VISUALIZER
    _tags = frozenset({NodeTag.RGB})

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


class ClassMapToRGB(Node):
    """Colourise an integer class-index map ``[B, H, W]`` into an RGB image ``[B, H, W, 3]``.

    Each class id indexes a palette colour; pixels equal to ``background_value`` (and, when a
    ``mask`` is connected, pixels where ``mask == 0``) render black. The explicit ``palette``
    lets it colourise arbitrary integer id-maps (compartment ids, cluster ids, class indices).

    Parameters
    ----------
    palette : list[tuple[int, int, int]] | None
        Per-id RGB colours in 0-255, indexed by class id. When ``None``, a Tableau-20 palette is
        cycled. The lookup wraps modulo the palette length, so ids beyond it reuse colours.
    background_value : int
        Class id rendered black (default ``-1``, so id ``0`` stays a valid class for clustering).
    """

    _category = NodeCategory.VISUALIZER
    _tags = frozenset({NodeTag.MASK, NodeTag.RGB})

    INPUT_SPECS = {
        "class_map": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Integer class-index map [B, H, W]; background_value pixels render black.",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Optional foreground mask [B, H, W]; pixels where mask == 0 render black.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "label_rgb": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Colourised RGB image [B, H, W, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        palette: list[tuple[int, int, int]] | None = None,
        background_value: int = -1,
        **kwargs: Any,
    ) -> None:
        colors = list(palette) if palette is not None else list(_TAB20)
        if not colors:
            raise ValueError("palette must be a non-empty list of (r, g, b)")
        self.background_value = int(background_value)
        super().__init__(
            palette=[[int(c) for c in rgb] for rgb in colors],
            background_value=self.background_value,
            **kwargs,
        )
        self._lut = torch.tensor(
            [[c / 255.0 for c in rgb] for rgb in colors], dtype=torch.float32
        )  # [P, 3] in [0, 1]

    @torch.no_grad()
    def forward(self, class_map: Tensor, mask: Tensor | None = None, **_: Any) -> dict[str, Tensor]:
        """Look the palette up per pixel; background and masked-out pixels stay black."""
        lut = self._lut.to(class_map.device)
        idx = class_map.clamp(min=0) % lut.shape[0]
        rgb = lut[idx]  # [B, H, W, 3]
        valid = class_map != self.background_value
        if mask is not None:
            valid = valid & (mask.to(class_map.device) != 0)
        return {"label_rgb": torch.where(valid.unsqueeze(-1), rgb, torch.zeros_like(rgb))}


__all__ = ["ClassMapToRGB", "ScalarHSVColormapNode", "render_scalar_hsv_colormap"]
