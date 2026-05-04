"""Image-compositing nodes.

- ``ROIZoomNode``: crops a region defined by a bbox and resizes it to a fixed
  output frame size. Emits its own RGB stream (a standalone zoom video).

- ``InsetComposer``: pastes a fixed-size inset frame into a corner of a larger
  base frame with optional border, for picture-in-picture video output.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_core.node import Node


class ROIZoomNode(Node):
    """Crop a region defined by a bbox and resize it to a fixed output frame.

    Emits one RGB frame per input frame at ``(zoom_height, zoom_width)``.
    When ``valid`` is provided and equals ``0`` for a frame, the output is a
    solid ``bg_color`` frame.

    Parameters
    ----------
    zoom_height, zoom_width : int
        Output frame dimensions in pixels. Defaults 320 x 320.
    bg_color : tuple[float, float, float]
        Background RGB (in [0, 1]) used when ``valid == 0``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.IMAGE, NodeTag.PREPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "source": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Source RGB frames [B, H, W, 3] in [0, 1].",
        ),
        "bbox": PortSpec(
            dtype=torch.float32,
            shape=(-1, 4),
            description="Per-frame bbox [B, 4] in xyxy pixel coordinates.",
        ),
        "valid": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame validity [B]; 0 yields a blank zoom frame.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "zoom": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Zoomed RGB frames [B, zoom_height, zoom_width, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        zoom_height: int = 320,
        zoom_width: int = 320,
        bg_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        **kwargs: Any,
    ) -> None:
        if zoom_height < 8 or zoom_width < 8:
            raise ValueError("zoom dimensions must be >= 8 px")
        if len(bg_color) != 3:
            raise ValueError("bg_color must have 3 channels")

        self.zoom_height = int(zoom_height)
        self.zoom_width = int(zoom_width)
        self.bg_color = tuple(float(c) for c in bg_color)

        super().__init__(
            zoom_height=self.zoom_height,
            zoom_width=self.zoom_width,
            bg_color=self.bg_color,
            **kwargs,
        )

    @torch.no_grad()
    def forward(
        self,
        source: torch.Tensor,
        bbox: torch.Tensor,
        valid: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        b, src_h, src_w, _ = source.shape
        bboxes = bbox.round().to(torch.int64).clamp_min(0).tolist()
        valids = valid.to(torch.int64).tolist() if valid is not None else [1] * b

        bg = torch.tensor(self.bg_color, dtype=source.dtype, device=source.device)
        out = bg.view(1, 1, 1, 3).expand(b, self.zoom_height, self.zoom_width, 3).clone()

        for i, (coords, v) in enumerate(zip(bboxes, valids, strict=True)):
            if v == 0:
                continue
            x0, y0, x1, y1 = coords
            x1 = min(src_w, x1)
            y1 = min(src_h, y1)
            if x1 <= x0 or y1 <= y0:
                continue

            crop_nchw = source[i : i + 1, y0:y1, x0:x1, :].permute(0, 3, 1, 2).contiguous()
            resized = F.interpolate(
                crop_nchw,
                size=(self.zoom_height, self.zoom_width),
                mode="bilinear",
                align_corners=False,
            )
            out[i] = resized.permute(0, 2, 3, 1).squeeze(0).clamp(0.0, 1.0)

        return {"zoom": out}


_CORNERS = ("top-left", "top-right", "bottom-left", "bottom-right")


class InsetComposer(Node):
    """Paste a fixed-size inset frame into a corner of a larger base frame.

    Picture-in-picture compositor. The inset is expected to already be at its
    final pixel size (e.g. produced by :class:`ROIZoomNode`); this node only
    places it onto the base, optionally with a coloured border. When
    ``valid == 0`` for a frame the base passes through untouched, so the
    inset never lies about a stale ROI.

    Parameters
    ----------
    corner : str
        One of ``"top-left"``, ``"top-right"``, ``"bottom-left"``,
        ``"bottom-right"``. Default ``"top-right"``.
    margin_px : int
        Distance in pixels between the inset and the closest base edges.
        Default ``16``.
    border_px : int
        Border thickness in pixels. ``0`` disables the border. Default ``2``.
    border_color : tuple[float, float, float]
        Border RGB in [0, 1]. Default white ``(1, 1, 1)``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.IMAGE, NodeTag.PREPROCESSING, NodeTag.NUMPY})

    INPUT_SPECS = {
        "base": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Base RGB frames [B, H, W, 3] in [0, 1].",
        ),
        "inset": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Inset RGB frames [B, h, w, 3] in [0, 1] at final size.",
        ),
        "valid": PortSpec(
            dtype=torch.int32,
            shape=(-1,),
            description="Per-frame validity [B]; 0 leaves the base untouched.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "composite": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Base frames with inset pasted in [B, H, W, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        corner: str = "top-right",
        margin_px: int = 16,
        border_px: int = 2,
        border_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        **kwargs: Any,
    ) -> None:
        if corner not in _CORNERS:
            raise ValueError(f"corner must be one of {_CORNERS}, got {corner!r}")
        if margin_px < 0:
            raise ValueError("margin_px must be >= 0")
        if border_px < 0:
            raise ValueError("border_px must be >= 0")
        if len(border_color) != 3:
            raise ValueError("border_color must have 3 channels")

        self.corner = corner
        self.margin_px = int(margin_px)
        self.border_px = int(border_px)
        self.border_color = tuple(float(c) for c in border_color)

        super().__init__(
            corner=self.corner,
            margin_px=self.margin_px,
            border_px=self.border_px,
            border_color=self.border_color,
            **kwargs,
        )

    def _placement(self, base_h: int, base_w: int, ins_h: int, ins_w: int) -> tuple[int, int]:
        """Top-left (y, x) corner where the (bordered) inset block starts."""
        block_h = ins_h + 2 * self.border_px
        block_w = ins_w + 2 * self.border_px
        if block_h + 2 * self.margin_px > base_h or block_w + 2 * self.margin_px > base_w:
            raise ValueError(
                f"inset block ({block_h}x{block_w}) plus margins does not fit in "
                f"base ({base_h}x{base_w}); shrink the zoom size or margin."
            )

        if self.corner == "top-left":
            return self.margin_px, self.margin_px
        if self.corner == "top-right":
            return self.margin_px, base_w - self.margin_px - block_w
        if self.corner == "bottom-left":
            return base_h - self.margin_px - block_h, self.margin_px
        # bottom-right
        return base_h - self.margin_px - block_h, base_w - self.margin_px - block_w

    @torch.no_grad()
    def forward(
        self,
        base: torch.Tensor,
        inset: torch.Tensor,
        valid: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        b, base_h, base_w, _ = base.shape
        _, ins_h, ins_w, _ = inset.shape

        y0, x0 = self._placement(base_h, base_w, ins_h, ins_w)
        bp = self.border_px
        ix0, iy0 = x0 + bp, y0 + bp
        ix1, iy1 = ix0 + ins_w, iy0 + ins_h

        out = base.clone()
        valids = valid.to(torch.int64).tolist() if valid is not None else [1] * b

        if bp > 0:
            color = torch.tensor(self.border_color, dtype=base.dtype, device=base.device)
        else:
            color = None

        for i, v in enumerate(valids):
            if v == 0:
                continue
            if color is not None:
                # Paint the full bordered block with the border colour, then
                # overwrite the inner region with the inset on top.
                out[i, y0 : y0 + ins_h + 2 * bp, x0 : x0 + ins_w + 2 * bp] = color
            out[i, iy0:iy1, ix0:ix1] = inset[i].clamp(0.0, 1.0)

        return {"composite": out}
