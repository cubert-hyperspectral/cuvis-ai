"""Image-compositing nodes.

Provides ``ROIZoomNode``: crops a region defined by a bbox and resizes it to
a fixed output frame size. Emitted as its own RGB stream (for a standalone
zoom video); no compositing onto a base image.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec


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
