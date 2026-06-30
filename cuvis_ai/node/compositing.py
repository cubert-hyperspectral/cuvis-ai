"""Image-compositing nodes.

- ``ROIZoomNode``: crops a region defined by a bbox and resizes it to a fixed
  output frame size. Emits its own RGB stream (a standalone zoom video).

- ``InsetComposer``: pastes a fixed-size inset frame into a corner of a larger
  base frame with optional border, for picture-in-picture video output.

- ``ImageConcatenator``: stitches a variable number of RGB frames into one
  side-by-side (or stacked) strip, for a single node-native result image.

- ``LabelOverlay``: alpha-blends a colourised label map onto an RGB frame.

- ``TitleOverlay``: burns a text caption into the top-left of an RGB frame.

- ``LegendStrip``: appends a class-colour legend strip below an RGB frame.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from PIL import Image, ImageDraw, ImageFont

from cuvis_ai.utils.connected_components import label_connected_components
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


_AXES = ("horizontal", "vertical")
_ALIGNS = ("start", "center", "end")


class ImageConcatenator(Node):
    """Concatenate several RGB frames into one side-by-side or stacked strip.

    A fan-in node: connect any number of ``rgb_image`` sources to the single
    ``images`` port and they are concatenated in **connection order** (the
    order the edges were added with ``pipeline.connect``). Frames may differ on
    the cross axis (height for a horizontal strip, width for a vertical one);
    each is padded to the common size with ``bg_color`` and aligned per
    ``align``. An optional ``gap`` inserts a ``bg_color`` separator between
    frames. The whole batch is concatenated together, so every source must
    share the same batch size.

    Parameters
    ----------
    axis : str
        ``"horizontal"`` places frames left-to-right (pads heights);
        ``"vertical"`` stacks them top-to-bottom (pads widths). Default
        ``"horizontal"``.
    gap : int
        Width (horizontal) or height (vertical) in pixels of a ``bg_color``
        separator inserted between adjacent frames. ``0`` disables it. Default
        ``0``.
    bg_color : tuple[float, float, float]
        RGB in [0, 1] used for padding and gaps. Default white ``(1, 1, 1)``.
    align : str
        Cross-axis placement of a smaller frame: ``"start"`` (top / left),
        ``"center"``, or ``"end"`` (bottom / right). Default ``"center"``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.IMAGE, NodeTag.RGB})

    INPUT_SPECS = {
        "images": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frames to concatenate, [B, H, W, 3] in [0, 1]; one per connection.",
            variadic=True,
        ),
    }

    OUTPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Concatenated RGB strip [B, H', W', 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        axis: str = "horizontal",
        gap: int = 0,
        bg_color: tuple[float, float, float] = (1.0, 1.0, 1.0),
        align: str = "center",
        **kwargs: Any,
    ) -> None:
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {_AXES}, got {axis!r}")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if len(bg_color) != 3:
            raise ValueError("bg_color must have 3 channels")
        if align not in _ALIGNS:
            raise ValueError(f"align must be one of {_ALIGNS}, got {align!r}")

        self.axis = axis
        self.gap = int(gap)
        self.bg_color = tuple(float(c) for c in bg_color)
        self.align = align

        super().__init__(
            axis=self.axis,
            gap=self.gap,
            bg_color=self.bg_color,
            align=self.align,
            **kwargs,
        )

    def _pad_cross(self, img: torch.Tensor, target: int, bg: torch.Tensor) -> torch.Tensor:
        """Pad ``img`` along the cross axis to ``target`` with ``bg``, per ``align``."""
        cross = 1 if self.axis == "horizontal" else 2
        cur = img.shape[cross]
        if cur == target:
            return img
        pad_total = target - cur
        if self.align == "start":
            before = 0
        elif self.align == "center":
            before = pad_total // 2
        else:  # "end"
            before = pad_total

        shape = list(img.shape)
        shape[cross] = target
        canvas = bg.view(1, 1, 1, 3).expand(shape).clone()
        if cross == 1:
            canvas[:, before : before + cur, :, :] = img
        else:
            canvas[:, :, before : before + cur, :] = img
        return canvas

    @torch.no_grad()
    def forward(self, images: list[torch.Tensor], **_: Any) -> dict[str, torch.Tensor]:
        if not images:
            raise ValueError("ImageConcatenator received no images")

        ref = images[0]
        batch = ref.shape[0]
        for k, img in enumerate(images):
            if img.ndim != 4 or img.shape[-1] != 3:
                raise ValueError(f"images[{k}] must be [B, H, W, 3], got shape {tuple(img.shape)}")
            if img.shape[0] != batch:
                raise ValueError(
                    f"images[{k}] has batch {img.shape[0]}, expected {batch} (all must match)"
                )

        bg = torch.tensor(self.bg_color, dtype=ref.dtype, device=ref.device)
        horizontal = self.axis == "horizontal"
        cross = 1 if horizontal else 2
        cat_dim = 2 if horizontal else 1

        target = max(img.shape[cross] for img in images)
        padded = [self._pad_cross(img.to(ref.device), target, bg) for img in images]

        gap_block: torch.Tensor | None = None
        if self.gap > 0:
            gap_shape = [batch, target, self.gap, 3] if horizontal else [batch, self.gap, target, 3]
            gap_block = bg.view(1, 1, 1, 3).expand(gap_shape).clone()

        pieces: list[torch.Tensor] = []
        for i, img in enumerate(padded):
            if i > 0 and gap_block is not None:
                pieces.append(gap_block)
            pieces.append(img)

        out = torch.cat(pieces, dim=cat_dim).clamp(0.0, 1.0)
        return {"rgb_image": out}


# TEMP: lifted verbatim from the cuvis_ai_metalscrapes experiment plugin
# (node/compositor.py). Remove once those viz nodes ship in the catalog/plugin.
class LabelOverlay(Node):
    """Alpha-blend a colourised label map onto an RGB image on its foreground pixels.

    A pixel is "foreground" when its ``label_rgb`` differs from ``background_color``;
    background pixels keep the original RGB. Returns a single blended frame, so several
    overlays can be montaged column-by-column.

    Parameters
    ----------
    alpha : float
        Blend factor for the label colour over the base image (default 0.55).
    background_color : tuple[float, float, float]
        Label-map background colour in [0, 1]; pixels equal to it are left unblended.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.IMAGE, NodeTag.RGB})

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Base RGB image [B, H, W, 3] in [0, 1].",
        ),
        "label_rgb": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Colourised label map [B, H, W, 3] in [0, 1].",
        ),
    }
    OUTPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Blended RGB image [B, H, W, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        alpha: float = 0.55,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1]; got {alpha}")
        if len(background_color) != 3:
            raise ValueError("background_color must have 3 channels")
        self.alpha = float(alpha)
        self.background_color = tuple(float(c) for c in background_color)
        super().__init__(alpha=self.alpha, background_color=self.background_color, **kwargs)

    @torch.no_grad()
    def forward(
        self, rgb_image: torch.Tensor, label_rgb: torch.Tensor, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Blend ``label_rgb`` onto ``rgb_image`` where the label is non-background."""
        if not torch.is_floating_point(label_rgb):
            label_rgb = label_rgb.to(torch.float32) / 255.0
        label_rgb = label_rgb.to(device=rgb_image.device, dtype=rgb_image.dtype)
        if rgb_image.shape != label_rgb.shape:
            raise ValueError(
                f"rgb_image {tuple(rgb_image.shape)} != label_rgb {tuple(label_rgb.shape)}"
            )
        bg = torch.tensor(self.background_color, device=rgb_image.device, dtype=rgb_image.dtype)
        fg_mask = (label_rgb - bg).abs().sum(dim=-1, keepdim=True) > 1e-6
        blend_weight = self.alpha * fg_mask.to(rgb_image.dtype)
        frame = (1.0 - blend_weight) * rgb_image + blend_weight * label_rgb
        return {"frame": frame.clamp(0.0, 1.0)}


# TEMP: lifted verbatim from the cuvis_ai_metalscrapes experiment plugin
# (node/compositor.py). Remove once those viz nodes ship in the catalog/plugin.
class TitleOverlay(Node):
    """Burn a text caption into the top-left of each RGB frame, over a translucent box.

    The caption comes from one of three places, in priority order: the per-frame
    ``caption`` input port (a ``list[str]``, one entry per frame, so a DataModule can
    title each montage column), the ``text`` argument to :meth:`forward`, or the
    constructor ``text`` default. Drawn with PIL over a semi-transparent box so it stays
    legible on any background.

    Parameters
    ----------
    text : str
        Default caption drawn into every frame when no per-frame ``caption`` is wired.
    font_size : int
        Caption font size in points (default 20).
    pad_px : int
        Inset of the caption box from the top-left corner (default 8).
    text_color, box_color : tuple[int, int, int]
        Text and box RGB colours in 0-255.
    box_alpha : float
        Opacity of the box behind the text (default 0.5).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.IMAGE, NodeTag.RGB})

    INPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame [B, H, W, 3] in [0, 1].",
        ),
        "caption": PortSpec(
            dtype=list,
            shape=(),
            description="Optional per-frame captions (list[str], one per batch element).",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB frame with the caption drawn in, [B, H, W, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        text: str = "",
        font_size: int = 20,
        pad_px: int = 8,
        text_color: tuple[int, int, int] = (255, 255, 255),
        box_color: tuple[int, int, int] = (0, 0, 0),
        box_alpha: float = 0.5,
        **kwargs: Any,
    ) -> None:
        if not 0.0 <= box_alpha <= 1.0:
            raise ValueError(f"box_alpha must be in [0, 1]; got {box_alpha}")
        self.text = str(text)
        self.pad_px = int(pad_px)
        self.text_color = tuple(int(c) for c in text_color)
        self.box_color = tuple(int(c) for c in box_color)
        self.box_alpha = float(box_alpha)
        super().__init__(
            text=self.text,
            font_size=int(font_size),
            pad_px=self.pad_px,
            text_color=list(self.text_color),
            box_color=list(self.box_color),
            box_alpha=self.box_alpha,
            **kwargs,
        )
        try:
            self._font = ImageFont.truetype("arial.ttf", int(font_size))
        except OSError:
            self._font = ImageFont.load_default()

    def _draw(self, frame_hw3: np.ndarray, text: str) -> np.ndarray:
        """Draw ``text`` over a translucent box on one [H, W, 3] uint8 frame."""
        img = Image.fromarray(frame_hw3).convert("RGBA")
        layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(layer)
        x0, y0 = self.pad_px, self.pad_px
        left, top, right, bottom = draw.textbbox((x0, y0), text, font=self._font)
        draw.rectangle(
            [left - 5, top - 3, right + 5, bottom + 3],
            fill=(*self.box_color, int(255 * self.box_alpha)),
        )
        draw.text((x0, y0), text, fill=(*self.text_color, 255), font=self._font)
        merged = Image.alpha_composite(img, layer).convert("RGB")
        return np.asarray(merged, dtype=np.float32) / 255.0

    @torch.no_grad()
    def forward(
        self,
        frame: torch.Tensor,
        caption: list[str] | None = None,
        text: str | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Caption each frame from the per-frame ``caption`` port, ``text``, or the default.

        Parameters
        ----------
        frame : torch.Tensor
            RGB frames ``[B, H, W, 3]`` in ``[0, 1]``.
        caption : list[str] or None, optional
            Per-frame captions, one per batch element; takes priority over ``text`` and the
            constructor default. Must have length ``B``.
        text : str or None, optional
            Single caption applied to every frame, overriding the constructor default.
        **_ : Any
            Additional unused keyword arguments (e.g. the pipeline ``context``).

        Returns
        -------
        dict[str, torch.Tensor]
            ``frame`` float32 ``[B, H, W, 3]`` with each caption drawn in; an empty caption
            leaves its frame unchanged.
        """
        batch = frame.shape[0]
        if caption is not None:
            if len(caption) != batch:
                raise ValueError(
                    f"caption has {len(caption)} entries but the batch has {batch} frames."
                )
            labels = [str(c) for c in caption]
        else:
            single = self.text if text is None else str(text)
            labels = [single] * batch
        out = torch.empty_like(frame)
        for i in range(batch):
            clamped = frame[i].clamp(0.0, 1.0)
            if not labels[i].strip():
                # An empty caption is a no-op: draw no box and pass the frame through.
                out[i] = clamped
                continue
            arr = (clamped * 255).round().to(torch.uint8).cpu().numpy()
            out[i] = torch.from_numpy(self._draw(arr, labels[i])).to(frame.device, frame.dtype)
        return {"frame": out}


class LegendStrip(Node):
    """Append a horizontal class-colour legend strip below the input frame.

    Each ``(label, rgb)`` entry renders as a swatch plus its text label, wrapped over
    ``n_columns``. When the optional ``label_rgb`` mask is connected, the legend appends
    a connected-component instance count ``(N)`` per class for the current frame and dims
    rows whose count is zero. The legend is built from an explicit ``entries`` list of
    ``(label, rgb)`` rows.

    Parameters
    ----------
    entries : list[tuple[str, tuple[int, int, int]]]
        Ordered ``(label, (r, g, b))`` legend rows; colours in 0-255.
    n_columns : int
        Number of legend columns before wrapping to a new row (default 6).
    tile_height_px, swatch_width_px, text_padding_px, font_size : int
        Legend layout dimensions in pixels.
    background_color : tuple[float, float, float]
        Strip background colour in [0, 1].
    text_color, dim_text_color : tuple[int, int, int]
        Text colour for present / zero-count rows, in 0-255.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.IMAGE, NodeTag.RGB})

    INPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Input frame [B, H, W, 3] in [0, 1].",
        ),
        "label_rgb": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Optional colourised label map [B, H, W, 3] in [0, 1] for instance counts.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "frame": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Frame with the legend strip appended [B, H + legend_h, W, 3] in [0, 1].",
        ),
    }

    def __init__(
        self,
        entries: list[tuple[str, tuple[int, int, int]]],
        n_columns: int = 6,
        tile_height_px: int = 22,
        swatch_width_px: int = 28,
        text_padding_px: int = 6,
        background_color: tuple[float, float, float] = (0.08, 0.08, 0.08),
        text_color: tuple[int, int, int] = (240, 240, 240),
        dim_text_color: tuple[int, int, int] = (110, 110, 110),
        font_size: int = 12,
        **kwargs: Any,
    ) -> None:
        if not entries:
            raise ValueError("entries must be a non-empty list of (label, (r, g, b))")
        self._entries = [(str(name), tuple(int(c) for c in rgb)) for name, rgb in entries]
        self.n_columns = int(n_columns)
        self.tile_height_px = int(tile_height_px)
        self.swatch_width_px = int(swatch_width_px)
        self.text_padding_px = int(text_padding_px)
        self.background_color = tuple(float(c) for c in background_color)
        self.text_color = tuple(int(c) for c in text_color)
        self.dim_text_color = tuple(int(c) for c in dim_text_color)
        self.font_size = int(font_size)
        super().__init__(
            entries=[[name, list(rgb)] for name, rgb in self._entries],
            n_columns=self.n_columns,
            tile_height_px=self.tile_height_px,
            swatch_width_px=self.swatch_width_px,
            text_padding_px=self.text_padding_px,
            background_color=list(self.background_color),
            text_color=list(self.text_color),
            dim_text_color=list(self.dim_text_color),
            font_size=self.font_size,
            **kwargs,
        )
        try:
            self._font = ImageFont.truetype("arial.ttf", self.font_size)
        except OSError:
            self._font = ImageFont.load_default()
        n_rows = (len(self._entries) + self.n_columns - 1) // self.n_columns
        self._legend_h = n_rows * self.tile_height_px + 2 * self.text_padding_px

    @staticmethod
    def _count_instances(label_rgb_u8: np.ndarray, color: tuple[int, int, int]) -> int:
        """Connected-component count for one class colour on an [H, W, 3] uint8 image."""
        r, g, b = color
        mask = (
            (label_rgb_u8[..., 0] == r) & (label_rgb_u8[..., 1] == g) & (label_rgb_u8[..., 2] == b)
        )
        if not mask.any():
            return 0
        labels = label_connected_components(torch.from_numpy(mask), connectivity=8)
        return int(labels.max().item())

    def _render_strip(self, width: int, counts: list[int] | None) -> torch.Tensor:
        """Render the legend strip [legend_h, width, 3] in [0, 1] for the given per-class counts."""
        bg = tuple(int(c * 255) for c in self.background_color)
        img = Image.new("RGB", (width, self._legend_h), bg)
        draw = ImageDraw.Draw(img)
        col_w = width // self.n_columns
        for i, (name, rgb) in enumerate(self._entries):
            row = i // self.n_columns
            col = i % self.n_columns
            x0 = col * col_w + self.text_padding_px
            y0 = self.text_padding_px + row * self.tile_height_px
            swatch_x1 = x0 + self.swatch_width_px
            swatch_y1 = y0 + self.tile_height_px - 4
            n = counts[i] if counts is not None else None
            present = n is None or n > 0
            swatch_fill = rgb if present else tuple(c // 3 for c in rgb)
            draw.rectangle(
                [x0, y0, swatch_x1, swatch_y1], fill=swatch_fill, outline=(255, 255, 255)
            )
            label_text = f"{name} ({n})" if n is not None else name
            text_color = self.text_color if present else self.dim_text_color
            text_x = swatch_x1 + self.text_padding_px
            text_y = y0 + max(0, (self.tile_height_px - self.font_size) // 2 - 2)
            draw.text((text_x, text_y), label_text, fill=text_color, font=self._font)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr)

    @torch.no_grad()
    def forward(
        self, frame: torch.Tensor, label_rgb: torch.Tensor | None = None, **_: Any
    ) -> dict[str, torch.Tensor]:
        """Append the legend strip below ``frame``; optionally count instances per class."""
        b, _, w, _ = frame.shape
        counts: list[int] | None = None
        if label_rgb is not None:
            lab = label_rgb[0].detach().cpu()
            if torch.is_floating_point(lab):
                arr_u8 = (lab.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).numpy()
            else:
                arr_u8 = lab.to(torch.uint8).numpy()
            counts = [self._count_instances(arr_u8, color) for _, color in self._entries]
        strip = self._render_strip(w, counts).to(device=frame.device, dtype=frame.dtype)
        strip = strip.unsqueeze(0).expand(b, -1, -1, -1)
        return {"frame": torch.cat([frame, strip], dim=1).clamp(0.0, 1.0)}
