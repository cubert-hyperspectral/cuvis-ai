"""Pure-torch drawing helpers for uint8 HWC images."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

# Keep in sync with vis_helpers.OBJECT_PALETTE for compatibility.
_DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (230, 25, 75),
    (60, 180, 75),
    (255, 225, 25),
    (0, 130, 200),
    (245, 130, 48),
    (145, 30, 180),
    (70, 240, 240),
    (240, 50, 230),
    (210, 245, 60),
    (250, 190, 212),
    (0, 128, 128),
    (220, 190, 255),
]

_FONT_5x7: dict[str, list[str]] = {
    "0": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["01110", "10001", "00001", "00110", "00001", "10001", "01110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
    "6": ["01110", "10000", "11110", "10001", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00010", "11100"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "00110", "00110"],
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
}


def _as_color_tensor(
    color: torch.Tensor | tuple[int, int, int], device: torch.device
) -> torch.Tensor:
    """Normalize an RGB color input to a uint8 tensor on the target device."""
    color_t = torch.as_tensor(color, dtype=torch.uint8, device=device)
    if color_t.shape != (3,):
        raise ValueError(f"Expected color shape (3,), got {tuple(color_t.shape)}")
    return color_t


def _glyph(text: str, device: torch.device) -> torch.Tensor:
    """Render text into a 5x7 bitmap tensor using the built-in glyph table."""
    if not text:
        return torch.zeros((7, 0), dtype=torch.uint8, device=device)

    parts: list[torch.Tensor] = []
    for idx, char in enumerate(text):
        rows = _FONT_5x7.get(char, _FONT_5x7[" "])
        glyph = torch.tensor(
            [[1 if bit == "1" else 0 for bit in row] for row in rows],
            dtype=torch.uint8,
            device=device,
        )
        parts.append(glyph)
        if idx < len(text) - 1:
            parts.append(torch.zeros((7, 1), dtype=torch.uint8, device=device))
    return torch.cat(parts, dim=1)


@torch.no_grad()
def mask_edge(mask: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    """Compute edge pixels from a binary mask."""
    if mask.ndim != 2:
        raise ValueError(f"Expected mask shape (H, W), got {tuple(mask.shape)}")

    mask_bool = mask.to(torch.bool)
    if not torch.any(mask_bool):
        return torch.zeros_like(mask_bool)

    t = max(1, int(thickness))
    kernel = 2 * t + 1
    inv = (~mask_bool).to(torch.float32).unsqueeze(0).unsqueeze(0)
    dilated_inv = F.max_pool2d(inv, kernel_size=kernel, stride=1, padding=t)
    eroded = dilated_inv.eq(0).squeeze(0).squeeze(0)
    return mask_bool & ~eroded


@torch.no_grad()
def draw_box(
    img: torch.Tensor,
    box_xyxy: tuple[int, int, int, int],
    color: torch.Tensor | tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw rectangle edges in-place on a uint8 HWC image."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return

    x1, y1, x2, y2 = [int(v) for v in box_xyxy]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 < x1 or y2 < y1:
        return

    t = max(1, int(thickness))
    color_t = _as_color_tensor(color, img.device)

    y_top_end = min(h, y1 + t)
    y_bottom_start = max(y1, y2 - t + 1)
    x_left_end = min(w, x1 + t)
    x_right_start = max(x1, x2 - t + 1)

    img[y1:y_top_end, x1 : x2 + 1, :] = color_t
    img[y_bottom_start : y2 + 1, x1 : x2 + 1, :] = color_t
    img[y1 : y2 + 1, x1:x_left_end, :] = color_t
    img[y1 : y2 + 1, x_right_start : x2 + 1, :] = color_t


@torch.no_grad()
def draw_text(
    img: torch.Tensor,
    x: int,
    y: int,
    text: str,
    color: torch.Tensor | tuple[int, int, int],
    scale: int = 2,
    bg: bool = True,
) -> None:
    """Draw bitmap text in-place on a uint8 HWC image."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    glyph = _glyph(text, img.device)
    s = max(1, int(scale))
    if s > 1:
        glyph = glyph.repeat_interleave(s, dim=0).repeat_interleave(s, dim=1)

    gh, gw = int(glyph.shape[0]), int(glyph.shape[1])
    if gh == 0 or gw == 0:
        return

    h, w = int(img.shape[0]), int(img.shape[1])
    x_i, y_i = int(x), int(y)
    color_t = _as_color_tensor(color, img.device)

    if bg:
        pad = max(1, s)
        rx0 = max(0, x_i - pad)
        ry0 = max(0, y_i - pad)
        rx1 = min(w, x_i + gw + pad)
        ry1 = min(h, y_i + gh + pad)
        if rx1 > rx0 and ry1 > ry0:
            region = img[ry0:ry1, rx0:rx1, :].to(torch.float32)
            img[ry0:ry1, rx0:rx1, :] = torch.round(region * 0.25).to(torch.uint8)

    x0 = max(0, x_i)
    y0 = max(0, y_i)
    x1 = min(w, x_i + gw)
    y1 = min(h, y_i + gh)
    if x1 <= x0 or y1 <= y0:
        return

    gx0 = x0 - x_i
    gy0 = y0 - y_i
    gx1 = gx0 + (x1 - x0)
    gy1 = gy0 + (y1 - y0)

    mask_crop = glyph[gy0:gy1, gx0:gx1].to(torch.bool)
    if not torch.any(mask_crop):
        return

    region = img[y0:y1, x0:x1, :]
    region[mask_crop] = color_t


def _fill_triangle(
    img: torch.Tensor,
    vertices: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    color: torch.Tensor | tuple[int, int, int],
) -> None:
    """Rasterize a filled triangle in-place on a uint8 HWC image."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return

    xs = [float(v[0]) for v in vertices]
    ys = [float(v[1]) for v in vertices]
    x0 = max(0, min(w - 1, int(math.floor(min(xs)))))
    x1 = max(0, min(w - 1, int(math.ceil(max(xs)))))
    y0 = max(0, min(h - 1, int(math.floor(min(ys)))))
    y1 = max(0, min(h - 1, int(math.ceil(max(ys)))))
    if x1 < x0 or y1 < y0:
        return

    grid_y, grid_x = torch.meshgrid(
        torch.arange(y0, y1 + 1, device=img.device, dtype=torch.float32),
        torch.arange(x0, x1 + 1, device=img.device, dtype=torch.float32),
        indexing="ij",
    )
    px = grid_x + 0.5
    py = grid_y + 0.5

    (ax, ay), (bx, by), (cx, cy) = vertices
    denom = (cx - ax) * (by - ay) - (bx - ax) * (cy - ay)
    if abs(denom) < 1e-6:
        return

    bary_u = ((px - ax) * (by - ay) - (py - ay) * (bx - ax)) / denom
    bary_v = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / -denom
    inside = (bary_u >= 0.0) & (bary_v >= 0.0) & ((bary_u + bary_v) <= 1.0)

    if not torch.any(inside):
        return

    region = img[y0 : y1 + 1, x0 : x1 + 1, :]
    region[inside] = _as_color_tensor(color, img.device)


@torch.no_grad()
def draw_downward_triangle(
    img: torch.Tensor,
    tip_x: int,
    tip_y: int,
    width: int,
    height: int,
    color: torch.Tensor | tuple[int, int, int],
    *,
    outline_color: torch.Tensor | tuple[int, int, int] | None = None,
    outline_thickness: int = 1,
) -> None:
    """Draw a filled downward-pointing isosceles triangle in-place."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    if h == 0 or w == 0:
        return

    tri_w = max(1, int(width))
    tri_h = max(1, int(height))
    tip_x_i = int(tip_x)
    tip_y_i = int(tip_y)
    half_w = tri_w / 2.0

    outer = (
        (tip_x_i, tip_y_i),
        (tip_x_i - half_w, tip_y_i - tri_h),
        (tip_x_i + half_w, tip_y_i - tri_h),
    )

    if outline_color is not None:
        _fill_triangle(img, outer, outline_color)

        t = max(0, int(outline_thickness))
        inner_w = tri_w - 2 * t
        inner_h = tri_h - 2 * t
        if inner_w > 0 and inner_h > 0:
            inner_half_w = inner_w / 2.0
            inner_tip_y = tip_y_i - t
            inner = (
                (tip_x_i, inner_tip_y),
                (tip_x_i - inner_half_w, inner_tip_y - inner_h),
                (tip_x_i + inner_half_w, inner_tip_y - inner_h),
            )
            _fill_triangle(img, inner, color)
        return

    _fill_triangle(img, outer, color)


@torch.no_grad()
def id_to_color(ids: torch.Tensor) -> torch.Tensor:
    """Map integer IDs to deterministic uint8 RGB colors (hybrid policy)."""
    if ids.ndim != 1:
        raise ValueError(f"Expected ids shape (N,), got {tuple(ids.shape)}")

    ids_i64 = ids.to(torch.int64)
    n = int(ids_i64.numel())
    out = torch.empty((n, 3), dtype=torch.uint8, device=ids.device)
    if n == 0:
        return out

    palette = torch.tensor(_DEFAULT_PALETTE, dtype=torch.uint8, device=ids.device)
    palette_len = int(palette.shape[0])

    in_palette = (ids_i64 >= 0) & (ids_i64 < palette_len)
    if torch.any(in_palette):
        out[in_palette] = palette[ids_i64[in_palette]]

    out_of_palette = ~in_palette
    if torch.any(out_of_palette):
        hashed = ids_i64[out_of_palette] * 1103515245 + 12345
        raw = (
            torch.stack(
                [((hashed >> 16) & 0xFF), ((hashed >> 8) & 0xFF), (hashed & 0xFF)],
                dim=1,
            ).to(torch.float32)
            / 255.0
        )
        biased = (0.35 + 0.65 * raw).clamp(0.0, 1.0)
        out[out_of_palette] = torch.round(biased * 255.0).to(torch.uint8)

    return out


@torch.no_grad()
def overlay_instances(
    image: torch.Tensor,
    masks: list[tuple[int, torch.Tensor]],
    *,
    alpha: float = 0.4,
    draw_edges: bool = True,
    draw_ids: bool = True,
    edge_thickness: int = 2,
    text_scale: int = 2,
) -> torch.Tensor:
    """Blend instance masks, draw optional edges, and draw optional object IDs."""
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(image.shape)} {image.dtype}"
        )
    if not (0.0 <= float(alpha) <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")

    out = image.clone()
    if not masks:
        return out

    edge_t = max(1, int(edge_thickness))
    txt_scale = max(1, int(text_scale))
    white = torch.tensor([255, 255, 255], dtype=torch.uint8, device=out.device)

    for obj_id, mask in masks:
        if mask.ndim != 2:
            raise ValueError(f"Expected mask shape (H, W), got {tuple(mask.shape)}")
        if tuple(mask.shape) != tuple(out.shape[:2]):
            raise ValueError(
                f"Mask shape {tuple(mask.shape)} does not match image shape {tuple(out.shape[:2])}"
            )
        if mask.device != out.device:
            raise ValueError(f"Mask device {mask.device} must match image device {out.device}")

        fg = mask.to(torch.bool)
        if not torch.any(fg):
            continue

        color = id_to_color(torch.tensor([int(obj_id)], device=out.device, dtype=torch.int64))[0]

        if alpha > 0.0:
            current = out[fg].to(torch.float32)
            tint = color.to(torch.float32)
            out[fg] = (
                torch.round((1.0 - alpha) * current + alpha * tint).clamp(0, 255).to(torch.uint8)
            )

        if draw_edges:
            edges = mask_edge(fg, thickness=edge_t)
            out[edges] = color

        if draw_ids:
            ys, xs = torch.where(fg)
            if ys.numel() > 0:
                x_min = int(xs.min().item())
                y_min = int(ys.min().item())
                label = str(int(obj_id))
                label_mask = _glyph(label, out.device)
                if txt_scale > 1:
                    label_mask = label_mask.repeat_interleave(txt_scale, dim=0).repeat_interleave(
                        txt_scale, dim=1
                    )
                label_h = int(label_mask.shape[0])
                pad = max(1, txt_scale)
                text_x = max(0, x_min - 1)
                text_y = max(0, y_min - label_h - 2 * pad)
                draw_text(out, text_x, text_y, label, white, scale=txt_scale, bg=True)

    return out


@torch.no_grad()
def draw_sparkline(
    img: torch.Tensor,
    x1: int,
    y1: int,
    width: int,
    height: int,
    values: torch.Tensor,
    color: torch.Tensor | tuple[int, int, int],
    bg_alpha: float = 0.5,
) -> None:
    """Render a filled area sparkline chart on a uint8 HWC image (in-place).

    Draws a mini filled area chart of ``values`` within the rectangle
    ``(x1, y1, x1+width, y1+height)``. The values are min-max normalized
    internally; the chart is filled from the curve down to the bottom edge.

    Parameters
    ----------
    img : Tensor
        ``(H, W, 3)`` uint8 image, modified in-place.
    x1, y1 : int
        Top-left corner of the sparkline region.
    width, height : int
        Dimensions of the sparkline region in pixels.
    values : Tensor
        ``(C,)`` float — the spectral signature or any 1-D signal.
    color : Tensor or tuple
        RGB color for the filled area.
    bg_alpha : float
        Background darkening factor (0=black, 1=no darkening).
    """
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != torch.uint8:
        raise ValueError(
            f"Expected image shape (H, W, 3) uint8, got {tuple(img.shape)} {img.dtype}"
        )

    h, w = int(img.shape[0]), int(img.shape[1])
    num_vals = int(values.numel())
    if num_vals < 2 or width < 2 or height < 2:
        return

    # Clamp region to image bounds
    rx1 = max(0, int(x1))
    ry1 = max(0, int(y1))
    rx2 = min(w, int(x1) + int(width))
    ry2 = min(h, int(y1) + int(height))
    if rx2 <= rx1 or ry2 <= ry1:
        return

    rw = rx2 - rx1
    rh = ry2 - ry1

    # Darken background region for readability
    region = img[ry1:ry2, rx1:rx2, :]
    darkened = (region.to(torch.float32) * bg_alpha).clamp(0, 255).to(torch.uint8)
    img[ry1:ry2, rx1:rx2, :] = darkened

    # Min-max normalize values to [0, 1]
    vals = values.to(torch.float32).detach()
    v_min = vals.min()
    v_max = vals.max()
    v_range = v_max - v_min
    if v_range < 1e-12:
        # Flat signal — draw a horizontal line at mid-height
        norm_vals = torch.full_like(vals, 0.5)
    else:
        norm_vals = (vals - v_min) / v_range

    # Map each column to a band index and compute y-position
    color_t = _as_color_tensor(color, img.device)
    for col in range(rw):
        # Map column to band index (linear interpolation)
        band_idx_f = col * (num_vals - 1) / max(1, rw - 1)
        band_lo = int(band_idx_f)
        band_hi = min(band_lo + 1, num_vals - 1)
        frac = band_idx_f - band_lo
        val = float(norm_vals[band_lo]) * (1.0 - frac) + float(norm_vals[band_hi]) * frac

        # y=0 at top of region, y=rh-1 at bottom
        # val=1 → top of region, val=0 → bottom
        curve_y = int(round((1.0 - val) * (rh - 1)))
        curve_y = max(0, min(curve_y, rh - 1))

        # Fill from curve_y down to bottom
        abs_x = rx1 + col
        abs_y_start = ry1 + curve_y
        if abs_y_start < ry2:
            img[abs_y_start:ry2, abs_x, :] = color_t


__all__ = [
    "mask_edge",
    "draw_box",
    "draw_text",
    "draw_downward_triangle",
    "draw_sparkline",
    "id_to_color",
    "overlay_instances",
]
