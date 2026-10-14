"""Tile an image batch into a T x T grid and stitch per-tile feature grids back together.

A ViT backbone at a fixed input resolution (e.g. 336 px, a 24 x 24 patch grid) sees a small
object inside one coarse patch, together with its context. Running the backbone on the T x T tiles
of a frame (the backbone node resizes each tile to its resolution) gives a T times finer feature
grid. ``ImageTiler`` stacks the tiles along the batch dimension, so any batched per-image node runs
on them unchanged; ``GridStitcher`` reassembles the per-tile grids in the same image-major,
row-major tile order::

    PercentileNormalizer -> ImageTiler(tiles=2) -> feature extractor -> GridStitcher(tiles=2)
        -> patch-level anomaly detector

Both nodes are pure reshapes: stateless, differentiable and device-agnostic. They cut equal,
non-overlapping tiles; overlapping tiles with blended seams are ``cuvis_ai_core.utils.tiling``'s
``sliding_window_inference``.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node


def _check_tiles(node: str, tiles: Any) -> int:
    t = int(tiles)
    if t < 1 or t != tiles:
        raise ValueError(f"{node}: tiles must be a positive int, got {tiles!r}.")
    return t


class ImageTiler(Node):
    """Split every image of a batch into a T x T grid of equal tiles stacked along the batch."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.IMAGE,
            NodeTag.PREPROCESSING,
            NodeTag.BATCHED,
            NodeTag.DIFFERENTIABLE,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Image batch [B, H, W, C] (any channel count); H and W must be divisible "
            "by `tiles`.",
        ),
    }
    OUTPUT_SPECS = {
        "tiles": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Tiles [B * T * T, H / T, W / T, C]; tile (i, j) of image b sits at index "
            "b * T * T + i * T + j (image-major, then row-major), the order GridStitcher expects.",
        ),
    }

    def __init__(self, tiles: int = 2, **kwargs: Any) -> None:
        """Create a tiler; ``tiles`` = T, the number of tiles per image side (T x T tiles)."""
        tiles = _check_tiles(type(self).__name__, tiles)
        super().__init__(tiles=tiles, **kwargs)
        self.tiles = tiles

    def forward(self, image: Tensor, **_: Any) -> dict[str, Tensor]:
        """Cut ``image`` [B, H, W, C] into [B * T * T, H / T, W / T, C] tiles."""
        b, h, w, c = image.shape
        t = self.tiles
        if h % t or w % t:
            raise ValueError(
                f"ImageTiler: a {h} x {w} frame does not split into {t} x {t} equal tiles; "
                "crop or pad it first."
            )
        x = image.reshape(b, t, h // t, t, w // t, c).permute(0, 1, 3, 2, 4, 5)
        return {"tiles": x.reshape(b * t * t, h // t, w // t, c)}


class GridStitcher(Node):
    """Reassemble per-tile grids produced in ImageTiler order into one grid per image."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset(
        {
            NodeTag.EMBEDDING,
            NodeTag.POSTPROCESSING,
            NodeTag.BATCHED,
            NodeTag.DIFFERENTIABLE,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "tiles": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Per-tile grids [B * T * T, g_h, g_w, D] in ImageTiler order (e.g. "
            "ViT patch features).",
        ),
    }
    OUTPUT_SPECS = {
        "grid": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Stitched grid [B, T * g_h, T * g_w, D]: tile (i, j) fills rows i * g_h.. "
            "and columns j * g_w..",
        ),
    }

    def __init__(self, tiles: int = 2, **kwargs: Any) -> None:
        """Create a stitcher; ``tiles`` = T, as on the ImageTiler that produced the tiles."""
        tiles = _check_tiles(type(self).__name__, tiles)
        super().__init__(tiles=tiles, **kwargs)
        self.tiles = tiles

    def forward(self, tiles: Tensor, **_: Any) -> dict[str, Tensor]:
        """Stitch [B * T * T, g_h, g_w, D] tile grids into [B, T * g_h, T * g_w, D]."""
        n, gh, gw, d = tiles.shape
        t = self.tiles
        if n % (t * t):
            raise ValueError(f"GridStitcher: {n} tiles is not a whole number of {t} x {t} images.")
        b = n // (t * t)
        x = tiles.reshape(b, t, t, gh, gw, d).permute(0, 1, 3, 2, 4, 5)
        return {"grid": x.reshape(b, t * gh, t * gw, d)}


__all__ = ["GridStitcher", "ImageTiler"]
