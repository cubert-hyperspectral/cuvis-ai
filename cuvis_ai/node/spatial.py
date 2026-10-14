"""Spatial resampling: score a coarse grid of a cube, then bring the map back to the cube's size.

A per-pixel spectral model (e.g. a Gaussian mixture over the bands) costs the same per pixel, so on
a 1000 x 1080 cube it pays to score every ``s``-th pixel in both axes and interpolate the map back:
``GridSubsample`` takes the grid (``cube[:, ::s, ::s, :]``) and ``ScoreUpsample`` resizes the
grid's score map to the height and width of a reference tensor (bilinear by default).
``ScoreMapSmoothing`` convolves a score map with a Gaussian, the usual post-processing of
patch-level anomaly detectors: isolated one-patch peaks drop, regions that several patches agree
on keep their level. All three are stateless, torch-native and differentiable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node.metric_utils import subsample_hw
from cuvis_ai_core.node.node import Node

_UPSAMPLE_MODES = ("bilinear", "bicubic", "nearest")


class GridSubsample(Node):
    """Keep every ``stride``-th pixel of a cube in both spatial axes: ``cube[:, ::s, ::s, :]``."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Cube [B, H, W, C] (a hyperspectral cube or any per-pixel feature map).",
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="The grid [B, ceil(H / stride), ceil(W / stride), C], starting at pixel 0.",
        ),
    }

    def __init__(self, stride: int = 4, **kwargs: Any) -> None:
        """Create a subsampler.

        Parameters
        ----------
        stride : keep every ``stride``-th row and column (>= 1; 1 passes the cube through).
        """
        if isinstance(stride, bool) or int(stride) != stride or int(stride) < 1:
            raise ValueError(f"GridSubsample: stride must be an integer >= 1, got {stride!r}.")
        self.stride = int(stride)
        super().__init__(stride=self.stride, **kwargs)

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the stride grid of the cube."""
        return {"cube": subsample_hw(cube, self.stride)}


class ScoreUpsample(Node):
    """Resize a score map [B, h, w, C] to the height and width of a reference tensor."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Score map [B, h, w, C] of a coarse grid (e.g. a GridSubsample branch).",
        ),
        "reference": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Any tensor [B, H, W, *] of the target size; only its height and width are "
            "read (e.g. the full-resolution cube the grid was taken from).",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="The map resized to [B, H, W, C] (align_corners=False).",
        ),
    }

    def __init__(self, mode: str = "bilinear", **kwargs: Any) -> None:
        """Create a resizer.

        Parameters
        ----------
        mode : ``"bilinear"`` (default), ``"bicubic"`` or ``"nearest"`` interpolation.
        """
        if mode not in _UPSAMPLE_MODES:
            raise ValueError(f"ScoreUpsample: mode must be one of {_UPSAMPLE_MODES}, got {mode!r}.")
        self.mode = mode
        super().__init__(mode=self.mode, **kwargs)

    def forward(self, scores: Tensor, reference: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the map interpolated to the reference's height and width."""
        size = (int(reference.shape[1]), int(reference.shape[2]))
        align = None if self.mode == "nearest" else False
        out = F.interpolate(
            scores.permute(0, 3, 1, 2), size=size, mode=self.mode, align_corners=align
        )
        return {"scores": out.permute(0, 2, 3, 1)}


class ScoreMapSmoothing(Node):
    """Smooth a score map [B, H, W, C] with a Gaussian of ``sigma_px`` pixels."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Score map [B, H, W, C], e.g. a fused anomaly map.",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Same shape: each channel convolved with a normalised Gaussian of "
            "`sigma_px` (radius round(4 sigma), mirrored border).",
        ),
    }

    def __init__(self, sigma_px: float = 8.0, **kwargs: Any) -> None:
        """Create a Gaussian smoothing of score maps.

        Parameters
        ----------
        sigma_px : the Gaussian's standard deviation in pixels (default 8.0); ``0`` passes the map
            through. The kernel spans round(4 sigma) pixels on each side and is applied separably;
            the border is mirrored (OpenCV's BORDER_REFLECT_101), or repeated when the map is
            narrower than the kernel.
        """
        if (
            isinstance(sigma_px, bool)
            or not isinstance(sigma_px, (int, float))
            or not 0.0 <= float(sigma_px) < float("inf")
        ):
            raise ValueError(
                f"ScoreMapSmoothing: sigma_px must be a finite number >= 0, got {sigma_px!r}."
            )
        self.sigma_px = float(sigma_px)
        super().__init__(sigma_px=self.sigma_px, **kwargs)
        self.radius = int(round(4.0 * self.sigma_px))
        x = torch.arange(-self.radius, self.radius + 1, dtype=torch.float32)
        k = torch.exp(-0.5 * (x / max(self.sigma_px, 1e-12)) ** 2)
        self.register_buffer("_kernel", k / k.sum(), persistent=False)  # not saved in the .pt

    def forward(self, scores: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the Gaussian-smoothed map."""
        if self.radius == 0:
            return {"scores": scores.clone()}
        b, h, w, c = scores.shape
        x = scores.permute(0, 3, 1, 2).reshape(b * c, 1, h, w)
        k = self._kernel.to(dtype=x.dtype)
        r = self.radius
        mode_w = "reflect" if r < w else "replicate"
        mode_h = "reflect" if r < h else "replicate"
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode_w), k.view(1, 1, 1, -1))
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode=mode_h), k.view(1, 1, -1, 1))
        return {"scores": x.reshape(b, c, h, w).permute(0, 2, 3, 1)}


__all__ = ["GridSubsample", "ScoreMapSmoothing", "ScoreUpsample"]
