"""Spatial resampling: score a coarse grid of a cube, then bring the map back to the cube's size.

A per-pixel spectral model (e.g. a Gaussian mixture over the bands) costs the same per pixel, so on
a 1000 x 1080 cube it pays to score every ``s``-th pixel in both axes and interpolate the map back:
``GridSubsample`` takes the grid (``cube[:, ::s, ::s, :]``) and ``ScoreUpsample`` resizes the
grid's score map to the height and width of a reference tensor (bilinear by default).
``ScoreMapSmoothing`` convolves a score map with a Gaussian, the usual post-processing of
patch-level anomaly detectors: isolated one-patch peaks drop, regions that several patches agree
on keep their level. ``LocalMaximaSuppression`` keeps the local maxima of a score map and zeroes
everything else, a peak-picking step for a vote map, for example. All four are stateless and
torch-native. The first three are differentiable; ``LocalMaximaSuppression`` passes the gradient
to the values it keeps and none to the choice of which values those are.
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
    _tags = frozenset({NodeTag.ONNX, NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

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
    _tags = frozenset({NodeTag.ONNX, NodeTag.ANOMALY, NodeTag.POSTPROCESSING, NodeTag.TORCH})

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
    _tags = frozenset({NodeTag.ONNX, NodeTag.ANOMALY, NodeTag.POSTPROCESSING, NodeTag.TORCH})

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


def _window_sizes(window: int | list[int]) -> tuple[int, int]:
    """Validate and return ``(height, width)`` for the ``window`` constructor argument."""
    if isinstance(window, (list, tuple)):
        if len(window) != 2:
            raise ValueError(
                f"LocalMaximaSuppression: a list window must have 2 entries [height, width], "
                f"got {window!r}."
            )
        sizes = (window[0], window[1])
    else:
        sizes = (window, window)
    for size in sizes:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1 or size % 2 == 0:
            raise ValueError(
                f"LocalMaximaSuppression: window sizes must be odd positive integers, "
                f"got {window!r}."
            )
    return sizes[0], sizes[1]


def _block_max(x: Tensor, top: int, bottom: int, left: int, right: int) -> Tensor:
    """Max over a rectangle placed relative to every pixel of ``x`` ``[N, 1, H, W]``.

    The rectangle spans the row offsets ``-top..bottom`` and the column offsets
    ``-left..right``. A negative extent moves that edge to the other side of the pixel:
    ``bottom=-1`` ends the rectangle one row above it, ``right=-1`` one column to its left.
    Cells past the image border count as ``-inf``, so they never win. An empty rectangle
    (for example ``top=0, bottom=-1``) gives ``-inf`` for every pixel.
    """
    kh, kw = top + bottom + 1, left + right + 1
    if kh < 1 or kw < 1:
        return torch.full_like(x, float("-inf"))
    padded = F.pad(x, (left, right, top, bottom), value=float("-inf"))
    return F.max_pool2d(padded, kernel_size=(kh, kw), stride=1)


class LocalMaximaSuppression(Node):
    """Keep the local maxima of a score map ``[B, H, W, C]`` and zero everything else.

    A value stays when it equals the maximum of the ``window`` centred on it; every other
    value becomes zero (including a suppressed negative value: zero is only the fill, not a
    floor on what a surviving value can be). The window is treated as filled with minus
    infinity past the image border, so the border neither creates nor hides a maximum.

    With ``keep_plateaus=False``, of several equal maxima inside one window only the first
    in row-major order (top row first, then left to right) stays: a value also needs to be
    strictly greater than the maximum of the window cells that come before it in that order.
    Those earlier cells form two rectangles relative to the pixel: every row above it across
    the full window width, and the cells to its left in its own row; no pool over the later
    cells is needed, since a later cell's own earlier-cells test already covers the tie
    against this one. With ``keep_plateaus=True`` every tied maximum stays.

    This tie rule is local, not global: of several equal maxima, one survives only if no
    equal value comes before it inside its own window. A plateau that fits inside one window
    radius collapses to its single first point; a long, bent or branching plateau can leave
    several surviving points, because a later point's window may not reach back far enough to
    see an earlier tied value. The node makes no stronger guarantee than that.

    This differs from :class:`cuvis_ai.node.mask_ops.ScoreMapSuppression`, which lowers
    scores inside a given mask; this node has no mask input and decides purely from the
    score map's own local structure.

    The suppression decision is a comparison of pooled values: a kept value passes its
    gradient through unchanged, a suppressed one gets none, and the choice between the two
    has no gradient. The node is a peak-picking postprocessing step, not a layer to train
    through.

    Parameters
    ----------
    window : int or list[int], optional
        Window size (default ``3``): either one odd positive integer for a square window,
        or ``[height, width]`` with both entries odd and positive.
    keep_plateaus : bool, optional
        Keep every tied maximum of a plateau instead of only the first in row-major order
        (default ``False``).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.ONNX, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Score map [B, H, W, C], e.g. a vote map.",
        ),
    }
    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Same shape: the local maxima of `scores` kept at their value, "
            "everything else set to 0.",
        ),
    }

    def __init__(
        self, window: int | list[int] = 3, keep_plateaus: bool = False, **kwargs: Any
    ) -> None:
        self.window_height, self.window_width = _window_sizes(window)
        self.window = window if isinstance(window, int) else list(window)
        if not isinstance(keep_plateaus, bool):
            raise ValueError(
                f"LocalMaximaSuppression: keep_plateaus must be a bool, got {keep_plateaus!r}."
            )
        self.keep_plateaus = keep_plateaus
        super().__init__(window=self.window, keep_plateaus=self.keep_plateaus, **kwargs)

    def forward(self, scores: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return ``scores`` with every non-local-maximum value set to 0; see the class
        docstring for the tie rule."""
        b, h, w, c = scores.shape
        rh, rw = self.window_height // 2, self.window_width // 2
        x = scores.permute(0, 3, 1, 2).reshape(b * c, 1, h, w)

        full_max = _block_max(x, rh, rh, rw, rw)
        is_max = x >= full_max

        if self.keep_plateaus:
            kept = torch.where(is_max, x, torch.zeros_like(x))
            return {"scores": kept.reshape(b, c, h, w).permute(0, 2, 3, 1)}

        # Earlier cells: all rows above (full window width) union the same row's left cells.
        above_max = _block_max(x, rh, -1, rw, rw)
        left_max = _block_max(x, 0, 0, rw, -1)
        earlier_max = torch.maximum(above_max, left_max)
        is_first = x > earlier_max

        kept = torch.where(is_max & is_first, x, torch.zeros_like(x))
        return {"scores": kept.reshape(b, c, h, w).permute(0, 2, 3, 1)}


__all__ = ["GridSubsample", "LocalMaximaSuppression", "ScoreMapSmoothing", "ScoreUpsample"]
