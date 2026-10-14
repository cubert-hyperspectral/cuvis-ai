"""Spectral Angle Mapper node."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node


class SpectralAngleMapper(Node):
    """Compute per-pixel spectral angle against one or more reference spectra."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.CLASSIFICATION, NodeTag.STATEFUL, NodeTag.NUMPY}
    )

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "spectral_signature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Reference spectra [N, 1, 1, C]",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Spectral angle scores [B, H, W, N] in radians",
        ),
        "best_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Best score per pixel [B, H, W, 1]",
        ),
        "identity_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="1-based best-matching identity [B, H, W]",
        ),
    }

    def __init__(self, num_channels: int, eps: float = 1e-12, **kwargs: Any) -> None:
        if int(num_channels) <= 0:
            raise ValueError(f"num_channels must be > 0, got {num_channels}")
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        super().__init__(num_channels=self.num_channels, eps=self.eps, **kwargs)

    @torch.no_grad()
    def forward(
        self,
        cube: torch.Tensor,
        spectral_signature: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run spectral-angle scoring for all references."""
        ref = spectral_signature.squeeze(1).squeeze(1)  # [N, C]
        channel_count = int(ref.shape[-1])
        ref_mean = ref.mean(dim=-1, keepdim=True)
        ref_norm = ref / (ref_mean + self.eps)

        pixel_mean = cube.mean(dim=-1, keepdim=True)
        cube_norm = cube / (pixel_mean + self.eps)

        ref_expanded = ref_norm.view(1, 1, 1, ref_norm.shape[0], channel_count)
        cube_expanded = cube_norm.unsqueeze(-2)

        dot = (cube_expanded * ref_expanded).sum(dim=-1)
        norms = cube_norm.norm(dim=-1, keepdim=True) * ref_norm.norm(dim=-1).view(1, 1, 1, -1)
        cos_sim = dot / (norms + self.eps)
        scores = torch.acos(cos_sim.clamp(-1.0, 1.0))

        best_scores = scores.amin(dim=-1, keepdim=True)
        identity_mask = scores.argmin(dim=-1).to(torch.int32) + 1

        return {
            "scores": scores,
            "best_scores": best_scores,
            "identity_mask": identity_mask,
        }


_THRESHOLDS = ("fixed", "otsu")


def _otsu_deg(angle: np.ndarray) -> float:
    """OpenCV's Otsu level of an angle map quantised to 0.1 deg (0-25.5 deg), in degrees."""
    q = np.clip(angle * 10.0, 0, 255).astype(np.uint8)
    level, _ = cv2.threshold(q, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(level) / 10.0


def _fill_holes(m: np.ndarray) -> np.ndarray:
    """m [h, w] (bool) with its holes filled: background not 4-connected to the border is set."""
    h, w = m.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1:-1, 1:-1] = m
    flood = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood, (0, 0), 1, flags=4)  # the outside background becomes 1
    return m | (pad[1:-1, 1:-1] == 0)


class SpectralObjectMask(Node):
    """Mark the pixels whose spectral angle to the frame's median spectrum exceeds a threshold."""

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.MASK, NodeTag.SEGMENTATION, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] (e.g. reflectance) whose background "
            "covers most of the frame.",
        ),
    }
    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Object mask [B, H, W, 1]: spectral angle to the frame's median spectrum "
            "above the threshold (on the stride grid, nearest neighbour to H, W), filled and grown "
            "when `fill` / `dilate_px` are set.",
        ),
        "angle": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="The angle in degrees on the stride grid [B, ceil(H / s), ceil(W / s), 1].",
        ),
    }

    def __init__(
        self,
        min_angle_deg: float = 6.0,
        stride: int = 4,
        median_stride: int = 8,
        threshold: str = "fixed",
        otsu_floor_deg: float = 3.0,
        otsu_ceiling_deg: float = 12.0,
        fill: bool = False,
        dilate_px: int = 0,
        **kwargs: Any,
    ) -> None:
        """Create a spectral objectness mask.

        Parameters
        ----------
        min_angle_deg : smallest angle to the median spectrum that counts as an object, in degrees
            (default 6.0).
        stride : grid step in pixels on which the angle is computed (default 4); 1 computes it on every pixel.
        median_stride : grid step of the pixels whose per-band median is the background spectrum
            (default 8: 16 875 pixels of a 1000 x 1080 frame, plenty for a median).
        threshold : ``"fixed"`` (``min_angle_deg``, default) or ``"otsu"``: each frame's Otsu level
            of the angle map, clipped to ``[otsu_floor_deg, otsu_ceiling_deg]`` (defaults 3 / 12).
        fill : close 1-cell gaps (3 x 3) and fill the holes of the objects on the stride grid
            (default False).
        dilate_px : grow the full-size object mask by this many pixels (square window, default 0).
        """
        if (
            isinstance(min_angle_deg, bool)
            or not isinstance(min_angle_deg, (int, float))
            or not 0.0 <= float(min_angle_deg) <= 180.0
        ):
            raise ValueError(
                f"SpectralObjectMask: min_angle_deg must be in [0, 180], got {min_angle_deg!r}."
            )
        for name, val in (("stride", stride), ("median_stride", median_stride)):
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ValueError(
                    f"SpectralObjectMask: {name} must be an integer >= 1, got {val!r}."
                )
        if threshold not in _THRESHOLDS:
            raise ValueError(
                f"SpectralObjectMask: threshold must be one of {_THRESHOLDS}, got {threshold!r}."
            )
        for name, val in (
            ("otsu_floor_deg", otsu_floor_deg),
            ("otsu_ceiling_deg", otsu_ceiling_deg),
        ):
            if (
                isinstance(val, bool)
                or not isinstance(val, (int, float))
                or not 0.0 <= val <= 180.0
            ):
                raise ValueError(f"SpectralObjectMask: {name} must be in [0, 180], got {val!r}.")
        if otsu_floor_deg > otsu_ceiling_deg:
            raise ValueError("SpectralObjectMask: otsu_floor_deg must not exceed otsu_ceiling_deg.")
        if not isinstance(fill, bool):
            raise ValueError(f"SpectralObjectMask: fill must be a bool, got {fill!r}.")
        if isinstance(dilate_px, bool) or not isinstance(dilate_px, int) or dilate_px < 0:
            raise ValueError(
                f"SpectralObjectMask: dilate_px must be an integer >= 0, got {dilate_px!r}."
            )
        self.min_angle_deg = float(min_angle_deg)
        self.stride = int(stride)
        self.median_stride = int(median_stride)
        self.threshold = threshold
        self.otsu_floor_deg = float(otsu_floor_deg)
        self.otsu_ceiling_deg = float(otsu_ceiling_deg)
        self.fill = fill
        self.dilate_px = int(dilate_px)
        super().__init__(
            min_angle_deg=self.min_angle_deg,
            stride=self.stride,
            median_stride=self.median_stride,
            threshold=self.threshold,
            otsu_floor_deg=self.otsu_floor_deg,
            otsu_ceiling_deg=self.otsu_ceiling_deg,
            fill=self.fill,
            dilate_px=self.dilate_px,
            **kwargs,
        )

    def _objects(self, angle: Tensor) -> Tensor:
        """[B, 1, h, w] float (0 / 1) objects on the stride grid from the angle map [B, 1, h, w]."""
        if self.threshold == "fixed" and not self.fill:
            return (angle > self.min_angle_deg).to(torch.float32)
        a = angle[:, 0].detach().cpu().numpy()  # small grid: threshold, closing, holes on the CPU
        out = np.zeros(a.shape, np.float32)
        for i in range(a.shape[0]):
            thr = self.min_angle_deg
            if self.threshold == "otsu":
                thr = min(max(_otsu_deg(a[i]), self.otsu_floor_deg), self.otsu_ceiling_deg)
            o = a[i] > thr
            if self.fill:
                k = np.ones((3, 3), np.uint8)
                o = cv2.morphologyEx(o.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
                o = _fill_holes(o)
            out[i] = o
        return torch.from_numpy(out).to(device=angle.device)[:, None]

    def forward(self, cube: Tensor, **_: Any) -> dict[str, Tensor]:
        """Return the object mask at full size and the angle map on the stride grid."""
        b, h, w, c = cube.shape
        grid = cube[:, :: self.stride, :: self.stride, :].to(torch.float32)  # a strided view
        sample = cube[:, :: self.median_stride, :: self.median_stride, :].to(torch.float32)
        # [B, C]: the background's spectrum (the median along a contiguous axis: 4x faster on a GPU)
        ref = sample.permute(0, 3, 1, 2).reshape(b, c, -1).median(dim=2).values
        # per frame a matrix-vector product reads the strided grid once, without copying it
        dot = torch.stack([grid[i] @ ref[i] for i in range(b)])
        norm = torch.linalg.vector_norm(grid, dim=-1) * ref.norm(dim=-1)[:, None, None]
        cos = (dot / (norm + 1e-6)).clamp(-1.0, 1.0)
        angle = torch.rad2deg(torch.arccos(cos))[:, None]  # [B, 1, h / s, w / s]
        objects = self._objects(angle)
        r = self.dilate_px
        if r and r % self.stride == 0 and h % self.stride == 0 and w % self.stride == 0:
            # cell-aligned objects grown by r = k x stride pixels are the grid grown by k cells
            k = r // self.stride
            objects = F.max_pool2d(objects, kernel_size=2 * k + 1, stride=1, padding=k)
            r = 0
        full = F.interpolate(objects, size=(h, w), mode="nearest")
        if r:
            full = F.max_pool2d(full, kernel_size=2 * r + 1, stride=1, padding=r)
        return {"decisions": full.permute(0, 2, 3, 1) > 0.5, "angle": angle.permute(0, 2, 3, 1)}


__all__ = ["SpectralAngleMapper", "SpectralObjectMask"]
