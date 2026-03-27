"""Spectral signature extraction nodes for hyperspectral cubes."""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec


class BBoxSpectralExtractor(Node):
    """Extract per-bbox spectral signatures with trimmed median/mean and std.

    Given an HSI cube ``[B, H, W, C]`` and detection bboxes ``[B, N, 4]``
    (xyxy format), extracts a center-cropped spectral signature for each bbox.
    Outputs the per-band aggregated signature, per-band std, and a binary
    validity mask.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, 4),
            description="Detection bboxes [B, N, 4] in xyxy format.",
        ),
    }

    OUTPUT_SPECS = {
        "spectral_signatures": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Per-bbox spectral signatures [B, N, C].",
        ),
        "spectral_std": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Per-bbox spectral std [B, N, C].",
        ),
        "spectral_valid": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1),
            description="Validity mask [B, N]. 1=valid, 0=invalid.",
        ),
    }

    def __init__(
        self,
        center_crop_scale: float = 0.65,
        min_crop_pixels: int = 4,
        trim_fraction: float = 0.10,
        l2_normalize: bool = True,
        aggregation: str = "median",
        **kwargs: Any,
    ) -> None:
        if not (0.0 < center_crop_scale <= 1.0):
            raise ValueError("center_crop_scale must be in (0.0, 1.0].")
        if min_crop_pixels < 1:
            raise ValueError("min_crop_pixels must be >= 1.")
        if not (0.0 <= trim_fraction < 0.5):
            raise ValueError("trim_fraction must be in [0.0, 0.5).")
        if aggregation not in ("median", "mean"):
            raise ValueError("aggregation must be 'median' or 'mean'.")

        self.center_crop_scale = float(center_crop_scale)
        self.min_crop_pixels = int(min_crop_pixels)
        self.trim_fraction = float(trim_fraction)
        self.l2_normalize = bool(l2_normalize)
        self.aggregation = str(aggregation)

        super().__init__(
            center_crop_scale=center_crop_scale,
            min_crop_pixels=min_crop_pixels,
            trim_fraction=trim_fraction,
            l2_normalize=l2_normalize,
            aggregation=aggregation,
            **kwargs,
        )

    def _trimmed_stats(
        self, pixels: torch.Tensor, num_channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute trimmed median/mean and std over pixel rows.

        Parameters
        ----------
        pixels : Tensor
            Spectral vectors ``[P, C]`` gathered from the crop.
        num_channels : int
            Number of spectral channels ``C``.

        Returns
        -------
        (signature, std) : tuple of Tensors, each ``[C]``
        """
        zeros = torch.zeros(num_channels, dtype=pixels.dtype, device=pixels.device)
        if pixels.numel() == 0:
            return zeros, zeros

        # Filter non-finite rows
        valid_rows = torch.isfinite(pixels).all(dim=1)
        pixels = pixels[valid_rows]
        if pixels.shape[0] < self.min_crop_pixels:
            return zeros, zeros

        # Filter near-zero-norm pixels
        norms = torch.linalg.vector_norm(pixels, dim=1)
        pixels = pixels[norms >= 1e-8]
        if pixels.shape[0] < self.min_crop_pixels:
            return zeros, zeros

        # Sort per-band and trim
        sorted_vals, _ = torch.sort(pixels, dim=0)
        num_pixels = sorted_vals.shape[0]
        trim_k = int(math.floor(num_pixels * self.trim_fraction))
        if trim_k > 0 and (num_pixels - 2 * trim_k) > 0:
            sorted_vals = sorted_vals[trim_k : num_pixels - trim_k]

        # Aggregate
        if self.aggregation == "median":
            signature = sorted_vals.median(dim=0).values
        else:
            signature = sorted_vals.mean(dim=0)

        std = sorted_vals.std(dim=0, unbiased=False)
        return signature, std

    def _center_crop_bbox(
        self, x1: int, y1: int, x2: int, y2: int, img_h: int, img_w: int
    ) -> tuple[int, int, int, int]:
        """Compute center-cropped bbox clamped to image bounds."""
        # Clamp to image
        x1 = max(0, min(x1, img_w))
        y1 = max(0, min(y1, img_h))
        x2 = max(0, min(x2, img_w))
        y2 = max(0, min(y2, img_h))

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return x1, y1, x2, y2

        # Center crop
        margin_x = (1.0 - self.center_crop_scale) / 2.0
        margin_y = (1.0 - self.center_crop_scale) / 2.0
        cx1 = x1 + int(math.floor(bw * margin_x))
        cy1 = y1 + int(math.floor(bh * margin_y))
        cx2 = x2 - int(math.floor(bw * margin_x))
        cy2 = y2 - int(math.floor(bh * margin_y))

        crop_area = (cx2 - cx1) * (cy2 - cy1)
        if crop_area < self.min_crop_pixels:
            # Fall back to full bbox
            return x1, y1, x2, y2

        return cx1, cy1, cx2, cy2

    @torch.no_grad()
    def forward(
        self,
        cube: torch.Tensor,
        bboxes: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Extract per-bbox spectral signatures for the first batch element."""
        if cube.ndim != 4:
            raise ValueError(f"cube must have shape [B, H, W, C], got {tuple(cube.shape)}.")
        if cube.shape[0] < 1:
            raise ValueError("cube must have B >= 1.")
        if bboxes.ndim != 3 or bboxes.shape[2] != 4:
            raise ValueError(f"bboxes must have shape [B, N, 4], got {tuple(bboxes.shape)}.")

        cube_0 = cube[0]  # [H, W, C]
        img_h, img_w, num_channels = (
            int(cube_0.shape[0]),
            int(cube_0.shape[1]),
            int(cube_0.shape[2]),
        )

        num_boxes = int(bboxes.shape[1])

        # Empty detections
        if num_boxes == 0:
            empty_sig = torch.empty((1, 0, num_channels), dtype=torch.float32, device=cube.device)
            empty_valid = torch.empty((1, 0), dtype=torch.int32, device=cube.device)
            return {
                "spectral_signatures": empty_sig,
                "spectral_std": empty_sig.clone(),
                "spectral_valid": empty_valid,
            }

        signatures: list[torch.Tensor] = []
        stds: list[torch.Tensor] = []
        valids: list[int] = []

        for i in range(num_boxes):
            bx1, by1, bx2, by2 = [int(v) for v in bboxes[0, i].round().tolist()]

            cx1, cy1, cx2, cy2 = self._center_crop_bbox(bx1, by1, bx2, by2, img_h, img_w)

            cw = cx2 - cx1
            ch = cy2 - cy1
            if cw <= 0 or ch <= 0:
                # Bbox fully outside image
                zeros = torch.zeros(num_channels, dtype=cube_0.dtype, device=cube_0.device)
                signatures.append(zeros)
                stds.append(zeros.clone())
                valids.append(0)
                continue

            # Gather pixels from crop region: [P, C]
            pixels = cube_0[cy1:cy2, cx1:cx2, :].reshape(-1, num_channels)

            sig, std = self._trimmed_stats(pixels, num_channels)

            is_valid = sig.norm() >= 1e-8
            if is_valid and self.l2_normalize:
                sig_norm = sig.norm()
                if sig_norm >= 1e-8:
                    sig = sig / sig_norm

            signatures.append(sig)
            stds.append(std)
            valids.append(1 if is_valid else 0)

        signatures_t = torch.stack(signatures, dim=0).unsqueeze(0)  # [1, N, C]
        stds_t = torch.stack(stds, dim=0).unsqueeze(0)  # [1, N, C]
        valids_t = torch.tensor(valids, dtype=torch.int32, device=cube.device).unsqueeze(
            0
        )  # [1, N]

        return {
            "spectral_signatures": signatures_t.to(torch.float32),
            "spectral_std": stds_t.to(torch.float32),
            "spectral_valid": valids_t,
        }


class SpectralSignatureExtractor(Node):
    """Extract per-object spectral signatures from SAM-style label masks."""

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C].",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description="Label map [1, H, W], values are object IDs (0 is background).",
        ),
        "object_ids": PortSpec(
            dtype=torch.int64,
            shape=(1, -1),
            description="Active object IDs [1, N]. If omitted, infer from mask.",
            optional=True,
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Optional wavelengths [C] in nanometers.",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "signatures": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1),
            description="Per-object mean spectral signatures [1, N, C].",
        ),
        "signatures_std": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1),
            description="Per-object spectral std vectors [1, N, C].",
        ),
    }

    def __init__(
        self,
        trim_fraction: float = 0.1,
        min_mask_pixels: int = 10,
        zero_norm_threshold: float = 1e-8,
        **kwargs: Any,
    ) -> None:
        if not (0.0 <= trim_fraction < 0.5):
            raise ValueError("trim_fraction must be in [0.0, 0.5).")
        if min_mask_pixels < 1:
            raise ValueError("min_mask_pixels must be >= 1.")
        if zero_norm_threshold < 0.0:
            raise ValueError("zero_norm_threshold must be non-negative.")

        self.trim_fraction = float(trim_fraction)
        self.min_mask_pixels = int(min_mask_pixels)
        self.zero_norm_threshold = float(zero_norm_threshold)

        super().__init__(
            trim_fraction=trim_fraction,
            min_mask_pixels=min_mask_pixels,
            zero_norm_threshold=zero_norm_threshold,
            **kwargs,
        )

    @staticmethod
    def _parse_mask(mask: torch.Tensor) -> torch.Tensor:
        """Squeeze a [1, H, W] or [H, W] mask to 2-D."""
        if mask.ndim == 3:
            if mask.shape[0] != 1:
                raise ValueError(
                    f"mask must have shape [1, H, W] or [H, W], got {tuple(mask.shape)}."
                )
            return mask[0]
        if mask.ndim == 2:
            return mask
        raise ValueError(f"mask must have shape [1, H, W] or [H, W], got {tuple(mask.shape)}.")

    @staticmethod
    def _parse_object_ids(object_ids: torch.Tensor | None) -> torch.Tensor | None:
        """Squeeze a [1, N] or [N] object-ID tensor to 1-D."""
        if object_ids is None:
            return None
        if object_ids.ndim == 2:
            if object_ids.shape[0] != 1:
                raise ValueError(
                    f"object_ids must have shape [1, N] or [N], got {tuple(object_ids.shape)}."
                )
            return object_ids[0]
        if object_ids.ndim == 1:
            return object_ids
        raise ValueError(
            f"object_ids must have shape [1, N] or [N], got {tuple(object_ids.shape)}."
        )

    @staticmethod
    def _resize_mask_if_needed(mask_2d: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Resize mask to (height, width) using nearest-neighbor if sizes differ."""
        if tuple(mask_2d.shape) == (height, width):
            return mask_2d
        mask_np = mask_2d.detach().cpu().numpy().astype(np.int32, copy=False)
        resized = cv2.resize(mask_np, (width, height), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy(resized).to(mask_2d.device)

    def _trimmed_stats(
        self, pixels: torch.Tensor, num_channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute trimmed mean and std over pixel rows, filtering outliers."""
        zeros = torch.zeros(num_channels, dtype=pixels.dtype, device=pixels.device)
        if pixels.numel() == 0:
            return zeros, zeros

        valid_rows = torch.isfinite(pixels).all(dim=1)
        pixels = pixels[valid_rows]
        if pixels.shape[0] < self.min_mask_pixels:
            return zeros, zeros

        norms = torch.linalg.vector_norm(pixels, dim=1)
        pixels = pixels[norms >= self.zero_norm_threshold]
        if pixels.shape[0] < self.min_mask_pixels:
            return zeros, zeros

        sorted_vals, _ = torch.sort(pixels, dim=0)
        num_pixels = sorted_vals.shape[0]
        trim_k = int(math.floor(num_pixels * self.trim_fraction))
        if trim_k > 0 and (num_pixels - 2 * trim_k) > 0:
            sorted_vals = sorted_vals[trim_k : num_pixels - trim_k]

        mean = sorted_vals.mean(dim=0)
        std = sorted_vals.std(dim=0, unbiased=False)
        return mean, std

    def forward(
        self,
        cube: torch.Tensor,
        mask: torch.Tensor,
        object_ids: torch.Tensor | None = None,
        wavelengths: np.ndarray | torch.Tensor | None = None,  # noqa: ARG002
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Extract per-object signatures for the first batch element."""
        if cube.ndim != 4:
            raise ValueError(f"cube must have shape [B, H, W, C], got {tuple(cube.shape)}.")
        if cube.shape[0] < 1:
            raise ValueError("cube must have B >= 1.")

        cube_0 = cube[0]
        height, width, num_channels = (
            int(cube_0.shape[0]),
            int(cube_0.shape[1]),
            int(cube_0.shape[2]),
        )

        mask_2d = self._parse_mask(mask).to(device=cube_0.device)
        mask_2d = self._resize_mask_if_needed(mask_2d, height=height, width=width)

        parsed_ids = self._parse_object_ids(object_ids)
        if parsed_ids is None:
            resolved_ids = torch.unique(mask_2d[mask_2d != 0], sorted=True).to(torch.int64)
        else:
            resolved_ids = parsed_ids.to(device=cube_0.device, dtype=torch.int64)

        if resolved_ids.numel() == 0:
            empty = torch.empty((1, 0, num_channels), dtype=cube_0.dtype, device=cube_0.device)
            return {"signatures": empty, "signatures_std": empty.clone()}

        signatures: list[torch.Tensor] = []
        signatures_std: list[torch.Tensor] = []
        for obj_id in resolved_ids.tolist():
            obj_mask = mask_2d == int(obj_id)
            if not bool(obj_mask.any()):
                zeros = torch.zeros(num_channels, dtype=cube_0.dtype, device=cube_0.device)
                signatures.append(zeros)
                signatures_std.append(zeros.clone())
                continue

            pixels = cube_0[obj_mask]
            mean, std = self._trimmed_stats(pixels, num_channels=num_channels)
            signatures.append(mean)
            signatures_std.append(std)

        signatures_t = torch.stack(signatures, dim=0).unsqueeze(0)
        signatures_std_t = torch.stack(signatures_std, dim=0).unsqueeze(0)
        return {
            "signatures": signatures_t.to(torch.float32),
            "signatures_std": signatures_std_t.to(torch.float32),
        }


__all__ = ["BBoxSpectralExtractor", "SpectralSignatureExtractor"]
