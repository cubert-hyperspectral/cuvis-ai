"""
Preprocessing Nodes.

This module provides nodes for preprocessing hyperspectral data, including
wavelength-based band selection and filtering. These nodes help reduce
dimensionality and focus analysis on specific spectral regions of interest.

See Also
--------
cuvis_ai.node.channel_selector : Advanced channel selection methods
cuvis_ai.node.normalization : Normalization and standardization nodes
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor


class BandpassByWavelength(Node):
    """Select channels by wavelength interval from BHWC tensors.

    This node filters hyperspectral data by keeping only channels within a specified
    wavelength range. Wavelengths must be provided via the input port.

    Parameters
    ----------
    min_wavelength_nm : float
        Minimum wavelength (inclusive) to keep, in nanometers
    max_wavelength_nm : float | None, optional
        Maximum wavelength (inclusive) to keep. If None, selects all wavelengths
        >= min_wavelength_nm. Default: None

    Examples
    --------
    >>> # Create bandpass node
    >>> bandpass = BandpassByWavelength(
    ...     min_wavelength_nm=500.0,
    ...     max_wavelength_nm=700.0,
    ... )
    >>> # Filter cube in BHWC format with wavelengths from input port
    >>> wavelengths_tensor = torch.from_numpy(wavelengths).float()
    >>> filtered = bandpass.forward(data=cube_bhwc, wavelengths=wavelengths_tensor)["filtered"]
    >>>
    >>> # For single HWC images, add a batch dimension first:
    >>> # filtered = bandpass.forward(data=cube_hwc.unsqueeze(0), wavelengths=wavelengths_tensor)["filtered"]
    >>>
    >>> # Use with wavelengths from upstream node
    >>> pipeline.connect(
    ...     (data_node.outputs.cube, bandpass.data),
    ...     (data_node.outputs.wavelengths, bandpass.wavelengths),
    ... )
    """

    INPUT_SPECS = {
        "data": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelengths for each channel [C] in nanometers (can be int32 or float32)",
        ),
    }

    OUTPUT_SPECS = {
        "filtered": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Filtered hyperspectral cube with selected channels [B, H, W, C_filtered]",
        ),
    }

    def __init__(
        self,
        min_wavelength_nm: float,
        max_wavelength_nm: float | None = None,
        **kwargs,
    ) -> None:
        self.min_wavelength_nm = float(min_wavelength_nm)
        self.max_wavelength_nm = float(max_wavelength_nm) if max_wavelength_nm is not None else None

        super().__init__(
            min_wavelength_nm=self.min_wavelength_nm,
            max_wavelength_nm=self.max_wavelength_nm,
            **kwargs,
        )

    def forward(self, data: Tensor, wavelengths: Tensor, **kwargs: Any) -> dict[str, Tensor]:
        """Filter cube by wavelength range.

        Parameters
        ----------
        data : Tensor
            Input hyperspectral cube [B, H, W, C].
        wavelengths : Tensor
            Wavelengths tensor [C] in nanometers.
        **kwargs : Any
            Additional keyword arguments (unused).

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "filtered" key containing filtered cube [B, H, W, C_filtered]

        Raises
        ------
        ValueError
            If no channels are selected by the provided wavelength range
        """

        # Create mask for wavelength range
        if self.max_wavelength_nm is None:
            keep_mask = wavelengths >= self.min_wavelength_nm
        else:
            keep_mask = (wavelengths >= self.min_wavelength_nm) & (
                wavelengths <= self.max_wavelength_nm
            )

        if keep_mask.sum().item() == 0:
            raise ValueError("No channels selected by the provided wavelength range")

        # Filter cube
        filtered = data[..., keep_mask]

        return {"filtered": filtered}


class SpatialRotateNode(Node):
    """Rotate spatial dimensions of cubes, masks, and RGB images.

    Applies a fixed rotation (90, -90, or 180 degrees) to the H and W
    dimensions of all provided inputs.  Wavelengths pass through unchanged.

    Place immediately after a data node so all downstream consumers see
    correctly oriented data.

    Parameters
    ----------
    rotation : int | None
        Rotation in degrees.  Supported: 90, -90, 180
        (and aliases 270, -270, -180).  None or 0 means passthrough.
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Segmentation mask [B, H, W]",
            optional=True,
        ),
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3]",
            optional=True,
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Rotated hyperspectral cube [B, H', W', C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Rotated segmentation mask [B, H', W']",
            optional=True,
        ),
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="Rotated RGB image [B, H', W', 3]",
            optional=True,
        ),
    }

    _VALID_ROTATIONS = {None, 0, 90, -90, 180, -180, 270, -270}

    def __init__(self, rotation: int | None = None, **kwargs: Any) -> None:
        if rotation not in self._VALID_ROTATIONS:
            raise ValueError(
                f"rotation must be one of {sorted(r for r in self._VALID_ROTATIONS if r is not None)}"
                f" or None, got {rotation}"
            )
        self.rotation = self._normalize(rotation)
        super().__init__(rotation=rotation, **kwargs)

    @staticmethod
    def _normalize(rotation: int | None) -> int | None:
        """Normalize equivalent rotation values to a canonical form (None, 90, -90, or 180)."""
        if rotation in (None, 0):
            return None
        if rotation in (180, -180):
            return 180
        if rotation in (90, -270):
            return 90
        if rotation in (-90, 270):
            return -90
        return rotation

    @torch.no_grad()
    def forward(
        self,
        cube: Tensor,
        mask: Tensor | None = None,
        rgb_image: Tensor | None = None,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply the configured rotation to the cube, mask, and rgb_image tensors."""
        k = {None: 0, 90: 1, -90: -1, 180: 2}[self.rotation]

        result: dict[str, Tensor] = {}
        result["cube"] = torch.rot90(cube, k=k, dims=(1, 2)).contiguous() if k else cube
        if mask is not None:
            result["mask"] = torch.rot90(mask, k=k, dims=(1, 2)).contiguous() if k else mask
        if rgb_image is not None:
            result["rgb_image"] = (
                torch.rot90(rgb_image, k=k, dims=(1, 2)).contiguous() if k else rgb_image
            )
        return result


class BBoxRoiCropNode(Node):
    """Differentiable bbox cropping via torchvision roi_align.

    Accepts BHWC images and xyxy bboxes, outputs NCHW crops resized to a
    fixed ``output_size``.  Padding rows (all coords <= 0) are filtered out,
    so the output N equals the number of valid detections.

    Parameters
    ----------
    output_size : tuple[int, int]
        Target crop size ``(H, W)`` for roi_align.
    aligned : bool
        Use sub-pixel aligned roi_align (recommended).
    """

    INPUT_SPECS = {
        "images": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input images [B, H, W, C] in BHWC format, values in [0, 1].",
        ),
        "bboxes": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, 4),
            description="Detection bboxes [B, N, 4] in xyxy pixel coordinates.",
        ),
    }

    OUTPUT_SPECS = {
        "crops": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description=(
                "RoI-aligned crops [N, C, crop_h, crop_w] in NCHW format. "
                "N = number of valid (non-padding) bboxes."
            ),
        ),
    }

    def __init__(
        self,
        output_size: tuple[int, int] = (256, 128),
        aligned: bool = True,
        **kwargs: Any,
    ) -> None:
        self.output_size = tuple(output_size)
        self.aligned = bool(aligned)
        super().__init__(output_size=list(output_size), aligned=aligned, **kwargs)

    def forward(self, images: Tensor, bboxes: Tensor, **_: Any) -> dict[str, Tensor]:
        """Crop and resize bounding-box regions from images.

        Parameters
        ----------
        images : Tensor
            ``[B, H, W, C]`` float32, values in [0, 1].
        bboxes : Tensor
            ``[B, N_padded, 4]`` float32 xyxy pixel coordinates.

        Returns
        -------
        dict
            ``{"crops": Tensor [N, C, crop_h, crop_w]}``
        """
        from torchvision.ops import roi_align

        B, _H, _W, C = images.shape
        crop_h, crop_w = self.output_size

        # BHWC → BCHW
        images_bchw = images.permute(0, 3, 1, 2).contiguous()

        # Build batch indices and flatten bboxes
        N_padded = bboxes.shape[1]
        batch_idx = (
            torch.arange(B, device=bboxes.device).unsqueeze(1).expand(B, N_padded).reshape(-1)
        )
        flat_bboxes = bboxes.reshape(-1, 4)  # [B*N_padded, 4]

        # Filter padding rows (all coords <= 0)
        valid_mask = (flat_bboxes > 0).any(dim=1)
        valid_bboxes = flat_bboxes[valid_mask]
        valid_batch_idx = batch_idx[valid_mask]

        N = valid_bboxes.shape[0]
        if N == 0:
            return {
                "crops": torch.empty(0, C, crop_h, crop_w, device=images.device, dtype=images.dtype)
            }

        # Build [N, 5] roi tensor: [batch_index, x1, y1, x2, y2]
        rois = torch.cat([valid_batch_idx.unsqueeze(1).to(valid_bboxes.dtype), valid_bboxes], dim=1)

        crops = roi_align(
            images_bchw,
            rois,
            output_size=self.output_size,
            spatial_scale=1.0,
            aligned=self.aligned,
        )

        return {"crops": crops}


class ChannelNormalizeNode(Node):
    """Per-channel mean/std normalization for NCHW tensors.

    Defaults to ImageNet statistics but accepts any per-channel values.

    Parameters
    ----------
    mean : tuple[float, ...]
        Per-channel mean.
    std : tuple[float, ...]
        Per-channel std.
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    INPUT_SPECS = {
        "images": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input images [N, C, H, W] in NCHW format.",
        ),
    }

    OUTPUT_SPECS = {
        "normalized": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Normalized images [N, C, H, W] in NCHW format.",
        ),
    }

    def __init__(
        self,
        mean: tuple[float, ...] = IMAGENET_MEAN,
        std: tuple[float, ...] = IMAGENET_STD,
        **kwargs: Any,
    ) -> None:
        self._mean_vals = tuple(float(v) for v in mean)
        self._std_vals = tuple(float(v) for v in std)

        super().__init__(mean=list(self._mean_vals), std=list(self._std_vals), **kwargs)

        # Register as buffers so they auto-move with .to(device)
        self.register_buffer(
            "_mean_buf",
            torch.tensor(self._mean_vals, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "_std_buf",
            torch.tensor(self._std_vals, dtype=torch.float32).view(1, -1, 1, 1),
        )

    def forward(self, images: Tensor, **_: Any) -> dict[str, Tensor]:
        """Normalize images per channel.

        Parameters
        ----------
        images : Tensor
            ``[N, C, H, W]`` float32.

        Returns
        -------
        dict
            ``{"normalized": Tensor [N, C, H, W]}``
        """
        normalized = (images - self._mean_buf) / self._std_buf
        return {"normalized": normalized}


__all__ = [
    "BandpassByWavelength",
    "BBoxRoiCropNode",
    "ChannelNormalizeNode",
    "SpatialRotateNode",
]
