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


__all__ = ["BandpassByWavelength", "SpatialRotateNode"]
