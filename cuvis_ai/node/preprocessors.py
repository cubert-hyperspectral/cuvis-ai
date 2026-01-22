from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec
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


__all__ = ["BandpassByWavelength"]
