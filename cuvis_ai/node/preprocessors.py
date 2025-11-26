from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from cuvis_ai.node import Node
from cuvis_ai.pipeline.ports import PortSpec


class BandpassByWavelength(Node):
    """Select channels by wavelength interval from BHWC tensors.

    This node filters hyperspectral data by keeping only channels within a specified
    wavelength range. Wavelengths can be provided via an input port, constructor
    parameter, or will be cached from the first forward pass.

    Parameters
    ----------
    min_wavelength_nm : float
        Minimum wavelength (inclusive) to keep, in nanometers
    max_wavelength_nm : float | None, optional
        Maximum wavelength (inclusive) to keep. If None, selects all wavelengths
        >= min_wavelength_nm. Default: None
    wavelengths : np.ndarray | list[float] | None, optional
        Optional cached wavelengths array. If provided, will be used for filtering
        without requiring wavelengths input port. Default: None

    Examples
    --------
    >>> # Create bandpass node with cached wavelengths
    >>> bandpass = BandpassByWavelength(
    ...     min_wavelength_nm=500.0,
    ...     max_wavelength_nm=700.0,
    ...     wavelengths=np.array([400, 450, 500, 550, 600, 650, 700, 750, 800])
    ... )
    >>> # Filter cube in BHWC format
    >>> filtered = bandpass.forward(data=cube_bhwc)["filtered"]
    >>>
    >>> # For single HWC images, add a batch dimension first:
    >>> # filtered = bandpass.forward(data=cube_hwc.unsqueeze(0))["filtered"]
    >>>
    >>> # Or use with wavelengths from upstream node
    >>> canvas.connect(
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
            optional=True,
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
        wavelengths: np.ndarray | list[float] | None = None,
        **kwargs,
    ) -> None:
        self.min_wavelength_nm = float(min_wavelength_nm)
        self.max_wavelength_nm = float(max_wavelength_nm) if max_wavelength_nm is not None else None

        # Cache wavelengths if provided
        self._cached_wavelengths: np.ndarray | None = None
        if wavelengths is not None:
            cached_array = np.asarray(wavelengths, dtype=np.float32).ravel()
            self._cached_wavelengths = cached_array

        super().__init__(
            min_wavelength_nm=self.min_wavelength_nm,
            max_wavelength_nm=self.max_wavelength_nm,
            wavelengths=wavelengths.tolist()
            if isinstance(wavelengths, np.ndarray)
            else wavelengths,
            **kwargs,
        )

    def forward(
        self, data: Tensor, wavelengths: Tensor | np.ndarray | None = None, **kwargs: Any
    ) -> dict[str, Tensor]:
        """Filter cube by wavelength range.

        Parameters
        ----------
        data : Tensor
            Input hyperspectral cube [B, H, W, C].
        wavelengths : Tensor | np.ndarray | None, optional
            Wavelengths array [C] in nanometers. If not provided, uses cached
            wavelengths or raises error. Can also be passed via kwargs.
        **kwargs : Any
            Additional keyword arguments. Can include 'wavelengths' for backward
            compatibility.

        Returns
        -------
        dict[str, Tensor]
            Dictionary with "filtered" key containing filtered cube [B, H, W, C_filtered]

        Raises
        ------
        ValueError
            If input shape is invalid or no channels are selected
        RuntimeError
            If wavelengths are not provided and not cached
        """
        # Get wavelengths from various sources (priority order)
        # 1. Direct parameter
        # 2. kwargs (for backward compatibility)
        # 3. Cached wavelengths
        if wavelengths is None:
            wavelengths = kwargs.get("wavelengths", None)
        if wavelengths is None and self._cached_wavelengths is not None:
            wavelengths = self._cached_wavelengths

        if wavelengths is None:
            raise RuntimeError(
                "BandpassByWavelength requires 'wavelengths' input port, constructor parameter, "
                "or cached wavelengths from previous forward pass"
            )

        # Convert to tensor if needed - handle both np.int32 and np.float32/torch.float32
        if torch.is_tensor(wavelengths):
            wl = wavelengths.to(dtype=torch.float32)
            # Cache as float32 numpy array
            if self._cached_wavelengths is None:
                self._cached_wavelengths = wl.detach().cpu().numpy().astype(np.float32)
        else:
            # Cache original numpy array preserving its dtype (int32 or float32)
            if self._cached_wavelengths is None:
                self._cached_wavelengths = wavelengths.copy()
            # Convert to float32 tensor for computation (needed for float comparisons)
            wl = torch.as_tensor(wavelengths, dtype=torch.float32)

        # Create mask for wavelength range
        if self.max_wavelength_nm is None:
            keep_mask = wl >= self.min_wavelength_nm
        else:
            keep_mask = (wl >= self.min_wavelength_nm) & (wl <= self.max_wavelength_nm)

        if keep_mask.sum().item() == 0:
            raise ValueError("No channels selected by the provided wavelength range")

        # Filter cube
        filtered = data[..., keep_mask]

        return {"filtered": filtered}

    def load(self, params: dict, serial_dir: str) -> None:
        """Load BandpassByWavelength state from serialized data.

        Parameters
        ----------
        params : dict
            Serialized parameters dictionary
        serial_dir : str
            Directory containing serialized data
        """
        config = params.get("params", params)
        self.min_wavelength_nm = float(config.get("min_wavelength_nm", self.min_wavelength_nm))
        max_wl = config.get("max_wavelength_nm", self.max_wavelength_nm)
        self.max_wavelength_nm = float(max_wl) if max_wl is not None else None

        wavelengths = config.get("wavelengths")
        self._cached_wavelengths = (
            np.asarray(wavelengths, dtype=np.float32).ravel() if wavelengths is not None else None
        )


__all__ = ["BandpassByWavelength"]
