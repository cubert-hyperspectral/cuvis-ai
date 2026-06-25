"""Savitzky-Golay spectral filtering node.

Applies a Savitzky-Golay convolution along the spectral axis of a
hyperspectral cube to smooth spectra or compute spectral derivatives. The
filter coefficients are computed once at construction with
:func:`scipy.signal.savgol_coeffs` and stored as a frozen buffer; the forward
pass is pure ``torch`` (no SciPy import at inference time).
"""

import numpy as np
import torch
import torch.nn.functional as F
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from scipy.signal import savgol_coeffs

from cuvis_ai_core.node import Node

# Map this node's boundary modes onto torch ``F.pad`` modes.
_PAD_MODE = {
    "nearest": "replicate",
    "mirror": "reflect",
    "constant": "constant",
}


class SavitzkyGolay(Node):
    """Savitzky-Golay smoothing / derivative filter over the spectral axis.

    A polynomial of degree ``polyorder`` is least-squares fitted within a
    sliding window of ``window_length`` bands; the fitted value (or its
    ``deriv``-th derivative) replaces the centre band. Coefficients are built
    once via :func:`scipy.signal.savgol_coeffs` and applied with a single
    ``conv1d`` along the channel axis.

    Notes
    -----
    The SciPy coefficients are convolution-oriented, while ``torch`` ``conv1d``
    is a cross-correlation; the kernel is therefore stored already flipped so
    the result matches ``scipy.signal.savgol_filter(..., mode="nearest")`` on
    the interior. This filter does not reproduce SciPy's ``mode="interp"``
    boundary handling.

    Parameters
    ----------
    window_length : int, optional
        Length of the filter window in bands; should be odd (default: 11).
    polyorder : int, optional
        Order of the fitted polynomial; must be less than ``window_length``
        (default: 2).
    deriv : int, optional
        Order of the derivative to compute; ``0`` smooths (default: 0).
    delta : float, optional
        Spacing of the samples the filter is applied to; only used when
        ``deriv > 0`` (default: 1.0).
    mode : str, optional
        Boundary handling: ``"nearest"`` (replicate), ``"mirror"`` (reflect),
        or ``"constant"`` (zero pad) (default: ``"nearest"``).
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.HYPERSPECTRAL, NodeTag.PREPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelength array [C] in nanometers",
        ),
    }

    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Filtered hyperspectral cube [B, H, W, C]",
        )
    }

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 2,
        deriv: int = 0,
        delta: float = 1.0,
        mode: str = "nearest",
        **kwargs,
    ) -> None:
        self.window_length = int(window_length)
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.delta = float(delta)
        self.mode = str(mode)
        super().__init__(
            window_length=self.window_length,
            polyorder=self.polyorder,
            deriv=self.deriv,
            delta=self.delta,
            mode=self.mode,
            **kwargs,
        )
        self._pad_mode = _PAD_MODE[self.mode]
        self._pad = self.window_length // 2
        # savgol_coeffs are convolution-oriented; flip for conv1d cross-correlation.
        coefs = savgol_coeffs(
            self.window_length, self.polyorder, deriv=self.deriv, delta=self.delta
        )
        kernel = torch.tensor(coefs, dtype=torch.float32).flip(0).view(1, 1, -1)
        self.register_buffer("coefs", kernel)

    def forward(self, cube: torch.Tensor, **_) -> dict[str, torch.Tensor]:
        """Apply the Savitzky-Golay filter along the spectral axis.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.

        Returns
        -------
        dict[str, torch.Tensor]
            ``{"cube": filtered}`` with the same shape as the input.
        """
        B, H, W, C = cube.shape
        signal = cube.reshape(B * H * W, 1, C)
        if self._pad_mode == "constant":
            padded = F.pad(signal, (self._pad, self._pad), mode="constant", value=0.0)
        else:
            padded = F.pad(signal, (self._pad, self._pad), mode=self._pad_mode)
        kernel = self.coefs.to(dtype=signal.dtype)
        filtered = F.conv1d(padded, kernel)
        return {"cube": filtered.reshape(B, H, W, C)}
