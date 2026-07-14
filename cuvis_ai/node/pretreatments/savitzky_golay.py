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
from loguru import logger
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

    Sample spacing (``deriv > 0`` only)
    -----------------------------------
    A Savitzky-Golay kernel is a fixed convolution, so it assumes uniform band
    spacing. When the optional ``wavelengths`` port is connected, the effective
    spacing is taken from it (the median of the band steps) and the derivative
    is rescaled accordingly, so the ``delta`` parameter is only used as a
    fallback when ``wavelengths`` is absent. If the bands are not uniformly
    spaced, a single kernel cannot be exact and a warning is emitted; use
    :class:`~cuvis_ai.node.pretreatments.spectral_derivative.SpectralDerivative`
    for a coordinate-aware derivative.

    Parameters
    ----------
    window_length : int, optional
        Length of the filter window in bands; must be odd (default: 11).
    polyorder : int, optional
        Order of the fitted polynomial; must be less than ``window_length``
        (default: 2).
    deriv : int, optional
        Order of the derivative to compute; ``0`` smooths (default: 0).
    delta : float, optional
        Fallback sample spacing in nm, used for ``deriv > 0`` when the
        ``wavelengths`` port is not connected (default: 1.0).
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
            description="Optional wavelength array [C] in nanometers; when connected it "
            "sets the sample spacing for deriv > 0 (median band step).",
            optional=True,
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
        if self.window_length % 2 == 0:
            raise ValueError(f"window_length must be odd, got {self.window_length}")
        self.polyorder = int(polyorder)
        self.deriv = int(deriv)
        self.delta = float(delta)
        self.mode = str(mode)
        if self.mode not in _PAD_MODE:
            raise ValueError(f"mode must be one of {sorted(_PAD_MODE)}, got {self.mode!r}")
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

    def forward(self, cube: torch.Tensor, wavelengths=None, **_) -> dict[str, torch.Tensor]:
        """Apply the Savitzky-Golay filter along the spectral axis.

        Parameters
        ----------
        cube : torch.Tensor
            Input cube in BHWC format.
        wavelengths : array-like, optional
            Band wavelengths in nm. For ``deriv > 0`` the derivative is rescaled
            to the median band spacing taken from this array (overriding the
            ``delta`` parameter); ignored for smoothing (``deriv == 0``).

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
        filtered = F.conv1d(padded, kernel).reshape(B, H, W, C)
        if self.deriv > 0 and wavelengths is not None:
            filtered = filtered * self._spacing_rescale(wavelengths, cube)
        return {"cube": filtered}

    def _spacing_rescale(self, wavelengths, cube: torch.Tensor) -> float:
        """Return the factor mapping the kernel's ``delta`` to the true spacing.

        The kernel is built for a sample spacing of ``self.delta``; a ``deriv``-th
        derivative scales as ``1 / spacing**deriv``, so rescaling by
        ``(self.delta / true_spacing)**deriv`` gives the derivative for the
        wavelengths' actual (median) band step. A warning is emitted when the
        spacing is non-uniform, since one kernel cannot honor it exactly.
        """
        wl = torch.as_tensor(np.asarray(wavelengths), dtype=torch.float64).reshape(-1)
        if wl.numel() < 2:
            return 1.0
        steps = torch.diff(wl)
        spacing = float(torch.median(steps))
        if spacing == 0.0:
            return 1.0
        spread = float(steps.max() - steps.min())
        if spread > 1e-3 * abs(spacing):
            logger.warning(
                "SavitzkyGolay received non-uniform band spacing (steps range "
                f"{float(steps.min()):.3g}..{float(steps.max()):.3g} nm); a fixed "
                "kernel cannot honor it exactly, using the median step "
                f"{spacing:.3g} nm. Use SpectralDerivative for exact non-uniform "
                "derivatives."
            )
        return (self.delta / spacing) ** self.deriv
