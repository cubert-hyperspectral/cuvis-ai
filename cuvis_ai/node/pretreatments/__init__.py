"""Spectral pretreatment nodes.

A family of ``cube`` in / ``cube`` out transforms that preprocess
hyperspectral spectra before downstream modelling. All are chainable:

- :class:`SavitzkyGolay`: polynomial smoothing / derivative filter.
- :class:`ContinuumRemoval`: upper convex-hull continuum normalization.
- :class:`SpectralDerivative`: first/second derivative w.r.t. wavelength.
- :class:`Logarithm`: base-10 or natural log transform.
- :class:`Standardize`: ``(x - mean) / std`` per channel, spectrum, sample or location, or globally.
- :class:`BandCrosstalkCorrector`: a learned square band-mixing matrix applied per pixel
  spectrum to remove crosstalk between bands, the identity at construction.
"""

from cuvis_ai.node.pretreatments.continuum_removal import ContinuumRemoval
from cuvis_ai.node.pretreatments.crosstalk import BandCrosstalkCorrector
from cuvis_ai.node.pretreatments.logarithm import Logarithm
from cuvis_ai.node.pretreatments.savitzky_golay import SavitzkyGolay
from cuvis_ai.node.pretreatments.spectral_derivative import SpectralDerivative
from cuvis_ai.node.pretreatments.standardize import Standardize

__all__ = [
    "SavitzkyGolay",
    "ContinuumRemoval",
    "SpectralDerivative",
    "Logarithm",
    "Standardize",
    "BandCrosstalkCorrector",
]
