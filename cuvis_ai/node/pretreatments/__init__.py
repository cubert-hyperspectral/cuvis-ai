"""Spectral pretreatment nodes.

A family of ``cube`` in / ``cube`` out transforms that preprocess
hyperspectral spectra before downstream modelling. All are chainable:

- :class:`SavitzkyGolay` -- polynomial smoothing / derivative filter.
- :class:`ContinuumRemoval` -- upper convex-hull continuum normalization.
- :class:`SpectralDerivative` -- first/second derivative w.r.t. wavelength.
- :class:`SNVCorrection` -- per-pixel standard normal variate scatter correction.
- :class:`Logarithm` -- base-10 or natural log transform.
- :class:`MeanCenter` -- globally-fitted per-channel mean centring.
- :class:`UnitVarianceScaling` -- globally-fitted per-channel unit-variance scaling.
"""

from cuvis_ai.node.pretreatments.continuum_removal import ContinuumRemoval
from cuvis_ai.node.pretreatments.logarithm import Logarithm
from cuvis_ai.node.pretreatments.savitzky_golay import SavitzkyGolay
from cuvis_ai.node.pretreatments.scaling import MeanCenter, UnitVarianceScaling
from cuvis_ai.node.pretreatments.snv import SNVCorrection
from cuvis_ai.node.pretreatments.spectral_derivative import SpectralDerivative

__all__ = [
    "SavitzkyGolay",
    "ContinuumRemoval",
    "SpectralDerivative",
    "SNVCorrection",
    "Logarithm",
    "MeanCenter",
    "UnitVarianceScaling",
]
