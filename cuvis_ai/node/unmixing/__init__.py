"""Spectral unmixing nodes.

This package provides per-pixel spectral unmixing nodes for hyperspectral data:

- :class:`NNLSUnmixing` -- stateless non-negative least squares unmixing against runtime-supplied endmember spectra.
- :class:`NMFUnmixing` -- blind unmixing that learns endmembers by non-negative matrix factorization during statistical initialization, then solves per-pixel abundances against the frozen endmembers.

See Also
--------
cuvis_ai.node.unmixing.nnls : Known-endmember NNLS unmixing.
cuvis_ai.node.unmixing.nmf : Blind NMF unmixing.
"""

from cuvis_ai.node.unmixing.nmf import NMFUnmixing
from cuvis_ai.node.unmixing.nnls import NNLSUnmixing

__all__ = ["NNLSUnmixing", "NMFUnmixing"]
