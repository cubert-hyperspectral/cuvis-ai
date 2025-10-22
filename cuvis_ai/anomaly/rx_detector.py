import warnings

import numpy as np

from cuvis_ai.anomaly.abstract_detector import AbstractDetector
from cuvis_ai.utils.numpy import flatten_spatial


class RXDetector(AbstractDetector):
    """
    Reed-Xiaoli (RX) anomaly detector using Mahalanobis distance.

    Works on either a 3D data cube (H x W x B) or flattened pixel array (N x B).
    If no reference spectra are provided, the detector estimates background
    statistics from the input data itself.
    """

    def __init__(self, ref_spectra: list = None):
        super().__init__(ref_spectra or [])
        self.mu = None
        self.cov = None
        self.cov_inv = None
        self._fit = False

    @property
    def _allow_refless(self) -> bool:
        return True

    def score(self, data: np.ndarray) -> np.ndarray:
        """Compute anomaly score(s) for flat data or full cube."""
        # Handle full data cube input
        # Now 'data' is flattened: (N, B)
        # Estimate background mean/covariance
        # Iterate over the first axis
        output = []
        # Warn if unnormalized data
        if np.percentile(data, 90) > 2.0:
            warnings.warn(
                "RX detector is being used without properly normalized data. "
                "Unexpected behavior may occur!"
            )
        for x in range(data.shape[0]):
            if not self._fit:
                mu = np.mean(data[x, :, :], axis=0)
                cov = np.cov(data[x, :, :], rowvar=False)
                cov_inv = np.linalg.pinv(cov)
            else:
                mu = self.mu
                cov_inv = self.cov_inv
                # Invert covariance
            diff = data[x, :, :] - mu
            # Mahalanobis distance: diff * cov_inv * diff^T per sample
            m_dist = np.einsum("ij,jk,ik->i", diff, cov_inv, diff)
            # Return shape (N, 1)
            output.append(m_dist[:, np.newaxis])
        return np.array(output)

    def fit(self, data: np.ndarray, *args, **kwargs):
        """Calculate statistics to reuse for RX detector-like datasets"""
        means = []
        covs = []
        data = flatten_spatial(data)
        for x in range(data.shape[0]):
            mu = np.mean(data[x, :, :], axis=0)
            means.append(mu)
            cov = np.cov(data[x, :, :], rowvar=False)
            covs.append(cov)
        self.mu = np.mean(np.array(means), axis=0)
        self.cov = np.mean(np.array(covs), axis=0)
        self.cov_inv = np.linalg.pinv(self.cov)
        # Mark the node as having been fit
        self._fit = True
        return self
