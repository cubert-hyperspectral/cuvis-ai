"""Clustering nodes for hyperspectral cubes.

These nodes are fitted once during statistical initialization with
scikit-learn, freeze the learned parameters into torch buffers, and run a
pure-torch forward pass (no sklearn dependency at inference).

- ``KMeansClusterer`` assigns each pixel to its nearest K-means centroid.
- ``GaussianMixtureClusterer`` evaluates a Gaussian mixture posterior per pixel.
"""

from cuvis_ai.node.clustering.gmm import GaussianMixtureClusterer
from cuvis_ai.node.clustering.kmeans import KMeansClusterer

__all__ = [
    "KMeansClusterer",
    "GaussianMixtureClusterer",
]
