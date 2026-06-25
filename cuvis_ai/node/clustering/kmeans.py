"""K-means clustering node for hyperspectral cubes.

``KMeansClusterer`` fits scikit-learn's K-means during statistical
initialization, copies the learned cluster centroids into a torch buffer, and
runs a pure-torch nearest-centroid assignment in ``forward``. The sklearn
import is lazy and confined to the fit step, so inference has no sklearn
dependency.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node._statistical_fit import _StatisticalFitNode


class KMeansClusterer(_StatisticalFitNode):
    """Partition pixel spectra into ``n_clusters`` groups by nearest centroid.

    The node is fitted once via ``statistical_initialization`` (scikit-learn's
    ``KMeans``); the resulting centroids are stored as a torch buffer. At
    inference each pixel is assigned to its nearest centroid in Euclidean
    space, emitting the 0-based cluster id and the distance to that centroid.

    Cluster ids are emitted directly in the range ``0 .. n_clusters - 1`` (no
    background / ``-1`` sentinel is used).

    Parameters
    ----------
    n_clusters : int, optional
        Number of clusters to fit (default: 8).
    init : str, optional
        scikit-learn ``KMeans`` initialization method (default: "k-means++").
    n_init : int, optional
        Number of seeded re-initializations sklearn runs at fit (default: 10).
    max_iter : int, optional
        Maximum Lloyd iterations per run (default: 300).
    random_state : int, optional
        Seed for the sklearn fit, for reproducible centroids (default: 0).
    **kwargs : Any
        Forwarded to ``_StatisticalFitNode`` (``max_fit_pixels``, ``fit_seed``)
        and the ``Node`` base.

    Attributes
    ----------
    centroids : torch.Tensor
        Fitted cluster centers, shape ``[n_clusters, C]`` after fit; a length-0
        placeholder before fit.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {NodeTag.HYPERSPECTRAL, NodeTag.CLASSIFICATION, NodeTag.STATEFUL, NodeTag.TORCH}
    )

    OUTPUT_SPECS = {
        "class_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="0-based cluster id per pixel [B, H, W]",
        ),
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Euclidean distance to the assigned centroid [B, H, W, 1]",
        ),
    }

    def __init__(
        self,
        n_clusters: int = 8,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        random_state: int = 0,
        **kwargs: Any,
    ) -> None:
        """Store K-means hyperparameters and register the centroid buffer."""
        self.n_clusters = int(n_clusters)
        self.init = str(init)
        self.n_init = int(n_init)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)
        super().__init__(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
            **kwargs,
        )
        self.register_buffer("centroids", torch.zeros(0, dtype=torch.float32))

    def _fit(self, pixels: torch.Tensor) -> None:
        """Fit scikit-learn K-means and freeze the centroids as a torch buffer.

        Parameters
        ----------
        pixels : torch.Tensor
            ``[N, C]`` float32 training-pixel matrix.
        """
        from sklearn.cluster import KMeans

        model = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(pixels.cpu().numpy())
        self.centroids = torch.tensor(model.cluster_centers_, dtype=torch.float32)

    @torch.no_grad()
    def forward(self, cube: torch.Tensor) -> dict[str, torch.Tensor]:
        """Assign each pixel to its nearest centroid.

        Parameters
        ----------
        cube : torch.Tensor
            Input hyperspectral cube ``[B, H, W, C]``.

        Returns
        -------
        dict[str, torch.Tensor]
            ``class_mask`` int32 ``[B, H, W]`` (0-based cluster id) and
            ``scores`` float32 ``[B, H, W, 1]`` (distance to the assigned
            centroid).
        """
        self._require_initialized()
        B, H, W, C = cube.shape
        flat = cube.reshape(-1, C).to(torch.float32)
        d = torch.cdist(flat, self.centroids.to(flat.dtype))
        class_mask = d.argmin(dim=1).reshape(B, H, W).to(torch.int32)
        scores = d.min(dim=1).values.reshape(B, H, W, 1).to(torch.float32)
        return {"class_mask": class_mask, "scores": scores}
