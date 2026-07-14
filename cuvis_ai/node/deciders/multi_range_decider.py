"""Multi-range bucketing node for score maps.

This module provides a stateless node that slices a continuous per-pixel score
map into ordered integer buckets defined by a list of edges. It generalizes the
binary threshold deciders to an arbitrary number of ranges.
"""

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from torch import Tensor

from cuvis_ai_core.node import Node


class MultiRangeSlicer(Node):
    """Bucket a per-pixel score map into ordered class indices by edges.

    Each pixel score is assigned the index of the half-open range it falls into,
    delegating to :func:`torch.bucketize`. With ``edges = [e0, e1, ..., e_{k-1}]``
    the output index is ``0`` for scores below the first edge and ``k`` for
    scores at/above the last edge.

    Convention
    ----------
    ``torch.bucketize(x, edges, right=False)`` is equivalent to
    ``numpy.digitize(x, edges, right=True)`` (and ``right=True`` corresponds to
    ``numpy.digitize(..., right=False)``). The ``right`` flag selects whether a
    value exactly equal to an edge falls into the lower or upper bucket: with
    ``right=False`` (the default here) an edge value goes to the upper bucket,
    matching ``numpy.digitize(..., right=True)``.

    Parameters
    ----------
    edges : list[float]
        Monotonically increasing bucket boundaries. Default: ``[0.25, 0.5, 0.75]``.
    right : bool
        Passed through to :func:`torch.bucketize`. Controls edge-equality
        behavior as described above. Default: ``False``.

    Examples
    --------
    >>> slicer = MultiRangeSlicer(edges=[0.25, 0.5, 0.75])
    >>> scores = torch.tensor([[[[0.1], [0.3], [0.6], [0.9]]]])
    >>> slicer.forward(scores=scores)["class_mask"]
    tensor([[[0, 1, 2, 3]]], dtype=torch.int32)
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.CLASSIFICATION, NodeTag.POSTPROCESSING, NodeTag.TORCH})

    INPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Per-pixel score map [B, H, W, 1]",
        ),
    }

    OUTPUT_SPECS = {
        "class_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Per-pixel bucket index [B, H, W]; 0 = below the first edge",
        ),
    }

    def __init__(
        self,
        edges: list[float] | None = None,
        right: bool = False,
        **kwargs: Any,
    ) -> None:
        self.edges = [float(e) for e in (edges if edges is not None else [0.25, 0.5, 0.75])]
        if any(b <= a for a, b in zip(self.edges, self.edges[1:], strict=False)):
            raise ValueError(f"edges must be strictly increasing, got {self.edges}")
        self.right = bool(right)
        super().__init__(edges=self.edges, right=self.right, **kwargs)

    @torch.no_grad()
    def forward(self, scores: Tensor, **_: Any) -> dict[str, Tensor]:
        """Slice the score map into ordered bucket indices.

        Parameters
        ----------
        scores : Tensor
            Per-pixel score map ``[B, H, W, 1]``.

        Returns
        -------
        dict[str, Tensor]
            ``{"class_mask": Tensor [B, H, W]}`` of int32 bucket indices.
        """
        boundaries = torch.tensor(self.edges, dtype=scores.dtype, device=scores.device)
        idx = torch.bucketize(scores.squeeze(-1), boundaries, right=self.right)
        return {"class_mask": idx.to(torch.int32)}
