"""Shape-morphology descriptor node.

``ShapeMorphology`` labels the connected components of a per-frame mask (reusing
the repo's OpenCV connected-components path) and computes per-object geometric
descriptors (area, centroid, axis lengths, eccentricity, orientation, bbox area)
in pure torch.  The descriptor conventions match
``skimage.measure.regionprops``.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.utils.connected_components import label_connected_components
from cuvis_ai_core.node import Node

_SUPPORTED_PROPERTIES = (
    "area",
    "centroid_y",
    "centroid_x",
    "major_axis",
    "minor_axis",
    "eccentricity",
    "orientation",
    "bbox_area",
)


class ShapeMorphology(Node):
    """Per-object shape descriptors from a binary or labeled mask.

    For each batch element the mask is reduced to integer instance labels
    (connected components when ``binarize`` is True, otherwise the input is
    taken to already carry labels) and a fixed set of geometric descriptors is
    computed per object.  Descriptors follow the
    ``skimage.measure.regionprops`` conventions:

    * ``area`` - pixel count of the region.
    * ``centroid_y`` / ``centroid_x`` - mean pixel coordinates.
    * ``major_axis`` / ``minor_axis`` - ``4 * sqrt(lambda)`` of the two covariance eigenvalues (descending).
    * ``eccentricity`` - ``sqrt(1 - lambda_2 / lambda_1)``.
    * ``orientation`` - ``0.5 * atan2(2 * cov_xy, cov_yy - cov_xx)``.
    * ``bbox_area`` - area of the axis-aligned bounding box.

    Rows are padded to ``max_objects``; ``valid`` marks the first ``N`` rows
    True (real objects) and the remainder False (padding).

    Parameters
    ----------
    properties : list of str
        Descriptor names to emit, in order along the ``P`` axis.  Defaults to
        all supported descriptors.
    max_objects : int
        Number of object rows in the output; objects beyond this are dropped
        and fewer objects are zero-padded.  Default ``256``.
    connectivity : int
        Pixel connectivity (``4`` or ``8``) for connected-component labeling
        when ``binarize`` is True.  Default ``8``.
    binarize : bool
        If True, treat any nonzero pixel as foreground and run connected-
        component labeling.  If False, treat the input as an already-labeled
        mask and use its nonzero values as instance ids.  Default ``True``.
    """

    _category = NodeCategory.TRANSFORM
    _tags = frozenset({NodeTag.MASK, NodeTag.TORCH})

    INPUT_SPECS = {
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Mask [B, H, W]; binary (>0 foreground) or already-labeled.",
        ),
    }

    OUTPUT_SPECS = {
        "identity_mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Instance labels [B, H, W]; 0 background, 1..N objects.",
        ),
        "properties": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Per-object descriptors [B, max_objects, P].",
        ),
        "valid": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1),
            description="Validity per slot [B, max_objects]; True for real objects.",
        ),
    }

    def __init__(
        self,
        properties: list[str] | None = None,
        max_objects: int = 256,
        connectivity: int = 8,
        binarize: bool = True,
        **kwargs: Any,
    ) -> None:
        if properties is None:
            properties = list(_SUPPORTED_PROPERTIES)
        unknown = [p for p in properties if p not in _SUPPORTED_PROPERTIES]
        if unknown:
            raise ValueError(f"unknown properties: {unknown}; supported: {_SUPPORTED_PROPERTIES}")
        if len(properties) == 0:
            raise ValueError("properties must list at least one descriptor")
        if max_objects < 1:
            raise ValueError("max_objects must be >= 1")
        if connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")

        self.properties = list(properties)
        self.max_objects = int(max_objects)
        self.connectivity = int(connectivity)
        self.binarize = bool(binarize)

        super().__init__(
            properties=self.properties,
            max_objects=self.max_objects,
            connectivity=self.connectivity,
            binarize=self.binarize,
            **kwargs,
        )

    def _labels_for_frame(self, mask_2d: torch.Tensor) -> torch.Tensor:
        """Return a dense 1..N relabeled int32 label map for one frame."""
        if self.binarize:
            return label_connected_components(mask_2d, connectivity=self.connectivity)
        # Already-labeled: compact nonzero values to a dense 1..N range so the
        # downstream gather over contiguous ids works regardless of the input
        # label values.
        labels = mask_2d.to(torch.int64)
        unique = torch.unique(labels[labels != 0], sorted=True)
        dense = torch.zeros_like(labels)
        for new_id, old_id in enumerate(unique.tolist(), start=1):
            dense[labels == old_id] = new_id
        return dense.to(torch.int32)

    def _descriptors_for_frame(
        self, labels: torch.Tensor, num_objects: int, device: torch.device
    ) -> torch.Tensor:
        """Compute per-object descriptors for a labeled frame.

        Returns a ``[num_objects, len(self.properties)]`` float32 tensor with
        one row per instance id ``1..num_objects`` (in id order).
        """
        out = torch.zeros((num_objects, len(self.properties)), dtype=torch.float32, device=device)
        if num_objects == 0:
            return out

        labels = labels.to(device)
        height, width = labels.shape
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=device),
            torch.arange(width, dtype=torch.float64, device=device),
            indexing="ij",
        )
        flat_labels = labels.reshape(-1).to(torch.int64)
        fg = flat_labels > 0
        ids = flat_labels[fg] - 1  # 0-based region index
        y = ys.reshape(-1)[fg]
        x = xs.reshape(-1)[fg]

        n = num_objects
        area = torch.zeros(n, dtype=torch.float64, device=device)
        area.scatter_add_(0, ids, torch.ones_like(y))
        safe_area = area.clamp_min(1.0)

        sum_y = torch.zeros(n, dtype=torch.float64, device=device).scatter_add_(0, ids, y)
        sum_x = torch.zeros(n, dtype=torch.float64, device=device).scatter_add_(0, ids, x)
        cy = sum_y / safe_area
        cx = sum_x / safe_area

        # Second central moments via E[.] - mean*mean (population, ddof=0).
        sum_yy = torch.zeros(n, dtype=torch.float64, device=device).scatter_add_(0, ids, y * y)
        sum_xx = torch.zeros(n, dtype=torch.float64, device=device).scatter_add_(0, ids, x * x)
        sum_yx = torch.zeros(n, dtype=torch.float64, device=device).scatter_add_(0, ids, y * x)
        cov_yy = sum_yy / safe_area - cy * cy
        cov_xx = sum_xx / safe_area - cx * cx
        cov_yx = sum_yx / safe_area - cy * cx

        # Eigenvalues of the symmetric 2x2 [[cov_yy, cov_yx], [cov_yx, cov_xx]].
        tr = cov_yy + cov_xx
        diff = cov_yy - cov_xx
        disc = torch.sqrt((diff * diff + 4.0 * cov_yx * cov_yx).clamp_min(0.0))
        lam1 = (tr + disc) * 0.5  # >= lam2
        lam2 = (tr - disc) * 0.5
        lam1 = lam1.clamp_min(0.0)
        lam2 = lam2.clamp_min(0.0)

        major = 4.0 * torch.sqrt(lam1)
        minor = 4.0 * torch.sqrt(lam2)
        eccentricity = torch.sqrt((1.0 - lam2 / lam1.clamp_min(1e-12)).clamp_min(0.0))
        eccentricity = torch.where(lam1 > 0, eccentricity, torch.zeros_like(eccentricity))
        orientation = 0.5 * torch.atan2(2.0 * cov_yx, cov_yy - cov_xx)

        # Axis-aligned bbox extent per region.
        min_y = torch.full((n,), float(height), dtype=torch.float64, device=device)
        max_y = torch.full((n,), -1.0, dtype=torch.float64, device=device)
        min_x = torch.full((n,), float(width), dtype=torch.float64, device=device)
        max_x = torch.full((n,), -1.0, dtype=torch.float64, device=device)
        min_y.scatter_reduce_(0, ids, y, reduce="amin", include_self=True)
        max_y.scatter_reduce_(0, ids, y, reduce="amax", include_self=True)
        min_x.scatter_reduce_(0, ids, x, reduce="amin", include_self=True)
        max_x.scatter_reduce_(0, ids, x, reduce="amax", include_self=True)
        bbox_area = (max_y - min_y + 1.0) * (max_x - min_x + 1.0)
        bbox_area = torch.where(area > 0, bbox_area, torch.zeros_like(bbox_area))

        values: dict[str, torch.Tensor] = {
            "area": area,
            "centroid_y": cy,
            "centroid_x": cx,
            "major_axis": major,
            "minor_axis": minor,
            "eccentricity": eccentricity,
            "orientation": orientation,
            "bbox_area": bbox_area,
        }
        for col, name in enumerate(self.properties):
            out[:, col] = values[name].to(torch.float32)
        return out

    @torch.no_grad()
    def forward(self, mask: torch.Tensor, **_: Any) -> dict[str, torch.Tensor]:
        """Label objects and emit per-object descriptors, padded to ``max_objects``."""
        device = mask.device
        b, h, w = mask.shape
        p = len(self.properties)

        identity = torch.zeros((b, h, w), dtype=torch.int32, device=device)
        properties = torch.zeros((b, self.max_objects, p), dtype=torch.float32, device=device)
        valid = torch.zeros((b, self.max_objects), dtype=torch.bool, device=device)

        for i in range(b):
            labels = self._labels_for_frame(mask[i]).to(device)
            num_objects = int(labels.max().item())
            n_keep = min(num_objects, self.max_objects)

            # Drop labels beyond max_objects from the emitted identity map.
            if num_objects > self.max_objects:
                labels = torch.where(labels > self.max_objects, torch.zeros_like(labels), labels)
            identity[i] = labels.to(torch.int32)

            if num_objects == 0:
                continue

            desc = self._descriptors_for_frame(labels, num_objects, device)
            properties[i, :n_keep] = desc[:n_keep]
            valid[i, :n_keep] = True

        return {
            "identity_mask": identity,
            "properties": properties,
            "valid": valid,
        }


__all__ = ["ShapeMorphology"]
