"""Data preparation nodes for CU3S hyperspectral pipelines."""

from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.labels import BinaryAnomalyLabelMapper


class CU3SDataNode(Node):
    """General-purpose data node for CU3S hyperspectral sequences.

    This node normalizes common CU3S batch inputs for pipelines:
    - converts `cube` from uint16 to float32
    - passes optional `mask` through unchanged
    - extracts 1D `wavelengths` from batched input
    """

    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.Tensor,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Segmentation mask [B, H, W]",
            optional=True,
        ),
        "wavelengths": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1),
            description="Wavelengths for each channel [B, C]",
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Hyperspectral cube [B, H, W, C] as float32",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Segmentation mask passthrough [B, H, W]",
            optional=True,
        ),
        "wavelengths": PortSpec(
            dtype=np.int32,
            shape=(-1,),
            description="Wavelengths [C] in nm",
            optional=True,
        ),
        "mesu_index": PortSpec(
            dtype=torch.int64,
            shape=(-1,),
            description="Measurement/frame index per batch element [B]",
            optional=True,
        ),
    }

    def forward(
        self,
        cube: torch.Tensor,
        mask: torch.Tensor | None = None,
        wavelengths: torch.Tensor | None = None,
        mesu_index: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Normalize CU3S batch data for pipeline consumption."""
        result: dict[str, torch.Tensor | np.ndarray] = {"cube": cube.to(torch.float32)}

        # Keep the same behavior as existing data nodes: use first batch entry.
        if wavelengths is not None:
            result["wavelengths"] = wavelengths[0].cpu().numpy()

        if mask is not None:
            result["mask"] = mask

        if mesu_index is not None:
            result["mesu_index"] = mesu_index

        return result


class LentilsAnomalyDataNode(CU3SDataNode):
    """Lentils-specific CU3S data node with binary anomaly label mapping.

    Inherits shared CU3S normalization (cube + wavelengths) and additionally maps
    multi-class masks to binary anomaly masks.
    """

    INPUT_SPECS = {
        **CU3SDataNode.INPUT_SPECS,
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Multi-class segmentation mask [B, H, W]",
            optional=True,  # Explicit for readability
        ),
    }
    OUTPUT_SPECS = {
        **CU3SDataNode.OUTPUT_SPECS,
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask (0=normal, 1=anomaly) [B, H, W, 1]",
            optional=True,
        ),
    }

    def __init__(
        self, normal_class_ids: list[int], anomaly_class_ids: list[int] | None = None, **kwargs
    ) -> None:
        # Keep node params on the base Node for config/serialization compatibility.
        super().__init__(
            normal_class_ids=normal_class_ids, anomaly_class_ids=anomaly_class_ids, **kwargs
        )
        self._binary_mapper = BinaryAnomalyLabelMapper(
            normal_class_ids=normal_class_ids,
            anomaly_class_ids=anomaly_class_ids,
        )

    def forward(
        self,
        cube: torch.Tensor,
        mask: torch.Tensor | None = None,
        wavelengths: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, torch.Tensor | np.ndarray]:
        """Apply CU3S normalization and optional Lentils binary mask mapping."""
        result = super().forward(cube=cube, mask=None, wavelengths=wavelengths, **_)

        if mask is not None:
            # Mapper expects channel-last mask: BHW -> BHWC.
            mask_4d = mask.unsqueeze(-1)
            mapped = self._binary_mapper.forward(cube=cube, mask=mask_4d, **_)
            result["mask"] = mapped["mask"]

        return result
