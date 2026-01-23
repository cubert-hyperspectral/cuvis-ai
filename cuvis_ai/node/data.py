from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import PortSpec

from cuvis_ai.node.labels import BinaryAnomalyLabelMapper


class LentilsAnomalyDataNode(Node):
    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.uint16,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Multi-class segmentation mask [B, H, W]",
            optional=True,
        ),
        "wavelengths": PortSpec(
            dtype=torch.int32, shape=(-1, -1), description="Wavelengths for each channel"
        ),
    }
    OUTPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary anomaly mask (0=normal, 1=anomaly) [B, H, W, 1]",
            optional=True,
        ),
        # wavelength must be a required input port and not optional
        "wavelengths": PortSpec(
            dtype=np.int32, shape=(-1,), description="Wavelengths for each channel", optional=True
        ),
    }

    def __init__(
        self, normal_class_ids: list[int], anomaly_class_ids: list[int] | None = None, **kwargs
    ) -> None:
        super().__init__(
            normal_class_ids=normal_class_ids, anomaly_class_ids=anomaly_class_ids, **kwargs
        )

        self._binary_mapper = BinaryAnomalyLabelMapper(  # could have be used as a node as well
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
        result: dict[str, torch.Tensor | np.ndarray] = {"cube": cube.to(torch.float32)}

        # wavelengths passthrough, could check that in all batch elements the same wavelengths are used
        # input B x C -> output C
        if wavelengths is not None:
            result["wavelengths"] = wavelengths[0].cpu().numpy()

        if mask is not None:
            # Add channel dimension for mapper: BHW -> BHWC
            mask_4d = mask.unsqueeze(-1)

            # Always apply binary mapper
            mapped = self._binary_mapper.forward(
                cube=cube,
                mask=mask_4d,
                **_,  # Pass through additional kwargs
            )
            result["mask"] = mapped["mask"]  # Already BHWC bool

        return result
