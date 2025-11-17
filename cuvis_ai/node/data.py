from typing import Any

import numpy as np
import torch

from cuvis_ai.node import Node
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper
from cuvis_ai.pipeline.ports import PortSpec


class LentilsAnomalyDataNode(Node):
    INPUT_SPECS = {
        "cube": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input hyperspectral cube [B, H, W, C]",
        ),
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(-1, -1, -1),
            description="Multi-class segmentation mask [B, H, W]",
            optional=True,
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
        "wavelengths": PortSpec(
            dtype=np.int32, shape=(-1,), description="Wavelengths for each channel"
        ),
    }

    def __init__(
        self, normal_class_ids: list[int], wavelengths: np.ndarray | None = None, **kwargs
    ) -> None:
        super().__init__(normal_class_ids=normal_class_ids, wavelengths=wavelengths, **kwargs)

        if wavelengths is not None:
            wavelengths = wavelengths.astype(np.int32)
        self.wavelengths = wavelengths

        self._binary_mapper = BinaryAnomalyLabelMapper(  # could have be used as a node as well
            normal_class_ids=normal_class_ids,
        )

    def forward(
        self, cube: torch.Tensor, mask: torch.Tensor | None = None, **_: Any
    ) -> dict[str, torch.Tensor | np.ndarray]:
        result: dict[str, torch.Tensor | np.ndarray] = {"cube": cube}

        if self.wavelengths is None:
            # Infer wavelengths from cube shape if not provided
            num_channels = cube.shape[-1]  # C from BHWC
            self.wavelengths = np.arange(num_channels, dtype=np.int32)
        result["wavelengths"] = self.wavelengths

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

    def load(self, params, serial_dir) -> bool:
        return True
