from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.utils.general import _ensure_channels_last


class MultiClassDecider(BaseDecider):
    """Multi-class argmax/argmin decider operating on channels-last torch tensors."""

    def __init__(
        self,
        class_count: int | None = None,
        use_min: bool = False,
    ) -> None:
        super().__init__()
        if class_count is None:
            class_count = 1
        self.class_count = class_count
        self.use_min = use_min

    def forward(
        self,
        logits_bhwc: Tensor,
        y: Tensor | None = None,
        m: Any = None,
        **_: Any,
    ) -> Tensor:
        """Select the winning class per pixel using torch operations."""

        tensor = _ensure_channels_last(logits_bhwc)
        num_channels = tensor.shape[-1]

        assert num_channels == self.class_count, (
            f"Input channel dimension ({num_channels}) does not match configured class_count ({self.class_count})."
        )

        if self.use_min:
            indices = torch.argmin(tensor, dim=-1, keepdim=True)
        else:
            indices = torch.argmax(tensor, dim=-1, keepdim=True)

        return indices.to(torch.int32)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1, -1, -1, self.class_count]

    @BaseDecider.output_dim.getter
    def output_dim(self):
        return [-1, -1, -1, 1]

    def serialize(self, directory: str):
        return {
            "class_count": int(self.class_count),
            "use_min": bool(self.use_min),
        }

    def load(self, params: dict, filepath: str):
        """Load this node from a serialized graph."""
        try:
            self.use_min = bool(params["use_min"])
            self.class_count = int(params["class_count"])
        except Exception as e:
            raise ValueError(f"Error loading MultiClassDecider from params: {params}") from e
