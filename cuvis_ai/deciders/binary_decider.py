from typing import Any

import torch
from torch import Tensor

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.pipeline.ports import PortSpec


class BinaryDecider(BaseDecider):
    """Simple decider node using a static threshold to classify data.

    Accepts logits as input, applies sigmoid transformation to convert to
    probabilities [0, 1], then applies threshold to produce binary decisions.

    Parameters
    ----------
    threshold : float
        The threshold to use for classification after sigmoid:
        result = (sigmoid(input) >= threshold)
    apply_sigmoid : bool, optional
        Whether to apply sigmoid transformation before thresholding (default: True).
        Set to False if input is already probabilities.
    """

    INPUT_SPECS = {
        "logits": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Input logits to threshold (BHWC format)",
        )
    }

    OUTPUT_SPECS = {
        "decisions": PortSpec(
            dtype=torch.bool,
            shape=(-1, -1, -1, 1),
            description="Binary decision mask (BHWC format)",
        )
    }

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        super().__init__()

    def forward(
        self,
        logits: Tensor,
        **_: Any,
    ) -> dict[str, Tensor]:
        """Apply sigmoid and threshold-based decisioning on channels-last data.

        Args:
            logits: Tensor shaped (B, H, W, C) containing logits.

        Returns:
            Dictionary with "decisions" key containing (B, H, W, 1) decision mask.
        """

        # Apply sigmoid if needed to convert logits to probabilities
        tensor = torch.sigmoid(logits)

        # Apply threshold to get binary decisions
        decisions = tensor >= self.threshold
        return {"decisions": decisions}

    def serialize(self, directory: str) -> dict[str, float]:
        """
        Convert the class into a serialized representation
        """
        return {
            "threshold": self.threshold,
        }

    def load(self, params: dict, filepath: str) -> None:
        """Load this node from a serialized graph."""
        try:
            self.threshold = float(params["threshold"])
        except Exception as e:
            raise ValueError(f"Error loading BinaryDecider from params: {params}") from e
