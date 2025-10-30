from typing import Any

from torch import Tensor

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.utils.general import _ensure_channels_last


class BinaryDecider(BaseDecider):
    """Simple decider node using a static threshold to classify data.

    Parameters
    ----------
    threshold : Any
        The threshold to use for classification: result = (input >= threshold)
    """

    def __init__(self, threshold: float = 0.5) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(
        self,
        logits_bhwc: Tensor,
        y: Tensor | None = None,
        m: Any = None,
        **_: Any,
    ) -> Tensor:
        """Apply threshold-based decisioning on channels-last data.

        Args:
            logits_bhwc: Tensor shaped (B, H, W) or (B, H, W, C).
            binary_decider_params: Configuration with threshold and optional dtype.

        Returns:
            Boolean (B, H, W, 1) decision mask unless a dtype override is supplied.
        """
        tensor = _ensure_channels_last(logits_bhwc)
        decisions = (tensor >= self.threshold).to(tensor.dtype)
        return decisions

    @BaseDecider.input_dim.getter
    def input_dim(self):
        """
        Returns the needed shape for the input data.
        If a dimension is not important it will return -1 in the specific position.

        Returns
        -------
        tuple
            Needed shape for data
        """
        return [-1, -1, -1, 1]

    @BaseDecider.output_dim.getter
    def output_dim(self):
        """
        Returns the provided shape for the output data.
        If a dimension is not important it will return -1 in the specific position.

        Returns
        -------
        tuple
            Provided shape for data
        """
        return [-1, -1, -1, 1]

    def serialize(self, directory: str):
        """
        Convert the class into a serialized representation
        """
        return {
            "threshold": self.threshold,
        }

    def load(self, params: dict, filepath: str):
        """Load this node from a serialized graph."""
        try:
            self.threshold = float(params["threshold"])
        except Exception as e:
            raise ValueError(f"Error loading BinaryDecider from params: {params}") from e
