import numpy as np

from cuvis_ai.deciders.base_decider import BaseDecider


class Cascaded(BaseDecider):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1, -1, 1]

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
        return [-1, -1, 1]

    def serialize(self, directory: str):
        """
        Convert the class into a serialized representation
        """
        data = {}
        return data

    def load(self, params: dict, filepath: str):
        """Load this node from a serialized graph."""
        pass  # No attributes
