import numpy as np

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.utils.numpy import flatten_batch_and_spatial, unflatten_batch_and_spatial


class BinaryDecider(BaseDecider):
    """Simple decider node using a static threshold to classify data.

    Parameters
    ----------
    threshold : Any
        The threshold to use for classification: result = (input >= threshold)
    """

    def __init__(self, threshold: float = 1.0) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply binary decision on input data.

        Paramaters
        ----------
        X : np.ndarray
            Input data as numpy array

        Returns
        -------
        np.ndarray
            Classified input data. Where the datapoints are False if smaller than the threshold or True if larger or equal.
        """
        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = flatten_soft_output >= self.threshold
        return unflatten_batch_and_spatial(decisions, X.shape)

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
        data = {
            "threshold": self.threshold,
        }
        return data

    def load(self, params: dict, filepath: str):
        """Load this node from a serialized graph."""
        try:
            self.threshold = float(params["threshold"])
        except:
            raise ValueError(
                f"Could not read attribute 'threshold' as float. Read '{params}' from save file!"
            )


# TODO: How would this functionality be integrated into Deep Learning Methods and Models
