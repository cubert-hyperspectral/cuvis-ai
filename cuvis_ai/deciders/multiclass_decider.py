import numpy as np

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.utils.numpy import (
    flatten_batch_and_spatial,
    get_shape_without_batch,
    unflatten_batch_and_spatial,
)


class MultiClassDecider(BaseDecider):
    """Simple multi-class maximum decider.
    Given a matrix with N channels, chooses the channel with the highest value per spatial location.
    The result will be a single channel matrix with the indices of the chosen channels as values."""

    def __init__(self, n: int, use_min: bool = False) -> None:
        """Create multi-class decider instance

        Parameters
        ----------
        n : int
            Number of classes
        use_min : bool
            Use the minimizing value to decide
        """
        super().__init__()
        self.n = n
        self.use_min = use_min

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply the maximum classification on the data.
        Parameters
        ----------
        X : np.ndarray
            Data to apply the classification on.
        Returns
        -------
        np.ndarray
            Classified data. Single channel matrix comprised of the channel indices of the chosen classes.
        """
        self._input_dim = get_shape_without_batch(X, ignore=[0, 1])
        flatten_soft_output = flatten_batch_and_spatial(X)
        if self.use_min:
            decisions = np.argmin(flatten_soft_output, axis=1)
        else:
            decisions = np.argmax(flatten_soft_output, axis=1)
        return unflatten_batch_and_spatial(decisions, X.shape)

    @BaseDecider.input_dim.getter
    def input_dim(self):
        return [-1, -1, self.n]

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
            "class_count": self.n,
        }
        return data

    def load(self, params: dict, filepath: str):
        """Load this node from a serialized graph."""
        try:
            self.n = int(params["class_count"])
        except:
            raise ValueError(
                f"Could not read attribute 'class_count' as int. Read '{params}' from save file!"
            )


# TODO: How would this functionality be integrated into Deep Learning Methods and Models
