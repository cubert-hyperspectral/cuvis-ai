import pickle as pk
from collections.abc import Callable

import numpy as np

from cuvis_ai.deciders.base_decider import BaseDecider
from cuvis_ai.utils.numpy import flatten_batch_and_spatial, unflatten_batch_and_spatial


def all_agree(decisions: np.ndarray) -> bool:
    return np.all(decisions == decisions[0])


def at_least_n_agree(n: int) -> Callable[[np.ndarray], bool]:
    return lambda decisions: np.count_nonzero(decisions) >= n


class CombiningDecider(BaseDecider):
    """Decider using values of multiple channels to classify the result.
    The data of all channels at a spatial location are utilized in the chosen decision strategy to classify each data point.

    Parameters
    ----------
    channel_count : int
        The number of channels to expect
    rule : Callable[[np.ndarray], bool]
        The decision strategy to use. :meth:`all_agree` and :meth:`at_least_n_agree` are provided here.
        Custom strategies may also be used.
    """

    def __init__(
        self, channel_count: int = None, rule: Callable[[np.ndarray], bool] = None
    ) -> None:
        super().__init__()

        self.n = channel_count
        self.rule = np.vectorize(rule)
        self.initialized = bool(rule is not None and channel_count is not None)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Apply the chosen :arg:`rule` to the input data.
        Parameters
        ----------
        X : np.ndarray
            Data to classify.
        Returns
        -------
        np.ndarray :
            Data classified to a single channel boolean matrix.
        """
        flatten_soft_output = flatten_batch_and_spatial(X)
        decisions = self.rule(flatten_soft_output)
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
        if not self.initialized:
            print("Module not fully initialized, skipping output!")
            return
        # Write pickle object to file
        dump_file = f"{hash(self.rule)}_pca.pkl"
        pk.dump(self.rule, open(dump_file, "wb"))
        data = {"class_count": self.n, "rules_file": os.path.join(directory, dump_file)}
        return data

    def load(self, params: dict, filepath: str):
        """Load this node from a serialized graph."""
        try:
            self.n = int(params["class_count"])
        except:
            raise ValueError(
                f"Could not read attribute 'class_count' as int. Read '{params}' from save file!"
            )
        try:
            dump_file = os.path.join(filepath, params["rules_file"])
            self.rule = pk.load(open(dump_file, "rb"))
        except:
            raise ValueError("Failed to restore attribute 'rule' from save file!")
        self.initialized = True
