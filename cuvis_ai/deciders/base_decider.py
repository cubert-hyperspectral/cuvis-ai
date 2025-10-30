from abc import ABC, abstractmethod
from typing import Any

from torch import Tensor

from cuvis_ai.node import Node


class BaseDecider(Node, ABC):
    """
    Abstract class for Decision Making Nodes.

    The decider nodes transform a prediction state into a final prediction
    based on the task that needs to be accomplished.
    """

    def __init__(self):
        super().__init__()

    def fit(self, x, *args, **kwargs):
        # TODO refactor the thing with the empty fits
        pass

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        y: Tensor | None = None,
        m: Any = None,
        **kwargs: Any,
    ) -> Tensor:
        """
        Predict labels based on the input labels.

        Parameters
        ----------
        x : array-like
            Input data.

        Returns
        -------
        Any
            Transformed data.
        """
        pass

    @abstractmethod
    def serialize(self):
        """
        Convert the class into a serialized representation
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load from serialized format into an object
        """
        pass
