from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from torch import Tensor, nn

from cuvis_ai.utils.serializer import Serializable
from cuvis_ai.utils.torch import check_array_shape

LabelLike = Tensor | Sequence[Any] | None
MetaLike = Any
NodeOutput = Tensor | tuple[Tensor, LabelLike, MetaLike]


class Node(nn.Module, ABC, Serializable):
    """
    Abstract class for data preprocessing.
    """

    def __init__(self, *args, **kwargs) -> None:
        # Initialize Serializable first to capture hparams
        Serializable.__init__(self, *args, **kwargs)
        # Then initialize nn.Module without any args/kwargs
        nn.Module.__init__(self)
        self.id = f"{type(self).__name__}-{str(uuid.uuid4())}"
        self.__forward_metadata = {}
        self.__fit_metadata = {}
        self.__forward_inputs = {}
        self.__fit_inputs = {}
        self._initialized = False
        self.freezed = False

    @property
    def requires_initial_fit(self) -> bool:
        """Whether this node requires statistical initialization before training.
        
        Returns
        -------
        bool
            True if node needs initialize_from_data() to be called before training
        """
        return False

    @property
    def is_trainable(self) -> bool:
        """Whether this node's parameters should receive gradients during training.
        
        Returns
        -------
        bool
            True if parameters should be trainable
        """
        return False

    def initialize_from_data(self, iterator) -> None:
        """Initialize node parameters from a data iterator (statistical fitting).
        
        This method is called during the statistical initialization phase for nodes
        that have requires_initial_fit=True. It should compute and store any
        statistics needed from the data (e.g., mean, covariance for RX detector).
        
        Parameters
        ----------
        iterator : Iterator
            Iterator yielding batches of (x, y, m) tuples where:
            - x: input tensor
            - y: labels (optional)
            - m: metadata dict (optional)
            
        Raises
        ------
        NotImplementedError
            If the node requires initialization but doesn't implement this method
        """
        if self.requires_initial_fit:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires initial fit but does not "
                "implement initialize_from_data()"
            )

    def prepare_for_train(self) -> None:
        """Prepare node for gradient-based training.
        
        This method is called after statistical initialization for nodes that
        have is_trainable=True. It should convert any buffers that need gradients
        into nn.Parameters.
        
        For example, RXGlobal converts mu and cov from buffers to parameters
        when trainable_stats=True.
        """
        pass

    def freeze(self) -> None:
        """Freeze all parameters in this node (disable gradient computation).
        
        This is called for statistical nodes that should remain fixed after
        initialization (is_trainable=False).
        """
        self.freezed = True
        self.requires_grad_(False)

    @abstractmethod
    def forward(
        self,
        x: Tensor,
        y: LabelLike = None,
        m: MetaLike = None,
        **_: Any,
    ) -> NodeOutput:
        """
        Transform the input data.

        Parameters:
        x (array-like): Input data.

        Returns:
        Transformed data.
        """
        raise NotImplementedError("All Nodes must implement a torch-native forward.")

    def check_output_dim(self, x):
        """
        Check that the parameters for the output data data match user
        expectations

        Parameters:
        x (array-like): Input data.

        Returns:
        (Bool) Valid data
        """
        return check_array_shape(x, self.output_dim)

    def check_input_dim(self, x):
        """
        Check that the parameters for the input data data match user
        expectations

        Parameters:
        x (array-like): Input data.

        Returns:
        (Bool) Valid data
        """
        return check_array_shape(x, self.input_dim)

    def set_forward_meta_request(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, bool):
                raise ValueError("Invalid usage of Metadata Routing")
            self.__forward_metadata[k] = v

    def set_fit_meta_request(self, **kwargs):
        for k, v in kwargs.items():
            if not isinstance(v, bool):
                raise ValueError("Invalid usage of Metadata Routing")
            self.__fit_metadata[k] = v

    def get_forward_requested_meta(self):
        return self.__forward_metadata

    def get_fit_requested_meta(self):
        return self.__fit_metadata

    @property
    @abstractmethod
    def input_dim(self) -> tuple[int, int, int, int]:
        """
        Returns the needed shape for the input data.
        If a dimension is not important, it will return -1 in the specific position.

        Returns:
        (tuple) needed shape for data
        """
        ...

    @property
    @abstractmethod
    def output_dim(self) -> tuple[int, int, int, int]:
        """
        Returns the shape for the output data.
        If a dimension is dependent on the input, it will return -1 in the specific position.

        Returns:
        (tuple) expected output shape for data
        """
        ...

    # @abstractmethod
    def serialize(self, serial_dir: str) -> dict:
        """
        Convert the class into a serialized representation
        """
        # ...
        return {**self.hparams}

    @abstractmethod
    def load(self, params: dict, serial_dir: str) -> None:
        """
        Load from serialized format into an object
        """
        ...
