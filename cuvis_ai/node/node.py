import uuid
from abc import ABC, abstractmethod

from cuvis_ai.utils.numpy import check_array_shape, get_shape_without_batch


class Node(ABC):
    """
    Abstract class for data preprocessing.
    """

    def __init__(self):
        super(Node, self).__init__()
        self.id = f"{type(self).__name__}-{str(uuid.uuid4())}"
        self.__forward_metadata = {}
        self.__fit_metadata = {}
        self.__forward_inputs = {}
        self.__fit_inputs = {}
        self.initialized = False
        self.freezed = False

    @abstractmethod
    def forward(self, X):
        """
        Transform the input data.

        Parameters:
        X (array-like): Input data.

        Returns:
        Transformed data.
        """
        pass

    def check_output_dim(self, X):
        """
        Check that the parameters for the output data data match user
        expectations

        Parameters:
        X (array-like): Input data.

        Returns:
        (Bool) Valid data
        """
        return check_array_shape(get_shape_without_batch(X), self.output_dim)

    def check_input_dim(self, X):
        """
        Check that the parameters for the input data data match user
        expectations

        Parameters:
        X (array-like): Input data.

        Returns:
        (Bool) Valid data
        """
        return check_array_shape(get_shape_without_batch(X), self.input_dim)

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
    def input_dim(self) -> tuple[int, int, int]:
        """
        Returns the needed shape for the input data.
        If a dimension is not important, it will return -1 in the specific position.

        Returns:
        (tuple) needed shape for data
        """
        pass

    @property
    @abstractmethod
    def output_dim(self) -> tuple[int, int, int]:
        """
        Returns the shape for the output data.
        If a dimension is dependent on the input, it will return -1 in the specific position.

        Returns:
        (tuple) expected output shape for data
        """
        pass

    @abstractmethod
    def serialize(self, serial_dir: str) -> dict:
        """
        Convert the class into a serialized representation
        """
        pass

    @abstractmethod
    def load(self, params: dict, serial_dir: str) -> None:
        """
        Load from serialized format into an object
        """
        pass
