from abc import ABC, abstractmethod
from typing import Any


class Preprocessor(ABC):
    """
    Abstract class for data preprocessing.
    """

    @abstractmethod
    def fit(self, X: Any) -> "Preprocessor":
        """
        Fit the preprocessor to the data.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        self
        """
        pass

    @abstractmethod
    def forward(self, X: Any, y: Any | None = None, m: Any = None, **kwargs: Any) -> Any:
        pass


class BaseSupervised(ABC):
    @abstractmethod
    def fit(self, X: Any, Y: Any) -> "BaseSupervised":
        pass

    @abstractmethod
    def forward(self, X: Any, y: Any | None = None, m: Any = None, **kwargs: Any) -> Any:
        pass


class BaseUnsupervised(ABC):
    """Abstract node for all unsupervised classifiers to follow.

    Parameters
    ----------
    ABC : ABC
        Defines node as a base class.
    """

    @abstractmethod
    def fit(self, X: Any) -> "BaseUnsupervised":
        """_summary_

        Parameters
        ----------
        X : Any
            Generic method to initialize a classifier with data.
        """
        pass

    @abstractmethod
    def forward(self, X: Any, y: Any | None = None, m: Any = None, **kwargs: Any) -> Any:
        """Transform

        Parameters
        ----------
        X : Any
            Generic method to pass new data through the unsupervised classifier.

        Returns
        -------
        Any
            Return type and shape must be defined by the implemented child classes.
        """
        pass


class BaseTransformation:
    @abstractmethod
    def forward(self, X: Any, y: Any | None = None, m: Any = None, **kwargs: Any) -> Any:
        pass
