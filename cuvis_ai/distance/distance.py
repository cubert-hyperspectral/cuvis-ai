import warnings
from abc import abstractmethod
from copy import deepcopy

import numpy as np
import yaml

from cuvis_ai.node import CubeConsumer, Node
from cuvis_ai.utils.numpy import flatten_spatial


class AbstractDistance(Node, CubeConsumer):
    """
    Abstract class for data preprocessing.

    There are two ways spectral distance can be used

    1) Distance relative to a reference spectra, which returns a single scored image
    2) Distance relative to multiple reference spectral, which returns an array of scored images
    (This can then be passed into a decider algorithm)
    # TODO: Should this behavior be allowed? This might break the notion of input/output dimensions
    - If the spectra are known, they can be stored with the object initially

    Parameters
    ----------
    Node : Node
        Defines the distance measure as a type of node.
    """

    def __init__(self, ref_spectra: list = []):
        """Initialize distance metric

        Parameters
        ----------
        ref_spectra : list, optional
            List of reference spectra to compare against, by default []
        """
        super().__init__()
        # Assign any reference spectra, if they exist
        self.ref_spectra = self.spectra_to_array(ref_spectra)

    @staticmethod
    def spectra_to_array(ref_spectra: np.ndarray | list) -> np.ndarray:
        """Convert list of spectra to a numpy array

        Parameters
        ----------
        ref_spectra : np.ndarray | list
            Object of reference spectra

        Returns
        -------
        np.ndarray
            Spectra stored in singular, indexable, sequential array.
        """
        if type(ref_spectra) == list:
            # Cast to a numpy
            ref_spectra = np.array(ref_spectra)
            if ref_spectra.ndim == 1:
                ref_spectra = np.reshape(ref_spectra, (1, ref_spectra.shape[-1]))
        if ref_spectra.shape == 1:
            # Squeeze an extra dimension
            ref_spectra = np.expand_dims(ref_spectra, axis=0)
        return ref_spectra

    def fit(self, X):
        self.initialized = True
        pass

    def forward(self, X: np.ndarray, ref_spectra: list = None) -> np.ndarray:
        """Pass the data through comparative function

        Parameters
        ----------
        X : np.ndarray
            Input data.
        ref_spectra : list, optional
            List of spectra to compare against

        Returns
        -------
        np.ndarray
            Distance maps for each of the reference spectra.

        Raises
        ------
        ValueError
            Mismatch in input data and reference spectra provided on function call.
        ValueError
            Mismatch in input data and reference spectra provided on node initialization.
        ValueError
            No reference spectra provided in init or on forward function pass.
        """
        ref = self.ref_spectra if ref_spectra is None else ref_spectra

        # Default behavior is to use the ref_spectra passed to the function
        if len(ref) > 0:
            if X.shape[-1] != ref.shape[-1]:
                raise ValueError("Mismatch in input data and reference spectra!")
            # Process and return
            res = self.score(flatten_spatial(X), ref)
            return res.reshape(*X.shape[:-1], ref.shape[0])
        else:
            raise ValueError("No reference spectra provided!")

    def serialize(self, working_dir: str) -> str:
        """Convert distance node to serializable format

        Parameters
        ----------
        working_dir : str
            Directory where node metadata should be saved.

        Returns
        -------
        str
            YAML parameterization of node.
        """
        data = deepcopy(self.__dict__)
        data["type"] = type(self).__name__
        data["ref_spectra"] = data["ref_spectra"].tolist()
        # Dump to a string
        return yaml.dump(data, default_flow_style=False)

    def load(self, params: dict, filepath: str = None):
        """Load dumped parameters to recreate the distance object

        Parameters
        ----------
        params : dict
           Dictionary containing node values
        filepath : str, optional
            Directory containing node metadata, by default None
        """
        # Delete the type param
        del params["type"]
        self.__dict__ = params
        # Cast reference spectra back to numpy type
        self.ref_spectra = np.array(self.ref_spectra)

    @abstractmethod
    def score(self, data: np.ndarray, X: np.ndarray):
        """Abstract distance method implemented by every distance type

        Parameters
        ----------
        data : np.ndarray
            Current data to compare.
        X : np.ndarray
            Reference to compare data against.
        """
        pass

    @Node.input_dim.getter
    def input_dim(self) -> list:
        """Get required input dimension

        Returns
        -------
        list
            List defining which input dimensions should be checked in graph.
        """
        # Note: This denotes by default we don't care about the input image shape
        # so long as the wavelengths match
        if len(self.ref_spectra) > 0:
            return [-1, -1, self.ref_spectra.shape[1]]
        else:
            return [-1, -1, -1]

    @Node.output_dim.getter
    def output_dim(self):
        """Get required output dimension

        Returns
        -------
        list
            List defining which input dimensions should be checked in graph.
        """
        # Note: Output can be arbitrary length depending on the reference spectra
        return [-1, -1, 1]


class SpectralAngle(AbstractDistance):
    """Cosine distance between spectra according to the Spectral Angle Mapper (SAM) formula.

    Nota Bene: Measurements should be normalized as large values skews this calculation towards π/2.

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type
    """

    def __init__(self, ref_spectra: list = []):
        """Construct SAM

        Parameters
        ----------
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against.

        Returns
        -------
        np.ndarray
            Distance scores.
        """
        # Throw a warning for a large number of unnormalized values
        if np.percentile(data, 90) > 2.0:
            # 10% of the data exceeds 200%
            warnings.warn(
                "Spectral angle mapper is being used without properly normalized data. Unexpected behavior may occur!"
            )

        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores.append(
                np.arccos(
                    np.dot(data, ref_spectra[idx, :])
                    / (np.linalg.norm(data, axis=-1) * np.linalg.norm(ref_spectra[idx]))
                )
            )

        output_scores = np.stack(output_scores, axis=-1)
        # print(f"Output scores shape: {output_scores.shape}, min: {output_scores.min()}, max: {output_scores.max()}")
        return output_scores


class Euclidean(AbstractDistance):
    """Calculate L2 (Euclidean) Distance.

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type.
    """

    def __init__(self, ref_spectra: list = []):
        """Construct Euclidean distance node.

        Parameters
        ----------
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against

        Returns
        -------
        np.ndarray
            Distance scores.
        """
        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            delta = data - ref_spectra[idx, :]
            score = np.sqrt(np.sum(delta**2, axis=-1))
            output_scores.append(score)

        output_scores = np.stack(output_scores, axis=-1)
        return output_scores


class Manhattan(AbstractDistance):
    """Calculate L1 (Manhattan) Distance.

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type.
    """

    def __init__(self, ref_spectra: list = []):
        """Construct Manhattan distance node.

        Parameters
        ----------
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against.

        Returns
        -------
        np.ndarray
            Distance scores
        """

        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores.append(np.sum(np.abs(data - ref_spectra[idx, :]), axis=-1))

        output_scores = np.stack(output_scores, axis=-1)
        return output_scores


class Canberra(AbstractDistance):
    """Calculate  Weighted L1 (Canberra) Distance.

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type
    """

    def __init__(self, ref_spectra: list = []):
        """Construct Canberra distance node.

        Parameters
        ----------
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)

    @staticmethod
    def score(data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against.

        Returns
        -------
        np.ndarray
            Distance scores
        """
        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores.append(
                np.sum(
                    np.abs(data - ref_spectra[idx, :])
                    / (np.abs(data) + np.abs(ref_spectra[idx, :])),
                    axis=-1,
                )
            )
        output_scores = np.stack(output_scores, axis=-1)
        return output_scores


class Minkowski(AbstractDistance):
    """N-th degree normed vector space (Minkowski) Distance.

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type
    """

    def __init__(self, degree: int, ref_spectra: list = []):
        """Construct Minkowski distance node.

        Parameters
        ----------
        degree : int
            Order of Minkowski distance
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)
        self.degree = degree

    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against.

        Returns
        -------
        np.ndarray
            Distance scores
        """
        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores.append(
                (np.sum((data - ref_spectra[idx, :]) ** self.degree, axis=-1))
                ** (1.0 / float(self.degree))
            )
        output_scores = np.stack(output_scores, axis=-1)
        return output_scores


class GFC(AbstractDistance):
    """Goodness-of-fit Coefficient (GFC)

    Citation:

    Hernández-Andrés, J., Romero, J., García-Beltrán, A., & Nieves, J. L. (1998). Testing linear models on spectral daylight measurements. Applied Optics, 37(6), 971-977.

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type
    """

    def __init__(self, ref_spectra: list = []):
        """Construct GFC distance node.

        Parameters
        ----------
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)

    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against.

        Returns
        -------
        np.ndarray
            Distance scores
        """
        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores.append(
                1.0
                - (
                    np.dot(data, ref_spectra[idx, :])
                    / (np.linalg.norm(data, axis=-1) * np.linalg.norm(ref_spectra[idx]))
                )
            )
        output_scores = np.stack(output_scores, axis=-1)
        return output_scores


class ECS(AbstractDistance):
    """Euclidean Distance of Cumulative Spectrum (ECS)

    Parameters
    ----------
    AbstractDistance : AbstractDistance
        Defines the node as AbstractDistance node type
    """

    def __init__(self, wavelengths: np.ndarray | list, ref_spectra: list = []):
        """Initialize an ECS distance node.

        Parameters
        ----------
        wavelengths : np.ndarray | list
            Array defining positioning of wavelength channels (typically given in nm).
            This length of this vector must equal the number of channels in inputted datacubes.
        ref_spectra : list, optional
            Reference spectra to compare against, by default []
        """
        super().__init__(ref_spectra)
        # Cast this to a list, necessary for serialization
        self.wavelengths = list(wavelengths)

    @Node.input_dim.getter
    def input_dim(self) -> list:
        """Return the required input dimension

        Returns
        -------
        list
            Required input shape, which can vary in datacube height and width, but must be of consistent channel size.
        """
        # Note: this function depends on the wavelengths, so we need to match the dimension
        return [-1, -1, len(self.wavelengths)]

    def score(self, data: np.ndarray, ref_spectra: np.ndarray) -> np.ndarray:
        """Score new datacubes against reference spectra.

        Parameters
        ----------
        data : np.ndarray
            Input data.
        ref_spectra : np.ndarray
            Reference spectra to compare against.

        Returns
        -------
        np.ndarray
            Distance scores
        """
        output_scores = []
        for idx in range(ref_spectra.shape[0]):
            # Calculate the distances
            output_scores.append(
                np.sqrt(
                    (
                        np.trapz(data, self.wavelengths, axis=-1)
                        - np.trapz(ref_spectra[idx, :], self.wavelengths)
                    )
                    ** 2
                )
            )
        output_scores = np.stack(output_scores, axis=-1)
        return output_scores
