"""Mock nodes for testing serialization patterns."""

import pytest
import torch

from cuvis_ai.node.dimensionality_reduction import TrainablePCA

# This file previously contained MockStatisticalTrainableNode which was unused
# and has been removed. If additional mock nodes are needed for testing,
# they should be added here.


@pytest.fixture(scope="session")
def trainable_pca():
    """Create a TrainablePCA node for testing.

    Creates a PCA node initialized with 5-channel dummy data.
    The node is initialized and unfrozen for gradient training.

    Returns
    -------
    TrainablePCA
        Initialized and unfrozen PCA node with 3 components
    """
    pca = TrainablePCA(num_channels=5, n_components=3)

    # Initialize with dummy data (using port-based dict format)
    data_iterator = ({"data": torch.randn(2, 10, 10, 5)} for _ in range(3))
    pca.statistical_initialization(data_iterator)
    pca.unfreeze()  # Convert buffers to parameters for gradient training

    return pca
