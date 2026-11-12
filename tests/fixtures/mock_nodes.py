"""Mock nodes for testing serialization patterns."""

import pytest
import torch
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from cuvis_ai_core.node import Node

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


class SeededTensorSource(Node):
    """Source node that emits one fixed random tensor on ``data``.

    For pipeline save and reload tests: the tensor is ``torch.randn(shape)`` from a
    generator seeded with ``seed``, so the pipeline rebuilt from YAML emits the same
    values. A saved pipeline stores the class path, which is why the class lives in an
    importable module and takes its shape as a constructor argument.

    Parameters
    ----------
    shape : list[int]
        Shape of the emitted tensor, ``[B, H, W, C]``.
    seed : int, optional
        Seed of the generator (default ``0``).
    """

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def __init__(self, shape: list[int], seed: int = 0, **kwargs) -> None:
        self.shape = list(shape)
        self.seed = int(seed)
        super().__init__(shape=self.shape, seed=self.seed, **kwargs)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        """Return the seeded tensor."""
        generator = torch.Generator().manual_seed(self.seed)
        return {"data": torch.randn(self.shape, generator=generator)}
