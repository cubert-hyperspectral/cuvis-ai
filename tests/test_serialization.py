from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.preprocessor import NMF, PCA
from cuvis_ai.supervised import LDA
from cuvis_ai.unsupervised import GMM, KMeans, MeanShift
from cuvis_ai.utils.serializer import YamlSerializer
from cuvis_ai.utils.test import get_np_dummy_data
from cuvis_ai.anomoly.rx_v2 import RXPerBatch

TYPES_TO_COMPARE = (int, float, str, bool, list, tuple, np.ndarray)
N_COMPONENTS = 3
CLUSTERS = 3
BATCH_SHAPE = (4, 6, 4, 5)
SAMPLE_SHAPE = (6, 4, 5)
LABEL_SHAPE = (6, 4, 1)


def _fit_preprocessor(node):
    node.fit(get_np_dummy_data(BATCH_SHAPE))
    return node


def _fit_supervised(node):
    data = get_np_dummy_data(SAMPLE_SHAPE)
    labels = np.where(get_np_dummy_data(LABEL_SHAPE) > 0.5, 1, 0)
    node.fit(data, labels)
    return node


NODE_FACTORIES = [
    pytest.param(lambda: _fit_preprocessor(PCA(N_COMPONENTS)), id="pca"),
    pytest.param(
        lambda: _fit_preprocessor(NMF(N_COMPONENTS, max_iter=50, tol=1e-2)),
        id="nmf",
    ),
    pytest.param(lambda: _fit_preprocessor(KMeans(CLUSTERS)), id="kmeans"),
    pytest.param(
        lambda: _fit_preprocessor(GMM(CLUSTERS, max_iter=50, random_state=0)),
        id="gmm",
    ),
    pytest.param(lambda: _fit_preprocessor(MeanShift()), id="mean_shift"),
    pytest.param(lambda: _fit_supervised(LDA()), id="lda"),
]


@pytest.fixture(params=NODE_FACTORIES)
def trained_node(request):
    return request.param()


def test_serialization_roundtrip(trained_node, tmp_path):
    """Serialize a node to disk and ensure loading reproduces key state."""
    test_dir = tmp_path / "serde"
    test_dir.mkdir()

    params = trained_node.serialize(str(test_dir))

    serializer = YamlSerializer(test_dir, "test_node")
    serializer.serialize(params)

    reloaded = trained_node.__class__()
    reloaded.id = trained_node.id
    reloaded.load(serializer.load(), str(test_dir))

    mismatches = []

    for attr, value in reloaded.__dict__.items():
        if not isinstance(value, TYPES_TO_COMPARE):
            continue
        original = getattr(trained_node, attr)

        if isinstance(value, np.ndarray) or isinstance(original, np.ndarray):
            if not np.array_equal(value, original):
                mismatches.append(attr)
        elif value != original:
            mismatches.append(attr)

    assert not mismatches, f"Attributes differed after roundtrip: {mismatches}"


def test_node_hparams_auto_population():
    """Test that Node subclasses automatically populate hparams from init parameters."""
    # Test with a single parameter
    eps_value = 1e-99
    model = RXPerBatch(eps=eps_value)
    
    assert hasattr(model, "hparams"), "Node should have hparams attribute"
    assert "eps" in model.hparams, "hparams should contain eps parameter"
    assert model.hparams["eps"] == eps_value, f"Expected eps={eps_value}, got {model.hparams['eps']}"
    
    # Test with default parameter
    model_default = RXPerBatch()
    assert hasattr(model_default, "hparams"), "Node with defaults should have hparams attribute"
    assert "eps" in model_default.hparams, "hparams should contain eps parameter with default"
    assert model_default.hparams["eps"] == 1e-6, "Default eps value should be 1e-6"


def test_node_hparams_persistence():
    """Test that hparams are preserved and can be used for node recreation."""
    # Create a node with specific parameters
    eps_value = 1e-8
    original_node = RXPerBatch(eps=eps_value)
    
    # Verify hparams
    assert original_node.hparams == {"eps": eps_value}
    
    # Create a new node with the same hparams
    recreated_node = RXPerBatch(**original_node.hparams)
    
    # Verify the recreated node has the same hyperparameters
    assert recreated_node.hparams == original_node.hparams
    assert recreated_node.eps == original_node.eps
