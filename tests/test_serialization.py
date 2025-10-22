from __future__ import annotations

import numpy as np
import pytest

from cuvis_ai.preprocessor import NMF, PCA
from cuvis_ai.supervised import LDA
from cuvis_ai.unsupervised import GMM, KMeans, MeanShift
from cuvis_ai.utils.serializer import YamlSerializer
from cuvis_ai.utils.test import get_np_dummy_data

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
