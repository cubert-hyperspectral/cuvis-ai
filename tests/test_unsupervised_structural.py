import pytest

from cuvis_ai.unsupervised import GMM, KMeans, MeanShift
from cuvis_ai.utils.test import get_np_dummy_data

CLUSTERS = 3
BATCH_SHAPE = (4, 6, 4, 5)
SAMPLE_SHAPE = (6, 4, 5)
OUTPUT_SHAPE = (SAMPLE_SHAPE[0], SAMPLE_SHAPE[1], 1)


@pytest.fixture
def batch_data():
    return get_np_dummy_data(BATCH_SHAPE)


@pytest.fixture(
    params=[
        pytest.param(lambda: KMeans(CLUSTERS), id="kmeans"),
        pytest.param(lambda: GMM(CLUSTERS, max_iter=50, random_state=0), id="gmm"),
        pytest.param(lambda: MeanShift(), id="mean_shift"),
    ]
)
def fitted_node(request, batch_data):
    node = request.param()
    node.fit(batch_data)
    return node


def test_initialization(fitted_node):
    assert fitted_node.initialized
    assert fitted_node.input_dim
    assert fitted_node.output_dim


def test_correct_input_dim(fitted_node, batch_data):
    assert fitted_node.check_input_dim(SAMPLE_SHAPE)
    assert fitted_node.check_input_dim(batch_data)
    smaller_data = get_np_dummy_data(SAMPLE_SHAPE)
    assert fitted_node.check_input_dim(smaller_data)


def test_incorrect_input_dim(fitted_node):
    assert not fitted_node.check_input_dim((SAMPLE_SHAPE[0], SAMPLE_SHAPE[1], 2))


def test_correct_output_dim(fitted_node):
    assert fitted_node.check_output_dim(OUTPUT_SHAPE)


def test_passthrough(fitted_node, batch_data):
    """Forward pass should produce cluster indices in the last dimension."""
    output = fitted_node.forward(batch_data)
    assert output.shape == (BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2], 1)

    no_batch_data = get_np_dummy_data(SAMPLE_SHAPE)
    output = fitted_node.forward(no_batch_data)
    assert output.shape == OUTPUT_SHAPE
