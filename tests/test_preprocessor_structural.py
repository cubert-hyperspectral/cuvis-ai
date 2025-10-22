import pytest

from cuvis_ai.preprocessor import NMF, PCA
from cuvis_ai.utils.test import get_np_dummy_data

N_COMPONENTS = 3
BATCH_SHAPE = (4, 6, 4, 5)
SAMPLE_SHAPE = (6, 4, 5)


@pytest.fixture
def batch_data():
    return get_np_dummy_data(BATCH_SHAPE)


@pytest.fixture(
    params=[
        pytest.param((PCA, {"n_components": N_COMPONENTS}), id="pca"),
        pytest.param(
            (NMF, {"n_components": N_COMPONENTS, "max_iter": 50, "tol": 1e-2}),
            id="nmf",
        ),
    ]
)
def fitted_node(request, batch_data):
    node_cls, kwargs = request.param
    node = node_cls(**kwargs)
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
    assert not fitted_node.check_input_dim((6, 4, 4))


def test_correct_output_dim(fitted_node):
    assert fitted_node.check_output_dim((6, 4, N_COMPONENTS))


def test_passthrough(fitted_node, batch_data):
    """Forward pass should preserve batch dimension and output components."""
    output = fitted_node.forward(batch_data)
    assert output.shape == (BATCH_SHAPE[0], BATCH_SHAPE[1], BATCH_SHAPE[2], N_COMPONENTS)

    no_batch_data = get_np_dummy_data(SAMPLE_SHAPE)
    output = fitted_node.forward(no_batch_data)
    assert output.shape == (SAMPLE_SHAPE[0], SAMPLE_SHAPE[1], N_COMPONENTS)
