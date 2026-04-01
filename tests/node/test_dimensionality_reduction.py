"""Tests for PCA nodes."""

from __future__ import annotations

import copy

import pytest
import torch

from cuvis_ai.node.dimensionality_reduction import PCA, TrainablePCA

pytestmark = pytest.mark.unit


@pytest.fixture
def structured_pca_batch() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [2.0, 1.0, 3.0]],
                [[3.0, 4.0, 1.0], [4.0, 2.0, 5.0]],
            ],
            [
                [[0.0, 1.0, 4.0], [1.0, 3.0, 0.0]],
                [[2.0, 0.0, 3.0], [3.0, 5.0, 1.0]],
            ],
        ],
        dtype=torch.float32,
    )


def test_pca_does_not_require_initial_fit() -> None:
    node = PCA(n_components=3)
    assert node.requires_initial_fit is False


def test_pca_forward_returns_expected_shapes(create_test_cube) -> None:
    node = PCA(n_components=3)
    data, _ = create_test_cube(
        batch_size=2,
        height=4,
        width=5,
        num_channels=6,
        mode="synthetic",
        seed=7,
        dtype=torch.float32,
    )

    outputs = node.forward(data)

    assert outputs["projected"].shape == torch.Size([2, 4, 5, 3])
    assert outputs["explained_variance_ratio"].shape == torch.Size([3])
    assert outputs["components"].shape == torch.Size([3, 6])


def test_pca_variance_ratio_sums_to_one(create_test_cube) -> None:
    node = PCA(n_components=3)
    data, _ = create_test_cube(
        batch_size=2,
        height=4,
        width=5,
        num_channels=6,
        mode="synthetic",
        seed=11,
        dtype=torch.float32,
    )
    outputs = node.forward(data)
    ratio = outputs["explained_variance_ratio"]

    assert torch.all(ratio >= 0)
    assert torch.isclose(ratio.sum(), torch.tensor(1.0), atol=1e-5)


def test_pca_uses_last_frame_for_auxiliary_outputs(structured_pca_batch: torch.Tensor) -> None:
    node = PCA(n_components=2)
    outputs = node.forward(structured_pca_batch)
    expected_components, mean, eigenvalues = node._fit_frame(structured_pca_batch[-1])
    expected_projected = node._project(
        structured_pca_batch[-1].reshape(-1, structured_pca_batch.shape[-1]),
        mean,
        expected_components,
    ).reshape(
        structured_pca_batch.shape[1],
        structured_pca_batch.shape[2],
        node.n_components,
    )

    assert torch.allclose(outputs["components"], expected_components, atol=1e-5)
    assert torch.allclose(
        outputs["explained_variance_ratio"],
        node._variance_ratio(eigenvalues),
        atol=1e-5,
    )
    assert torch.allclose(outputs["projected"][-1], expected_projected, atol=1e-5)


def test_trainable_pca_still_requires_initial_fit() -> None:
    node = TrainablePCA(num_channels=5, n_components=3)
    assert node.requires_initial_fit is True


def test_trainable_pca_initializes_and_projects_with_shared_helpers(
    trainable_pca: TrainablePCA,
    create_test_cube,
) -> None:
    node = copy.deepcopy(trainable_pca)
    data, _ = create_test_cube(
        batch_size=2,
        height=4,
        width=4,
        num_channels=5,
        mode="synthetic",
        seed=19,
        dtype=torch.float32,
    )
    outputs = node.forward(data)
    expected = node._project(data.reshape(-1, 5), node._mean, node._components).reshape(2, 4, 4, 3)

    assert node._statistically_initialized is True
    assert outputs["projected"].shape == torch.Size([2, 4, 4, 3])
    assert torch.allclose(outputs["projected"], expected, atol=1e-5)
    assert torch.allclose(outputs["components"], node._components, atol=1e-5)
    assert torch.isclose(outputs["explained_variance_ratio"].sum(), torch.tensor(1.0), atol=1e-5)
