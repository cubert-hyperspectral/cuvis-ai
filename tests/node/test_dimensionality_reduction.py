"""Tests for PCA nodes."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA as SklearnPCA

from cuvis_ai.node.dimensionality_reduction import PCA, TrainablePCA

pytestmark = pytest.mark.unit

N_CHANNELS = 6
N_COMPONENTS = 3


def _correlated_cubes(seed: int = 0, n_batches: int = 3) -> list[torch.Tensor]:
    """Seeded float32 cubes [2, 6, 5, C] with a well separated spectrum and a non-zero mean."""
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(N_CHANNELS, N_CHANNELS)))[0]
    scales = np.array([9.0, 4.0, 2.0, 0.3, 0.2, 0.1])
    mean = rng.normal(size=N_CHANNELS) * 2.0
    cubes = []
    for _ in range(n_batches):
        latent = rng.normal(size=(2 * 6 * 5, N_CHANNELS)) * scales
        cube = (latent @ basis.T + mean).reshape(2, 6, 5, N_CHANNELS)
        cubes.append(torch.from_numpy(cube.astype(np.float32)))
    return cubes


def _pixels(cubes: list[torch.Tensor]) -> np.ndarray:
    """All pixels of the cubes as one float64 [N, C] matrix, the form sklearn fits on."""
    return np.concatenate([c.numpy().reshape(-1, c.shape[-1]) for c in cubes]).astype(np.float64)


def _fitted_pca(cubes: list[torch.Tensor], whiten: bool = False) -> TrainablePCA:
    node = TrainablePCA(num_channels=N_CHANNELS, n_components=N_COMPONENTS, whiten=whiten)
    node.statistical_initialization({"data": cube} for cube in cubes)
    return node


def _signs(components: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Per-component sign that aligns ``components`` with ``reference`` (eigenvectors are signless)."""
    return np.sign(np.sum(components * reference, axis=1))


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


def test_trainable_pca_matches_sklearn_pca() -> None:
    cubes = _correlated_cubes()
    node = _fitted_pca(cubes)
    reference = SklearnPCA(n_components=N_COMPONENTS, svd_solver="full").fit(_pixels(cubes))

    components = node._components.numpy()
    signs = _signs(components, reference.components_)
    assert np.allclose(node._mean.numpy(), reference.mean_, atol=1e-5)
    assert np.allclose(node._explained_variance.numpy(), reference.explained_variance_, rtol=1e-4)
    assert np.allclose(components * signs[:, None], reference.components_, atol=1e-4)

    outputs = node.forward(cubes[0])
    projected = outputs["projected"].numpy().reshape(-1, N_COMPONENTS) * signs
    assert np.allclose(projected, reference.transform(_pixels(cubes[:1])), rtol=1e-4, atol=1e-3)
    # The node's ratio is taken over the retained components and sums to 1; sklearn's
    # explained_variance_ratio_ divides by the total variance of all channels instead.
    retained = reference.explained_variance_ / reference.explained_variance_.sum()
    assert np.allclose(outputs["explained_variance_ratio"].numpy(), retained, atol=1e-5)


def test_trainable_pca_whitened_projection_matches_sklearn() -> None:
    cubes = _correlated_cubes(seed=1)
    node = _fitted_pca(cubes, whiten=True)
    reference = SklearnPCA(n_components=N_COMPONENTS, svd_solver="full", whiten=True).fit(
        _pixels(cubes)
    )

    signs = _signs(node._components.numpy(), reference.components_)
    projected = node.forward(cubes[0])["projected"].numpy().reshape(-1, N_COMPONENTS) * signs
    assert np.allclose(projected, reference.transform(_pixels(cubes[:1])), rtol=1e-4, atol=1e-3)


@pytest.mark.parametrize("whiten", [False, True])
def test_trainable_pca_reconstructed_matches_sklearn_inverse_transform(whiten: bool) -> None:
    cubes = _correlated_cubes(seed=2)
    node = _fitted_pca(cubes, whiten=whiten)
    reference = SklearnPCA(n_components=N_COMPONENTS, svd_solver="full").fit(_pixels(cubes))

    reconstructed = node.forward(cubes[0])["reconstructed"]
    pixels = _pixels(cubes[:1])
    expected = reference.inverse_transform(reference.transform(pixels)).reshape(cubes[0].shape)

    assert reconstructed.shape == cubes[0].shape
    assert reconstructed.dtype == torch.float32
    assert np.allclose(reconstructed.numpy(), expected, rtol=1e-4, atol=1e-3)


def test_trainable_pca_reconstructed_is_an_optional_output_port() -> None:
    spec = TrainablePCA.OUTPUT_SPECS["reconstructed"]
    assert spec.optional is True
    assert spec.dtype == torch.float32
    assert tuple(spec.shape) == (-1, -1, -1, -1)
    assert "reconstructed" not in PCA.OUTPUT_SPECS


@pytest.mark.parametrize("whiten", [False, True])
def test_trainable_pca_existing_outputs_keep_their_formula(whiten: bool) -> None:
    """Adding ``reconstructed`` leaves the three existing outputs as they were."""
    cubes = _correlated_cubes(seed=3)
    node = _fitted_pca(cubes, whiten=whiten)
    cube = cubes[1]
    outputs = node.forward(cube)

    expected = (cube.reshape(-1, N_CHANNELS) - node._mean) @ node._components.T
    if whiten:
        expected = expected * (1.0 / torch.sqrt(node._explained_variance + node.eps))
    ratio = node._explained_variance / (node._explained_variance.sum() + node.eps)

    assert torch.allclose(outputs["projected"].reshape(-1, N_COMPONENTS), expected, atol=1e-6)
    assert torch.equal(outputs["components"], node._components)
    assert torch.allclose(outputs["explained_variance_ratio"], ratio, atol=1e-7)


def test_pca_per_frame_matches_sklearn_on_one_frame() -> None:
    frame = _correlated_cubes(seed=4, n_batches=1)[0][:1]
    node = PCA(n_components=N_COMPONENTS)
    reference = SklearnPCA(n_components=N_COMPONENTS, svd_solver="full").fit(_pixels([frame]))

    outputs = node.forward(frame)
    signs = _signs(outputs["components"].numpy(), reference.components_)
    projected = outputs["projected"].numpy().reshape(-1, N_COMPONENTS) * signs
    assert np.allclose(projected, reference.transform(_pixels([frame])), rtol=1e-4, atol=1e-3)


@pytest.mark.gpu
def test_trainable_pca_fits_over_two_cuda_batches() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cubes = [cube.to("cuda") for cube in _correlated_cubes(seed=5, n_batches=2)]
    node = TrainablePCA(num_channels=N_CHANNELS, n_components=N_COMPONENTS).to("cuda")
    node.statistical_initialization({"data": cube} for cube in cubes)

    outputs = node.forward(cubes[0])
    assert outputs["projected"].device.type == "cuda"
    assert outputs["reconstructed"].device.type == "cuda"
