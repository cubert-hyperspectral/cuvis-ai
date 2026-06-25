from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.clustering import GaussianMixtureClusterer

pytestmark = pytest.mark.unit


def _three_gaussian_blob_pixels() -> np.ndarray:
    """Return [N, C] float32 pixels from three well-separated Gaussian blobs."""
    rng = np.random.default_rng(0)
    centers = np.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0], [20.0, 0.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    blobs = [c + rng.normal(0.0, 0.4, size=(200, 4)) for c in centers]
    return np.concatenate(blobs, axis=0).astype(np.float32)


def _as_cube(pixels: np.ndarray) -> torch.Tensor:
    """Reshape an [N, C] pixel matrix to a [1, N, 1, C] BHWC cube."""
    n, c = pixels.shape
    return torch.tensor(pixels).reshape(1, n, 1, c)


@torch.no_grad()
def test_gmm_forward_matches_sklearn_predict_proba_and_score() -> None:
    from sklearn.mixture import GaussianMixture

    pixels = _three_gaussian_blob_pixels()
    n = pixels.shape[0]
    cube = _as_cube(pixels)

    node = GaussianMixtureClusterer(n_components=3, random_state=0)
    node.statistical_initialization([{"cube": cube}])
    assert node.is_initialized

    # Pin the node's parameters to a freshly fitted model so the closed-form
    # torch forward is compared against that exact model.
    model = GaussianMixture(
        n_components=3,
        covariance_type="full",
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        random_state=0,
    ).fit(pixels)
    node.means = torch.tensor(model.means_, dtype=torch.float32)
    node.weights = torch.tensor(model.weights_, dtype=torch.float32)
    node.precisions_chol = torch.tensor(model.precisions_cholesky_, dtype=torch.float32)

    out = node.forward(cube=cube)
    assert out["class_mask"].shape == (1, n, 1)
    assert out["class_mask"].dtype == torch.int32
    assert out["abundances"].shape == (1, n, 1, 3)
    assert out["abundances"].dtype == torch.float32
    assert out["scores"].shape == (1, n, 1, 1)
    assert out["scores"].dtype == torch.float32

    node_labels = out["class_mask"].reshape(-1).numpy()
    assert np.array_equal(node_labels, model.predict(pixels))

    node_proba = out["abundances"].reshape(n, 3).numpy()
    assert np.allclose(node_proba, model.predict_proba(pixels), atol=1e-4)
    # Responsibilities sum to 1 over components.
    assert np.allclose(node_proba.sum(axis=1), 1.0, atol=1e-5)

    node_scores = out["scores"].reshape(n).numpy()
    assert np.allclose(node_scores, model.score_samples(pixels), atol=1e-4)


@torch.no_grad()
def test_gmm_empty_stream_raises_and_stays_uninitialized() -> None:
    node = GaussianMixtureClusterer(n_components=3, random_state=0)
    with pytest.raises(RuntimeError):
        node.statistical_initialization([])
    assert not node.is_initialized
    with pytest.raises(RuntimeError):
        node.forward(cube=_as_cube(_three_gaussian_blob_pixels()))


@torch.no_grad()
def test_gmm_state_dict_roundtrip_identical() -> None:
    pixels = _three_gaussian_blob_pixels()
    cube = _as_cube(pixels)

    node = GaussianMixtureClusterer(n_components=3, random_state=0)
    node.statistical_initialization([{"cube": cube}])
    before = node.forward(cube=cube)

    state = node.state_dict()
    fresh = GaussianMixtureClusterer(n_components=3, random_state=0)
    fresh.load_state_dict(state)
    assert fresh.is_initialized
    after = fresh.forward(cube=cube)

    assert torch.equal(before["class_mask"], after["class_mask"])
    assert torch.equal(before["abundances"], after["abundances"])
    assert torch.equal(before["scores"], after["scores"])
