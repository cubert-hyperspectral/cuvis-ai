from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.clustering import KMeansClusterer

pytestmark = pytest.mark.unit


def _three_blob_pixels() -> np.ndarray:
    """Return [N, C] float32 pixels drawn from three well-separated blobs."""
    rng = np.random.default_rng(0)
    centers = np.array(
        [[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0], [20.0, 0.0, 20.0, 0.0]],
        dtype=np.float32,
    )
    blobs = [c + rng.normal(0.0, 0.3, size=(200, 4)) for c in centers]
    return np.concatenate(blobs, axis=0).astype(np.float32)


def _as_cube(pixels: np.ndarray) -> torch.Tensor:
    """Reshape an [N, C] pixel matrix to a [1, N, 1, C] BHWC cube."""
    n, c = pixels.shape
    return torch.tensor(pixels).reshape(1, n, 1, c)


@torch.no_grad()
def test_kmeans_forward_reproduces_sklearn_predict_exactly() -> None:
    from sklearn.cluster import KMeans

    pixels = _three_blob_pixels()
    cube = _as_cube(pixels)

    node = KMeansClusterer(n_clusters=3, random_state=0)
    node.statistical_initialization([{"cube": cube}])
    assert node.is_initialized

    # Pin the node's centroids to a freshly fitted model so forward (nearest
    # centroid) is compared against that exact model's predict labels.
    model = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0).fit(
        pixels
    )
    node.centroids = torch.tensor(model.cluster_centers_, dtype=torch.float32)

    out = node.forward(cube=cube)
    assert out["class_mask"].shape == (1, pixels.shape[0], 1)
    assert out["class_mask"].dtype == torch.int32
    assert out["scores"].shape == (1, pixels.shape[0], 1, 1)
    assert out["scores"].dtype == torch.float32

    node_labels = out["class_mask"].reshape(-1).numpy()
    sklearn_labels = model.predict(pixels)
    assert np.array_equal(node_labels, sklearn_labels)


@torch.no_grad()
def test_kmeans_empty_stream_raises_and_stays_uninitialized() -> None:
    node = KMeansClusterer(n_clusters=3, random_state=0)
    with pytest.raises(RuntimeError):
        node.statistical_initialization([])
    assert not node.is_initialized
    with pytest.raises(RuntimeError):
        node.forward(cube=_as_cube(_three_blob_pixels()))


@torch.no_grad()
def test_kmeans_state_dict_roundtrip_identical() -> None:
    pixels = _three_blob_pixels()
    cube = _as_cube(pixels)

    node = KMeansClusterer(n_clusters=3, random_state=0)
    node.statistical_initialization([{"cube": cube}])
    before = node.forward(cube=cube)

    state = node.state_dict()
    fresh = KMeansClusterer(n_clusters=3, random_state=0)
    fresh.load_state_dict(state)
    assert fresh.is_initialized
    after = fresh.forward(cube=cube)

    assert torch.equal(before["class_mask"], after["class_mask"])
    assert torch.equal(before["scores"], after["scores"])


@torch.no_grad()
def test_mask_restricts_fit_to_foreground_pixels() -> None:
    pixels = _three_blob_pixels()  # 600 pixels
    cube = _as_cube(pixels)  # [1, 600, 1, C]
    n = pixels.shape[0]
    mask = torch.zeros(1, n, 1, dtype=torch.int32)
    mask[0, :150, 0] = 1  # only the first blob is foreground

    node = KMeansClusterer(n_clusters=3, random_state=0)
    # No mask -> all pixels; with a connected mask -> only the foreground rows.
    assert node._collect_pixels([{"cube": cube}]).shape[0] == n
    assert node._collect_pixels([{"cube": cube, "mask": mask}]).shape[0] == 150


@torch.no_grad()
def test_mask_is_declared_as_optional_input() -> None:
    spec = KMeansClusterer.INPUT_SPECS["mask"]
    assert spec.optional is True


@torch.no_grad()
def test_collect_pixels_skips_batches_without_cube_port() -> None:
    cube = _as_cube(_three_blob_pixels())
    node = KMeansClusterer(n_clusters=3, random_state=0)
    pixels = node._collect_pixels([{}, {"cube": cube}, {"other": torch.rand(2, 2)}])
    assert pixels.shape[0] == cube.shape[1]


@torch.no_grad()
def test_collect_pixels_subsamples_to_budget_deterministically() -> None:
    cube = _as_cube(_three_blob_pixels())  # 600 pixels
    node = KMeansClusterer(n_clusters=3, random_state=0, max_fit_pixels=50, fit_seed=7)
    first = node._collect_pixels([{"cube": cube}])
    second = node._collect_pixels([{"cube": cube}])
    assert first.shape == (50, cube.shape[-1])
    # The seeded permutation makes the subsample reproducible.
    assert torch.equal(first, second)
