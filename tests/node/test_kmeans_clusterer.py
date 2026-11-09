from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.clustering import KMeansClusterer
from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

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


def _positive_spectra_pixels() -> np.ndarray:
    """Return [N, C] float32 pixels from three positive spectra of different shape.

    Spectral angles need positive spectra: ``SpectralAngleMapper`` divides each spectrum by its
    mean, which flips or blows up a spectrum whose mean is zero or negative.
    """
    rng = np.random.default_rng(1)
    shapes = np.array(
        [[10.0, 20.0, 30.0, 40.0], [40.0, 30.0, 20.0, 10.0], [20.0, 50.0, 20.0, 50.0]],
        dtype=np.float32,
    )
    blobs = [c + rng.normal(0.0, 1.0, size=(200, 4)) for c in shapes]
    return np.concatenate(blobs, axis=0).astype(np.float32)


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
    # No fit mask -> all pixels; with a connected fit mask -> only the foreground rows.
    assert node._collect_pixels([{"cube": cube}]).shape[0] == n
    assert node._collect_pixels([{"cube": cube, "fit_mask": mask}]).shape[0] == 150


@torch.no_grad()
def test_fit_mask_is_declared_as_optional_input() -> None:
    spec = KMeansClusterer.INPUT_SPECS["fit_mask"]
    assert spec.optional is True


def test_no_input_port_is_named_like_the_loaders_annotation_mask() -> None:
    """A pipeline fills every unconnected input whose name is a batch key; the cu3s loaders send
    their annotation mask as ``mask`` (all zero on normal frames), so a port of that name would
    silently restrict the fit to annotated pixels."""
    assert "mask" not in KMeansClusterer.INPUT_SPECS


@torch.no_grad()
def test_an_annotation_mask_in_the_batch_does_not_restrict_the_fit() -> None:
    pixels = _three_blob_pixels()
    cube = _as_cube(pixels)
    zeros = torch.zeros(
        1, pixels.shape[0], 1, dtype=torch.int32
    )  # a normal frame's annotation mask
    node = KMeansClusterer(n_clusters=3, random_state=0)
    assert node._collect_pixels([{"cube": cube, "mask": zeros}]).shape[0] == pixels.shape[0]


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


# ----- spectral_signature output port (the centroids) --------------------------------------------


def _fitted(n_clusters: int = 3) -> KMeansClusterer:
    node = KMeansClusterer(n_clusters=n_clusters, random_state=0)
    node.statistical_initialization([{"cube": _as_cube(_three_blob_pixels())}])
    return node


@torch.no_grad()
def test_centroids_port_matches_the_fitted_buffer_reshaped() -> None:
    node = _fitted(n_clusters=3)
    cube = _as_cube(_three_blob_pixels())
    out = node.forward(cube=cube)
    k, c = node.centroids.shape
    assert out["spectral_signature"].shape == (k, 1, 1, c)
    assert out["spectral_signature"].dtype == torch.float32
    assert torch.equal(out["spectral_signature"], node.centroids.reshape(k, 1, 1, c))


@torch.no_grad()
def test_centroids_port_is_unaffected_by_batch_size() -> None:
    node = _fitted(n_clusters=3)
    pixels = _three_blob_pixels()
    cube_b1 = _as_cube(pixels)  # [1, N, 1, C]
    cube_b2 = torch.cat([cube_b1, cube_b1], dim=0)  # [2, N, 1, C]

    out_b1 = node.forward(cube=cube_b1)
    out_b2 = node.forward(cube=cube_b2)

    k, c = node.centroids.shape
    assert out_b2["spectral_signature"].shape == (k, 1, 1, c)
    assert torch.equal(out_b1["spectral_signature"], out_b2["spectral_signature"])


def test_spectral_signature_is_declared_in_output_specs() -> None:
    spec = KMeansClusterer.OUTPUT_SPECS["spectral_signature"]
    assert spec.dtype == torch.float32
    assert spec.shape == (-1, 1, 1, -1)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@torch.no_grad()
def test_centroids_follow_the_cube_to_cuda() -> None:
    node = _fitted(n_clusters=3)
    node = node.to("cuda")
    cube = _as_cube(_three_blob_pixels()).to("cuda")
    out = node.forward(cube=cube)
    assert out["spectral_signature"].device.type == "cuda"
    assert out["class_mask"].device.type == "cuda"
    assert out["scores"].device.type == "cuda"


@torch.no_grad()
def test_centroids_survive_state_dict_roundtrip() -> None:
    node = _fitted(n_clusters=3)
    cube = _as_cube(_three_blob_pixels())
    before = node.forward(cube=cube)

    state = node.state_dict()
    fresh = KMeansClusterer(n_clusters=3, random_state=0)
    fresh.load_state_dict(state)
    after = fresh.forward(cube=cube)

    assert torch.equal(before["spectral_signature"], after["spectral_signature"])


@torch.no_grad()
def test_state_dict_has_no_new_buffer_for_the_centroids_port() -> None:
    """The ``spectral_signature`` port reshapes the existing ``centroids`` buffer and adds no
    new buffer, so a state dict saved before this change (same keys: ``_initialized``,
    ``centroids``) still loads without a missing- or unexpected-key error."""
    node = _fitted(n_clusters=3)
    state = node.state_dict()
    assert set(state) == {"_initialized", "centroids"}
    fresh = KMeansClusterer(n_clusters=3, random_state=0)
    fresh.load_state_dict(state)  # strict by default: would raise on any key mismatch
    assert fresh.is_initialized


class _CubeSource(Node):
    """Module-scope test source emitting a fixed BHWC cube of positive spectra."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"cube": _as_cube(_positive_spectra_pixels())}


@torch.no_grad()
def test_kmeans_centroids_wire_into_spectral_angle_mapper_as_the_minimum_angle() -> None:
    pixels = _positive_spectra_pixels()  # [N, 4]; _CubeSource emits this same array
    cube = _as_cube(pixels)  # [1, N, 1, 4]

    kmeans = KMeansClusterer(n_clusters=3, random_state=0, name="kmeans")
    kmeans.statistical_initialization([{"cube": cube}])

    pipe = CuvisPipeline("kmeans_sam")
    src = _CubeSource(name="src")
    sam = SpectralAngleMapper(name="sam")
    pipe.connect(src.outputs.cube, kmeans.inputs.cube)
    pipe.connect(src.outputs.cube, sam.inputs.cube)
    pipe.connect(kmeans.outputs.spectral_signature, sam.inputs.spectral_signature)

    out = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)
    best_scores = out[("sam", "best_scores")]

    # Reference: the minimum spectral angle to any centroid, computed in numpy. SpectralAngleMapper
    # divides pixel and reference by their own mean before the angle; that rescaling does not
    # change the angle between two vectors, so the raw cosine angle is the right reference.
    centroids = kmeans.centroids.numpy()  # [K, C]
    px = pixels.astype(np.float64)
    ref = centroids.astype(np.float64)
    dot = px @ ref.T
    norms = np.linalg.norm(px, axis=-1, keepdims=True) * np.linalg.norm(ref, axis=-1)[None, :]
    cos = np.clip(dot / norms, -1.0, 1.0)
    angles = np.arccos(cos)  # [N, K]
    expected_best = angles.min(axis=-1)  # the minimum angle over all centroids

    got = best_scores.reshape(-1).numpy()
    np.testing.assert_allclose(got, expected_best, atol=1e-4)
