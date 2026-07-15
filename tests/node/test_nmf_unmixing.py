from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.unmixing import NMFUnmixing

pytestmark = pytest.mark.unit


def _synthetic_pixels(
    n_components: int = 3,
    channels: int = 10,
    n_pixels: int = 600,
    seed: int = 0,
) -> torch.Tensor:
    """Build non-negative pixels as mixtures of ``n_components`` known spectra."""
    rng = np.random.default_rng(seed)
    components = rng.uniform(0.2, 1.0, size=(n_components, channels)).astype(np.float32)
    weights = rng.uniform(0.0, 1.0, size=(n_pixels, n_components)).astype(np.float32)
    pixels = weights @ components  # [N, C], non-negative
    return torch.from_numpy(pixels)


@torch.no_grad()
def test_nmf_reconstruction_parity_with_sklearn() -> None:
    n_components, channels = 3, 10
    height, width = 20, 30
    pixels = _synthetic_pixels(n_components, channels, height * width, seed=1)
    cube = pixels.reshape(1, height, width, channels)

    node = NMFUnmixing(
        n_components=n_components,
        init="nndsvda",
        beta_loss="frobenius",
        max_iter=1000,
        random_state=0,
        max_fit_pixels=0,  # use all pixels so the fit matches the sklearn reference
    )
    node.statistical_initialization(iter([{"cube": cube}]))
    assert node.is_initialized

    out = node.forward(cube=cube)
    assert out["abundances"].shape == (1, height, width, n_components)
    assert out["endmembers"].shape == (n_components, channels)
    assert out["scores"].shape == (1, height, width, 1)
    assert out["class_mask"].shape == (1, height, width)
    assert out["class_mask"].dtype == torch.int32

    # Mean per-pixel reconstruction residual should be small on data drawn from a
    # rank-3 non-negative model.
    mean_node_residual = out["scores"].mean().item()
    assert mean_node_residual < 1e-1

    # Reconstruction parity against sklearn's own NMF on the same pixels. NMF is
    # permutation/scale non-unique, so compare the aggregate reconstruction error,
    # not the raw factors. sklearn reports the total Frobenius norm of (X - W H).
    import sklearn.decomposition

    model = sklearn.decomposition.NMF(
        n_components=n_components,
        init="nndsvda",
        beta_loss="frobenius",
        max_iter=1000,
        random_state=0,
    )
    model.fit_transform(pixels.numpy())
    sklearn_total = float(model.reconstruction_err_)

    node_total = float(torch.linalg.vector_norm(out["scores"].reshape(-1)))
    assert np.isclose(node_total, sklearn_total, atol=1e-2, rtol=1e-2)


@torch.no_grad()
def test_nmf_empty_stream_raises_and_stays_uninitialized() -> None:
    node = NMFUnmixing(n_components=3)
    with pytest.raises(RuntimeError):
        node.statistical_initialization(iter([]))
    assert not node.is_initialized
    with pytest.raises(RuntimeError):
        node.forward(cube=torch.rand(1, 2, 2, 10))


@torch.no_grad()
def test_nmf_state_dict_round_trip() -> None:
    n_components, channels = 3, 10
    height, width = 12, 16
    pixels = _synthetic_pixels(n_components, channels, height * width, seed=2)
    cube = pixels.reshape(1, height, width, channels)

    node = NMFUnmixing(n_components=n_components, max_iter=400, max_fit_pixels=0)
    node.statistical_initialization(iter([{"cube": cube}]))
    out_a = node.forward(cube=cube)

    state = node.state_dict()

    # Fresh instance starts uninitialized with a placeholder buffer; loading the
    # checkpoint must resize the lazy buffer (base _load_from_state_dict hook).
    fresh = NMFUnmixing(n_components=n_components, max_iter=400, max_fit_pixels=0)
    assert not fresh.is_initialized
    fresh.load_state_dict(state)
    assert fresh.is_initialized

    out_b = fresh.forward(cube=cube)
    assert torch.allclose(out_a["abundances"], out_b["abundances"], atol=1e-5)
    assert torch.allclose(out_a["endmembers"], out_b["endmembers"], atol=1e-5)
    assert torch.allclose(out_a["scores"], out_b["scores"], atol=1e-5)
    assert torch.equal(out_a["class_mask"], out_b["class_mask"])


@torch.no_grad()
def test_nmf_forward_validates_cube_shape_and_channels() -> None:
    channels = 6
    pixels = _synthetic_pixels(2, channels, 60, seed=2)
    cube = pixels.reshape(1, 6, 10, channels)
    node = NMFUnmixing(n_components=2, max_iter=200, random_state=0, max_fit_pixels=0)
    node.statistical_initialization(iter([{"cube": cube}]))

    with pytest.raises(ValueError, match="Expected cube"):
        node.forward(cube=cube[0])
    with pytest.raises(ValueError, match="channels"):
        node.forward(cube=cube[..., : channels - 1])
