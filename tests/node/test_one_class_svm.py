"""Tests for the one-class SVM novelty detector.

Verifies the pure-torch RBF forward pass reproduces scikit-learn's
``decision_function`` / ``predict`` exactly, that the chunked forward matches a
single-shot computation, that an empty fit stream is rejected, and that a fitted
node survives a ``state_dict`` round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.svm import OneClassSVMDetector

pytestmark = pytest.mark.unit


def _cube_from_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Reshape an ``[N, C]`` pixel matrix into a ``[1, N, 1, C]`` cube."""
    n, c = pixels.shape
    return pixels.reshape(1, n, 1, c).to(torch.float32)


def _gaussian_blob(n: int, c: int, seed: int = 0) -> torch.Tensor:
    """Draw ``n`` samples of dimension ``c`` from a fixed-seed unit Gaussian."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(n, c, generator=gen, dtype=torch.float32)


def _fit_node(train: torch.Tensor, **kwargs) -> OneClassSVMDetector:
    """Fit a detector on a ``[N, C]`` training matrix via its fit stream."""
    node = OneClassSVMDetector(max_fit_pixels=0, **kwargs)
    node.statistical_initialization([{"cube": _cube_from_pixels(train)}])
    return node


@torch.no_grad()
def test_matches_sklearn_decision_function_and_predict() -> None:
    """Node scores/decisions reproduce sklearn decision_function/predict."""
    from sklearn.svm import OneClassSVM

    train = _gaussian_blob(200, 4, seed=0)
    inliers = _gaussian_blob(20, 4, seed=1)
    outliers = _gaussian_blob(20, 4, seed=2) + 50.0  # far from the blob
    test = torch.cat([inliers, outliers], dim=0)

    node = _fit_node(train, nu=0.5, gamma="scale")
    out = node.forward(cube=_cube_from_pixels(test))

    model = OneClassSVM(kernel="rbf", nu=0.5, gamma="scale").fit(train.numpy())
    expected_df = torch.as_tensor(model.decision_function(test.numpy()), dtype=torch.float32)
    expected_outlier = torch.as_tensor(model.predict(test.numpy()) == -1)

    scores = out["scores"].reshape(-1)
    decisions = out["decisions"].reshape(-1)

    assert torch.allclose(scores, expected_df, atol=1e-4)
    assert torch.equal(decisions, expected_outlier)
    # Sanity: the far outliers should actually be flagged.
    assert decisions[20:].all()


@torch.no_grad()
def test_chunked_forward_matches_single_shot() -> None:
    """A tiny chunk_size yields the same scores as a single-shot pass."""
    train = _gaussian_blob(150, 5, seed=3)
    test = torch.cat([_gaussian_blob(15, 5, seed=4), _gaussian_blob(15, 5, seed=5) + 30.0], dim=0)
    cube = _cube_from_pixels(test)

    node = _fit_node(train, nu=0.3, gamma="scale")

    node.chunk_size = 8
    chunked = node.forward(cube=cube)

    node.chunk_size = 10**9
    single = node.forward(cube=cube)

    # cdist accumulates in float32; chunk vs single-shot can differ by an ULP or
    # two, so allclose (not bit-exact), while the outlier mask must still match.
    assert torch.allclose(chunked["scores"], single["scores"], atol=1e-5, rtol=0.0)
    assert torch.equal(chunked["decisions"], single["decisions"])


@torch.no_grad()
def test_empty_stream_raises_and_stays_uninitialized() -> None:
    """An empty fit stream raises and leaves the node uninitialized."""
    node = OneClassSVMDetector()
    with pytest.raises(RuntimeError):
        node.statistical_initialization([])
    assert not node.is_initialized
    with pytest.raises(RuntimeError):
        node.forward(cube=torch.zeros(1, 2, 2, 4, dtype=torch.float32))


@torch.no_grad()
def test_non_rbf_kernel_raises_at_fit() -> None:
    """A non-rbf kernel is rejected at fit time."""
    train = _gaussian_blob(50, 3, seed=6)
    node = OneClassSVMDetector(kernel="linear", max_fit_pixels=0)
    with pytest.raises(ValueError, match="rbf"):
        node.statistical_initialization([{"cube": _cube_from_pixels(train)}])
    assert not node.is_initialized


@torch.no_grad()
def test_state_dict_round_trip_is_identical() -> None:
    """fit -> state_dict -> fresh load -> forward reproduces the same output."""
    train = _gaussian_blob(120, 6, seed=7)
    test = torch.cat([_gaussian_blob(10, 6, seed=8), _gaussian_blob(10, 6, seed=9) + 40.0], dim=0)
    cube = _cube_from_pixels(test)

    fitted = _fit_node(train, nu=0.2, gamma="scale", chunk_size=37)
    before = fitted.forward(cube=cube)

    state = fitted.state_dict()
    fresh = OneClassSVMDetector(nu=0.2, gamma="scale", chunk_size=37, max_fit_pixels=0)
    fresh.load_state_dict(state)

    assert fresh.is_initialized
    after = fresh.forward(cube=cube)

    assert torch.equal(before["scores"], after["scores"])
    assert torch.equal(before["decisions"], after["decisions"])


@torch.no_grad()
def test_gamma_float_matches_sklearn() -> None:
    """An explicit float gamma also reproduces sklearn's decision function."""
    from sklearn.svm import OneClassSVM

    train = _gaussian_blob(100, 4, seed=10)
    test = torch.cat([_gaussian_blob(10, 4, seed=11), _gaussian_blob(10, 4, seed=12) + 25.0], dim=0)

    node = _fit_node(train, nu=0.4, gamma=0.25)
    scores = node.forward(cube=_cube_from_pixels(test))["scores"].reshape(-1)

    model = OneClassSVM(kernel="rbf", nu=0.4, gamma=0.25).fit(train.numpy())
    expected = torch.as_tensor(model.decision_function(test.numpy()), dtype=torch.float32)

    assert torch.allclose(scores, expected, atol=1e-4)
    # Resolved gamma buffer should equal the requested float.
    assert np.isclose(float(node.gamma_buf.item()), 0.25)
