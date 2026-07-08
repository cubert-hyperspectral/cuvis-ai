from __future__ import annotations

import numpy as np
import pytest
import torch

from cuvis_ai.node.pretreatments import MeanCenter, UnitVarianceScaling

pytestmark = pytest.mark.unit


def _make_stream(seed: int) -> tuple[list[dict[str, torch.Tensor]], np.ndarray]:
    rng = np.random.default_rng(seed)
    cubes = [rng.random((1, 8, 8, 5)).astype(np.float32) for _ in range(3)]
    stream = [{"cube": torch.tensor(c)} for c in cubes]
    pixels = np.concatenate([c.reshape(-1, 5) for c in cubes], axis=0)
    return stream, pixels


@torch.no_grad()
def test_mean_center_fits_global_mean() -> None:
    stream, pixels = _make_stream(21)
    node = MeanCenter()
    node.statistical_initialization(iter(stream))

    assert node.is_initialized
    expected = pixels.mean(axis=0)
    assert np.allclose(node.mean_c.numpy(), expected, atol=1e-4)

    cube = stream[0]["cube"].clone()
    out = node.forward(cube=cube)["cube"]
    assert torch.allclose(out, cube - node.mean_c, atol=1e-6)


@torch.no_grad()
def test_unit_variance_fits_global_std_ddof1() -> None:
    stream, pixels = _make_stream(22)
    node = UnitVarianceScaling()
    node.statistical_initialization(iter(stream))

    assert node.is_initialized
    expected = pixels.std(axis=0, ddof=1)
    assert np.allclose(node.std_c.numpy(), expected, atol=1e-4)

    cube = stream[0]["cube"].clone()
    out = node.forward(cube=cube)["cube"]
    assert torch.allclose(out, cube / node.std_c.clamp_min(1e-8), atol=1e-6)


@torch.no_grad()
@pytest.mark.parametrize("cls", [MeanCenter, UnitVarianceScaling])
def test_empty_stream_rejected_and_stays_uninitialized(cls) -> None:
    node = cls()
    with pytest.raises(RuntimeError):
        node.statistical_initialization(iter([]))
    assert node.is_initialized is False
    with pytest.raises(RuntimeError):
        node.forward(cube=torch.rand(1, 2, 2, 5))


@torch.no_grad()
@pytest.mark.parametrize("cls", [MeanCenter, UnitVarianceScaling])
def test_batches_without_cube_port_are_skipped(cls) -> None:
    """Batches lacking the cube port are ignored; the fit uses the remaining ones."""
    stream, pixels = _make_stream(24)
    padded = [{}, *stream, {"other": torch.rand(1, 2, 2, 5)}]
    node = cls()
    node.statistical_initialization(iter(padded))

    assert node.is_initialized
    if cls is MeanCenter:
        assert np.allclose(node.mean_c.numpy(), pixels.mean(axis=0), atol=1e-4)
    else:
        assert np.allclose(node.std_c.numpy(), pixels.std(axis=0, ddof=1), atol=1e-4)


def test_base_fit_hook_is_not_implemented() -> None:
    """Streaming-moment nodes never call the base _fit; its default stays a stub."""
    with pytest.raises(NotImplementedError):
        MeanCenter()._fit(torch.rand(4, 5))


@torch.no_grad()
@pytest.mark.parametrize("cls", [MeanCenter, UnitVarianceScaling])
def test_state_dict_round_trip(cls) -> None:
    stream, _ = _make_stream(23)
    node = cls()
    node.statistical_initialization(iter(stream))

    cube = torch.rand(1, 4, 4, 5)
    expected = node.forward(cube=cube)["cube"]

    fresh = cls()
    fresh.load_state_dict(node.state_dict())
    assert fresh.is_initialized
    got = fresh.forward(cube=cube)["cube"]
    assert torch.allclose(got, expected, atol=1e-6)
