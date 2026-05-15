"""Unit tests for SpectralAngleMapper."""

from __future__ import annotations

import math

import pytest
import torch

from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper

pytestmark = pytest.mark.unit


def _sam(num_channels: int) -> SpectralAngleMapper:
    return SpectralAngleMapper(num_channels=num_channels)


def test_identical_spectrum_zero_angle() -> None:
    cube = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)  # [1,1,1,4]
    ref = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)  # [1,1,1,4]

    out = _sam(num_channels=4).forward(cube=cube, spectral_signature=ref)
    assert torch.allclose(out["scores"], torch.zeros_like(out["scores"]), atol=1e-6)
    assert torch.allclose(out["best_scores"], torch.zeros_like(out["best_scores"]), atol=1e-6)


def test_orthogonal_spectrum_high_angle() -> None:
    cube = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
    ref = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)

    out = _sam(num_channels=2).forward(cube=cube, spectral_signature=ref)
    expected = torch.full_like(out["scores"], math.pi / 2.0)
    assert torch.allclose(out["scores"], expected, atol=1e-6)


def test_scaled_spectrum_zero_angle() -> None:
    cube = torch.tensor([[[[2.0, 4.0, 6.0]]]], dtype=torch.float32)
    ref = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)

    out = _sam(num_channels=3).forward(cube=cube, spectral_signature=ref)
    assert torch.allclose(out["scores"], torch.zeros_like(out["scores"]), atol=1e-3)


def test_output_shapes_single_ref(create_test_cube) -> None:
    cube, _ = create_test_cube(
        batch_size=2,
        height=3,
        width=4,
        num_channels=5,
        mode="random",
        dtype=torch.float32,
    )
    ref = torch.ones((1, 1, 1, 5), dtype=torch.float32)

    out = _sam(num_channels=5).forward(cube=cube, spectral_signature=ref)
    assert out["scores"].shape == (2, 3, 4, 1)
    assert out["best_scores"].shape == (2, 3, 4, 1)
    assert out["identity_mask"].shape == (2, 3, 4)
    assert torch.all(out["identity_mask"] == 1)


def test_output_shapes_multi_ref(create_test_cube) -> None:
    cube, _ = create_test_cube(
        batch_size=2,
        height=4,
        width=5,
        num_channels=6,
        mode="random",
        dtype=torch.float32,
    )
    ref = torch.tensor(
        [
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    out = _sam(num_channels=6).forward(cube=cube, spectral_signature=ref)
    assert out["scores"].shape == (2, 4, 5, 3)
    assert out["best_scores"].shape == (2, 4, 5, 1)
    assert out["identity_mask"].shape == (2, 4, 5)
    assert out["identity_mask"].dtype == torch.int32
    assert torch.all((out["identity_mask"] >= 1) & (out["identity_mask"] <= 3))


def test_batch_independence() -> None:
    # Batch 0 matches ref 1; batch 1 matches ref 2.
    cube = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )
    ref = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    out = _sam(num_channels=3).forward(cube=cube, spectral_signature=ref)
    assert out["identity_mask"][0, 0, 0].item() == 1
    assert out["identity_mask"][1, 0, 0].item() == 2


def test_dark_pixel_stability() -> None:
    cube = torch.zeros((1, 2, 2, 4), dtype=torch.float32)
    ref = torch.ones((1, 1, 1, 4), dtype=torch.float32)

    out = _sam(num_channels=4).forward(cube=cube, spectral_signature=ref)
    assert torch.isfinite(out["scores"]).all()
    assert torch.isfinite(out["best_scores"]).all()


def test_multi_reference_identity_mask() -> None:
    cube = torch.tensor(
        [
            [
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            ]
        ],
        dtype=torch.float32,
    )  # [1,2,2,3]
    ref = torch.tensor(
        [
            [[[1.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0]]],
            [[[0.0, 0.0, 1.0]]],
        ],
        dtype=torch.float32,
    )  # [3,1,1,3]

    out = _sam(num_channels=3).forward(cube=cube, spectral_signature=ref)
    expected = torch.tensor([[[1, 2], [3, 1]]], dtype=torch.int32)
    assert torch.equal(out["identity_mask"], expected)


def test_best_scores_equals_min(create_test_cube) -> None:
    cube, _ = create_test_cube(
        batch_size=2,
        height=3,
        width=4,
        num_channels=8,
        mode="random",
        seed=123,
        dtype=torch.float32,
    )
    ref = torch.tensor(
        [
            [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    out = _sam(num_channels=8).forward(cube=cube, spectral_signature=ref)
    expected = out["scores"].amin(dim=-1, keepdim=True)
    assert torch.allclose(out["best_scores"], expected, atol=1e-6)
