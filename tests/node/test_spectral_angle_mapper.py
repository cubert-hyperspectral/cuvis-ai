"""Unit tests for SpectralAngleMapper."""

from __future__ import annotations

import math

import pytest
import torch

from cuvis_ai.node.spectral_angle_mapper import SpectralAngleMapper, StatefulSpectralAngleMapper

pytestmark = pytest.mark.unit


def _sam(num_channels: int) -> SpectralAngleMapper:
    return SpectralAngleMapper(num_channels=num_channels)


def test_identical_spectrum_zero_angle() -> None:
    cube = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)  # [1,1,1,4]
    ref = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]], dtype=torch.float32)  # [1,1,1,4]

    out = _sam(num_channels=4).forward(cube=cube, spectral_signature=ref)
    assert torch.allclose(out["scores"], torch.zeros_like(out["scores"]), atol=1e-6)
    assert torch.allclose(out["best_scores"], torch.zeros_like(out["best_scores"]), atol=1e-3)


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


def test_stateful_requires_signature_before_fit() -> None:
    node = StatefulSpectralAngleMapper(num_channels=3)
    cube = torch.ones((1, 1, 1, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="No learned_signature present"):
        node.forward(cube=cube)


def test_stateful_fit_signature_then_forward_uses_learned_reference() -> None:
    node = StatefulSpectralAngleMapper(num_channels=3)
    node.fit_signature(torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))
    cube = torch.tensor([[[[1.0, 2.0, 3.0]]]], dtype=torch.float32)
    out = node.forward(cube=cube)
    assert out["scores"].shape == (1, 1, 1, 1)
    assert out["best_scores"].shape == (1, 1, 1, 1)
    assert torch.allclose(out["best_scores"], torch.zeros_like(out["best_scores"]), atol=1e-3)
    assert torch.equal(out["identity_mask"], torch.ones((1, 1, 1), dtype=torch.int32))


def test_stateful_forward_prefers_explicit_signature_over_learned() -> None:
    node = StatefulSpectralAngleMapper(num_channels=2)
    node.fit_signature(torch.tensor([[1.0, 0.0]], dtype=torch.float32))
    cube = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)
    explicit = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)
    out = node.forward(cube=cube, spectral_signature=explicit)
    assert torch.allclose(out["best_scores"], torch.zeros_like(out["best_scores"]), atol=1e-6)


def test_stateful_fit_accepts_numpy_signature() -> None:
    node = StatefulSpectralAngleMapper(num_channels=4)
    sig = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32).numpy()
    node.fit_signature(sig)
    assert tuple(node.learned_signature.shape) == (1, 1, 1, 4)
    assert bool(node._has_learned_signature.item()) is True


def test_stateful_fit_rejects_wrong_shape() -> None:
    node = StatefulSpectralAngleMapper(num_channels=3)
    bad = torch.ones((1, 1, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="signature must have shape"):
        node.fit_signature(bad)


def test_stateful_fit_rejects_wrong_channel_count() -> None:
    node = StatefulSpectralAngleMapper(num_channels=3)
    with pytest.raises(ValueError, match="signature channel mismatch"):
        node.fit_signature(torch.ones((1, 2), dtype=torch.float32))


def test_stateful_fit_rejects_multiple_signatures_for_single_class_node() -> None:
    node = StatefulSpectralAngleMapper(num_channels=3)
    multi = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=torch.float32)
    with pytest.raises(ValueError, match="expects one signature"):
        node.fit_signature(multi)


def test_stateful_state_dict_roundtrip_preserves_learned_signature() -> None:
    node_a = StatefulSpectralAngleMapper(num_channels=3)
    learned = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float32)
    node_a.fit_signature(learned)
    state = node_a.state_dict()

    node_b = StatefulSpectralAngleMapper(num_channels=3)
    node_b.load_state_dict(state)

    assert bool(node_b._has_learned_signature.item()) is True
    assert torch.allclose(node_b.learned_signature[0, 0, 0], learned[0], atol=1e-6)

    cube = learned.view(1, 1, 1, 3)
    out = node_b.forward(cube=cube)
    assert torch.allclose(out["best_scores"], torch.zeros_like(out["best_scores"]), atol=1e-6)
