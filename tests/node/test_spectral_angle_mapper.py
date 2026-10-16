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


def _cube_and_refs(seed: int, n: int = 16, c: int = 6) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    cube = torch.rand(2, 5, 7, c, generator=g) + 0.05
    refs = torch.rand(n, 1, 1, c, generator=g) + 0.05
    return cube, refs


def test_matches_float64_golden_with_16_references() -> None:
    cube, refs = _cube_and_refs(seed=0)
    out = SpectralAngleMapper().forward(cube=cube, spectral_signature=refs)

    x = cube.double()
    r = refs.double().reshape(refs.shape[0], -1)
    cos = torch.einsum(
        "bhwc,nc->bhwn", x / x.norm(dim=-1, keepdim=True), r / r.norm(dim=-1, keepdim=True)
    )
    golden = torch.arccos(cos.clamp(-1.0, 1.0))
    torch.testing.assert_close(out["scores"].double(), golden, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        out["best_scores"].double(), golden.amin(dim=-1, keepdim=True), atol=1e-5, rtol=0
    )
    assert torch.equal(out["identity_mask"], (golden.argmin(dim=-1) + 1).to(torch.int32))


def test_num_channels_is_optional_and_rebuilds() -> None:
    node = SpectralAngleMapper()
    assert node.num_channels is None
    again = SpectralAngleMapper(**node.hparams)
    assert again.hparams == node.hparams
    cube, refs = _cube_and_refs(seed=1, n=3)
    assert again.forward(cube=cube, spectral_signature=refs)["scores"].shape == (2, 5, 7, 3)


def test_num_channels_mismatch_raises() -> None:
    cube, refs = _cube_and_refs(seed=2, c=6)
    with pytest.raises(ValueError, match="num_channels is 5"):
        SpectralAngleMapper(num_channels=5).forward(cube=cube, spectral_signature=refs)


def test_reference_channel_mismatch_raises() -> None:
    cube, refs = _cube_and_refs(seed=3, c=6)
    with pytest.raises(ValueError, match="6 channels but the references have 5"):
        SpectralAngleMapper().forward(cube=cube, spectral_signature=refs[..., :5])


def test_empty_signature_raises() -> None:
    cube, _ = _cube_and_refs(seed=4)
    with pytest.raises(ValueError, match="no reference spectrum"):
        SpectralAngleMapper().forward(cube=cube, spectral_signature=torch.zeros(0, 1, 1, 6))


def test_zero_reference_scores_right_angle_without_touching_the_others() -> None:
    # SignaturesToReferences passes zero rows for objects absent from a frame.
    cube, refs = _cube_and_refs(seed=5, n=3)
    with_zero = torch.cat([refs, torch.zeros(1, 1, 1, 6)])
    out = SpectralAngleMapper().forward(cube=cube, spectral_signature=with_zero)
    reference = SpectralAngleMapper().forward(cube=cube, spectral_signature=refs)
    torch.testing.assert_close(out["scores"][..., 3], torch.full((2, 5, 7), math.pi / 2))
    torch.testing.assert_close(out["scores"][..., :3], reference["scores"])
    assert torch.equal(out["identity_mask"], reference["identity_mask"])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_peak_memory_on_a_full_frame_with_16_references() -> None:
    cube = torch.rand(1, 1000, 1080, 61, device="cuda") + 0.05
    refs = torch.rand(16, 1, 1, 61, device="cuda") + 0.05
    node = SpectralAngleMapper().to("cuda")
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    node.forward(cube=cube, spectral_signature=refs)
    torch.cuda.synchronize()
    extra_mb = (torch.cuda.max_memory_allocated() - base) / 2**20
    assert extra_mb < 700, f"forward allocated {extra_mb:.0f} MB on top of its inputs"
