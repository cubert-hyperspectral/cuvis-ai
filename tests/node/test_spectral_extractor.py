"""Tests for BBoxSpectralExtractor node."""

from __future__ import annotations

import torch

from cuvis_ai.node.spectral_extractor import BBoxSpectralExtractor


def test_basic_extraction(create_test_cube, create_test_bboxes) -> None:
    """Known spectral regions produce correct median signatures and std."""
    cube, _ = create_test_cube(
        batch_size=1,
        height=32,
        width=32,
        num_channels=8,
        mode="wavelength_dependent",
        dtype=torch.float32,
    )
    bboxes = create_test_bboxes(n=1, height=32, width=32)

    node = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=False, aggregation="median")
    out = node.forward(cube=cube, bboxes=bboxes)

    sig = out["spectral_signatures"]
    std = out["spectral_std"]
    valid = out["spectral_valid"]

    assert sig.shape == (1, 1, 8)
    assert std.shape == (1, 1, 8)
    assert valid.shape == (1, 1)
    assert int(valid[0, 0].item()) == 1

    # Std should be near zero (all pixels in the wavelength_dependent cube are identical)
    assert float(std[0, 0].mean()) < 0.01


def test_center_crop_reduces_background(create_test_bboxes) -> None:
    """Center crop excludes border pixels with different spectra."""
    cube = torch.ones((1, 40, 40, 4), dtype=torch.float32) * 0.5
    # Inner 20x20 region has a distinct spectrum
    cube[0, 10:30, 10:30, :] = torch.tensor([1.0, 0.5, 0.25, 0.1])
    # Bbox covers both inner region and border
    bboxes = torch.tensor([[[5.0, 5.0, 35.0, 35.0]]], dtype=torch.float32)

    node_full = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=False)
    node_crop = BBoxSpectralExtractor(center_crop_scale=0.5, l2_normalize=False)

    out_full = node_full.forward(cube=cube, bboxes=bboxes)
    out_crop = node_crop.forward(cube=cube, bboxes=bboxes)

    # Center-cropped signature should be closer to the inner region spectrum
    inner_spec = torch.tensor([1.0, 0.5, 0.25, 0.1])
    dist_full = (out_full["spectral_signatures"][0, 0] - inner_spec).norm()
    dist_crop = (out_crop["spectral_signatures"][0, 0] - inner_spec).norm()
    assert float(dist_crop) < float(dist_full)


def test_empty_detections(create_test_cube) -> None:
    """N=0 detections produce correctly shaped empty outputs."""
    cube, _ = create_test_cube(
        batch_size=1,
        height=16,
        width=16,
        num_channels=5,
        mode="random",
        dtype=torch.float32,
    )
    bboxes = torch.empty((1, 0, 4), dtype=torch.float32)

    node = BBoxSpectralExtractor()
    out = node.forward(cube=cube, bboxes=bboxes)

    assert out["spectral_signatures"].shape == (1, 0, 5)
    assert out["spectral_std"].shape == (1, 0, 5)
    assert out["spectral_valid"].shape == (1, 0)


def test_bbox_outside_image(create_test_cube) -> None:
    """Bbox fully outside image produces zero vector and spectral_valid=0."""
    cube, _ = create_test_cube(
        batch_size=1,
        height=10,
        width=10,
        num_channels=4,
        mode="random",
        dtype=torch.float32,
    )
    bboxes = torch.tensor([[[20.0, 20.0, 30.0, 30.0]]], dtype=torch.float32)

    node = BBoxSpectralExtractor(l2_normalize=False)
    out = node.forward(cube=cube, bboxes=bboxes)

    assert int(out["spectral_valid"][0, 0].item()) == 0
    assert float(out["spectral_signatures"][0, 0].norm()) < 1e-7


def test_bbox_partially_outside() -> None:
    """Bbox partially outside image extracts from the valid portion."""
    cube = torch.ones((1, 10, 10, 4), dtype=torch.float32)
    # bbox extends beyond image on the right
    bboxes = torch.tensor([[[5.0, 2.0, 15.0, 8.0]]], dtype=torch.float32)

    node = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=False)
    out = node.forward(cube=cube, bboxes=bboxes)

    assert int(out["spectral_valid"][0, 0].item()) == 1
    assert float(out["spectral_signatures"][0, 0].norm()) > 0


def test_tiny_bbox_fallback() -> None:
    """Bbox smaller than min_crop_pixels after center crop falls back to full bbox."""
    cube = torch.ones((1, 10, 10, 3), dtype=torch.float32) * 0.7
    # Tiny bbox: 3x3 pixels, with center_crop_scale=0.5 → crop would be ~1x1 pixel
    bboxes = torch.tensor([[[2.0, 2.0, 5.0, 5.0]]], dtype=torch.float32)

    node = BBoxSpectralExtractor(center_crop_scale=0.5, min_crop_pixels=4, l2_normalize=False)
    out = node.forward(cube=cube, bboxes=bboxes)

    # Should still produce valid output (fell back to full bbox)
    assert int(out["spectral_valid"][0, 0].item()) == 1


def test_multiple_detections(create_test_bboxes) -> None:
    """Two bboxes over distinct spectral regions produce distinct signatures."""
    cube = torch.zeros((1, 20, 20, 4), dtype=torch.float32)
    cube[0, :10, :10, :] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # region A
    cube[0, 10:, 10:, :] = torch.tensor([0.0, 0.0, 0.0, 1.0])  # region B
    bboxes = torch.tensor([[[1.0, 1.0, 9.0, 9.0], [11.0, 11.0, 19.0, 19.0]]], dtype=torch.float32)

    node = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=False)
    out = node.forward(cube=cube, bboxes=bboxes)

    sig_a = out["spectral_signatures"][0, 0]
    sig_b = out["spectral_signatures"][0, 1]

    assert int(out["spectral_valid"][0, 0].item()) == 1
    assert int(out["spectral_valid"][0, 1].item()) == 1
    # Signatures should be distinctly different
    assert float((sig_a - sig_b).norm()) > 0.5


def test_l2_normalize(create_test_cube, create_test_bboxes) -> None:
    """With l2_normalize=True, output signatures have unit L2 norm."""
    cube, _ = create_test_cube(
        batch_size=1,
        height=32,
        width=32,
        num_channels=8,
        mode="wavelength_dependent",
        dtype=torch.float32,
    )
    bboxes = create_test_bboxes(n=1, height=32, width=32)

    node = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=True)
    out = node.forward(cube=cube, bboxes=bboxes)

    sig = out["spectral_signatures"][0, 0]
    assert int(out["spectral_valid"][0, 0].item()) == 1
    assert abs(float(sig.norm()) - 1.0) < 1e-5


def test_spectral_std_output(create_test_cube, create_test_bboxes) -> None:
    """Uniform crop has low std; mixed crop has higher std."""
    # Uniform cube
    cube_uniform = torch.ones((1, 20, 20, 4), dtype=torch.float32) * 0.5
    bboxes = create_test_bboxes(n=1, height=20, width=20)

    node = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=False)
    out_uniform = node.forward(cube=cube_uniform, bboxes=bboxes)

    # Mixed cube (noisy)
    cube_noisy, _ = create_test_cube(
        batch_size=1,
        height=20,
        width=20,
        num_channels=4,
        mode="random",
        dtype=torch.float32,
    )
    out_noisy = node.forward(cube=cube_noisy, bboxes=bboxes)

    std_uniform = float(out_uniform["spectral_std"][0, 0].mean())
    std_noisy = float(out_noisy["spectral_std"][0, 0].mean())

    assert std_uniform < std_noisy


def test_output_shape_contract(create_test_cube, create_test_bboxes) -> None:
    """Output shapes follow [1, N, C] sigs, [1, N, C] std, [1, N] valid contract."""
    B, H, W, C, N = 1, 64, 64, 16, 5
    cube, _ = create_test_cube(
        batch_size=B,
        height=H,
        width=W,
        num_channels=C,
        mode="random",
        dtype=torch.float32,
    )
    bboxes = create_test_bboxes(n=N, height=H, width=W)

    node = BBoxSpectralExtractor(center_crop_scale=0.65, l2_normalize=True)
    out = node.forward(cube=cube, bboxes=bboxes)

    sigs = out["spectral_signatures"]
    std = out["spectral_std"]
    valid = out["spectral_valid"]

    assert sigs.shape == (B, N, C)
    assert std.shape == (B, N, C)
    assert valid.shape == (B, N)
    assert sigs.dtype == torch.float32
    assert std.dtype == torch.float32
    assert valid.dtype == torch.int32
    # B and N dimensions align across all outputs
    assert sigs.shape[:2] == valid.shape
    assert sigs.shape[:2] == std.shape[:2]


def test_aggregation_mean_mode(create_test_cube, create_test_bboxes) -> None:
    """With aggregation='mean', returns mean instead of median."""
    cube, _ = create_test_cube(
        batch_size=1,
        height=32,
        width=32,
        num_channels=8,
        mode="wavelength_dependent",
        dtype=torch.float32,
    )
    bboxes = create_test_bboxes(n=1, height=32, width=32)

    node_median = BBoxSpectralExtractor(
        center_crop_scale=1.0, l2_normalize=False, aggregation="median"
    )
    node_mean = BBoxSpectralExtractor(center_crop_scale=1.0, l2_normalize=False, aggregation="mean")

    out_median = node_median.forward(cube=cube, bboxes=bboxes)
    out_mean = node_mean.forward(cube=cube, bboxes=bboxes)

    # For a uniform region, median and mean should be nearly identical
    assert torch.allclose(
        out_median["spectral_signatures"], out_mean["spectral_signatures"], atol=0.05
    )
