"""Characterization of the trimmed per-object statistics of the spectral extractors."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.spectral_extractor import BBoxSpectralExtractor, SpectralSignatureExtractor

pytestmark = pytest.mark.unit


def _cube_with_two_objects() -> tuple[torch.Tensor, torch.Tensor]:
    """A 6x6x3 cube: object 1 (18 px) has spectrum (1, 2, 3), object 2 (18 px) has (4, 5, 6)."""
    cube = torch.zeros(1, 6, 6, 3)
    mask = torch.zeros(1, 6, 6, dtype=torch.int32)
    cube[0, :3, :, :] = torch.tensor([1.0, 2.0, 3.0])
    mask[0, :3, :] = 1
    cube[0, 3:, :, :] = torch.tensor([4.0, 5.0, 6.0])
    mask[0, 3:, :] = 2
    return cube, mask


def test_signature_extractor_reports_the_trimmed_mean_and_std_per_object():
    cube, mask = _cube_with_two_objects()
    out = SpectralSignatureExtractor()(cube=cube, mask=mask)
    assert out["signatures"].shape == (1, 2, 3)
    assert torch.allclose(out["signatures"][0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.allclose(out["signatures"][0, 1], torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(out["signatures_std"], torch.zeros(1, 2, 3))


def test_signature_extractor_zeroes_objects_below_the_pixel_floor_and_absent_ids():
    cube, mask = _cube_with_two_objects()
    mask[0, 3:, :] = 0
    mask[0, 3, 0] = 2  # one pixel is below the default floor of ten
    out = SpectralSignatureExtractor()(
        cube=cube, mask=mask, object_ids=torch.tensor([[1, 2, 7]], dtype=torch.int64)
    )
    assert out["signatures"].shape == (1, 3, 3)
    assert torch.allclose(out["signatures"][0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(out["signatures"][0, 1], torch.zeros(3))
    assert torch.equal(out["signatures"][0, 2], torch.zeros(3))


def test_signature_extractor_trims_the_outliers_of_each_band():
    cube = torch.zeros(1, 1, 20, 2)
    cube[0, 0, :, 0] = torch.arange(20, dtype=torch.float32)  # 0..19, 10 % trimmed each side
    cube[0, 0, :, 1] = 5.0
    mask = torch.ones(1, 1, 20, dtype=torch.int32)
    out = SpectralSignatureExtractor(trim_fraction=0.1)(cube=cube, mask=mask)
    assert torch.allclose(out["signatures"][0, 0], torch.tensor([9.5, 5.0]))


def test_signature_extractor_returns_empty_for_an_empty_mask():
    cube, mask = _cube_with_two_objects()
    out = SpectralSignatureExtractor()(cube=cube, mask=torch.zeros_like(mask))
    assert out["signatures"].shape == (1, 0, 3)
    assert out["signatures_std"].shape == (1, 0, 3)


def test_bbox_extractor_mean_aggregation_averages_the_trimmed_pixels():
    cube = torch.zeros(1, 1, 20, 2)
    cube[0, 0, :, 0] = torch.arange(20, dtype=torch.float32)
    cube[0, 0, :, 1] = 5.0
    bboxes = torch.tensor([[[0.0, 0.0, 20.0, 1.0]]])
    node = BBoxSpectralExtractor(
        center_crop_scale=1.0, l2_normalize=False, aggregation="mean", trim_fraction=0.1
    )
    out = node(cube=cube, bboxes=bboxes)
    assert torch.allclose(out["spectral_signatures"][0, 0], torch.tensor([9.5, 5.0]))
