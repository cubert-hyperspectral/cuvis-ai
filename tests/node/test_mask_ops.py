from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.mask_ops import MaskRobustifier, MaskToBBoxKalman


def test_mask_robustifier_keeps_largest_component_and_preserves_labels() -> None:
    mask = torch.zeros((1, 6, 6), dtype=torch.int32)
    mask[0, 0:1, 0:1] = 2
    mask[0, 2:5, 2:5] = 5
    node = MaskRobustifier(opening_kernel=0, closing_kernel=0, min_area=0, keep_largest=True)

    out = node.forward(mask=mask)["mask"]

    assert out[0, 0, 0].item() == 0
    assert torch.all(out[0, 2:5, 2:5] == 5)
    assert int((out > 0).sum().item()) == 9


def test_mask_robustifier_filters_small_components_without_largest_only() -> None:
    mask = torch.zeros((2, 6, 6), dtype=torch.int32)
    mask[0, 0, 0] = 1
    mask[0, 2:4, 2:4] = 3
    node = MaskRobustifier(opening_kernel=0, closing_kernel=0, min_area=4, keep_largest=False)

    out = node.forward(mask=mask)["mask"]

    assert out[0, 0, 0].item() == 0
    assert torch.all(out[0, 2:4, 2:4] == 3)
    assert not bool(out[1].any())


def test_mask_robustifier_opening_and_closing_paths_run() -> None:
    mask = torch.zeros((1, 7, 7), dtype=torch.int32)
    mask[0, 2:5, 2:5] = 1
    mask[0, 3, 3] = 0
    mask[0, 0, 0] = 1
    node = MaskRobustifier(opening_kernel=3, closing_kernel=3, min_area=0, keep_largest=False)

    out = node.forward(mask=mask)["mask"]

    assert out.shape == mask.shape
    assert out.dtype == torch.int32
    assert out[0, 0, 0].item() == 0


def test_mask_robustifier_fast_path_preserves_mask_when_filters_disabled() -> None:
    mask = torch.tensor([[[0, 2], [3, 0]]], dtype=torch.int32)
    node = MaskRobustifier(opening_kernel=0, closing_kernel=0, min_area=0, keep_largest=False)

    out = node.forward(mask=mask)["mask"]

    assert torch.equal(out, mask)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"opening_kernel": -1}, "opening_kernel"),
        ({"closing_kernel": -1}, "closing_kernel"),
        ({"min_area": -1}, "min_area"),
    ],
)
def test_mask_robustifier_validates_constructor_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MaskRobustifier(**kwargs)


def test_mask_to_bbox_static_extractor_returns_extent_or_none() -> None:
    mask = torch.zeros((5, 6), dtype=torch.int32)
    assert MaskToBBoxKalman._bbox_from_mask_torch(mask) is None

    mask[1:4, 2:5] = 1
    assert MaskToBBoxKalman._bbox_from_mask_torch(mask) == (2, 1, 5, 4)


def test_mask_to_bbox_kalman_warmup_prediction_and_drop() -> None:
    masks = torch.zeros((4, 12, 12), dtype=torch.int32)
    masks[0, 3:5, 4:6] = 1
    masks[1, 3:5, 4:6] = 1
    node = MaskToBBoxKalman(
        padding_fraction=0.0,
        min_size_px=2,
        min_hits=2,
        max_predict_frames=1,
        process_noise=1e-4,
        measurement_noise=1e-4,
    )

    out = node.forward(mask=masks)

    assert out["bbox"].shape == (4, 4)
    assert out["valid"].tolist() == [0, 1, 2, 0]
    assert out["bbox"][1].tolist() == pytest.approx([4.0, 3.0, 6.0, 5.0], abs=0.5)


def test_mask_to_bbox_kalman_resets_warmup_after_miss() -> None:
    masks = torch.zeros((3, 10, 10), dtype=torch.int32)
    masks[0, 1:3, 1:3] = 1
    masks[2, 6:8, 6:8] = 1
    node = MaskToBBoxKalman(padding_fraction=0.0, min_size_px=2, min_hits=2)

    out = node.forward(mask=masks)

    assert out["valid"].tolist() == [0, 0, 0]
    assert torch.equal(out["bbox"], torch.zeros_like(out["bbox"]))


def test_mask_to_bbox_kalman_padding_min_size_and_clamp() -> None:
    masks = torch.zeros((1, 8, 8), dtype=torch.int32)
    masks[0, 0:1, 0:1] = 1
    node = MaskToBBoxKalman(padding_fraction=1.0, min_size_px=6, min_hits=1)

    out = node.forward(mask=masks)

    x0, y0, x1, y1 = out["bbox"][0].tolist()
    assert out["valid"].tolist() == [1]
    assert 0.0 <= x0 < x1 <= 8.0
    assert 0.0 <= y0 < y1 <= 8.0
    assert (x1 - x0) >= 6.0
    assert (y1 - y0) >= 6.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"padding_fraction": -0.1}, "padding_fraction"),
        ({"min_size_px": 0}, "min_size_px"),
        ({"min_hits": 0}, "min_hits"),
        ({"max_predict_frames": -1}, "max_predict_frames"),
    ],
)
def test_mask_to_bbox_kalman_validates_constructor_inputs(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        MaskToBBoxKalman(**kwargs)
