from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.mask_ops import (
    ClassMapRobustifier,
    MaskRobustifier,
    MaskToBBoxKalman,
    NearestLabelFill,
)


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


# --- ClassMapRobustifier -----------------------------------------------------


def test_class_map_robustifier_drops_speckle_keeps_block_and_echoes_source() -> None:
    cm = torch.full((1, 6, 6), -1, dtype=torch.int64)
    cm[0, 2:5, 2:5] = 1  # 9-px block of class 1
    cm[0, 0, 0] = 2  # 1-px speckle of class 2
    node = ClassMapRobustifier(
        opening_kernel=0, closing_kernel=0, min_area=4, keep_largest=False, background_value=-1
    )

    out = node.forward(class_map=cm)

    assert torch.all(out["class_map"][0, 2:5, 2:5] == 1)  # block kept
    assert out["class_map"][0, 0, 0].item() == -1  # speckle dropped -> background
    assert torch.equal(out["source"], cm)  # source echoed verbatim
    assert out["class_map"].dtype == torch.int64


def test_class_map_robustifier_keep_largest_per_class() -> None:
    cm = torch.full((1, 6, 8), -1, dtype=torch.int64)
    cm[0, 0:3, 0:3] = 1  # 9-px component of class 1
    cm[0, 5, 7] = 1  # separate 1-px component of class 1
    node = ClassMapRobustifier(
        opening_kernel=0, closing_kernel=0, min_area=0, keep_largest=True, background_value=-1
    )

    out = node.forward(class_map=cm)["class_map"]

    assert torch.all(out[0, 0:3, 0:3] == 1)  # largest component kept
    assert out[0, 5, 7].item() == -1  # smaller component dropped


def test_class_map_robustifier_all_background_returns_background() -> None:
    cm = torch.full((1, 4, 4), -1, dtype=torch.int64)
    node = ClassMapRobustifier(background_value=-1)

    out = node.forward(class_map=cm)

    assert torch.all(out["class_map"] == -1)
    assert torch.equal(out["source"], cm)


def test_class_map_robustifier_batch_items_independent() -> None:
    cm = torch.full((2, 5, 5), -1, dtype=torch.int64)
    cm[0, 1:4, 1:4] = 3  # frame 0 has a block; frame 1 stays all background
    node = ClassMapRobustifier(opening_kernel=0, closing_kernel=0, min_area=0, keep_largest=False)

    out = node.forward(class_map=cm)["class_map"]

    assert torch.all(out[0, 1:4, 1:4] == 3)
    assert torch.all(out[1] == -1)


def test_class_map_robustifier_validates_constructor_inputs() -> None:
    with pytest.raises(ValueError, match="min_area"):
        ClassMapRobustifier(min_area=-1)


# --- NearestLabelFill --------------------------------------------------------


def test_nearest_label_fill_fills_enclosed_hole() -> None:
    source = torch.zeros((1, 3, 3), dtype=torch.int64)  # class 0 everywhere (bg = -1)
    cm = source.clone()
    cm[0, 1, 1] = -1  # center is a hole

    out = NearestLabelFill(background_value=-1).forward(class_map=cm, source=source)["class_map"]

    assert torch.all(out == 0)  # hole filled from the enclosing class


def test_nearest_label_fill_leaves_background_untouched() -> None:
    source = torch.full((1, 3, 3), -1, dtype=torch.int64)
    source[0, 0, 0] = 2  # one labelled pixel; the rest is true background
    cm = source.clone()

    out = NearestLabelFill(background_value=-1).forward(class_map=cm, source=source)["class_map"]

    assert torch.equal(out, source)  # no holes -> background stays background


def test_nearest_label_fill_falls_back_to_source_when_unreachable() -> None:
    source = torch.full((1, 3, 3), -1, dtype=torch.int64)
    source[0, 1, 1] = 4  # a foreground pixel with NO surviving label anywhere
    cm = torch.full((1, 3, 3), -1, dtype=torch.int64)

    out = NearestLabelFill(background_value=-1).forward(class_map=cm, source=source)["class_map"]

    assert out[0, 1, 1].item() == 4  # filled from the source fallback
    assert int((out != -1).sum().item()) == 1  # nothing else added


def test_nearest_label_fill_tie_break_prefers_larger_class_id() -> None:
    source = torch.tensor([[[1, 1, 5]]], dtype=torch.int64)  # center foreground (labelled)
    cm = torch.tensor([[[1, -1, 5]]], dtype=torch.int64)  # center is a hole between 1 and 5

    out = NearestLabelFill(background_value=-1).forward(class_map=cm, source=source)["class_map"]

    assert out[0, 0, 1].item() == 5  # max class id wins the tie


def test_nearest_label_fill_identity_when_no_holes() -> None:
    cm = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int64)

    out = NearestLabelFill(background_value=-1).forward(class_map=cm, source=cm)["class_map"]

    assert torch.equal(out, cm)


def test_nearest_label_fill_respects_non_default_background_value() -> None:
    source = torch.tensor([[[1, 1, 2]]], dtype=torch.int64)  # bg = 0, all foreground
    cm = torch.tensor([[[1, 0, 2]]], dtype=torch.int64)  # center hole (0 == bg)

    out = NearestLabelFill(background_value=0).forward(class_map=cm, source=source)["class_map"]

    assert out[0, 0, 1].item() == 2  # 0 treated as background; larger id wins the tie


# --- end-to-end cleanup pipeline ---------------------------------------------


def test_class_map_cleanup_pipeline_runs_via_forward() -> None:
    from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

    cm = torch.full((1, 5, 5), 1, dtype=torch.int64)  # block of class 1...
    cm[0, 2, 2] = 2  # ...with a 1-px class-2 speckle
    robe = ClassMapRobustifier(
        opening_kernel=0, closing_kernel=0, min_area=2, keep_largest=False, background_value=-1
    )
    fill = NearestLabelFill(name="gap_fill", background_value=-1)
    pipe = CuvisPipeline("cleanup_test")
    pipe.connect(
        (robe.outputs.class_map, fill.inputs.class_map),
        (robe.outputs.source, fill.inputs.source),
    )

    out = pipe.forward(batch={"class_map": cm})
    res = out[(fill.name, "class_map")]

    assert res.shape == cm.shape
    assert res[0, 2, 2].item() == 1  # speckle dropped then gap-filled from neighbours
    assert bool((res != -1).all())  # foreground is dense (no leftover holes)
