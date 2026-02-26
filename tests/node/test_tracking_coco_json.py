from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cuvis_ai.node.json_writer import TrackingCocoJsonNode


def _build_inputs(
    frame_idx: int,
    mask_2d: torch.Tensor,
    object_ids: list[int],
    detection_scores: list[float],
) -> dict[str, torch.Tensor]:
    return {
        "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
        "mask": mask_2d.to(dtype=torch.int32).unsqueeze(0),
        "object_ids": torch.tensor([object_ids], dtype=torch.int64),
        "detection_scores": torch.tensor([detection_scores], dtype=torch.float32),
    }


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def test_tracking_coco_json_writes_valid_json_after_each_frame(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    frames = (
        _build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.95],
        ),
        _build_inputs(
            frame_idx=1,
            mask_2d=torch.zeros((2, 2), dtype=torch.int32),
            object_ids=[],
            detection_scores=[],
        ),
        _build_inputs(
            frame_idx=2,
            mask_2d=torch.tensor([[0, 2], [2, 2]], dtype=torch.int32),
            object_ids=[2],
            detection_scores=[0.88],
        ),
    )

    for frame in frames:
        node.forward(**frame)
        parsed = _read_json(json_path)
        assert set(parsed.keys()) == {"info", "images", "annotations", "categories"}

    parsed = _read_json(json_path)
    assert [image["id"] for image in parsed["images"]] == [0, 1, 2]
    assert len(parsed["annotations"]) == 2
    assert parsed["categories"] == [{"id": 1, "name": "person"}]


def test_tracking_coco_json_replaces_existing_frame_idempotently(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=4,
            mask_2d=torch.tensor([[1, 1], [0, 0]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.25],
        )
    )
    node.forward(
        **_build_inputs(
            frame_idx=4,
            mask_2d=torch.tensor([[1, 1], [0, 0]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.91],
        )
    )

    parsed = _read_json(json_path)
    assert len(parsed["images"]) == 1
    assert parsed["images"][0]["id"] == 4
    assert len(parsed["annotations"]) == 1
    assert parsed["annotations"][0]["score"] == pytest.approx(0.91)


def test_tracking_coco_json_writes_empty_frame_entry(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=7,
            mask_2d=torch.zeros((3, 4), dtype=torch.int32),
            object_ids=[],
            detection_scores=[],
        )
    )

    parsed = _read_json(json_path)
    assert len(parsed["images"]) == 1
    assert parsed["images"][0]["id"] == 7
    assert parsed["annotations"] == []


def test_tracking_coco_json_validates_alignment(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = TrackingCocoJsonNode(output_json_path=str(json_path), category_name="person")

    with pytest.raises(ValueError, match="identical lengths"):
        node.forward(
            **_build_inputs(
                frame_idx=0,
                mask_2d=torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
                object_ids=[1, 2],
                detection_scores=[0.9],
            )
        )


def test_tracking_coco_json_atomic_write_is_parseable(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = TrackingCocoJsonNode(
        output_json_path=str(json_path),
        category_name="person",
        atomic_write=True,
    )

    for frame_idx in range(12):
        has_obj = frame_idx % 2 == 0
        node.forward(
            **_build_inputs(
                frame_idx=frame_idx,
                mask_2d=torch.tensor([[0, 1], [1, 0]], dtype=torch.int32)
                if has_obj
                else torch.zeros((2, 2), dtype=torch.int32),
                object_ids=[1] if has_obj else [],
                detection_scores=[0.8] if has_obj else [],
            )
        )
        parsed = _read_json(json_path)
        assert "images" in parsed
        assert "annotations" in parsed
