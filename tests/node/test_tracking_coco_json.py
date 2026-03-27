from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from cuvis_ai.node.json_writer import CocoTrackMaskWriter


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
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path), category_name="person", flush_interval=1
    )

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
        assert set(parsed.keys()) == {"info", "videos", "annotations", "categories"}
        assert parsed["videos"][0]["length"] == len(parsed["videos"][0]["frame_indices"])

    node.close()
    parsed = _read_json(json_path)
    assert parsed["videos"][0]["frame_indices"] == [0, 1, 2]
    assert len(parsed["annotations"]) == 2
    assert parsed["categories"] == [{"id": 1, "name": "person"}]
    for ann in parsed["annotations"]:
        assert len(ann["segmentations"]) == 3
        assert len(ann["detection_scores"]) == 3
        assert len(ann["bboxes"]) == 3
        assert len(ann["areas"]) == 3


def test_tracking_coco_json_replaces_existing_frame_idempotently(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), category_name="person")

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

    node.close()
    parsed = _read_json(json_path)
    assert parsed["videos"][0]["frame_indices"] == [4]
    assert len(parsed["annotations"]) == 1
    assert parsed["annotations"][0]["detection_scores"] == pytest.approx([0.91])


def test_tracking_coco_json_writes_empty_frame_entry(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=7,
            mask_2d=torch.zeros((3, 4), dtype=torch.int32),
            object_ids=[],
            detection_scores=[],
        )
    )

    node.close()
    parsed = _read_json(json_path)
    assert parsed["videos"][0]["frame_indices"] == [7]
    assert parsed["videos"][0]["height"] == 3
    assert parsed["videos"][0]["width"] == 4
    assert parsed["annotations"] == []


def test_tracking_coco_json_validates_alignment(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), category_name="person")

    with pytest.raises(ValueError, match="identical lengths"):
        node.forward(
            **_build_inputs(
                frame_idx=0,
                mask_2d=torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
                object_ids=[1, 2],
                detection_scores=[0.9],
            )
        )


def test_tracking_coco_json_ignores_background_id_zero(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[0, 1],
            detection_scores=[0.99, 0.95],
        )
    )

    node.close()
    parsed = _read_json(json_path)
    assert len(parsed["annotations"]) == 1
    ann = parsed["annotations"][0]
    assert ann["track_id"] == 1
    assert ann["detection_scores"] == pytest.approx([0.95])


def test_tracking_coco_json_atomic_write_is_parseable(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path),
        category_name="person",
        atomic_write=True,
        flush_interval=1,
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
        assert "videos" in parsed
        assert "annotations" in parsed

    node.close()


def test_tracking_coco_json_deferred_write_on_close(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.95],
        )
    )

    assert not json_path.exists()

    node.close()
    assert json_path.exists()
    parsed = _read_json(json_path)
    assert parsed["videos"][0]["frame_indices"] == [0]
    assert len(parsed["annotations"]) == 1


def test_tracking_coco_json_close_is_idempotent(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.95],
        )
    )

    node.close()
    node.close()
    parsed = _read_json(json_path)
    assert parsed["videos"][0]["frame_indices"] == [0]


def test_tracking_coco_json_flush_interval(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path), category_name="person", flush_interval=3
    )

    for i in range(2):
        node.forward(
            **_build_inputs(
                frame_idx=i,
                mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
                object_ids=[1],
                detection_scores=[0.9],
            )
        )
    assert not json_path.exists()

    node.forward(
        **_build_inputs(
            frame_idx=2,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.9],
        )
    )
    assert json_path.exists()
    parsed = _read_json(json_path)
    assert parsed["videos"][0]["frame_indices"] == [0, 1, 2]

    node.close()


def test_tracking_coco_json_validates_flush_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flush_interval"):
        CocoTrackMaskWriter(
            output_json_path=str(tmp_path / "test.json"),
            flush_interval=-1,
        )
