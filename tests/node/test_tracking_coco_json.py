from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from cuvis_ai.node.json_file import CocoTrackMaskWriter
from cuvis_ai_core.data.rle import coco_rle_encode, coco_rle_to_bbox


def _build_inputs(
    frame_idx: int,
    mask_2d: torch.Tensor,
    object_ids: list[int],
    detection_scores: list[float],
    category_ids: list[int] | None = None,
    category_semantics: dict[int, str] | None = None,
) -> dict[str, torch.Tensor]:
    inputs: dict[str, torch.Tensor] = {
        "frame_id": torch.tensor([frame_idx], dtype=torch.int64),
        "mask": mask_2d.to(dtype=torch.int32).unsqueeze(0),
        "object_ids": torch.tensor([object_ids], dtype=torch.int64),
        "detection_scores": torch.tensor([detection_scores], dtype=torch.float32),
    }
    if category_ids is not None:
        inputs["category_ids"] = torch.tensor([category_ids], dtype=torch.int64)
    if category_semantics is not None:
        payload = json.dumps(
            {str(category_id): name for category_id, name in sorted(category_semantics.items())},
            separators=(",", ":"),
        ).encode("utf-8")
        inputs["category_semantics"] = torch.tensor(list(payload), dtype=torch.uint8)
    return inputs


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# ---------------------------------------------------------------------------
# Image dialect (default)
# ---------------------------------------------------------------------------


def test_tracking_coco_json_writes_valid_json_after_each_frame(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path), default_category_name="person", flush_interval=1
    )

    frames = (
        _build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.5],
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
            detection_scores=[0.25],
        ),
    )

    for frame in frames:
        node.forward(**frame)
        parsed = _read_json(json_path)
        assert set(parsed.keys()) == {"info", "images", "annotations", "categories"}

    node.close()
    parsed = _read_json(json_path)
    assert [img["id"] for img in parsed["images"]] == [0, 1, 2]
    for img in parsed["images"]:
        assert img["file_name"] == f"frame_{img['id']:06d}"
        assert img["height"] == 2 and img["width"] == 2
    assert len(parsed["annotations"]) == 2
    assert parsed["categories"] == [{"id": 1, "name": "person"}]
    for ann in parsed["annotations"]:
        assert set(ann.keys()) == {
            "id",
            "image_id",
            "category_id",
            "segmentation",
            "bbox",
            "area",
            "iscrowd",
            "score",
            "track_id",
        }
        assert isinstance(ann["segmentation"], dict)
        assert ann["iscrowd"] == 1
    annotations_by_track = {ann["track_id"]: ann for ann in parsed["annotations"]}
    assert annotations_by_track[1]["image_id"] == 0
    assert annotations_by_track[1]["score"] == 0.5
    assert annotations_by_track[2]["image_id"] == 2
    assert annotations_by_track[2]["area"] == 3.0


def test_tracking_coco_json_replaces_existing_frame_idempotently(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

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
            detection_scores=[0.75],
        )
    )

    node.close()
    parsed = _read_json(json_path)
    assert [img["id"] for img in parsed["images"]] == [4]
    assert len(parsed["annotations"]) == 1
    assert parsed["annotations"][0]["score"] == 0.75


def test_tracking_coco_json_writes_empty_frame_entry(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

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
    assert parsed["images"] == [{"id": 7, "file_name": "frame_000007", "height": 3, "width": 4}]
    assert parsed["annotations"] == []


def test_tracking_coco_json_records_per_frame_dimensions(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.5],
        )
    )
    node.forward(
        **_build_inputs(
            frame_idx=1,
            mask_2d=torch.zeros((3, 4), dtype=torch.int32),
            object_ids=[],
            detection_scores=[],
        )
    )

    node.close()
    parsed = _read_json(json_path)
    dims = {img["id"]: (img["height"], img["width"]) for img in parsed["images"]}
    assert dims == {0: (2, 2), 1: (3, 4)}


def test_tracking_coco_json_validates_alignment(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    with pytest.raises(ValueError, match="identical lengths"):
        node.forward(
            **_build_inputs(
                frame_idx=0,
                mask_2d=torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
                object_ids=[1, 2],
                detection_scores=[0.9],
            )
        )


def test_tracking_coco_json_writes_multi_category_tracks_and_header(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="object")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.5],
            category_ids=[1],
            category_semantics={1: "person"},
        )
    )
    node.forward(
        **_build_inputs(
            frame_idx=1,
            mask_2d=torch.tensor([[2, 2], [0, 1]], dtype=torch.int32),
            object_ids=[1, 2],
            detection_scores=[0.5, 0.25],
            category_ids=[1, 2],
            category_semantics={1: "person", 2: "car"},
        )
    )

    node.close()
    parsed = _read_json(json_path)
    assert parsed["categories"] == [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}]
    track_categories = {(ann["track_id"], ann["category_id"]) for ann in parsed["annotations"]}
    assert track_categories == {(1, 1), (2, 2)}
    # Track 1 appears on both frames -> one annotation per (track, frame).
    assert sorted(ann["image_id"] for ann in parsed["annotations"] if ann["track_id"] == 1) == [
        0,
        1,
    ]


def test_tracking_coco_json_rejects_conflicting_track_category_ids(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="object")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.95],
            category_ids=[1],
            category_semantics={1: "person"},
        )
    )

    with pytest.raises(ValueError, match="conflicting category IDs"):
        node.forward(
            **_build_inputs(
                frame_idx=1,
                mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
                object_ids=[1],
                detection_scores=[0.94],
                category_ids=[2],
                category_semantics={1: "person", 2: "car"},
            )
        )


def test_tracking_coco_json_ignores_background_id_zero(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[0, 1],
            detection_scores=[0.75, 0.5],
        )
    )

    node.close()
    parsed = _read_json(json_path)
    assert len(parsed["annotations"]) == 1
    ann = parsed["annotations"][0]
    assert ann["track_id"] == 1
    assert ann["score"] == 0.5


def test_tracking_coco_json_atomic_write_is_parseable(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path),
        default_category_name="person",
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
        assert "images" in parsed
        assert "annotations" in parsed

    node.close()


def test_tracking_coco_json_deferred_write_on_close(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

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
    assert [img["id"] for img in parsed["images"]] == [0]
    assert len(parsed["annotations"]) == 1


def test_tracking_coco_json_close_is_idempotent(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path), default_category_name="person")

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
    assert [img["id"] for img in parsed["images"]] == [0]


def test_tracking_coco_json_flush_interval(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path), default_category_name="person", flush_interval=3
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
    assert [img["id"] for img in parsed["images"]] == [0, 1, 2]

    node.close()


def test_tracking_coco_json_validates_flush_interval(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="flush_interval"):
        CocoTrackMaskWriter(
            output_json_path=str(tmp_path / "test.json"),
            flush_interval=-1,
        )


def test_tracking_coco_json_default_category_name_defaults_to_object(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path))

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.95],
        )
    )

    node.close()
    parsed = _read_json(json_path)
    assert parsed["categories"] == [{"id": 1, "name": "object"}]


def test_tracking_coco_json_rejects_invalid_dialect(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dialect"):
        CocoTrackMaskWriter(
            output_json_path=str(tmp_path / "test.json"),
            dialect="cocovid",
        )


def test_tracking_coco_json_no_frames_writes_no_file(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(output_json_path=str(json_path))
    node.close()
    assert not json_path.exists()


# ---------------------------------------------------------------------------
# Video dialect (opt-in legacy shape)
# ---------------------------------------------------------------------------


def test_tracking_coco_json_video_dialect_writes_track_shape(tmp_path: Path) -> None:
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path),
        dialect="video",
        default_category_name="person",
        flush_interval=1,
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


def test_tracking_coco_json_video_dialect_golden_payload(tmp_path: Path) -> None:
    """Pin the exact legacy payload so the opt-in video dialect cannot drift."""
    json_path = tmp_path / "tracking_results.json"
    node = CocoTrackMaskWriter(
        output_json_path=str(json_path),
        dialect="video",
        default_category_name="person",
    )

    node.forward(
        **_build_inputs(
            frame_idx=0,
            mask_2d=torch.tensor([[0, 1], [0, 1]], dtype=torch.int32),
            object_ids=[1],
            detection_scores=[0.5],
        )
    )
    node.close()

    rle = coco_rle_encode(np.array([[0, 1], [0, 1]], dtype=np.uint8))
    expected = {
        "info": {"description": "Mask tracking results", "version": "1.0"},
        "videos": [
            {
                "id": 1,
                "name": "tracking_results",
                "frame_indices": [0],
                "start_frame": 0,
                "length": 1,
                "height": 2,
                "width": 2,
            }
        ],
        "annotations": [
            {
                "id": 1,
                "track_id": 1,
                "category_id": 1,
                "segmentations": [rle],
                "detection_scores": [0.5],
                "bboxes": [coco_rle_to_bbox(rle)],
                "areas": [2.0],
            }
        ],
        "categories": [{"id": 1, "name": "person"}],
    }
    assert _read_json(json_path) == expected
