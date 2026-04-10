"""Tests for cuvis_ai.utils.cli_helpers."""

from __future__ import annotations

import json
from pathlib import Path

import click
import pytest

from cuvis_ai.utils.cli_helpers import (
    append_tracking_metrics,
    resolve_end_frame,
    resolve_run_output_dir,
    write_experiment_info,
)


class TestResolveRunOutputDir:
    def test_uses_source_stem(self, tmp_path: Path) -> None:
        result = resolve_run_output_dir(
            output_root=tmp_path, source_path=Path("video.mp4"), out_basename=None
        )
        assert result == tmp_path / "video"

    def test_custom_basename(self, tmp_path: Path) -> None:
        result = resolve_run_output_dir(
            output_root=tmp_path, source_path=Path("video.mp4"), out_basename="my_run"
        )
        assert result == tmp_path / "my_run"

    def test_rejects_empty_basename(self, tmp_path: Path) -> None:
        with pytest.raises(click.BadParameter, match="empty"):
            resolve_run_output_dir(
                output_root=tmp_path, source_path=Path("video.mp4"), out_basename="  "
            )

    def test_rejects_path_separator(self, tmp_path: Path) -> None:
        with pytest.raises(click.BadParameter, match="folder name"):
            resolve_run_output_dir(
                output_root=tmp_path, source_path=Path("video.mp4"), out_basename="a/b"
            )

        with pytest.raises(click.BadParameter, match="folder name"):
            resolve_run_output_dir(
                output_root=tmp_path, source_path=Path("video.mp4"), out_basename="a\\b"
            )


class TestResolveEndFrame:
    def test_passthrough_when_max_frames_none(self) -> None:
        assert resolve_end_frame(start_frame=0, end_frame=100, max_frames=None) == 100

    def test_max_frames_derives_end(self) -> None:
        assert resolve_end_frame(start_frame=10, end_frame=-1, max_frames=50) == 60

    def test_max_frames_minus_one(self) -> None:
        assert resolve_end_frame(start_frame=0, end_frame=-1, max_frames=-1) == -1

    def test_conflict_raises(self) -> None:
        with pytest.raises(click.BadParameter, match="conflict"):
            resolve_end_frame(start_frame=0, end_frame=50, max_frames=100)

    def test_consistent_values_ok(self) -> None:
        assert resolve_end_frame(start_frame=0, end_frame=50, max_frames=50) == 50

    def test_zero_max_frames_raises(self) -> None:
        with pytest.raises(click.BadParameter, match="positive"):
            resolve_end_frame(start_frame=0, end_frame=-1, max_frames=0)


class TestWriteExperimentInfo:
    def test_roundtrip(self, tmp_path: Path) -> None:
        write_experiment_info(tmp_path, model="yolo26n", frames=100)
        text = (tmp_path / "experiment_info.txt").read_text(encoding="utf-8")
        assert "model: yolo26n" in text
        assert "frames: 100" in text
        assert f"Experiment: {tmp_path.name}" in text


class TestAppendTrackingMetrics:
    def test_appends_metrics(self, tmp_path: Path) -> None:
        info_path = tmp_path / "experiment_info.txt"
        info_path.write_text("Parameters:\n", encoding="utf-8")

        tracking_json = tmp_path / "tracking.json"
        data = {
            "images": [{"id": 0}, {"id": 1}, {"id": 2}],
            "annotations": [
                {"image_id": 0, "track_id": 1},
                {"image_id": 0, "track_id": 2},
                {"image_id": 1, "track_id": 1},
            ],
        }
        tracking_json.write_text(json.dumps(data), encoding="utf-8")

        append_tracking_metrics(info_path, tracking_json)

        text = info_path.read_text(encoding="utf-8")
        assert "frames: 3" in text
        assert "unique_track_ids: 2" in text
        assert "zero_track_frames: 1" in text

    def test_missing_json_is_silent(self, tmp_path: Path) -> None:
        info_path = tmp_path / "experiment_info.txt"
        info_path.write_text("", encoding="utf-8")
        append_tracking_metrics(info_path, tmp_path / "nonexistent.json")
        assert info_path.read_text(encoding="utf-8") == ""
