from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import click
import pytest
from click.testing import CliRunner
from torch.utils.data import Subset

import examples.object_tracking.sam3.sam3_tracker as sam3_mod


class _FakePredictDataset:
    def __init__(self, total_frames: int, fps: float = 15.0) -> None:
        self._total_frames = total_frames
        self.fps = fps

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> dict[str, int]:
        return {"frame_id": idx}


class _RecordingCU3SDataModule:
    init_calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.__class__.init_calls.append(dict(kwargs))
        predict_ids = kwargs.get("predict_ids")
        total_frames = len(predict_ids) if predict_ids is not None else 20
        self.predict_ds = _FakePredictDataset(total_frames=total_frames, fps=12.5)

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        return None


class _FakeCU3SDataNode:
    def __init__(self, **_: object) -> None:
        self.outputs = SimpleNamespace(
            cube="cube_out", wavelengths="wavelengths_out", mesu_index="mesu_out"
        )


class _FakeFalseRGBSelector:
    def __init__(self, **_: object) -> None:
        self.cube = "cube_in"
        self.wavelengths = "wavelengths_in"
        self.rgb_image = "false_rgb_out"


class _RecordingVideoDataModule:
    init_calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.__class__.init_calls.append(dict(kwargs))
        end_frame = int(kwargs["end_frame"])
        total_frames = 15 if end_frame == -1 else end_frame
        self.predict_ds = _FakePredictDataset(total_frames=total_frames, fps=14.0)
        self.fps = 24.0

    def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
        return None


class _FakeVideoFrameNode:
    def __init__(self, **_: object) -> None:
        self.outputs = SimpleNamespace(rgb_image="video_rgb_out", frame_id="video_frame_out")


class _FakePipeline:
    def format_profiling_summary(self, total_frames: int) -> str:
        return f"summary for {total_frames}"


def _fake_false_rgb_initializer(
    false_rgb: object, predict_ds: object, sample_fraction: float
) -> list[int]:  # noqa: ARG001
    assert sample_fraction == 0.05
    return [0, max(len(predict_ds) - 1, 0)]


def test_cli_requires_exactly_one_source(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(sam3_mod.main, [])
    assert result.exit_code != 0
    assert "Exactly one of --cu3s-path or --video-path must be provided." in result.output

    cu3s = tmp_path / "sample.cu3s"
    video = tmp_path / "sample.mp4"
    cu3s.write_bytes(b"")
    video.write_bytes(b"")
    result = runner.invoke(
        sam3_mod.main,
        ["--cu3s-path", str(cu3s), "--video-path", str(video)],
    )
    assert result.exit_code != 0
    assert "Exactly one of --cu3s-path or --video-path must be provided." in result.output


def test_cli_rejects_negative_start_frame(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    runner = CliRunner()
    result = runner.invoke(
        sam3_mod.main,
        ["--video-path", str(video), "--start-frame", "-1"],
    )
    assert result.exit_code != 0
    assert "--start-frame must be zero or positive" in result.output


def test_cli_rejects_invalid_end_frame(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    runner = CliRunner()
    result = runner.invoke(
        sam3_mod.main,
        ["--video-path", str(video), "--end-frame", "0"],
    )
    assert result.exit_code != 0
    assert "--end-frame must be -1 or positive" in result.output


def test_cli_rejects_end_frame_not_after_start_frame(tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")
    runner = CliRunner()
    result = runner.invoke(
        sam3_mod.main,
        ["--video-path", str(video), "--start-frame", "5", "--end-frame", "5"],
    )
    assert result.exit_code != 0
    assert "--end-frame must be greater than --start-frame" in result.output


def test_cli_help_smoke() -> None:
    runner = CliRunner()
    result = runner.invoke(sam3_mod.main, ["--help"])
    assert result.exit_code == 0, result.output
    assert "--cu3s-path" in result.output
    assert "--video-path" in result.output
    assert "--start-frame" in result.output
    assert "--out-basename" in result.output


def test_video_mode_logs_processing_mode_is_ignored(monkeypatch, tmp_path: Path) -> None:
    video = tmp_path / "sample.mp4"
    video.write_bytes(b"")

    recorded_messages: list[str] = []

    def fake_info(message: str, *args: object, **_: object) -> None:
        recorded_messages.append(message.format(*args))

    monkeypatch.setattr(sam3_mod.logger, "info", fake_info)
    monkeypatch.setattr(sam3_mod, "run_sam3_tracker", lambda **_: None)

    runner = CliRunner()
    result = runner.invoke(
        sam3_mod.main,
        ["--video-path", str(video), "--processing-mode", "Raw"],
    )
    assert result.exit_code == 0, result.output
    assert "Ignoring --processing-mode because --video-path is set" in recorded_messages


def test_build_source_context_uses_predict_ids_for_cu3s() -> None:
    _RecordingCU3SDataModule.init_calls.clear()

    context = sam3_mod._build_source_context(
        cu3s_path=Path("sample.cu3s"),
        video_path=None,
        processing_mode="SpectralRadiance",
        start_frame=3,
        end_frame=8,
        single_cu3s_datamodule_cls=_RecordingCU3SDataModule,
        cu3s_data_node_cls=_FakeCU3SDataNode,
        false_rgb_selector_cls=_FakeFalseRGBSelector,
        false_rgb_initializer=_fake_false_rgb_initializer,
    )

    assert len(_RecordingCU3SDataModule.init_calls) == 2
    assert _RecordingCU3SDataModule.init_calls[0]["cu3s_file_path"] == "sample.cu3s"
    assert _RecordingCU3SDataModule.init_calls[1]["predict_ids"] == [3, 4, 5, 6, 7]
    assert context.source_type == "cu3s"
    assert context.source_rgb_port == "false_rgb_out"
    assert context.source_frame_id_port == "mesu_out"
    assert context.source_connections == [
        ("cube_out", "cube_in"),
        ("wavelengths_out", "wavelengths_in"),
    ]
    assert context.target_frames == 5
    assert context.dataset_fps == 12.5


def test_build_source_context_uses_subset_for_video() -> None:
    _RecordingVideoDataModule.init_calls.clear()

    context = sam3_mod._build_source_context(
        cu3s_path=None,
        video_path=Path("sample.mp4"),
        processing_mode="SpectralRadiance",
        start_frame=4,
        end_frame=10,
        video_frame_datamodule_cls=_RecordingVideoDataModule,
        video_frame_node_cls=_FakeVideoFrameNode,
    )

    assert len(_RecordingVideoDataModule.init_calls) == 1
    assert _RecordingVideoDataModule.init_calls[0]["video_path"] == "sample.mp4"
    assert _RecordingVideoDataModule.init_calls[0]["end_frame"] == 10
    assert isinstance(context.datamodule.predict_ds, Subset)
    assert list(context.datamodule.predict_ds.indices) == [4, 5, 6, 7, 8, 9]
    assert context.source_type == "video"
    assert context.source_rgb_port == "video_rgb_out"
    assert context.source_frame_id_port == "video_frame_out"
    assert context.source_connections == []
    assert context.target_frames == 6
    assert context.dataset_fps == 24.0


def test_write_profiling_summary_writes_to_output_root(tmp_path: Path) -> None:
    output_path = sam3_mod._write_profiling_summary(tmp_path, _FakePipeline(), total_frames=7)

    assert output_path == tmp_path / "profiling_summary.txt"
    assert output_path.read_text() == "summary for 7"


def test_resolve_run_output_dir_uses_source_stem_by_default() -> None:
    resolved = sam3_mod._resolve_run_output_dir(
        output_root=Path("tracking_output"),
        source_path=Path("Auto_013+01.cu3s"),
        out_basename=None,
    )
    assert resolved == Path("tracking_output") / "Auto_013+01"


def test_resolve_run_output_dir_rejects_invalid_basename() -> None:
    with pytest.raises(click.BadParameter, match="whitespace only"):
        sam3_mod._resolve_run_output_dir(
            output_root=Path("tracking_output"),
            source_path=Path("Auto_013+01.cu3s"),
            out_basename="   ",
        )
    with pytest.raises(click.BadParameter, match="folder name, not a path"):
        sam3_mod._resolve_run_output_dir(
            output_root=Path("tracking_output"),
            source_path=Path("Auto_013+01.cu3s"),
            out_basename="bad/name",
        )
