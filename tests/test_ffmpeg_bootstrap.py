from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from cuvis_ai._ffmpeg_bootstrap import _register_ffmpeg_dll_dirs


def _force_windows(monkeypatch: pytest.MonkeyPatch, mock_add: MagicMock) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os, "add_dll_directory", mock_add, raising=False)


def test_noop_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_add = MagicMock()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(os, "add_dll_directory", mock_add, raising=False)

    _register_ffmpeg_dll_dirs()

    mock_add.assert_not_called()


def test_registers_avcodec_dirs_on_windows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "avcodec-60.dll").write_bytes(b"")
    mock_add = MagicMock()
    _force_windows(monkeypatch, mock_add)
    monkeypatch.setenv("PATH", str(tmp_path))

    _register_ffmpeg_dll_dirs()

    mock_add.assert_called_once_with(str(tmp_path))


def test_skips_dirs_without_avcodec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "ffmpeg.exe").write_bytes(b"")
    mock_add = MagicMock()
    _force_windows(monkeypatch, mock_add)
    monkeypatch.setenv("PATH", str(tmp_path))

    _register_ffmpeg_dll_dirs()

    mock_add.assert_not_called()


def test_dedupes_repeated_path_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "avcodec-60.dll").write_bytes(b"")
    mock_add = MagicMock()
    _force_windows(monkeypatch, mock_add)
    monkeypatch.setenv("PATH", os.pathsep.join([str(tmp_path), str(tmp_path)]))

    _register_ffmpeg_dll_dirs()

    mock_add.assert_called_once_with(str(tmp_path))


def test_swallows_oserror_and_continues(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bad_dir = tmp_path / "bad"
    good_dir = tmp_path / "good"
    bad_dir.mkdir()
    good_dir.mkdir()
    (good_dir / "avcodec-60.dll").write_bytes(b"")

    mock_add = MagicMock()
    _force_windows(monkeypatch, mock_add)
    monkeypatch.setenv("PATH", os.pathsep.join([str(bad_dir), str(good_dir)]))

    real_isdir = os.path.isdir

    def fake_isdir(path: str) -> bool:
        if path == str(bad_dir):
            raise OSError("simulated permission error")
        return real_isdir(path)

    monkeypatch.setattr("cuvis_ai._ffmpeg_bootstrap.os.path.isdir", fake_isdir)

    _register_ffmpeg_dll_dirs()

    mock_add.assert_called_once_with(str(good_dir))


def test_skips_blank_path_entries(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / "avcodec-60.dll").write_bytes(b"")
    mock_add = MagicMock()
    _force_windows(monkeypatch, mock_add)
    monkeypatch.setenv("PATH", os.pathsep.join(["", str(tmp_path), ""]))

    _register_ffmpeg_dll_dirs()

    mock_add.assert_called_once_with(str(tmp_path))
