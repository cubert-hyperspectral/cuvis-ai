"""Tests for tools/strip_notebook_videos.py — the notebook size guard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import strip_notebook_videos as snv  # noqa: E402


def _make_notebook(*outputs: dict) -> dict:
    """Build a minimal notebook JSON with one code cell carrying the given outputs."""
    return {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": list(outputs),
                "source": [],
            }
        ],
        "metadata": {"kernelspec": {"name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def _embedded_video_output() -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/html": (
                '<video controls><source src="data:video/mp4;base64,AAAA..." '
                'type="video/mp4"></video>'
            )
        },
        "metadata": {},
    }


def _file_reference_video_output() -> dict:
    return {
        "output_type": "display_data",
        "data": {
            "text/html": '<video src="output/blood_perfusion/ndvi.mp4" controls width="640"></video>'
        },
        "metadata": {},
    }


def _matplotlib_image_output() -> dict:
    return {
        "output_type": "display_data",
        "data": {"image/png": "iVBORw0KGgo..."},
        "metadata": {},
    }


@pytest.mark.unit
def test_strip_embedded_video(tmp_path: Path) -> None:
    """data:video/... base64 outputs are stripped."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook(_embedded_video_output())), encoding="utf-8")

    removed = snv.strip_videos(nb_path)

    assert removed == 1
    written = json.loads(nb_path.read_text(encoding="utf-8"))
    assert written["cells"][0]["outputs"] == []


@pytest.mark.unit
def test_keep_file_reference_video(tmp_path: Path) -> None:
    """`<video src="path.mp4">` (embed=False) outputs are kept."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook(_file_reference_video_output())), encoding="utf-8")

    removed = snv.strip_videos(nb_path)

    assert removed == 0
    written = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(written["cells"][0]["outputs"]) == 1


@pytest.mark.unit
def test_keep_unrelated_outputs(tmp_path: Path) -> None:
    """Image/text outputs survive and embedded video is removed alongside them."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(
        json.dumps(_make_notebook(_matplotlib_image_output(), _embedded_video_output())),
        encoding="utf-8",
    )

    removed = snv.strip_videos(nb_path)

    assert removed == 1
    written = json.loads(nb_path.read_text(encoding="utf-8"))
    assert len(written["cells"][0]["outputs"]) == 1
    assert "image/png" in written["cells"][0]["outputs"][0]["data"]


@pytest.mark.unit
def test_html_as_list(tmp_path: Path) -> None:
    """Jupyter sometimes stores text/html as a list of strings; detector joins them."""
    output = {
        "output_type": "display_data",
        "data": {
            "text/html": [
                "<video controls>",
                '<source src="data:video/mp4;base64,XYZ" type="video/mp4">',
                "</video>",
            ]
        },
        "metadata": {},
    }
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook(output)), encoding="utf-8")

    assert snv.strip_videos(nb_path) == 1


@pytest.mark.unit
def test_no_outputs(tmp_path: Path) -> None:
    """Notebook with no outputs is untouched."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook()), encoding="utf-8")

    assert snv.strip_videos(nb_path) == 0


@pytest.mark.unit
def test_check_mode_clean_exits_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--check` exits 0 when no embedded video is present."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook(_file_reference_video_output())), encoding="utf-8")

    rc = snv.main(["--check", str(nb_path)])

    assert rc == 0


@pytest.mark.unit
def test_check_mode_violation_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--check` exits 1 when an embedded video is present and reports the offender."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook(_embedded_video_output())), encoding="utf-8")

    rc = snv.main(["--check", str(nb_path)])

    assert rc == 1
    err = capsys.readouterr().err
    assert str(nb_path) in err


@pytest.mark.unit
def test_main_in_place_strips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Without `--check`, `main` strips and reports."""
    nb_path = tmp_path / "nb.ipynb"
    nb_path.write_text(json.dumps(_make_notebook(_embedded_video_output())), encoding="utf-8")

    rc = snv.main([str(nb_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "stripped" in out
    written = json.loads(nb_path.read_text(encoding="utf-8"))
    assert written["cells"][0]["outputs"] == []


@pytest.mark.unit
def test_skips_non_ipynb_paths(tmp_path: Path) -> None:
    """Paths without `.ipynb` extension are ignored silently."""
    py_path = tmp_path / "x.py"
    py_path.write_text("print('hi')\n", encoding="utf-8")

    rc = snv.main([str(py_path)])

    assert rc == 0


@pytest.mark.unit
def test_skips_missing_paths(tmp_path: Path) -> None:
    """Missing paths are skipped silently."""
    rc = snv.main([str(tmp_path / "does_not_exist.ipynb")])

    assert rc == 0
