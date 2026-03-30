"""Guard against Markdown and docstring list rendering regressions."""

from __future__ import annotations

import re
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
MARKDOWN_ROOTS = (ROOT_DIR / "docs", ROOT_DIR / "examples")
PYTHON_ROOT = ROOT_DIR / "cuvis_ai"

LIST_RE = re.compile(r"^[ \t]*(?:[-*+]|\d+\.)\s+")
FENCE_RE = re.compile(r"^[ \t]*(```|~~~)")


def _iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for root in MARKDOWN_ROOTS:
        files.extend(sorted(root.rglob("*.md")))
    return files


def _iter_python_files() -> list[Path]:
    return sorted(PYTHON_ROOT.rglob("*.py"))


def _find_markdown_spacing_issues(text: str) -> list[int]:
    lines = text.splitlines()
    in_fence = False
    bad_lines: list[int] = []

    for idx, line in enumerate(lines[1:], start=1):
        stripped = line.lstrip()
        if FENCE_RE.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence or not LIST_RE.match(line):
            continue

        prev = lines[idx - 1]
        if prev.strip() == "" or LIST_RE.match(prev):
            continue

        bad_lines.append(idx + 1)

    return bad_lines


def _find_python_spacing_issues(text: str) -> list[int]:
    lines = text.splitlines()
    bad_lines: list[int] = []

    for idx, line in enumerate(lines[1:], start=1):
        if not LIST_RE.match(line):
            continue

        prev = lines[idx - 1]
        if prev.strip() == "" or LIST_RE.match(prev):
            continue

        bad_lines.append(idx + 1)

    return bad_lines


def _format_error(path: Path, bad_lines: list[int]) -> str:
    relative = path.relative_to(ROOT_DIR)
    sample = ", ".join(str(line) for line in bad_lines[:10])
    return (
        f"{relative} is missing a blank line before list items at lines: {sample}. "
        "Insert one empty line between prose/labels and the list block."
    )


def test_markdown_files_have_blank_lines_before_lists():
    """Markdown prose should be separated from following list blocks."""
    failures: list[str] = []

    for path in _iter_markdown_files():
        bad_lines = _find_markdown_spacing_issues(path.read_text(encoding="utf-8"))
        if bad_lines:
            failures.append(_format_error(path, bad_lines))

    assert not failures, "\n".join(failures)


def test_python_docstrings_have_blank_lines_before_lists():
    """Docstrings that feed MkDocs should separate prose from following lists."""
    failures: list[str] = []

    for path in _iter_python_files():
        bad_lines = _find_python_spacing_issues(path.read_text(encoding="utf-8"))
        if bad_lines:
            failures.append(_format_error(path, bad_lines))

    assert not failures, "\n".join(failures)
