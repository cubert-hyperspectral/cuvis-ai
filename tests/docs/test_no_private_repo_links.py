"""Guard test: every GitHub blob/tree link from the public docs into this repo
must resolve to a real path (catches wrong replacement targets and renames).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"

# Files that are user-facing docs / the published README and contributing guide.
CHECKED_FILES = [
    ROOT / "mkdocs.yml",
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    *sorted(DOCS_DIR.rglob("*.md")),
]

# https://github.com/cubert-hyperspectral/cuvis-ai/(blob|tree)/main/<path>
REPO_LINK_PATTERN = re.compile(
    r"https://github\.com/cubert-hyperspectral/cuvis-ai/(?:blob|tree)/main/([^)\"\s]+)"
)


def _link_target(raw: str) -> str:
    """The repo-relative path a blob/tree link points at: fragment (`#L10`)
    and query (`?plain=1`) dropped, trailing slash removed."""
    return raw.split("#", 1)[0].split("?", 1)[0].rstrip("/")


@pytest.mark.unit
@pytest.mark.parametrize("path", CHECKED_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_repo_blob_links_resolve(path: Path) -> None:
    """Every github.com/cubert-hyperspectral/cuvis-ai blob/tree link points at
    a path that actually exists in this repo (catches wrong replacement
    targets and future renames)."""
    text = path.read_text(encoding="utf-8")
    for m in REPO_LINK_PATTERN.finditer(text):
        rel_path = _link_target(m.group(1))
        target = ROOT / rel_path
        assert target.exists(), (
            f"{path.relative_to(ROOT)} links to "
            f"github.com/cubert-hyperspectral/cuvis-ai/blob-or-tree/main/{rel_path}, "
            f"but that path does not exist in the repo"
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("docs/x.md#L10", "docs/x.md"),
        ("docs/#anchor", "docs"),
        ("x.md?plain=1", "x.md"),
        ("docs/", "docs"),
        ("docs/x.md", "docs/x.md"),
    ],
)
def test_link_target_strips_fragment_and_query(raw: str, expected: str) -> None:
    """Fragments and queries are not part of the path that has to exist."""
    assert _link_target(raw) == expected
