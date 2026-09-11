"""Guard test: the public docs must never reference the private cookbook repo,
and every GitHub blob/tree link into this repo must resolve to a real path.

Regression guard for the 2026-09 removal of cuvis-ai-cookbook links (that repo
is private; every link into it was a 404 for anonymous readers).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "docs"

# Files that are user-facing docs / the published README and contributing
# guide. Maintainer-only files (CLAUDE.md, .github/copilot-instructions.md,
# CHANGELOG.md) intentionally still reference the cookbook's history and are
# out of scope for this guard.
CHECKED_FILES = [
    ROOT / "mkdocs.yml",
    ROOT / "README.md",
    ROOT / "CONTRIBUTING.md",
    *sorted(DOCS_DIR.rglob("*.md")),
]

COOKBOOK_PATTERN = re.compile(r"cuvis-ai-cookbook", re.IGNORECASE)

# https://github.com/cubert-hyperspectral/cuvis-ai/(blob|tree)/main/<path>
REPO_LINK_PATTERN = re.compile(
    r"https://github\.com/cubert-hyperspectral/cuvis-ai/(?:blob|tree)/main/([^)\"\s]+)"
)


@pytest.mark.unit
@pytest.mark.parametrize("path", CHECKED_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_cookbook_references(path: Path) -> None:
    """No published doc, mkdocs.yml, README, or CONTRIBUTING.md links to the
    private cuvis-ai-cookbook repo."""
    text = path.read_text(encoding="utf-8")
    match = COOKBOOK_PATTERN.search(text)
    if match is not None:
        context = text[max(0, match.start() - 40) : match.end() + 40]
        pytest.fail(
            f"{path.relative_to(ROOT)} still references cuvis-ai-cookbook "
            f"(private repo, 404 for anonymous readers): {context!r}"
        )


@pytest.mark.unit
@pytest.mark.parametrize("path", CHECKED_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_repo_blob_links_resolve(path: Path) -> None:
    """Every github.com/cubert-hyperspectral/cuvis-ai blob/tree link points at
    a path that actually exists in this repo (catches wrong replacement
    targets and future renames)."""
    text = path.read_text(encoding="utf-8")
    for m in REPO_LINK_PATTERN.finditer(text):
        rel_path = m.group(1).rstrip("/")
        target = ROOT / rel_path
        assert target.exists(), (
            f"{path.relative_to(ROOT)} links to "
            f"github.com/cubert-hyperspectral/cuvis-ai/blob-or-tree/main/{rel_path}, "
            f"but that path does not exist in the repo"
        )
