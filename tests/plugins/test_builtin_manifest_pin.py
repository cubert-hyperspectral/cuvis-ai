"""Guard the ``cuvis_ai_builtin`` manifest pin against the release it ships in.

A composed child environment installs cuvis-ai from the manifest's ``tag``, so a stale pin
makes every child run an older release than the host that composed it. The pin stayed at
v0.16.0 through six releases; the release stamp writes the ``## X.Y.Z - date`` heading this
guard reads, so bumping one without the other fails here.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
RELEASE_HEADING = re.compile(r"^## (\d+\.\d+\.\d+) - \d{4}-\d{2}-\d{2}\s*$", re.MULTILINE)


def test_builtin_manifest_tag_is_the_newest_changelog_version() -> None:
    """``tag`` equals ``v<version>`` of the first release heading in CHANGELOG.md."""
    match = RELEASE_HEADING.search((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert match, "CHANGELOG.md has no '## X.Y.Z - YYYY-MM-DD' heading"
    manifest = yaml.safe_load(
        (ROOT / "cuvis_ai/configs/plugins/cuvis_ai_builtin.yaml").read_text(encoding="utf-8")
    )
    assert manifest["name"] == "cuvis_ai_builtin"
    assert manifest["tag"] == f"v{match.group(1)}", (
        f"manifest pins {manifest['tag']}, CHANGELOG.md leads with {match.group(1)}"
    )
