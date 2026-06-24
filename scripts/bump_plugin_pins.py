"""Refresh plugin manifest tag pins from each plugin's latest GitHub release.

Reads ``configs/plugins/*.yaml``; for every plugin sourced from a git ``repo`` + ``tag``,
queries the repo's latest published release on GitHub and, when that tag is newer than the
pinned one, rewrites the manifest's top-level ``tag:`` in place. Local-``path`` (dev /
self-reference) and untagged entries are skipped, as are entries whose pinned or latest
tag is not a plain ``vX.Y.Z`` semver.

Prints one ``bumped <name>: <old> -> <new>`` line per change and, when ``CHANGELOG.md``
has an ``## [Unreleased]`` section, appends a single bump bullet there (otherwise the
changelog is left to the human reviewer, since entries are hand-curated).

Used by ``.github/workflows/plugin_pin_bump.yml``; run from the repo root. Honors
``GITHUB_TOKEN`` / ``GH_TOKEN`` from the environment to lift the GitHub API rate limit.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import yaml

_GITHUB = re.compile(r"(?:git@github\.com:|https?://github\.com/)(.+?)(?:\.git)?$")
_SEMVER = re.compile(r"^v(\d+)\.(\d+)\.(\d+)$")
# Matches exactly one top-level (column-0) tag line; capabilities entries are indented
# and commented lines start with '#', so neither is matched.
_TAG_LINE = re.compile(r'^(tag:[ \t]*)(["\']?)v\d+\.\d+\.\d+\2[ \t]*$', re.MULTILINE)

CATALOG = Path("configs/plugins")
CHANGELOG = Path("CHANGELOG.md")


def _semver(tag: str) -> tuple[int, int, int] | None:
    """Return ``(major, minor, patch)`` for a ``vX.Y.Z`` tag, else ``None``."""
    match = _SEMVER.match(tag.strip())
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def _latest_release_tag(owner_repo: str) -> str | None:
    """Return the latest published release tag for ``owner/repo``, or ``None``."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "cuvis-ai-pin-bot"}
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{owner_repo}/releases/latest"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed github host
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return json.load(response).get("tag_name")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None  # no published release yet
        raise


def bump_pins() -> list[tuple[str, str, str]]:
    """Rewrite stale ``tag:`` pins in place; return the list of ``(name, old, new)``."""
    bumps: list[tuple[str, str, str]] = []
    for manifest in sorted(CATALOG.glob("*.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        name, repo, tag = data.get("name"), data.get("repo"), data.get("tag")
        if not (name and repo and tag):
            continue  # local-path / self-reference / non-tagged entry
        match = _GITHUB.match(repo)
        if not match:
            print(f"skip {name}: unsupported repo url {repo}")
            continue
        latest = _latest_release_tag(match.group(1))
        if not latest:
            print(f"skip {name}: no published release")
            continue
        current, newest = _semver(tag), _semver(latest)
        if current is None or newest is None:
            print(f"skip {name}: non-semver tag (pinned {tag!r}, latest {latest!r})")
            continue
        if newest <= current:
            continue
        text = manifest.read_text(encoding="utf-8")
        new_text, count = _TAG_LINE.subn(rf"\g<1>\g<2>{latest}\g<2>", text)
        if count != 1:
            print(f"WARN {name}: expected one top-level tag line, found {count}; skipped")
            continue
        manifest.write_text(new_text, encoding="utf-8")
        bumps.append((name, tag, latest))
        print(f"bumped {name}: {tag} -> {latest}")
    return bumps


def append_changelog(bumps: list[tuple[str, str, str]]) -> None:
    """Append a bump bullet under an existing ``## [Unreleased]`` heading, if present."""
    if not CHANGELOG.exists():
        return
    text = CHANGELOG.read_text(encoding="utf-8")
    marker = "## [Unreleased]"
    if marker not in text:
        print("CHANGELOG: no [Unreleased] section; leaving the changelog to the reviewer")
        return
    listed = ", ".join(f"{name} {old} -> {new}" for name, old, new in bumps)
    bullet = f"- Bumped plugin manifest pins: {listed}."
    CHANGELOG.write_text(text.replace(marker, f"{marker}\n\n{bullet}", 1), encoding="utf-8")
    print("CHANGELOG: appended bump bullet under [Unreleased]")


def main() -> None:
    """Bump stale pins and emit a summary (consumed by the PR-opening workflow step)."""
    bumps = bump_pins()
    if not bumps:
        print("no plugin pins to bump")
        return
    append_changelog(bumps)
    print(f"\n{len(bumps)} pin(s) bumped:")
    for name, old, new in bumps:
        print(f"  - {name}: {old} -> {new}")


if __name__ == "__main__":
    main()
