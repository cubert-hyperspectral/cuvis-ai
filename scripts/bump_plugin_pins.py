"""Refresh plugin manifest tag pins from each plugin's latest GitHub release.

Reads ``cuvis_ai/configs/plugins/*.yaml``; for every plugin sourced from a git ``repo`` + ``tag``,
queries the repo's latest published release on GitHub and, when that tag is newer than the
pinned one, rewrites the manifest's top-level ``tag:`` in place. Local-``path`` (dev /
self-reference) and untagged entries are skipped, as are entries whose pinned or latest
tag is not a plain ``vX.Y.Z`` semver.

This is a **tag-only** bump: the rich ``capabilities:`` block (ports, icon, category) is not
regenerated, because that needs the plugin installed and introspected. To avoid silently
shipping a stale node list when a release adds nodes, each bump runs a best-effort
**capability drift** check: the plugin's own declared node set at the new tag is fetched and
compared against the manifest's ``capabilities`` class names. The check is one-directional --
a class the release declares but the manifest omits is flagged as a likely new node, while the
manifest listing *more* nodes than the plugin's example is expected (example manifests are often
curated subsets) and is not flagged. A flagged or unverifiable node set signals the workflow to
mark the PR for manual capabilities regeneration; the tag is bumped either way. Removals are not
detected by this heuristic (they surface at provision/load time).

Prints one ``bumped <name>: <old> -> <new>`` line per change and, when ``CHANGELOG.md`` has an
``## [Unreleased]`` section, appends a single bump bullet there (otherwise the changelog is left
to the human reviewer, since entries are hand-curated).

Used by ``.github/workflows/plugin_pin_bump.yml``; run from the repo root. Honors
``GITHUB_TOKEN`` / ``GH_TOKEN`` from the environment to lift the GitHub API rate limit and to
read private plugin repos.
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
# and commented lines start with '#', so neither is matched. Group 3 captures any trailing
# whitespace + inline comment (e.g. `tag: "v0.2.1"  # floors ...`) so the bump preserves it.
_TAG_LINE = re.compile(r'^(tag:[ \t]*)(["\']?)v\d+\.\d+\.\d+\2([ \t]*(?:#.*)?)$', re.MULTILINE)

CATALOG = Path("cuvis_ai/configs/plugins")
CHANGELOG = Path("CHANGELOG.md")


def _token() -> str | None:
    """Return a GitHub token from the environment, if present."""
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _semver(tag: str) -> tuple[int, int, int] | None:
    """Return ``(major, minor, patch)`` for a ``vX.Y.Z`` tag, else ``None``."""
    match = _SEMVER.match(tag.strip())
    return (int(match[1]), int(match[2]), int(match[3])) if match else None


def _extract_class_names(entries: object) -> set[str]:
    """Pull node class names from a manifest ``capabilities`` / ``provides`` list."""
    names: set[str] = set()
    if not isinstance(entries, list):
        return names
    for entry in entries:
        if isinstance(entry, dict) and entry.get("class_name"):
            names.add(entry["class_name"])
        elif isinstance(entry, str):
            names.add(entry)
    return names


def _manifest_node_set(doc: dict) -> set[str] | None:
    """Class-name set from a plugin manifest doc (capabilities/provides, top-level or nested)."""
    for key in ("capabilities", "provides"):
        if doc.get(key):
            return _extract_class_names(doc[key])
    plugins = doc.get("plugins")
    if isinstance(plugins, dict):
        names: set[str] = set()
        for cfg in plugins.values():
            if isinstance(cfg, dict):
                names |= _extract_class_names(cfg.get("capabilities") or cfg.get("provides"))
        return names or None
    return None


def _latest_release_tag(owner_repo: str) -> str | None:
    """Return the latest published release tag for ``owner/repo``, or ``None``."""
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "cuvis-ai-pin-bot"}
    token = _token()
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


def _fetch_text(owner_repo: str, path: str, ref: str) -> str | None:
    """Return a repo file's raw text at ``ref`` via the contents API, or ``None`` if absent."""
    headers = {"Accept": "application/vnd.github.raw", "User-Agent": "cuvis-ai-pin-bot"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"https://api.github.com/repos/{owner_repo}/contents/{path}?ref={ref}"
    request = urllib.request.Request(url, headers=headers)  # noqa: S310 - fixed github host
    try:
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            return response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise


def _released_node_set(owner_repo: str, name: str, tag: str) -> set[str] | None:
    """Best-effort: the plugin's own declared node set at ``tag``, or ``None`` if not locatable."""
    candidates = [
        "examples/plugins.yaml",
        f"examples/{name}/plugins.yaml",
        f"cuvis_ai/configs/plugins/{name}.yaml",
        "plugins.yaml",
    ]
    for path in candidates:
        text = _fetch_text(owner_repo, path, tag)
        if text is None:
            continue
        nodes = _manifest_node_set(yaml.safe_load(text) or {})
        if nodes is not None:
            return nodes
    return None


def bump_pins() -> tuple[list[tuple[str, str, str]], list[tuple[str, set[str]]]]:
    """Rewrite stale ``tag:`` pins in place.

    Returns ``(bumps, drifts)`` where ``bumps`` is ``(name, old, new)`` and ``drifts`` is
    ``(name, added)``: node class names the release declares but the manifest omits. An empty
    ``added`` set is the sentinel for "node set could not be verified".
    """
    bumps: list[tuple[str, str, str]] = []
    drifts: list[tuple[str, set[str]]] = []
    for manifest in sorted(CATALOG.glob("*.yaml")):
        data = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
        name, repo, tag = data.get("name"), data.get("repo"), data.get("tag")
        if not (name and repo and tag):
            continue  # local-path / self-reference / non-tagged entry
        match = _GITHUB.match(repo)
        if not match:
            print(f"skip {name}: unsupported repo url {repo}")
            continue
        owner_repo = match.group(1)
        latest = _latest_release_tag(owner_repo)
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
        new_text, count = _TAG_LINE.subn(rf"\g<1>\g<2>{latest}\g<2>\g<3>", text)
        if count != 1:
            print(f"WARN {name}: expected one top-level tag line, found {count}; skipped")
            continue
        manifest.write_text(new_text, encoding="utf-8")
        bumps.append((name, tag, latest))
        print(f"bumped {name}: {tag} -> {latest}")

        manifest_nodes = _extract_class_names(data.get("capabilities"))
        released_nodes = _released_node_set(owner_repo, name, latest)
        if released_nodes is None:
            print(f"  ! {name}: could not verify node set at {latest}; check capabilities manually")
            drifts.append((name, set()))
        elif released_nodes - manifest_nodes:
            # One-directional: a class the release declares but the catalog manifest omits is a
            # likely new node. The reverse (manifest has more) is expected, since a plugin's own
            # example manifest is often a curated subset, so it is not treated as drift.
            added = released_nodes - manifest_nodes
            print(f"  ! {name}: manifest is missing node(s) this release declares: {sorted(added)}")
            drifts.append((name, added))
    return bumps, drifts


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


def _set_output(key: str, value: str) -> None:
    """Write a GitHub Actions step output, if running under Actions."""
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a", encoding="utf-8") as handle:
            handle.write(f"{key}={value}\n")


def main() -> None:
    """Bump stale pins, report drift, and emit outputs the PR-opening workflow consumes."""
    bumps, drifts = bump_pins()
    if not bumps:
        print("no plugin pins to bump")
        return
    append_changelog(bumps)
    print(f"\n{len(bumps)} pin(s) bumped:")
    for name, old, new in bumps:
        print(f"  - {name}: {old} -> {new}")
    if drifts:
        print("\n>>> CAPABILITIES REVIEW NEEDED <<<")
        print("A bumped release declares nodes the manifest may be missing (or its node set could")
        print("not be verified), so the capabilities block likely needs a manual regen for:")
        for name, added in drifts:
            if not added:
                print(f"  - {name}: node set could not be verified at the new tag")
            else:
                print(f"  - {name}: missing node(s): {sorted(added)}")
        _set_output("drift", "true")


if __name__ == "__main__":
    main()
