"""Post-process a gh-pages checkout after `mike deploy`: noindex every version
that is not the current `latest` alias, backfill the Material "outdated
version" banner into pages built without one, strip non-HTML files that
cannot carry a noindex tag, and write a root `robots.txt`.

Runs after `mike deploy` / `mike set-default` (both without `--push`) against
a `git worktree` of the local `gh-pages` branch, before that worktree is
committed and pushed. See `.github/actions/deploy-docs/action.yml`.

Usage
-----
    python scripts/docs_postprocess_gh_pages.py [--check] [--no-banner]
        [--alias latest] [--site-url https://docs.cuvis.ai] GH_PAGES_ROOT

`--check` exits non-zero if any of the files would be modified, without
writing them. Without `--check`, modifies in place and reports what changed.

The transform is byte-based throughout (never text-mode) so it never rewrites
line endings, and validates the whole tree before writing anything, so a
mid-run failure never leaves the branch half-processed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ALIAS = "latest"
DEFAULT_SITE_URL = "https://docs.cuvis.ai"

VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")

# The exact tag we inject and key idempotency on. A foreign
# `<meta name="robots" content="index,follow">` (if one ever appears) does
# NOT stop us from adding ours — the most restrictive robots directive wins,
# and we only ever look for our own marked tag to decide whether we already
# ran.
NOINDEX_TAG = b'<meta name="robots" content="noindex" data-cuvis-postprocess="noindex">'
MARKED_NOINDEX_RE = re.compile(
    rb'<meta\s+name="robots"\s+content="noindex"\s+data-cuvis-postprocess="noindex"\s*/?>'
)

CHARSET_RE = re.compile(rb'<meta\s+charset="utf-8"\s*/?>', re.IGNORECASE)
HEAD_OPEN_RE = re.compile(rb"<head(?:\s[^>]*)?>", re.IGNORECASE)

# Material's empty outdated-banner container, built without an
# `overrides/main.html` block: a hidden div with nothing inside (possibly
# whitespace). Pages built AFTER overrides/main.html was added already carry
# a real `<aside class="md-banner ...">` inside this div — we must not touch
# those.
EMPTY_OUTDATED_DIV_RE = re.compile(
    rb'(<div data-md-color-scheme="default" data-md-component="outdated" hidden>)'
    rb"(\s*)"
    rb"(</div>)"
)
BANNER_MARKER = b"data-cuvis-postprocess-banner"


def _banner_html() -> bytes:
    # href="/" is deliberate: the site is served from the domain root, and a
    # depth-relative link would break for 404.html, which is served at an
    # arbitrary missing URL rather than its own directory.
    return (
        b'\n          <aside class="md-banner md-banner--warning" ' + BANNER_MARKER + b">"
        b'\n            <div class="md-banner__inner md-grid md-typeset">'
        b"\n              You're not viewing the latest version."
        b' <a href="/"><strong>Click here to go to latest.</strong></a>'
        b"\n            </div>"
        b"\n          </aside>\n        "
    )


# Text/sitemap files that are per-build, per-version artifacts. They cannot
# carry a noindex tag, and an old version's sitemap advertises stale URLs, so
# they are deleted outright from every outdated version.
REMOVE_FROM_OUTDATED = ("llms.txt", "llms-full.txt", "sitemap.xml", "sitemap.xml.gz")


class PostprocessError(SystemExit):
    """Raised (as SystemExit) for any validation failure. Message goes to stderr."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class Report:
    changed: list[Path] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)
    unsupported: list[Path] = field(default_factory=list)
    robots_changed: bool = False


def _load_versions(root: Path) -> list[dict]:
    versions_path = root / "versions.json"
    if not versions_path.is_file():
        raise PostprocessError(f"{versions_path} does not exist")
    try:
        data = json.loads(versions_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PostprocessError(f"{versions_path} is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise PostprocessError(f"{versions_path} must contain a JSON list")
    return data


def _validate(root: Path, versions: list[dict], alias: str) -> tuple[list[str], list[str]]:
    """Validate the whole tree before any write. Returns (current, outdated)
    version name lists. Raises PostprocessError on any problem."""
    if not (root / "CNAME").is_file():
        raise PostprocessError(f"{root}/CNAME is missing")
    if not (root / ".nojekyll").is_file():
        raise PostprocessError(f"{root}/.nojekyll is missing")

    current: list[str] = []
    outdated: list[str] = []
    all_versions: list[str] = []

    for entry in versions:
        version = entry.get("version")
        if not isinstance(version, str) or not VERSION_RE.match(version):
            raise PostprocessError(f"versions.json has an invalid version entry: {entry!r}")
        aliases = entry.get("aliases", [])
        if not isinstance(aliases, list):
            raise PostprocessError(f"versions.json entry for {version} has non-list aliases")

        vdir = root / version
        if vdir.is_symlink():
            raise PostprocessError(f"{vdir} is unexpectedly a symlink; refusing to process")
        if not vdir.is_dir():
            raise PostprocessError(f"{vdir} is listed in versions.json but does not exist")

        all_versions.append(version)
        if alias in aliases:
            current.append(version)
        else:
            outdated.append(version)

    if len(current) == 0:
        raise PostprocessError(f"no version in versions.json carries alias {alias!r}")
    if len(current) > 1:
        raise PostprocessError(f"more than one version carries alias {alias!r}: {current!r}")

    highest = max(all_versions, key=lambda v: tuple(int(p) for p in v.split(".")))
    if current[0] != highest:
        raise PostprocessError(
            f"alias {alias!r} is on {current[0]!r}, but the highest version present is "
            f"{highest!r} — refusing to noindex a version newer than 'latest'"
        )

    return current, outdated


def _iter_html(version_dir: Path) -> list[Path]:
    out = []
    for p in sorted(version_dir.rglob("*.html")):
        if p.is_symlink() or not p.is_file():
            continue
        out.append(p)
    return out


def _inject_noindex(html: bytes) -> bytes | None:
    """Return the modified bytes, or None if already present / no insertion
    point found (caller distinguishes 'no anchor' by checking the original
    head tag separately)."""
    if MARKED_NOINDEX_RE.search(html):
        return None
    m = CHARSET_RE.search(html) or HEAD_OPEN_RE.search(html)
    if not m:
        return None
    return html[: m.end()] + b"\n      " + NOINDEX_TAG + html[m.end() :]


def _remove_noindex(html: bytes) -> bytes:
    return MARKED_NOINDEX_RE.sub(b"", html)


def _inject_banner(html: bytes) -> bytes:
    if BANNER_MARKER in html:
        return html
    return EMPTY_OUTDATED_DIV_RE.sub(
        lambda m: m.group(1) + _banner_html() + m.group(3), html, count=1
    )


def _has_head_anchor(html: bytes) -> bool:
    return bool(CHARSET_RE.search(html) or HEAD_OPEN_RE.search(html))


def _process_outdated_version(
    version_dir: Path, *, banner: bool, dry_run: bool, report: Report
) -> None:
    for page in _iter_html(version_dir):
        original = page.read_bytes()
        current = original

        if not _has_head_anchor(current) and not MARKED_NOINDEX_RE.search(current):
            report.unsupported.append(page)
            continue

        noindexed = _inject_noindex(current)
        if noindexed is not None:
            current = noindexed

        if banner:
            current = _inject_banner(current)

        if current != original:
            report.changed.append(page)
            if not dry_run:
                page.write_bytes(current)

    for name in REMOVE_FROM_OUTDATED:
        target = version_dir / name
        if target.is_file():
            report.removed.append(target)
            if not dry_run:
                target.unlink()


def _process_current_version(version_dir: Path, *, dry_run: bool, report: Report) -> None:
    for page in _iter_html(version_dir):
        original = page.read_bytes()
        current = _remove_noindex(original)
        if current != original:
            report.changed.append(page)
            if not dry_run:
                page.write_bytes(current)


def _robots_txt(site_url: str, alias: str) -> bytes:
    site_url = site_url.rstrip("/")
    return f"Sitemap: {site_url}/{alias}/sitemap.xml\n".encode("ascii")


def postprocess(
    root: Path,
    *,
    alias: str = DEFAULT_ALIAS,
    site_url: str = DEFAULT_SITE_URL,
    banner: bool = True,
    dry_run: bool = False,
) -> Report:
    """Validate, then transform, the gh-pages tree rooted at `root`.

    Raises PostprocessError (a SystemExit subclass) if the tree fails
    validation, or if any page has HTML this script cannot safely process
    (report.unsupported is non-empty) — nothing is written in that case
    either.
    """
    versions = _load_versions(root)
    current, outdated = _validate(root, versions, alias)

    report = Report()

    for version in current:
        _process_current_version(root / version, dry_run=True, report=report)
    for version in outdated:
        _process_outdated_version(root / version, banner=banner, dry_run=True, report=report)

    if report.unsupported:
        names = ", ".join(str(p.relative_to(root)) for p in report.unsupported)
        raise PostprocessError(
            f"{len(report.unsupported)} HTML file(s) have no supported <head> anchor "
            f"and no existing marked noindex tag; refusing to write anything: {names}"
        )

    if dry_run:
        want_robots = _robots_txt(site_url, alias)
        robots_path = root / "robots.txt"
        if not robots_path.is_file() or robots_path.read_bytes() != want_robots:
            report.robots_changed = True
        return report

    # Re-run for real now that validation (including the dry pass above)
    # passed clean.
    report = Report()
    for version in current:
        _process_current_version(root / version, dry_run=False, report=report)
    for version in outdated:
        _process_outdated_version(root / version, banner=banner, dry_run=False, report=report)

    want_robots = _robots_txt(site_url, alias)
    robots_path = root / "robots.txt"
    if not robots_path.is_file() or robots_path.read_bytes() != want_robots:
        report.robots_changed = True
        robots_path.write_bytes(want_robots)

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("root", type=Path, help="Path to the gh-pages checkout")
    parser.add_argument("--alias", default=DEFAULT_ALIAS)
    parser.add_argument("--site-url", default=DEFAULT_SITE_URL)
    parser.add_argument("--no-banner", action="store_true")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would change; write nothing; exit 1 if anything would change.",
    )
    args = parser.parse_args(argv)

    report = postprocess(
        args.root,
        alias=args.alias,
        site_url=args.site_url,
        banner=not args.no_banner,
        dry_run=args.check,
    )

    verb = "would change" if args.check else "changed"
    print(
        f"{len(report.changed)} html file(s) {verb}, "
        f"{len(report.removed)} file(s) {'would be removed' if args.check else 'removed'}, "
        f"robots.txt {'would change' if args.check and report.robots_changed else ('changed' if report.robots_changed else 'unchanged')}"
    )

    if args.check and (report.changed or report.removed or report.robots_changed):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
