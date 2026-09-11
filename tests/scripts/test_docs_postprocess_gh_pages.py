"""Tests for scripts/docs_postprocess_gh_pages.py — the gh-pages SEO guard
that noindexes outdated mike versions, backfills the outdated banner, strips
non-HTML per-version artifacts, and writes robots.txt.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import docs_postprocess_gh_pages as pp  # noqa: E402

pytestmark = pytest.mark.unit

SITE_URL = "https://docs.cuvis.ai"
ALIAS = "latest"


def _material_page(*, title: str = "Page", extra_head: bytes = b"") -> bytes:
    """A minimal page shaped like real Material-for-MkDocs output: the
    charset meta first in <head>, then an empty (pre-banner-fix) outdated
    div."""
    return (
        b'<!doctype html>\n<html lang="en" class="no-js">\n  <head>\n'
        b'    <meta charset="utf-8">\n'
        b'    <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        + extra_head
        + b"    <title>"
        + title.encode()
        + b"</title>\n"
        b"  </head>\n  <body>\n"
        b'    <div data-md-color-scheme="default" data-md-component="outdated" hidden></div>\n'
        b"    <p>content</p>\n"
        b"  </body>\n</html>\n"
    )


def _write_version(root: Path, version: str, pages: dict[str, bytes]) -> None:
    vdir = root / version
    vdir.mkdir(parents=True, exist_ok=True)
    for rel, content in pages.items():
        p = vdir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)


def _default_pages() -> dict[str, bytes]:
    return {
        "index.html": _material_page(title="Home"),
        "404.html": _material_page(title="Not Found"),
        "concepts/node/index.html": _material_page(title="Node"),
        "llms.txt": b"llms content\n",
        "llms-full.txt": b"llms full content\n",
        "sitemap.xml": b"<urlset></urlset>\n",
        "sitemap.xml.gz": b"\x1f\x8b\x00binary\n",
    }


def _link_or_copy_latest(root: Path, target_version: str) -> bool:
    """Create `root/latest` pointing at `target_version`. Returns True if a
    real symlink was created, False if a copytree fallback was used (e.g. no
    symlink privilege on Windows)."""
    latest = root / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_symlink() or latest.is_file():
            latest.unlink()
        else:
            shutil.rmtree(latest)
    try:
        os.symlink(target_version, latest, target_is_directory=True)
        return True
    except OSError:
        shutil.copytree(root / target_version, latest)
        return False


def _write_versions_json(root: Path, entries: list[dict]) -> None:
    (root / "versions.json").write_text(json.dumps(entries), encoding="utf-8")


@pytest.fixture
def gh_pages(tmp_path: Path) -> Path:
    root = tmp_path / "gh-pages"
    root.mkdir()
    (root / "CNAME").write_text("docs.cuvis.ai", encoding="utf-8")
    (root / ".nojekyll").write_text("", encoding="utf-8")
    (root / "index.html").write_bytes(b"<script>location.replace('latest/')</script>\n")

    _write_version(root, "0.16.2", _default_pages())
    _write_version(root, "0.16.1", _default_pages())
    _write_versions_json(
        root,
        [
            {"version": "0.16.2", "title": "0.16.2", "aliases": ["latest"]},
            {"version": "0.16.1", "title": "0.16.1", "aliases": []},
        ],
    )
    _link_or_copy_latest(root, "0.16.2")
    return root


def _all_html_bytes(root: Path) -> dict[Path, bytes]:
    snapshot = {}
    for version_dir in root.iterdir():
        if not version_dir.is_dir() or version_dir.is_symlink():
            continue
        for page in sorted(version_dir.rglob("*.html")):
            if page.is_file() and not page.is_symlink():
                snapshot[page] = page.read_bytes()
    return snapshot


class TestNoindexPlacement:
    def test_noindex_only_in_outdated_version(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        for rel in ("index.html", "404.html", "concepts/node/index.html"):
            outdated_bytes = (gh_pages / "0.16.1" / rel).read_bytes()
            assert outdated_bytes.count(pp.NOINDEX_TAG) == 1, rel

            current_bytes = (gh_pages / "0.16.2" / rel).read_bytes()
            assert pp.NOINDEX_TAG not in current_bytes, rel

    def test_idempotent_second_run(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)
        snapshot_after_first = _all_html_bytes(gh_pages)

        report = pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        assert report.changed == []
        assert report.robots_changed is False
        assert _all_html_bytes(gh_pages) == snapshot_after_first

    def test_robots_txt_content(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        assert (gh_pages / "robots.txt").read_bytes() == (
            b"Sitemap: https://docs.cuvis.ai/latest/sitemap.xml\n"
        )

    def test_fallback_to_head_tag_without_charset(self, gh_pages: Path) -> None:
        page = gh_pages / "0.16.1" / "index.html"
        no_charset = (
            b"<!doctype html>\n<html>\n  <head>\n    <title>X</title>\n  </head>\n"
            b'  <body><div data-md-color-scheme="default" data-md-component="outdated" hidden></div></body>\n</html>\n'
        )
        page.write_bytes(no_charset)

        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        result = page.read_bytes()
        assert pp.NOINDEX_TAG in result
        # inserted right after <head ...> since there is no charset meta
        head_end = result.index(b"<head>") + len(b"<head>")
        assert result[head_end : head_end + 200].lstrip(b"\n ").startswith(pp.NOINDEX_TAG)

    def test_unsupported_html_refuses_to_write(self, gh_pages: Path) -> None:
        page = gh_pages / "0.16.1" / "index.html"
        page.write_bytes(b"not html at all, no head tag anywhere\n")
        before = _all_html_bytes(gh_pages)

        with pytest.raises(SystemExit):
            pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        # nothing was written anywhere in the tree
        assert _all_html_bytes(gh_pages) == before
        assert not (gh_pages / "robots.txt").exists()


class TestBanner:
    def test_banner_injected_in_outdated_only(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        outdated = (gh_pages / "0.16.1" / "index.html").read_bytes()
        assert pp.BANNER_MARKER in outdated
        assert b'href="/"' in outdated

        current = (gh_pages / "0.16.2" / "index.html").read_bytes()
        assert pp.BANNER_MARKER not in current

    def test_no_banner_flag_leaves_div_empty(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL, banner=False)

        outdated = (gh_pages / "0.16.1" / "index.html").read_bytes()
        assert pp.BANNER_MARKER not in outdated
        assert pp.NOINDEX_TAG in outdated

    def test_existing_aside_is_not_duplicated(self, gh_pages: Path) -> None:
        # Simulate a page already built with overrides/main.html: the
        # outdated div already contains a real <aside>, not empty.
        already_banner = _material_page(title="AlreadyBannered").replace(
            b'<div data-md-color-scheme="default" data-md-component="outdated" hidden></div>',
            b'<div data-md-color-scheme="default" data-md-component="outdated" hidden>'
            b'<aside class="md-banner md-banner--warning">'
            b"<div>You're not viewing the latest version.</div></aside></div>",
        )
        page = gh_pages / "0.16.1" / "index.html"
        page.write_bytes(already_banner)

        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        result = page.read_bytes()
        assert result.count(b"md-banner--warning") == 1
        assert pp.BANNER_MARKER not in result  # our marker never got added
        assert pp.NOINDEX_TAG in result


class TestNonHtmlRemoval:
    def test_llms_and_sitemap_removed_only_from_outdated(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        for name in ("llms.txt", "llms-full.txt", "sitemap.xml", "sitemap.xml.gz"):
            assert not (gh_pages / "0.16.1" / name).exists(), name
            assert (gh_pages / "0.16.2" / name).exists(), name


class TestVersionTransitions:
    def test_previous_latest_gets_noindex_after_new_release(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)
        old_0161 = (gh_pages / "0.16.1" / "index.html").read_bytes()

        # Simulate `mike deploy 0.17.0 latest --update-aliases`
        _write_version(gh_pages, "0.17.0", _default_pages())
        _write_versions_json(
            gh_pages,
            [
                {"version": "0.17.0", "title": "0.17.0", "aliases": ["latest"]},
                {"version": "0.16.2", "title": "0.16.2", "aliases": []},
                {"version": "0.16.1", "title": "0.16.1", "aliases": []},
            ],
        )
        _link_or_copy_latest(gh_pages, "0.17.0")

        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        assert pp.NOINDEX_TAG in (gh_pages / "0.16.2" / "index.html").read_bytes()
        assert pp.NOINDEX_TAG not in (gh_pages / "0.17.0" / "index.html").read_bytes()
        # untouched old version stays byte-identical across the second run
        assert (gh_pages / "0.16.1" / "index.html").read_bytes() == old_0161

    def test_version_regaining_latest_loses_noindex(self, gh_pages: Path) -> None:
        # A version can only self-heal back to un-noindexed by *regaining*
        # the `latest` alias while remaining the highest version present --
        # e.g. mike re-running `deploy --update-aliases` for the same
        # release, or a transient CI hiccup that dropped the alias getting
        # corrected. (Moving `latest` onto an older version is a distinct,
        # invalid state that _validate() rejects outright -- see
        # TestValidation.test_refuses_when_latest_is_not_the_highest_version.)
        _write_versions_json(
            gh_pages,
            [
                {"version": "0.16.2", "title": "0.16.2", "aliases": []},
                {"version": "0.16.1", "title": "0.16.1", "aliases": []},
            ],
        )
        with pytest.raises(SystemExit):
            pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        # mike corrects it: latest goes back onto 0.16.2, still the highest.
        _write_versions_json(
            gh_pages,
            [
                {"version": "0.16.2", "title": "0.16.2", "aliases": ["latest"]},
                {"version": "0.16.1", "title": "0.16.1", "aliases": []},
            ],
        )

        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        assert pp.NOINDEX_TAG not in (gh_pages / "0.16.2" / "index.html").read_bytes()
        assert pp.NOINDEX_TAG in (gh_pages / "0.16.1" / "index.html").read_bytes()


class TestValidation:
    def test_refuses_when_no_latest_alias(self, gh_pages: Path) -> None:
        _write_versions_json(
            gh_pages,
            [
                {"version": "0.16.2", "title": "0.16.2", "aliases": []},
                {"version": "0.16.1", "title": "0.16.1", "aliases": []},
            ],
        )
        with pytest.raises(SystemExit):
            pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)
        assert not (gh_pages / "robots.txt").exists()

    def test_refuses_when_two_versions_carry_latest(self, gh_pages: Path) -> None:
        _write_versions_json(
            gh_pages,
            [
                {"version": "0.16.2", "title": "0.16.2", "aliases": ["latest"]},
                {"version": "0.16.1", "title": "0.16.1", "aliases": ["latest"]},
            ],
        )
        with pytest.raises(SystemExit):
            pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

    def test_refuses_when_latest_is_not_the_highest_version(self, gh_pages: Path) -> None:
        _write_versions_json(
            gh_pages,
            [
                {"version": "0.16.2", "title": "0.16.2", "aliases": []},
                {"version": "0.16.1", "title": "0.16.1", "aliases": ["latest"]},
            ],
        )
        with pytest.raises(SystemExit):
            pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

    def test_refuses_when_a_listed_dir_is_missing(self, gh_pages: Path) -> None:
        shutil.rmtree(gh_pages / "0.16.1")
        with pytest.raises(SystemExit):
            pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)


class TestCheckMode:
    def test_check_mode_reports_without_writing(self, gh_pages: Path) -> None:
        before = _all_html_bytes(gh_pages)

        exit_code = pp.main(["--check", "--site-url", SITE_URL, "--alias", ALIAS, str(gh_pages)])

        assert exit_code == 1
        assert _all_html_bytes(gh_pages) == before
        assert not (gh_pages / "robots.txt").exists()

    def test_check_mode_clean_after_real_run(self, gh_pages: Path) -> None:
        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        exit_code = pp.main(["--check", "--site-url", SITE_URL, "--alias", ALIAS, str(gh_pages)])

        assert exit_code == 0


class TestSymlinkHandling:
    def test_symlinked_alias_dir_never_walked(self, gh_pages: Path) -> None:
        latest = gh_pages / "latest"
        if not latest.is_symlink():
            pytest.skip("symlinks not supported/permitted on this platform")

        pp.postprocess(gh_pages, alias=ALIAS, site_url=SITE_URL)

        # the alias path stays a symlink -- never replaced by a real directory
        assert (gh_pages / "latest").is_symlink()
        # and nothing was written through it (0.16.2, the real target, is
        # correctly left un-noindexed; this assertion mainly guards against a
        # naive `rglob` over the whole root that would double-visit the
        # symlinked copy)
        assert pp.NOINDEX_TAG not in (gh_pages / "0.16.2" / "index.html").read_bytes()
