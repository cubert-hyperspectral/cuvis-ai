"""Browser E2E tests for the node-catalog filter toolbar (``/catalogs/nodes/``).

Builds the docs site once per session, serves it over local HTTP (the theme's
instant-navigation JS needs http, not file://), and drives the filter UI with
Playwright: fold panels, Escape handling, OR tag semantics, the active-filter
strip, hash restore/hardening, the empty state, and the debounced live region.

Marked ``slow``: skipped by default, run with ``--runslow`` after a one-time
``uv run playwright install chromium``.
"""

import re
import subprocess
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("mkdocs", reason="docs extra not installed")
pytest.importorskip("playwright.sync_api", reason="playwright not installed")
pytest.importorskip("pytest_playwright", reason="pytest-playwright not installed")

from playwright.sync_api import expect  # noqa: E402

pytestmark = pytest.mark.slow

REPO_ROOT = Path(__file__).parent.parent.parent
CATALOG_PATH = "/catalogs/nodes/"


@pytest.fixture(scope="session")
def built_site(tmp_path_factory) -> Path:
    """Build the docs site once for the whole E2E session."""
    site_dir = tmp_path_factory.mktemp("site")
    result = subprocess.run(
        [sys.executable, "-m", "mkdocs", "build", "--strict", "-d", str(site_dir)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode != 0:
        pytest.fail(f"mkdocs build failed:\n{result.stdout}\n{result.stderr}")
    return site_dir


@pytest.fixture(scope="session")
def site_url(built_site):
    """Serve the built site on an ephemeral localhost port."""
    handler = partial(SimpleHTTPRequestHandler, directory=str(built_site))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    thread.join(timeout=5)


@pytest.fixture
def catalog(page, site_url):
    """A page object already on the catalog with the filter JS initialized."""
    page.set_default_timeout(10_000)
    page.goto(site_url + CATALOG_PATH)
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")
    return page


def _count(page) -> int:
    """Visible-item count parsed from '#node-filter-count' ('N of M items' or 'M items')."""
    text = page.locator("#node-filter-count").inner_text()
    return int(re.match(r"(\d+)", text).group(1))


def _total(page) -> int:
    text = page.locator("#node-filter-count").inner_text()
    return int(re.findall(r"(\d+)", text)[-1])


def test_accordion_open_close_switch(catalog):
    cats = catalog.locator("#node-filter-categories")
    tags = catalog.locator("#node-filter-tags")
    cat_toggle = catalog.locator('[data-panel="node-filter-categories"]')
    tag_toggle = catalog.locator('[data-panel="node-filter-tags"]')

    expect(cats).to_be_hidden()
    cat_toggle.click()
    expect(cats).to_be_visible()
    expect(cat_toggle).to_have_attribute("aria-expanded", "true")

    tag_toggle.click()  # accordion: opening one closes the other
    expect(tags).to_be_visible()
    expect(cats).to_be_hidden()
    expect(cat_toggle).to_have_attribute("aria-expanded", "false")

    tag_toggle.click()  # toggling the open one closes it
    expect(tags).to_be_hidden()


def test_escape_closes_panel_and_refocuses_toggle(catalog):
    tag_toggle = catalog.locator('[data-panel="node-filter-tags"]')
    tag_toggle.click()
    chip = catalog.locator("#node-filter-tags .filter-chip").first
    chip.focus()
    catalog.keyboard.press("Escape")
    expect(catalog.locator("#node-filter-tags")).to_be_hidden()
    expect(tag_toggle).to_be_focused()


def test_chip_click_filters_once_and_keeps_panel_open(catalog):
    """One click filters, a second click unfilters — fails if init() ever
    double-binds listeners (document$ replay) and a click toggles twice."""
    total = _total(catalog)
    catalog.locator('[data-panel="node-filter-categories"]').click()
    chip = catalog.locator("#node-filter-categories .filter-chip").first
    chip.click()
    expect(catalog.locator("#node-filter-categories")).to_be_visible()  # stays open
    expect(chip).to_have_attribute("aria-pressed", "true")
    assert _count(catalog) < total
    chip.click()
    expect(chip).to_have_attribute("aria-pressed", "false")
    assert _count(catalog) == total


def test_tags_combine_as_or_within_facet(catalog):
    """Two tags in the same facet OR together (union), never AND (intersection).

    For every pair among the first few tags the combined count stays within the OR
    bounds ``max(a, b) <= a|b <= a + b`` — an AND-within regression (intersection,
    ``<= min``) breaks the lower bound. At least one pair must strictly grow past
    each single count, which a dead/duplicate second chip could never do.
    """
    catalog.locator('[data-panel="node-filter-tags"]').click()
    chips = catalog.locator("#node-filter-tags .filter-chip")
    k = min(chips.count(), 6)
    assert k >= 2, "need at least two tag chips to exercise OR semantics"

    # Independent single-tag counts (click on, read, click off).
    singles = []
    for i in range(k):
        chips.nth(i).click()
        singles.append(_count(catalog))
        chips.nth(i).click()

    grew = False
    for i in range(k):
        for j in range(i + 1, k):
            chips.nth(i).click()
            chips.nth(j).click()
            both = _count(catalog)
            chips.nth(i).click()
            chips.nth(j).click()
            assert both >= max(singles[i], singles[j]), (
                "tags must OR (union), not AND, within a facet"
            )
            assert both <= singles[i] + singles[j], "union cannot exceed the sum of the parts"
            if both > singles[i] and both > singles[j]:
                grew = True
    assert grew, "no tag pair widened the result set — OR-within not demonstrated"


def test_category_and_tag_intersect_across_facets(catalog):
    """A category and a tag AND together across facets (intersection).

    The combined result is no larger than either facet alone and strictly narrower
    than the whole catalog. An OR-across regression would widen the result past a
    single facet, breaking the ``<= min`` bound.
    """
    total = _total(catalog)

    catalog.locator('[data-panel="node-filter-categories"]').click()
    cat_chips = catalog.locator("#node-filter-categories .filter-chip")
    cat_chips.nth(0).click()
    cat_only = _count(catalog)
    cat_chips.nth(0).click()  # clear before measuring the tag alone

    catalog.locator('[data-panel="node-filter-tags"]').click()  # accordion closes categories
    tag_chips = catalog.locator("#node-filter-tags .filter-chip")
    tag_chips.nth(0).click()
    tag_only = _count(catalog)

    # Re-activate the category: both facets now constrain the grid (selection persists
    # in the active-filter strip even though opening categories closes the tags panel).
    catalog.locator('[data-panel="node-filter-categories"]').click()
    cat_chips.nth(0).click()
    both = _count(catalog)

    assert both <= min(cat_only, tag_only), "category and tag must AND (intersect) across facets"
    assert both < total, "an across-facet AND must narrow the full catalog"


def test_strip_builds_removes_and_manages_focus(catalog):
    total = _total(catalog)
    cat_toggle = catalog.locator('[data-panel="node-filter-categories"]')
    cat_toggle.click()
    catalog.locator("#node-filter-categories .filter-chip").first.click()

    strip = catalog.locator("#node-filter-active")
    expect(strip).to_be_visible()
    expect(strip.locator("button")).to_have_count(1)
    expect(cat_toggle.locator(".filter-group-badge")).to_have_text("1")

    strip.locator("button").first.click()
    expect(strip).to_be_hidden()
    assert _count(catalog) == total
    expect(cat_toggle).to_be_focused()  # strip emptied: focus falls to the facet toggle


def test_bogus_hash_value_is_visible_and_removable(page, site_url):
    page.set_default_timeout(10_000)
    page.goto(site_url + CATALOG_PATH + "#category=doesnotexist")
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")

    strip_chip = page.locator("#node-filter-active button")
    expect(strip_chip).to_have_text("doesnotexist")  # raw value, still removable
    expect(page.locator("#node-filter-empty")).to_be_visible()

    strip_chip.click()
    expect(page.locator("#node-filter-empty")).to_be_hidden()
    assert _count(page) == _total(page)


def test_malformed_hash_does_not_kill_the_filter(page, site_url):
    page.set_default_timeout(10_000)
    page.goto(site_url + CATALOG_PATH + "#q=%")
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")
    total = _total(page)
    page.locator("#node-filter-search").fill("zzz-no-such-item")
    expect(page.locator("#node-filter-empty")).to_be_visible()
    page.locator("#node-filter-search").fill("")
    assert _count(page) == total


def test_empty_state_echoes_query_and_reset_focuses_search(catalog):
    search = catalog.locator("#node-filter-search")
    search.fill("zzz-no-such-item")
    empty = catalog.locator("#node-filter-empty")
    expect(empty).to_be_visible()
    expect(catalog.locator("#node-filter-empty-msg")).to_have_text(
        'No items match "zzz-no-such-item".'
    )
    catalog.locator("#node-filter-empty-reset").click()
    expect(empty).to_be_hidden()
    expect(search).to_be_focused()
    expect(search).to_have_value("")


def test_clear_is_state_aware(catalog):
    reset = catalog.locator("#node-filter-reset")
    expect(reset).to_be_hidden()  # dead chrome stays hidden at rest
    catalog.locator('[data-panel="node-filter-sources"]').click()
    catalog.locator("#node-filter-sources .filter-chip").first.click()
    expect(reset).to_be_visible()
    reset.click()
    expect(reset).to_be_hidden()
    assert _count(catalog) == _total(catalog)


def test_deep_link_restores_state_with_panels_folded(page, site_url):
    page.set_default_timeout(10_000)
    page.goto(site_url + CATALOG_PATH)
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")
    tag = page.locator("#node-filter-tags .filter-chip").first.get_attribute("data-tag")

    page.goto(site_url + CATALOG_PATH + f"#tag={tag}")
    page.reload()  # hash-only navigation does not re-run init; force a fresh load
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")

    expect(page.locator("#node-filter-tags")).to_be_hidden()  # stays folded
    tag_toggle = page.locator('[data-panel="node-filter-tags"]')
    expect(tag_toggle.locator(".filter-group-badge")).to_have_text("1")
    expect(tag_toggle).to_have_attribute("aria-label", "Tags filters, 1 selected")
    expect(page.locator("#node-filter-active button")).to_have_count(1)
    assert _count(page) < _total(page)


def test_narrow_viewport_wraps_toggles_to_second_row(page, site_url):
    page.set_viewport_size({"width": 375, "height": 800})
    page.set_default_timeout(10_000)
    page.goto(site_url + CATALOG_PATH)
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")

    search_box = page.locator("#node-filter-search").bounding_box()
    count_box = page.locator("#node-filter-count").bounding_box()
    group_box = page.locator(".filter-group-buttons").bounding_box()
    assert abs(search_box["y"] - count_box["y"]) < search_box["height"], (
        "search and count share row 1"
    )
    assert group_box["y"] > search_box["y"] + search_box["height"] / 2, "fold buttons wrap to row 2"


def test_status_live_region_catches_up_after_pause(catalog):
    status = catalog.locator("#node-filter-status")
    expect(status).to_have_attribute("aria-live", "polite")
    catalog.locator("#node-filter-search").fill("sam")
    count_text = catalog.locator("#node-filter-count").inner_text()
    expect(status).to_have_text(count_text)  # auto-waits past the ~400ms debounce


# ─── Row layout ──────────────────────────────────────────────────────────────

# Colour helpers evaluated in the page: Chromium reports color-mix() results as
# "color(srgb r g b)" and plain colours as "rgb(...)"; both parse to 0..1 channels.
_COLOR_JS = r"""
window.__cat = window.__cat || (() => {
  const parse = (s) => {
    let m = s.match(/^color\(srgb ([\d.e-]+) ([\d.e-]+) ([\d.e-]+)(?: \/ ([\d.e-]+))?\)$/);
    if (m) return [+m[1], +m[2], +m[3], m[4] === undefined ? 1 : +m[4]];
    m = s.match(/^rgba?\(([\d.]+),\s*([\d.]+),\s*([\d.]+)(?:,\s*([\d.]+))?\)$/);
    if (m) return [m[1] / 255, m[2] / 255, m[3] / 255, m[4] === undefined ? 1 : +m[4]];
    throw new Error("unparsed colour " + s);
  };
  const lum = (c) => {
    const ch = c.slice(0, 3).map((v) => (v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4));
    return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2];
  };
  const background = (el) => {
    for (let n = el; n; n = n.parentElement) {
      const c = parse(getComputedStyle(n).backgroundColor);
      if (c[3] > 0.5) return c;
    }
    return [1, 1, 1, 1];
  };
  const contrast = (fg, bg) => {
    const a = lum(fg), b = lum(bg);
    return (Math.max(a, b) + 0.05) / (Math.min(a, b) + 0.05);
  };
  const resolve = (value) => {
    const probe = document.createElement("span");
    probe.style.color = value;
    document.body.appendChild(probe);
    const out = getComputedStyle(probe).color;
    probe.remove();
    return out;
  };
  return { parse, background, contrast, resolve };
})();
"""


def _open(page, site_url, width, scheme=None, path=CATALOG_PATH):
    page.set_viewport_size({"width": width, "height": 900})
    page.set_default_timeout(10_000)
    page.goto(site_url + path)
    if path == CATALOG_PATH:
        page.wait_for_selector("#node-catalog-grid[data-filter-bound]")
    if scheme:
        page.evaluate(f"document.body.setAttribute('data-md-color-scheme', '{scheme}')")
    page.evaluate(_COLOR_JS)
    return page


def test_phone_rows_are_compact_and_coloured(page, site_url):
    """No admonition margin, padding or blue border on catalog rows at phone width."""
    _open(page, site_url, 390)
    geo = page.evaluate(
        """() => {
          const list = document.querySelector('#node-catalog-grid').getBoundingClientRect();
          const rows = Array.from(document.querySelectorAll('details.node-row')).slice(0, 2);
          const name = rows[0].querySelector('.row-name').getBoundingClientRect();
          const cat = rows[0].dataset.category;
          return {
            nameInset: name.left - list.left,
            gap: rows[1].getBoundingClientRect().top - rows[0].getBoundingClientRect().bottom,
            edge: getComputedStyle(rows[0]).borderLeftColor,
            expected: window.__cat.resolve(`var(--node-cat-${cat}-border)`),
          };
        }"""
    )
    assert geo["nameInset"] <= 24, geo
    assert abs(geo["gap"]) <= 1.5, geo
    assert geo["edge"] == geo["expected"], geo


def test_phone_rows_do_not_clip_content(page, site_url):
    """Long names and data-module pills wrap inside the row instead of overflowing it.

    The list clips overflow, so the check reads every summary descendant's box,
    not the row's width. The module path is the one element meant to truncate.
    """
    _open(page, site_url, 375)
    report = page.evaluate(
        """() => {
          const bad = [];
          for (const row of document.querySelectorAll('details.node-row')) {
            const summary = row.querySelector('summary');
            const box = summary.getBoundingClientRect();
            const inner = box.right - parseFloat(getComputedStyle(summary).paddingRight) + 0.5;
            for (const el of summary.querySelectorAll('*')) {
              if (el.closest('.row-module')) continue;
              const r = el.getBoundingClientRect();
              if (r.width && r.right > inner) {
                bad.push(row.querySelector('.row-name').textContent + ' ' + el.className);
                break;
              }
            }
          }
          const doc = document.documentElement;
          return { bad, pageScroll: doc.scrollWidth - doc.clientWidth };
        }"""
    )
    assert report["bad"] == [], report["bad"][:10]
    assert report["pageScroll"] <= 0, report


def test_desktop_rows_use_three_columns(page, site_url):
    _open(page, site_url, 1280)
    geo = page.evaluate(
        """() => {
          const row = document.querySelector('details.node-row');
          const top = (sel) => row.querySelector(sel).getBoundingClientRect();
          return { name: top('.row-name'), sum: top('.row-summary'), pill: top('.row-category') };
        }"""
    )
    assert abs(geo["name"]["y"] - geo["sum"]["y"]) <= 8, geo
    assert abs(geo["name"]["y"] - geo["pill"]["y"]) <= 8, geo
    assert geo["name"]["x"] < geo["sum"]["x"] < geo["pill"]["x"], geo


def test_opened_row_body_is_not_indented(page, site_url):
    _open(page, site_url, 390)
    row = page.locator('details.node-row[data-source="builtin"]').first
    row.locator(":scope > summary").click()
    expect(row.locator(".row-body")).to_be_visible()
    geo = row.evaluate(
        """(row) => {
          const rowBox = row.getBoundingClientRect();
          const body = row.querySelector('.row-body');
          const text = body.querySelector('p, .doc-signature, pre, h4') || body;
          const pre = body.querySelector('pre');
          return {
            inset: text.getBoundingClientRect().left - rowBox.left,
            preRight: pre ? pre.getBoundingClientRect().right : null,
            rowRight: rowBox.right,
          };
        }"""
    )
    assert geo["inset"] <= 28, geo
    if geo["preRight"] is not None:
        assert geo["preRight"] <= geo["rowRight"] + 0.5, geo


@pytest.mark.parametrize("scheme", ["default", "slate"])
def test_tag_text_meets_contrast(page, site_url, scheme):
    _open(page, site_url, 390, scheme=scheme)
    worst = page.evaluate(
        """() => {
          const seen = {};
          for (const el of document.querySelectorAll('.row-tags .tag-text')) {
            const tag = el.dataset.tag;
            if (tag in seen) continue;
            const fg = window.__cat.parse(getComputedStyle(el).color);
            seen[tag] = window.__cat.contrast(fg, window.__cat.background(el));
          }
          return Object.entries(seen).sort((a, b) => a[1] - b[1]).slice(0, 3);
        }"""
    )
    assert worst, "no tag text rendered"
    assert worst[0][1] >= 4.5, worst


@pytest.mark.parametrize("scheme", ["default", "slate"])
def test_overview_tiles_are_coloured_and_readable(page, site_url, scheme):
    _open(page, site_url, 1280, scheme=scheme, path="/catalogs/")
    tiles = page.evaluate(
        """() => Array.from(document.querySelectorAll('.cat-tile')).map((t) => {
          const label = t.querySelector('.cat-tile-label');
          const fg = window.__cat.parse(getComputedStyle(label).color);
          return {
            cat: t.dataset.category,
            border: getComputedStyle(t).borderTopColor,
            expected: window.__cat.resolve(`var(--node-cat-${t.dataset.category}-border)`),
            contrast: window.__cat.contrast(fg, window.__cat.background(t)),
          };
        })"""
    )
    assert tiles
    for tile in tiles:
        assert tile["border"] == tile["expected"], tile
        assert tile["contrast"] >= 4.5, tile


def test_deep_link_category_and_tag_values(page, site_url):
    """The hash takes stored tag values (``anomaly``), not the short chip labels."""
    _open(page, site_url, 1280)
    expected = page.evaluate(
        """() => Array.from(document.querySelectorAll('details.node-row')).filter((r) =>
          r.dataset.category === 'model' && r.dataset.tags.split(' ').includes('anomaly')).length"""
    )
    assert expected > 0
    page.goto(site_url + CATALOG_PATH + "#category=model&tag=anomaly")
    page.reload()
    page.wait_for_selector("#node-catalog-grid[data-filter-bound]")
    assert _count(page) == expected
