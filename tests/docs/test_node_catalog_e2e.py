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


def test_two_tags_widen_results_or_semantics(catalog):
    catalog.locator('[data-panel="node-filter-tags"]').click()
    chips = catalog.locator("#node-filter-tags .filter-chip")
    chips.nth(0).click()
    one_tag = _count(catalog)
    chips.nth(1).click()
    two_tags = _count(catalog)
    assert two_tags >= one_tag, "tags must combine as OR within the facet"


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
