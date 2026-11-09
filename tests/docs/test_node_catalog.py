"""Tests for the docs node-catalog generator (``scripts/generate_node_catalog.py``).

The generator feeds the Catalogs → Nodes page. These tests pin the plugin
side of the collection: capabilities must come from the repo's plugin
manifest YAMLs (nodes and data modules), and the built-in mirror manifest
must dedupe against the live built-in import.
"""

import importlib.util
import inspect
import sys
from importlib import import_module
from pathlib import Path

import pytest
from cuvis_ai_schemas.plugin import load_plugin_manifest

from cuvis_ai_core.data.public_datasets import PublicDatasets

pytest.importorskip("mkdocs_gen_files", reason="docs extra not installed")

# Plugin classes that share a short name with a built-in but are separate implementations, so
# the catalog lists both.
NAME_ONLY_MATCHES = {"DiceLoss", "CrossEntropyLoss", "ClassMapAccumulator", "PatchSampler"}


def _class_of(dotted_path: str) -> type:
    module_path, name = dotted_path.rsplit(".", 1)
    return getattr(import_module(module_path), name)


SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "generate_node_catalog.py"


@pytest.fixture(scope="module")
def catalog():
    """Import the generator script without triggering a docs build.

    The script only calls ``main()`` when ``__name__`` is ``__main__`` or
    ``<run_path>`` (mkdocs-gen-files runs it via ``runpy.run_path``), so a
    plain importlib load exposes the collection functions side-effect free.
    The module must be registered in ``sys.modules`` before execution so the
    dataclass decorator can resolve the PEP 563 string annotations.
    """
    spec = importlib.util.spec_from_file_location("_node_catalog_gen", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_plugin_nodes_collected_from_manifests(catalog):
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    assert entries, "no plugin capabilities collected from the plugin manifests"

    sam3_nodes = [e for e in entries if e.plugin_name == "sam3" and e.kind == "node"]
    assert sam3_nodes, "sam3 manifest capabilities missing from the catalog"
    sam3 = sam3_nodes[0]
    assert sam3.category in catalog._VALID_CATEGORIES
    assert sam3.tags, "manifest tags not carried into the entry"
    assert sam3.summary, "doc_summary not carried into the entry"
    assert sam3.input_specs and sam3.output_specs, "port specs not carried into the entry"
    assert sam3.repo_url and sam3.repo_url.startswith("https://")
    assert sam3.version, "git manifest tag not carried into the entry"
    assert sam3.source == "plugin"


def test_data_modules_collected(catalog):
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    data_modules = {e.data_module_name: e for e in entries if e.kind == "data_module"}
    assert "cu3s" in data_modules, "cu3s data module missing from the catalog"
    cu3s = data_modules["cu3s"]
    assert cu3s.plugin_name == "cuvis_ai_dataloader"
    assert "cu3s" in cu3s.extras
    assert cu3s.summary, "data-module doc_summary missing"
    assert cu3s.source == "data-module"


def test_builtin_mirror_dedupes_against_live_import(catalog):
    unfiltered = catalog.collect_plugin_nodes(exclude_dotted=set())
    mirror_paths = {e.dotted_path for e in unfiltered if e.plugin_name == "cuvis_ai_builtin"}
    assert mirror_paths, "cuvis_ai_builtin.yaml mirror entries not parsed"

    deduped = catalog.collect_plugin_nodes(exclude_dotted=mirror_paths)
    assert not any(e.plugin_name == "cuvis_ai_builtin" for e in deduped)
    assert any(e.plugin_name == "sam3" for e in deduped)


def test_browse_url_normalizes_git_remotes(catalog):
    assert (
        catalog._browse_url("https://github.com/cubert-hyperspectral/cuvis-ai-sam3.git")
        == "https://github.com/cubert-hyperspectral/cuvis-ai-sam3"
    )
    assert catalog._browse_url("git@gitlab.com:user/repo.git") == "https://gitlab.com/user/repo"


def test_plugin_collection_counts_have_a_floor(catalog):
    """Pin count floors so a silent regression to a handful of nodes fails.

    Current manifest-driven collection is ~70 plugin nodes across 16 plugins,
    8 data modules, and ~165 built-in mirror entries. These are floors, not
    exact counts, so adding or dropping a plugin does not break the test while a
    collapse back to "0 from plugins" (the bug this generator fixes) does.
    """
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    plugin_nodes = [e for e in entries if e.kind == "node" and e.plugin_name != "cuvis_ai_builtin"]
    builtin_mirror = [e for e in entries if e.plugin_name == "cuvis_ai_builtin"]
    data_modules = [e for e in entries if e.kind == "data_module"]

    assert len(plugin_nodes) >= 30, f"only {len(plugin_nodes)} plugin nodes collected"
    assert len(data_modules) >= 5, f"only {len(data_modules)} data modules collected"
    assert len(builtin_mirror) >= 100, f"only {len(builtin_mirror)} built-in mirror entries"
    assert len({e.plugin_name for e in plugin_nodes}) >= 6, "plugin diversity collapsed"


def test_local_path_prototypes_excluded_from_catalog(catalog):
    """Unreleased local-path prototype manifests never reach the published catalog.

    ``detr`` / ``turbovec`` / ``bytetrack`` ship ``path:`` dev pins (no installable
    ``repo:`` + ``tag:``); they would otherwise render as uninstallable, repo-less
    rows. Every catalogued *plugin* entry must be git-sourced (carry a ``repo_url``);
    the only permitted local-path entries are the built-in self-mirror.
    """
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    external_plugins = [e for e in entries if e.plugin_name != "cuvis_ai_builtin"]

    prototype_names = {"RTDETRDetection", "QuantizedSpectralSearch", "ByteTrack"}
    leaked = [e.dotted_path for e in external_plugins if e.name in prototype_names]
    assert not leaked, f"local-path prototype nodes leaked into the catalog: {leaked}"

    repo_less = [e.dotted_path for e in external_plugins if not e.repo_url]
    assert not repo_less, f"non-git plugin entries in the catalog: {repo_less}"


def test_empty_manifest_dir_raises(catalog, monkeypatch, tmp_path):
    """An empty plugins directory must fail the docs build, not ship an empty list."""
    monkeypatch.setattr(catalog, "PLUGIN_MANIFEST_DIR", tmp_path)
    with pytest.raises(RuntimeError, match="no plugin manifests"):
        catalog.collect_plugin_nodes(exclude_dotted=set())


def test_unparseable_manifest_raises(catalog, monkeypatch, tmp_path):
    """A malformed manifest must propagate an error, not be silently dropped."""
    (tmp_path / "broken.yaml").write_text("name: broken\ncapabilities: [oops\n")
    monkeypatch.setattr(catalog, "PLUGIN_MANIFEST_DIR", tmp_path)
    with pytest.raises(Exception):  # noqa: B017 - any load/validation error is acceptable
        catalog.collect_plugin_nodes(exclude_dotted=set())


def test_zero_collected_capabilities_raises(catalog, monkeypatch, tmp_path):
    """Parsing multiple manifests but collecting zero capabilities must raise.

    This is the exact "0 from plugins" regression the generator guards against;
    reproduced here by excluding every capability the manifests declare.
    """
    src = catalog.PLUGIN_MANIFEST_DIR
    picked = sorted(src.glob("*.yaml"), key=lambda p: p.stat().st_size)[:2]
    assert len(picked) >= 2, "need at least two manifests to exercise the guard"
    for manifest in picked:
        (tmp_path / manifest.name).write_bytes(manifest.read_bytes())

    monkeypatch.setattr(catalog, "PLUGIN_MANIFEST_DIR", tmp_path)
    all_dotted = {e.dotted_path for e in catalog.collect_plugin_nodes(exclude_dotted=set())}
    with pytest.raises(RuntimeError, match="zero plugin capabilities"):
        catalog.collect_plugin_nodes(exclude_dotted=all_dotted)


def test_index_page_filter_toolbar(catalog):
    """Pin the JS-facing HTML contract of the one-row filter toolbar.

    ``node_catalog_filter.js`` binds to these ids/attributes; a generator
    refactor that drops one ships a silently dead filter bar.
    """
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    for entry in entries:
        entry.search_text = catalog._build_search_text(entry)
    page = catalog._render_index_page(entries)
    n = len(entries)

    # Toolbar: labelled search with prerendered scale hint + count.
    assert 'id="node-filter-search"' in page
    assert 'aria-label="Search catalog items by name, tag, or module"' in page
    assert f"Search {n} items by name, tag, module" in page
    assert f'<span id="node-filter-count">{n} items</span>' in page

    # Three fold buttons wired to their panels, badges hidden from AT.
    assert 'class="filter-group-buttons"' in page
    for key in ("categories", "tags", "sources"):
        assert f'data-panel="node-filter-{key}"' in page
        assert f'aria-controls="node-filter-{key}"' in page
        assert f'id="node-filter-{key}" hidden' in page
    assert page.count('class="filter-group-badge" aria-hidden="true"') == 3
    assert page.count('aria-expanded="false"') == 3

    # Debounced live region, state-aware Clear, active strip, empty state.
    assert 'id="node-filter-status" aria-live="polite"' in page.replace(
        'class="node-filter-sr-only" ', ""
    )
    assert 'class="node-filter-sr-only"' in page
    assert 'id="node-filter-reset" class="filter-reset" hidden' in page
    assert 'id="node-filter-active" hidden' in page
    assert 'id="node-filter-empty" hidden' in page
    assert 'id="node-filter-empty-reset"' in page
    # Empty state lives inside the grid, after the rows.
    assert page.index('id="node-filter-empty"') > page.index('id="node-catalog-grid"')

    # Chip markup is unchanged (the filter logic keys off these attributes).
    assert 'data-category="' in page
    assert 'data-tag="' in page
    assert 'data-source="' in page

    # The old three-row layout is gone.
    assert "node-filter-row" not in page
    assert "filter-label" not in page


def test_plugin_card_renders_source_and_ports(catalog):
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    sam3 = next(
        e for e in entries if e.plugin_name == "sam3" and e.kind == "node" and e.input_specs
    )
    sam3.search_text = catalog._build_search_text(sam3)
    html = catalog._render_card(sam3)
    assert 'data-source="plugin"' in html
    assert 'class="ports-table"' in html
    assert "tree/" in html, "repo link should point at the pinned tag"

    cu3s = next(e for e in entries if e.data_module_name == "cu3s")
    cu3s.search_text = catalog._build_search_text(cu3s)
    dm_html = catalog._render_card(cu3s)
    assert 'data-source="data-module"' in dm_html
    assert "datamodule-pill" in dm_html


def _summary_of(card_html: str) -> str:
    return card_html.split("<summary>", 1)[1].split("</summary>", 1)[0]


def test_row_summary_layout_markup(catalog):
    """The row summary is head (name + pills), module, summary, tags; the CSS grid relies on it."""
    entries = catalog.collect_plugin_nodes(exclude_dotted=set())
    dm = next(e for e in entries if e.kind == "data_module" and e.repo_url)
    dm.search_text = catalog._build_search_text(dm)
    summary = _summary_of(catalog._render_card(dm))

    order = [
        'class="row-head"',
        'class="row-name"',
        'class="row-pills"',
        'class="row-module"',
        'class="row-summary"',
        'class="row-tags"',
    ]
    positions = [summary.index(marker) for marker in order]
    assert positions == sorted(positions)

    head = summary.split('class="row-head"', 1)[1].split('class="row-module"', 1)[0]
    pills = head.split('class="row-pills"', 1)[1]
    pill_order = ["datamodule-pill", "plugin-pill", "card-repo", "row-category"]
    pill_positions = [pills.index(marker) for marker in pill_order]
    assert pill_positions == sorted(pill_positions), "category pill comes last"

    assert "row-icon" not in summary
    assert "<img" not in summary


def test_row_tags_render_as_tag_text(catalog):
    builtins, _walked = catalog.collect_builtin_nodes()
    tagged = next(e for e in builtins if len(e.tags) >= 2)
    tagged.search_text = catalog._build_search_text(tagged)
    summary = _summary_of(catalog._render_card(tagged))
    tags = summary.split('class="row-tags"', 1)[1]
    for tag in tagged.tags:
        assert f'class="tag-chip tag-text" data-tag="{tag}"' in tags


def test_row_keeps_filter_data_attributes(catalog):
    """The filter JS reads these attributes from the <details> element."""
    builtins, _walked = catalog.collect_builtin_nodes()
    entry = next(e for e in builtins if e.tags)
    entry.search_text = catalog._build_search_text(entry)
    opening = catalog._render_card(entry).split(">", 1)[0]
    assert opening.startswith('<details class="node-row"')
    assert f'data-category="{entry.category}"' in opening
    assert f'data-tags="{" ".join(entry.tags)}"' in opening
    assert 'data-source="builtin"' in opening
    assert f'data-search="{entry.search_text}"' in opening


ASSETS_SCRIPT = SCRIPT.parent / "_node_catalog_assets.py"


def test_category_colours_target_rows_and_tiles(tmp_path, monkeypatch):
    """Every category's colour variables reach catalog rows and overview tiles."""
    from cuvis_ai_schemas.enums import NodeCategory

    spec = importlib.util.spec_from_file_location("_node_catalog_assets", ASSETS_SCRIPT)
    assets = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assets)
    css_path = tmp_path / "node_categories.css"
    monkeypatch.setattr(assets, "CSS_DST", css_path)

    assets.emit_css()
    css = css_path.read_text(encoding="utf-8")

    assert ".node-card" not in css
    for cat in NodeCategory:
        selector = f':is(.node-row, .cat-tile)[data-category="{cat.value}"]'
        assert selector in css, cat.value
        rule = css.split(selector, 1)[1].split("}", 1)[0]
        assert f"--card-fill: var(--node-cat-{cat.value}-fill);" in rule
        assert f"--card-border: var(--node-cat-{cat.value}-border);" in rule


def test_builtin_entries_are_public_and_concrete(catalog):
    """Private helpers and abstract bases are walked but never listed."""
    entries, walked = catalog.collect_builtin_nodes()
    assert "RXGlobal" in {e.name for e in entries}
    assert not [e.name for e in entries if e.name.startswith("_")]
    assert not [e.name for e in entries if inspect.isabstract(_class_of(e.dotted_path))]
    listed = {e.dotted_path for e in entries}
    assert listed < walked, "the walk should also report the skipped base classes"


def test_abstract_mirror_bases_do_not_return_as_plugin_rows(catalog):
    """The built-in mirror is deduped against every walked class, not just the listed ones."""
    entries, walked = catalog.collect_builtin_nodes()
    plugins = catalog.collect_plugin_nodes(exclude_dotted=walked)
    assert not [e.dotted_path for e in plugins if e.plugin_name == "cuvis_ai_builtin"]

    mirror = load_plugin_manifest(catalog.PLUGIN_MANIFEST_DIR / "cuvis_ai_builtin.yaml")
    concrete = {
        cap.class_name
        for cap in mirror.capabilities
        if not inspect.isabstract(_class_of(cap.class_name))
    }
    missing = sorted(concrete - {e.dotted_path for e in entries})
    assert not missing, f"concrete built-ins dropped from the catalog: {missing}"


def test_superseded_plugin_copies_are_hidden(catalog):
    """A plugin class that migrate-plugin-nodes maps to a built-in gets no row of its own."""
    assert catalog.SUPERSEDED, "the migrate-plugin-nodes map came back empty"
    entries = catalog.collect_entries()
    assert not [e.dotted_path for e in entries if e.dotted_path in catalog.SUPERSEDED]

    builtin_names = {e.name for e in entries if not e.is_plugin}
    twins = {
        e.name for e in entries if e.is_plugin and e.kind == "node" and e.name in builtin_names
    }
    assert not sorted(twins - NAME_ONLY_MATCHES), "plugin rows duplicate a built-in"


def test_every_plugin_entry_has_a_summary(catalog):
    entries = catalog.collect_entries()
    missing = [e.dotted_path for e in entries if e.is_plugin and not e.summary.strip()]
    assert not missing, f"catalog rows without a description: {missing}"


def test_catalogs_overview_counts_and_text(catalog):
    entries = catalog.collect_entries()
    page = catalog._render_catalogs_overview(entries)

    nodes = [e for e in entries if e.kind == "node"]
    plugins = {e.plugin_name for e in entries if e.is_plugin}
    assert f"**{len(nodes)} nodes**" in page
    assert f"{sum(1 for e in nodes if not e.is_plugin)} built-in" in page
    assert f"**{len(plugins)} plugins**" in page
    assert f"**{len(PublicDatasets.list_specs())} datasets**" in page
    assert "—" not in page and "→" not in page

    intro = catalog._render_index_page(entries).split('<div class="node-filter">')[0]
    assert "migrate-plugin-nodes" in intro
    assert "—" not in intro and "→" not in intro
