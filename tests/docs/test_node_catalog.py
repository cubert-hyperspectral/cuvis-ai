"""Tests for the docs node-catalog generator (``scripts/generate_node_catalog.py``).

The generator feeds the Catalogs → Nodes page. These tests pin the plugin
side of the collection: capabilities must come from the repo's plugin
manifest YAMLs (nodes and data modules), and the built-in mirror manifest
must dedupe against the live built-in import.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("mkdocs_gen_files", reason="docs extra not installed")

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

    Current manifest-driven collection is ~39 plugin nodes across 11 plugins,
    7 data modules, and ~147 built-in mirror entries. These are floors, not
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
