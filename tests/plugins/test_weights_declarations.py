"""The plugin manifests' ``weights:`` blocks and the committed ``weights.index.json``.

Every weight-bearing manifest carries the block ``emit_metadata`` projected from the
plugin's ``WEIGHTS`` tuple; the blocks must validate, pin fully, share one namespace with
each other and with core's built-in rows, and the index must be exactly what core
generates from them (``scripts/weights_index.py``).
"""

from __future__ import annotations

import json

import pytest
from cuvis_ai_schemas.plugin import PluginWeightEntry, load_plugin_manifest

from cuvis_ai_core.data.model_weights import TRAINED_PIPELINES, ModelRegistryConflict, ModelWeights
from scripts.weights_index import INDEX_PATH, PLUGINS_DIR, generate

pytestmark = pytest.mark.unit

WEIGHT_BEARING_PLUGINS = {"sam3", "rtsam2", "adaclip", "dinomaly"}


def _blocks() -> dict[str, list[PluginWeightEntry]]:
    blocks: dict[str, list[PluginWeightEntry]] = {}
    for path in sorted(PLUGINS_DIR.glob("*.yaml")):
        manifest = load_plugin_manifest(path)
        if manifest.weights:
            blocks[manifest.name] = list(manifest.weights)
    return blocks


def test_the_weight_bearing_plugins_declare_weights() -> None:
    assert set(_blocks()) == WEIGHT_BEARING_PLUGINS


def test_every_entry_is_fully_pinned_to_a_cubert_mirror() -> None:
    for plugin, entries in _blocks().items():
        for entry in entries:
            assert isinstance(entry, PluginWeightEntry)
            assert entry.repo_id.startswith("cubert-gmbh/"), (plugin, entry.name)
            assert len(entry.revision) == 40 and len(entry.sha256) == 64, (plugin, entry.name)
            assert entry.size_bytes > 0
            for aux in entry.aux_files:
                assert len(aux.sha256) == 64 and aux.size_bytes > 0, (plugin, entry.name, aux.path)
            assert entry.used_for and entry.summary and entry.description, (plugin, entry.name)


def test_names_and_aliases_are_unique_across_manifests_and_core() -> None:
    owner: dict[str, str] = {}
    for plugin, entries in _blocks().items():
        for entry in entries:
            for key in (entry.name, *entry.aliases):
                assert key not in owner, f"{key!r} declared by {plugin} and {owner[key]}"
                owner[key] = plugin
    core_keys = {e.name for e in TRAINED_PIPELINES} | {
        alias for e in TRAINED_PIPELINES for alias in e.aliases
    }
    assert not core_keys & set(owner), core_keys & set(owner)


def test_license_file_is_a_bare_filename_or_null() -> None:
    for entries in _blocks().values():
        for entry in entries:
            if entry.license_file is not None:
                assert "/" not in entry.license_file and "\\" not in entry.license_file


def test_manifests_load_into_the_registry_without_conflict() -> None:
    ModelWeights.reset()
    try:
        ModelWeights.load_manifests([PLUGINS_DIR])
    except ModelRegistryConflict as exc:
        pytest.fail(str(exc))
    rows = {row.entry.name: row for row in ModelWeights.rows()}
    for plugin, entries in _blocks().items():
        for entry in entries:
            assert rows[entry.name].plugin == plugin
            assert rows[entry.name].source == "manifest"
            assert rows[entry.name].entry == entry


def test_weights_index_matches_regenerated_output() -> None:
    assert INDEX_PATH.exists(), "run `uv run python -m scripts.weights_index`"
    assert INDEX_PATH.read_text(encoding="utf-8") == generate(), (
        "weights.index.json is stale; run `uv run python -m scripts.weights_index`"
    )


def test_weights_index_is_versioned_sorted_and_plugin_free() -> None:
    doc = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    assert doc["schema_version"] == 1
    assert doc["used_for_labels"] and doc["weights_hosts"]
    names = [row["name"] for row in doc["models"]]
    assert names == sorted(names)
    assert {row["source"] for row in doc["models"]} <= {"manifest", "dict"}
    assert not any(row["pin_mismatch"] for row in doc["models"])
    declared = {entry.name for entries in _blocks().values() for entry in entries}
    assert declared <= set(names)
