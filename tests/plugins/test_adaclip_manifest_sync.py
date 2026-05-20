"""Validate the selective AdaCLIP plugin manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from cuvis_ai_core.utils.plugin_config import PluginManifest

pytestmark = pytest.mark.unit

ADACLIP_MANIFEST_PATH = Path("configs/plugins/adaclip.yaml")
PLUGIN_NAME = "adaclip"
EXPECTED_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
EXPECTED_TAG = "v0.1.3"
EXPECTED_PROVIDES = [
    "cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector",
]


def test_adaclip_manifest_exists() -> None:
    assert ADACLIP_MANIFEST_PATH.exists(), f"Missing AdaCLIP manifest: {ADACLIP_MANIFEST_PATH}"


def test_adaclip_manifest_contains_only_adaclip_plugin() -> None:
    manifest = PluginManifest.from_yaml(ADACLIP_MANIFEST_PATH)
    assert set(manifest.plugins.keys()) == {PLUGIN_NAME}


def test_adaclip_manifest_matches_expected_release() -> None:
    manifest = PluginManifest.from_yaml(ADACLIP_MANIFEST_PATH)
    plugin = manifest.plugins[PLUGIN_NAME]

    assert getattr(plugin, "repo", None) == EXPECTED_REPO
    assert getattr(plugin, "tag", None) == EXPECTED_TAG
    assert plugin.provides == EXPECTED_PROVIDES
