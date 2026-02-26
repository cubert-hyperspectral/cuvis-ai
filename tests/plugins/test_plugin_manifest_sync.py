"""Guard against drift between registry and selective plugin manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.utils.plugin_config import PluginManifest

pytestmark = pytest.mark.unit

REGISTRY_PATH = Path("configs/plugins/registry.yaml")
ADACLIP_MANIFEST_PATH = Path("configs/plugins/adaclip.yaml")
PLUGIN_NAME = "adaclip"


def test_adaclip_manifest_exists() -> None:
    assert ADACLIP_MANIFEST_PATH.exists(), f"Missing AdaCLIP manifest: {ADACLIP_MANIFEST_PATH}"


def test_adaclip_manifest_contains_only_adaclip_plugin() -> None:
    manifest = PluginManifest.from_yaml(ADACLIP_MANIFEST_PATH)
    assert set(manifest.plugins.keys()) == {PLUGIN_NAME}


def test_adaclip_manifest_matches_registry_entry() -> None:
    registry_manifest = PluginManifest.from_yaml(REGISTRY_PATH)
    adaclip_manifest = PluginManifest.from_yaml(ADACLIP_MANIFEST_PATH)

    assert PLUGIN_NAME in registry_manifest.plugins, (
        f"Plugin '{PLUGIN_NAME}' must exist in registry: {REGISTRY_PATH}"
    )
    assert PLUGIN_NAME in adaclip_manifest.plugins, (
        f"Plugin '{PLUGIN_NAME}' must exist in selective manifest: {ADACLIP_MANIFEST_PATH}"
    )

    assert adaclip_manifest.plugins[PLUGIN_NAME].model_dump() == (
        registry_manifest.plugins[PLUGIN_NAME].model_dump()
    )
