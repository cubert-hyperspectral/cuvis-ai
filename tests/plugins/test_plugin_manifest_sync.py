"""Guard against drift between registry and selective plugin manifests."""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.utils.plugin_config import PluginManifest

pytestmark = pytest.mark.unit

REGISTRY_PATH = Path("configs/plugins/registry.yaml")
PLUGIN_MANIFESTS = {
    "adaclip": Path("configs/plugins/adaclip.yaml"),
    "trackeval": Path("configs/plugins/trackeval.yaml"),
}


@pytest.mark.parametrize(("plugin_name", "manifest_path"), PLUGIN_MANIFESTS.items())
def test_selective_manifest_exists(plugin_name: str, manifest_path: Path) -> None:
    assert manifest_path.exists(), f"Missing {plugin_name} manifest: {manifest_path}"


@pytest.mark.parametrize(("plugin_name", "manifest_path"), PLUGIN_MANIFESTS.items())
def test_selective_manifest_contains_only_target_plugin(
    plugin_name: str, manifest_path: Path
) -> None:
    manifest = PluginManifest.from_yaml(manifest_path)
    assert set(manifest.plugins.keys()) == {plugin_name}


@pytest.mark.parametrize(("plugin_name", "manifest_path"), PLUGIN_MANIFESTS.items())
def test_selective_manifest_matches_registry_entry(plugin_name: str, manifest_path: Path) -> None:
    registry_manifest = PluginManifest.from_yaml(REGISTRY_PATH)
    selective_manifest = PluginManifest.from_yaml(manifest_path)

    assert plugin_name in registry_manifest.plugins, (
        f"Plugin '{plugin_name}' must exist in registry: {REGISTRY_PATH}"
    )
    assert plugin_name in selective_manifest.plugins, (
        f"Plugin '{plugin_name}' must exist in selective manifest: {manifest_path}"
    )

    assert selective_manifest.plugins[plugin_name].model_dump() == (
        registry_manifest.plugins[plugin_name].model_dump()
    )
