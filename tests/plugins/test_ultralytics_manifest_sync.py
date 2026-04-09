"""Validate the selective Ultralytics plugin manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.utils.plugin_config import PluginManifest

pytestmark = pytest.mark.unit

ULTRALYTICS_MANIFEST_PATH = Path("configs/plugins/ultralytics.yaml")
PLUGIN_NAME = "ultralytics"
EXPECTED_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-ultralytics.git"
EXPECTED_TAG = "v0.1.0"
EXPECTED_PROVIDES = [
    "cuvis_ai_ultralytics.node.YOLOPreprocess",
    "cuvis_ai_ultralytics.node.YOLO26Detection",
    "cuvis_ai_ultralytics.node.YOLOPostprocess",
]


def test_ultralytics_manifest_exists() -> None:
    assert ULTRALYTICS_MANIFEST_PATH.exists(), (
        f"Missing Ultralytics manifest: {ULTRALYTICS_MANIFEST_PATH}"
    )


def test_ultralytics_manifest_contains_only_ultralytics_plugin() -> None:
    manifest = PluginManifest.from_yaml(ULTRALYTICS_MANIFEST_PATH)
    assert set(manifest.plugins.keys()) == {PLUGIN_NAME}


def test_ultralytics_manifest_matches_expected_release() -> None:
    manifest = PluginManifest.from_yaml(ULTRALYTICS_MANIFEST_PATH)
    plugin = manifest.plugins[PLUGIN_NAME]

    assert getattr(plugin, "repo", None) == EXPECTED_REPO
    assert getattr(plugin, "tag", None) == EXPECTED_TAG
    assert plugin.provides == EXPECTED_PROVIDES
