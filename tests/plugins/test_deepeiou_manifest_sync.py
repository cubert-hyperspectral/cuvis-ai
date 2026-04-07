"""Validate the selective DeepEIoU plugin manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.utils.plugin_config import PluginManifest

pytestmark = pytest.mark.unit

DEEPEIOU_MANIFEST_PATH = Path("configs/plugins/deepeiou.yaml")
PLUGIN_NAME = "deepeiou"
EXPECTED_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou.git"
EXPECTED_TAG = "v0.1.0"
EXPECTED_PROVIDES = [
    "cuvis_ai_deepeiou.node.DeepEIoUTrack",
    "cuvis_ai_deepeiou.node.OSNetExtractor",
    "cuvis_ai_deepeiou.node.ResNetExtractor",
]


def test_deepeiou_manifest_exists() -> None:
    assert DEEPEIOU_MANIFEST_PATH.exists(), f"Missing DeepEIoU manifest: {DEEPEIOU_MANIFEST_PATH}"


def test_deepeiou_manifest_contains_only_deepeiou_plugin() -> None:
    manifest = PluginManifest.from_yaml(DEEPEIOU_MANIFEST_PATH)
    assert set(manifest.plugins.keys()) == {PLUGIN_NAME}


def test_deepeiou_manifest_matches_expected_release() -> None:
    manifest = PluginManifest.from_yaml(DEEPEIOU_MANIFEST_PATH)
    plugin = manifest.plugins[PLUGIN_NAME]

    assert getattr(plugin, "repo", None) == EXPECTED_REPO
    assert getattr(plugin, "tag", None) == EXPECTED_TAG
    assert plugin.provides == EXPECTED_PROVIDES
