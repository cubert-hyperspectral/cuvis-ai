"""Validate the selective TrackEval plugin manifest."""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.utils.plugin_config import PluginManifest

pytestmark = pytest.mark.unit

TRACKEVAL_MANIFEST_PATH = Path("configs/plugins/trackeval.yaml")
PLUGIN_NAME = "trackeval"
EXPECTED_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-trackeval.git"
EXPECTED_TAG = "v0.1.0"
EXPECTED_PROVIDES = [
    "cuvis_ai_trackeval.node.HOTAMetricNode",
    "cuvis_ai_trackeval.node.CLEARMetricNode",
    "cuvis_ai_trackeval.node.IdentityMetricNode",
]


def test_trackeval_manifest_exists() -> None:
    assert TRACKEVAL_MANIFEST_PATH.exists(), (
        f"Missing TrackEval manifest: {TRACKEVAL_MANIFEST_PATH}"
    )


def test_trackeval_manifest_contains_only_trackeval_plugin() -> None:
    manifest = PluginManifest.from_yaml(TRACKEVAL_MANIFEST_PATH)
    assert set(manifest.plugins.keys()) == {PLUGIN_NAME}


def test_trackeval_manifest_matches_expected_release() -> None:
    manifest = PluginManifest.from_yaml(TRACKEVAL_MANIFEST_PATH)
    plugin = manifest.plugins[PLUGIN_NAME]

    assert getattr(plugin, "repo", None) == EXPECTED_REPO
    assert getattr(plugin, "tag", None) == EXPECTED_TAG
    assert plugin.provides == EXPECTED_PROVIDES
