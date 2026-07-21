"""Validate the selective DeepEIoU plugin manifest."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from cuvis_ai_schemas.plugin import load_plugin_manifest

pytestmark = pytest.mark.unit

# The pinned tag is intentionally not frozen here: pins are refreshed (manually or by
# the plugin-pin-bump workflow) whenever a plugin releases, so this asserts the tag is
# present and well-formed rather than a specific value. The node set below is the real
# guard against a plugin's exposed surface drifting out of sync with this manifest.
SEMVER_TAG = re.compile(r"v\d+\.\d+\.\d+")

DEEPEIOU_MANIFEST_PATH = Path("cuvis_ai/configs/plugins/deepeiou.yaml")
PLUGIN_NAME = "deepeiou"
EXPECTED_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-deepeiou.git"
EXPECTED_PROVIDES = [
    "cuvis_ai_deepeiou.node.DeepEIoUTrack",
    "cuvis_ai_deepeiou.node.OSNetExtractor",
    "cuvis_ai_deepeiou.node.ResNetExtractor",
]


def test_deepeiou_manifest_exists() -> None:
    assert DEEPEIOU_MANIFEST_PATH.exists(), f"Missing DeepEIoU manifest: {DEEPEIOU_MANIFEST_PATH}"


def test_deepeiou_manifest_contains_only_deepeiou_plugin() -> None:
    manifest = load_plugin_manifest(DEEPEIOU_MANIFEST_PATH)
    assert manifest.name == PLUGIN_NAME


def test_deepeiou_manifest_matches_expected_release() -> None:
    manifest = load_plugin_manifest(DEEPEIOU_MANIFEST_PATH)
    plugin = manifest

    assert getattr(plugin, "repo", None) == EXPECTED_REPO
    tag = getattr(plugin, "tag", None)
    assert tag is not None and SEMVER_TAG.fullmatch(tag), f"unexpected tag: {tag!r}"
    assert [node.class_name for node in plugin.capabilities] == EXPECTED_PROVIDES
