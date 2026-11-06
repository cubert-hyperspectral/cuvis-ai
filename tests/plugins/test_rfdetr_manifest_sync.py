"""Validate the rfdetr plugin manifest."""

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

RFDETR_MANIFEST_PATH = Path("cuvis_ai/configs/plugins/rfdetr.yaml")
PLUGIN_NAME = "rfdetr"
EXPECTED_REPO = "https://github.com/cubert-hyperspectral/cuvis-ai-rfdetr.git"
# The full surface of the plugin, including the classes cuvis-ai later took over as
# built-ins (fixed PCA, score fusion and intersection): pipelines that still name them
# must keep loading, and the docs catalog hides those three rows on its own.
EXPECTED_PROVIDES = [
    "cuvis_ai_rfdetr.node.rfdetr_detector.RFDETRDetector",
    "cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter",
    "cuvis_ai_rfdetr.node.rfdetr_trainable.RFDETRTrainable",
    "cuvis_ai_rfdetr.node.rfdetr_loss.RFDETRCriterionLoss",
    "cuvis_ai_rfdetr.node.percentile_composite.PercentileComposite",
    "cuvis_ai_rfdetr.node.scalar_minmax_bandslice.ScalarMinMaxBandSlice",
    "cuvis_ai_rfdetr.node.score_intersection.ScoreIntersection",
    "cuvis_ai_rfdetr.node.score_fusion.ScoreFusion",
    "cuvis_ai_rfdetr.node.fixed_pca_projection.FixedPCAProjection",
    "cuvis_ai_rfdetr.node.carl_segmenter.CarlSegmenter",
    "cuvis_ai_rfdetr.node.sam_shell_gate.SamShellGate",
]


def test_rfdetr_manifest_exists() -> None:
    assert RFDETR_MANIFEST_PATH.exists(), f"Missing rfdetr manifest: {RFDETR_MANIFEST_PATH}"


def test_rfdetr_manifest_contains_only_rfdetr_plugin() -> None:
    manifest = load_plugin_manifest(RFDETR_MANIFEST_PATH)
    assert manifest.name == PLUGIN_NAME


def test_rfdetr_manifest_matches_expected_release() -> None:
    manifest = load_plugin_manifest(RFDETR_MANIFEST_PATH)

    assert getattr(manifest, "repo", None) == EXPECTED_REPO
    tag = getattr(manifest, "tag", None)
    assert tag is not None and SEMVER_TAG.fullmatch(tag), f"unexpected tag: {tag!r}"
    assert [node.class_name for node in manifest.capabilities] == EXPECTED_PROVIDES
    assert all(node.doc_summary for node in manifest.capabilities), "capabilities not generated"
