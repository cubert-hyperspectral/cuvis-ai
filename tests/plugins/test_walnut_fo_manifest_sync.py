"""Validate the patchcore, steervit and efficientad manifests (the walnut foreign-object stack)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from cuvis_ai_schemas.plugin import load_plugin_manifest

pytestmark = pytest.mark.unit

# The pinned tag is intentionally not frozen here: pins are refreshed (manually or by
# the plugin-pin-bump workflow) whenever a plugin releases, so this asserts the tag is
# present and well-formed rather than a specific value. The node sets below are the real
# guard against a plugin's exposed surface drifting out of sync with its manifest.
SEMVER_TAG = re.compile(r"v\d+\.\d+\.\d+")
CATALOG = Path("cuvis_ai/configs/plugins")

EXPECTED_PROVIDES = {
    "patchcore": [
        "cuvis_ai_patchcore.node.patchcore.PatchCoreDetector",
        "cuvis_ai_patchcore.node.fusion.ScoreMapFusion",
        "cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer",
        "cuvis_ai_patchcore.node.gate.FrameScoreGate",
        "cuvis_ai_patchcore.node.fusion.DecisionFusion",
        "cuvis_ai_patchcore.node.fusion.MaskComposite",
        "cuvis_ai_patchcore.node.spatial.GridSubsample",
        "cuvis_ai_patchcore.node.spatial.ScoreUpsample",
        "cuvis_ai_patchcore.node.fusion.ScoreMapSuppression",
        "cuvis_ai_patchcore.node.temporal.MaskPersistence",
        "cuvis_ai_patchcore.node.morphology.MaskMinArea",
        "cuvis_ai_patchcore.node.spatial.ScoreMapSmoothing",
        "cuvis_ai_patchcore.node.morphology.MaskBlobGate",
        "cuvis_ai_patchcore.node.objectness.SpectralObjectMask",
        "cuvis_ai_patchcore.node.objectness.MaskBlobFilter",
        "cuvis_ai_patchcore.node.morphology.MaskPeakGate",
    ],
    "steervit": [
        "cuvis_ai_steervit.node.steervit.SteerViTExtractor",
        "cuvis_ai_steervit.node.stretch.JointPercentileStretch",
        "cuvis_ai_steervit.node.tiling.ImageTiler",
        "cuvis_ai_steervit.node.tiling.GridStitcher",
    ],
    "efficientad": [
        "cuvis_ai_efficientad.node.efficientad.EfficientAdDetector",
    ],
}
PLUGINS = sorted(EXPECTED_PROVIDES)


@pytest.mark.parametrize("name", PLUGINS)
def test_manifest_exists(name: str) -> None:
    assert (CATALOG / f"{name}.yaml").exists(), f"Missing {name} manifest: {CATALOG / name}.yaml"


@pytest.mark.parametrize("name", PLUGINS)
def test_manifest_names_its_plugin(name: str) -> None:
    manifest = load_plugin_manifest(CATALOG / f"{name}.yaml")
    assert manifest.name == name
    assert manifest.package_name == f"cuvis-ai-{name}"


@pytest.mark.parametrize("name", PLUGINS)
def test_manifest_matches_expected_release(name: str) -> None:
    plugin = load_plugin_manifest(CATALOG / f"{name}.yaml")

    assert getattr(plugin, "repo", None) == (
        f"https://github.com/cubert-hyperspectral/cuvis-ai-{name}.git"
    )
    tag = getattr(plugin, "tag", None)
    assert tag is not None and SEMVER_TAG.fullmatch(tag), f"unexpected tag: {tag!r}"
    assert [node.class_name for node in plugin.capabilities] == EXPECTED_PROVIDES[name]
