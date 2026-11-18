"""Standalone pipeline presets validate even when no trainrun references them.

``test_trainrun_pipeline_references_resolve`` only loads pipelines a trainrun
points at; a preset shipped for inference only (no trainrun) would never be
parsed by the suite and could rot silently. Presets whose node names are a
checkpoint contract pin those names here as well.
"""

from pathlib import Path

import pytest
from cuvis_ai_schemas.pipeline.config import PipelineConfig
from cuvis_ai_schemas.plugin import load_plugin_manifest

_PIPELINES = Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs" / "pipeline"
_PLUGINS = _PIPELINES.parent / "plugins"


def test_dinomaly_cir_lentils_preset_loads() -> None:
    """The lentils inference preset parses and keeps the checkpoint-matched spine."""
    pipeline = PipelineConfig.load_from_file(
        _PIPELINES / "anomaly" / "dinomaly" / "dinomaly_cir_lentils.yaml"
    )
    assert pipeline.metadata.name == "dinomaly_cir_lentils"
    node_names = {node.name for node in pipeline.nodes}
    assert node_names == {
        "anomaly_data",
        "MinMaxNormalizer",
        "cir_selector",
        "dinomaly_detector",
        "decider",
    }


def test_dinomaly_custom_training_preset_keeps_the_trained_spine() -> None:
    """The custom-selector training preset (the CuvisNEXT wizard's one trainrun) pins its spine.

    A trainrun checkpoint is a dict keyed by node name, so renaming a spine node would make a
    later inference preset load initialized weights instead of trained ones. The selector's
    frozen wavelengths are pinned for the same reason: they are what the detector was trained
    to see. The decider must be the two-stage one with both thresholds unset, so a saved run
    exposes the gate an operator calibrates after training.
    """
    pipeline = PipelineConfig.load_from_file(
        _PIPELINES / "anomaly" / "dinomaly" / "dinomaly_custom.yaml"
    )
    assert pipeline.metadata.name == "dinomaly_custom"
    nodes = {node.name: node for node in pipeline.nodes}
    assert {
        "anomaly_data",
        "MinMaxNormalizer",
        "custom_selector",
        "dinomaly_detector",
        "decider",
    } <= set(nodes)
    assert nodes["custom_selector"].hparams["target_wavelengths"] == [542.0, 902.0, 886.0]
    assert nodes["decider"].class_name.endswith("TwoStageBinaryDecider")
    assert nodes["decider"].hparams["image_threshold"] is None
    assert nodes["decider"].hparams["pixel_threshold"] is None


_SAM31_CLASS = "cuvis_ai_sam3.node.SAM31TextPropagation"
# Every shipped preset on the SAM 3.1 node, found by the class it names: a preset added later is
# held to the rules below without being listed here.
_SAM31_PRESETS = sorted(
    path.relative_to(_PIPELINES).as_posix()
    for path in _PIPELINES.rglob("*.yaml")
    if _SAM31_CLASS in path.read_text(encoding="utf-8")
)


def test_sam31_presets_are_found() -> None:
    """The search finds the three shipped SAM 3.1 presets, so the tests below do not run empty."""
    assert {
        "sam3/sam31_text_propagation.yaml",
        "sam3/sam31_text_propagation_video.yaml",
        "sam3/sam31_text_propagation_view.yaml",
    } <= set(_SAM31_PRESETS)


@pytest.mark.parametrize("preset", _SAM31_PRESETS)
def test_sam31_text_presets_call_their_tracker_sam31_tracker(preset: str) -> None:
    """The SAM 3.1 node is named ``sam31_tracker``, not ``sam3_tracker`` as in the SAM 3 presets.

    A host application shows the node name, not the class, in its profiling table and its logs.
    A SAM 3.1 preset that keeps the name of the SAM 3 preset it was copied from reads there as a
    SAM 3 run.
    """
    pipeline = PipelineConfig.load_from_file(_PIPELINES / preset)
    names = {node.name for node in pipeline.nodes if node.class_name == _SAM31_CLASS}

    assert names == {"sam31_tracker"}


@pytest.mark.parametrize("preset", _SAM31_PRESETS)
def test_sam31_text_presets_name_the_weights_their_node_selects(preset: str) -> None:
    """Each SAM 3.1 preset sets ``weight_name: sam3.1``, a row the sam3 manifest lets a node select.

    A host application gives a node the row its ``weight_name`` names and, where the hparam is
    left out, the default row of the group, which is the SAM 3 checkpoint. A preset that drops
    or misspells the hparam is provisioned with the wrong file and fails only when it runs.
    """
    manifest = load_plugin_manifest(_PLUGINS / "sam3.yaml")
    rows = {row.name: row for row in manifest.weights}
    pipeline = PipelineConfig.load_from_file(_PIPELINES / preset)
    (tracker,) = [node for node in pipeline.nodes if node.class_name == _SAM31_CLASS]

    assert tracker.class_name in {entry.class_name for entry in manifest.capabilities}
    assert tracker.hparams["weight_name"] == "sam3.1"
    assert rows["sam3.1"].selected_by == "weight_name" and not rows["sam3.1"].default


def test_sam3_manifest_lets_a_pipeline_need_one_checkpoint() -> None:
    """Both sam3 rows sit in the ``weight_name`` group and ``sam3`` is its default row.

    A host application requires a row without a selector for every node of the plugin, which
    would make a SAM 3.1 pipeline need the SAM 3 checkpoint too. With ``sam3`` as the default of
    the group, the pipelines on the SAM 3 nodes, which set no ``weight_name``, keep needing
    ``sam3`` and nothing else, and a pipeline that names ``sam3.1`` needs that file alone.
    """
    manifest = load_plugin_manifest(_PLUGINS / "sam3.yaml")

    assert [row.name for row in manifest.weights if row.selected_by is None] == []
    assert {row.selected_by for row in manifest.weights} == {"weight_name"}
    assert [row.name for row in manifest.weights if row.default] == ["sam3"]
