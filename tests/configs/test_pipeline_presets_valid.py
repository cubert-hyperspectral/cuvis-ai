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


@pytest.mark.parametrize(
    "preset", ["sam31_text_propagation.yaml", "sam31_text_propagation_video.yaml"]
)
def test_sam31_text_presets_name_the_weights_their_node_selects(preset: str) -> None:
    """Each SAM 3.1 preset sets ``weight_name`` to a row the sam3 manifest lets a node select.

    The ``sam3.1`` checkpoint is picked by that hparam and its group has no default row, so a
    preset that leaves the hparam out or misspells it is not provisioned by a host application
    and fails only when it runs.
    """
    manifest = load_plugin_manifest(_PLUGINS / "sam3.yaml")
    selectable = {row.name for row in manifest.weights if row.selected_by == "weight_name"}
    pipeline = PipelineConfig.load_from_file(_PIPELINES / "sam3" / preset)
    (tracker,) = [
        node
        for node in pipeline.nodes
        if node.class_name == "cuvis_ai_sam3.node.SAM31TextPropagation"
    ]

    assert tracker.class_name in {entry.class_name for entry in manifest.capabilities}
    assert tracker.hparams["weight_name"] in selectable
