"""Standalone pipeline presets validate even when no trainrun references them.

``test_trainrun_pipeline_references_resolve`` only loads pipelines a trainrun
points at; a preset shipped for inference only (no trainrun) would never be
parsed by the suite and could rot silently. Presets whose node names are a
checkpoint contract pin those names here as well.
"""

from pathlib import Path

from cuvis_ai_schemas.pipeline.config import PipelineConfig

_PIPELINES = Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs" / "pipeline"


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
