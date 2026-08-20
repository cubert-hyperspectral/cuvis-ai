"""Standalone pipeline presets validate even when no trainrun references them.

``test_trainrun_pipeline_references_resolve`` only loads pipelines a trainrun
points at; a preset shipped for inference only (no trainrun) would never be
parsed by the suite and could rot silently.
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
