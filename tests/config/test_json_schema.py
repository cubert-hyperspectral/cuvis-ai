import pytest

from cuvis_ai_core.training.config import OptimizerConfig, TrainingConfig
from cuvis_ai_core.utils.config_helpers import CONFIG_TYPE_REGISTRY, generate_json_schema


def test_optimizer_schema_generation():
    schema = OptimizerConfig.model_json_schema()

    assert "properties" in schema
    assert "lr" in schema["properties"]
    assert schema["properties"]["lr"]["type"] == "number"
    assert schema["properties"]["lr"]["minimum"] == pytest.approx(1e-6)
    assert schema["properties"]["lr"]["maximum"] == pytest.approx(1.0)


def test_training_schema_nested():
    schema = TrainingConfig.model_json_schema()

    assert "optimizer" in schema["properties"]
    assert "$defs" in schema or "definitions" in schema


def test_schema_generation_for_all_types():
    for config_type, _config_class in CONFIG_TYPE_REGISTRY.items():
        schema = generate_json_schema(config_type)
        assert isinstance(schema, dict)
        assert "properties" in schema
