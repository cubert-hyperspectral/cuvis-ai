import json

from cuvis_ai_core.training.config import DataConfig, PipelineConfig, TrainingConfig, TrainRunConfig


def test_complete_trainrun_serialization():
    """Test complete train run config serialization."""
    trainrun = TrainRunConfig(
        name="test_run",
        pipeline=PipelineConfig(
            name="test_pipeline",
            nodes=[{"name": "node1", "type": "test"}],
            connections=[{"from": "node1", "to": "output"}],
        ),
        data=DataConfig(
            cu3s_file_path="/path/to/data.cu3s",
            train_ids=[1, 2, 3],
            val_ids=[4, 5],
        ),
        training=TrainingConfig(max_epochs=10),
    )

    json_str = trainrun.model_dump_json()
    json_dict = json.loads(json_str)

    restored = TrainRunConfig.model_validate_json(json_str)

    assert restored.name == trainrun.name
    assert restored.pipeline.name == trainrun.pipeline.name
    assert restored.data.train_ids == trainrun.data.train_ids
    assert json_dict["training"]["max_epochs"] == 10


def test_yaml_to_pydantic():
    """Test loading YAML-like mapping into Pydantic model."""
    yaml_dict = {
        "name": "adamw",
        "lr": 0.001,
        "weight_decay": 0.01,
    }

    from cuvis_ai_core.training.config import OptimizerConfig

    config = OptimizerConfig(**yaml_dict)

    assert config.lr == 0.001
