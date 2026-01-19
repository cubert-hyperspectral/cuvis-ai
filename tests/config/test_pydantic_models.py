import json

import pytest
from pydantic import ValidationError

from cuvis_ai_core.training.config import (
    CallbacksConfig,
    DataConfig,
    EarlyStoppingConfig,
    ModelCheckpointConfig,
    OptimizerConfig,
    PipelineConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainRunConfig,
)


class TestOptimizerConfig:
    def test_valid_config(self):
        config = OptimizerConfig(name="adamw", lr=0.001, weight_decay=0.01)
        assert config.lr == 0.001

    def test_lr_min_constraint(self):
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=0.0)

    def test_lr_max_constraint(self):
        with pytest.raises(ValidationError):
            OptimizerConfig(lr=10.0)

    def test_weight_decay_constraint(self):
        with pytest.raises(ValidationError):
            OptimizerConfig(weight_decay=-0.1)

        with pytest.raises(ValidationError):
            OptimizerConfig(weight_decay=1.5)


class TestTrainingConfig:
    def test_nested_configs(self):
        config = TrainingConfig(
            seed=42,
            optimizer=OptimizerConfig(lr=0.001),
            max_epochs=100,
        )
        assert config.optimizer.lr == 0.001
        assert config.trainer.max_epochs == 100

    def test_max_epochs_constraint(self):
        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=0)

        with pytest.raises(ValidationError):
            TrainingConfig(max_epochs=20000)

    def test_optional_scheduler(self):
        config = TrainingConfig()
        assert config.scheduler is None

        config_with_scheduler = TrainingConfig(scheduler=SchedulerConfig(name="cosine", t_max=10))
        assert config_with_scheduler.scheduler.name == "cosine"


class TestCallbackConfig:
    def test_callbacks_structure(self):
        callbacks = CallbacksConfig(
            early_stopping=[EarlyStoppingConfig(monitor="val_loss")],
            checkpoint=ModelCheckpointConfig(dirpath="checkpoints", monitor="val_loss"),
        )
        assert callbacks.early_stopping[0].monitor == "val_loss"
        proto = callbacks.to_proto()
        restored = CallbacksConfig.from_proto(proto)
        assert restored.checkpoint.dirpath == "checkpoints"


class TestProtoSerialization:
    def test_optimizer_proto_roundtrip(self):
        original = OptimizerConfig(lr=0.005, weight_decay=0.02)
        proto = original.to_proto()
        restored = OptimizerConfig.from_proto(proto)

        assert restored.lr == original.lr
        assert restored.weight_decay == original.weight_decay

    def test_training_proto_roundtrip(self):
        original = TrainingConfig(
            seed=42,
            optimizer=OptimizerConfig(lr=0.001),
            max_epochs=50,
            scheduler=SchedulerConfig(name="cosine", t_max=5),
        )
        proto = original.to_proto()
        restored = TrainingConfig.from_proto(proto)

        assert restored.seed == original.seed
        assert restored.optimizer.lr == original.optimizer.lr
        assert restored.max_epochs == original.max_epochs
        assert restored.scheduler.name == "cosine"

    def test_trainrun_proto_roundtrip(self):
        trainrun = TrainRunConfig(
            name="test_run",
            pipeline=PipelineConfig(name="pipe", nodes=[], connections=[]),
            data=DataConfig(cu3s_file_path="/tmp/file.cu3s"),
            training=TrainingConfig(),
            loss_nodes=["loss1"],
            metric_nodes=["metric1"],
        )
        proto = trainrun.to_proto()
        restored = TrainRunConfig.from_proto(proto)
        assert restored.name == trainrun.name
        assert restored.data.cu3s_file_path == "/tmp/file.cu3s"
        assert restored.loss_nodes == ["loss1"]


def test_trainrun_json_roundtrip():
    trainrun = TrainRunConfig(
        name="json_run",
        pipeline=PipelineConfig(name="p", nodes=[], connections=[]),
        data=DataConfig(cu3s_file_path="/path/to/data.cu3s"),
        training=TrainingConfig(max_epochs=10),
        tags={"env": "dev"},
    )
    json_payload = trainrun.to_json()
    restored = TrainRunConfig.from_json(json_payload)
    assert restored.tags["env"] == "dev"

    # Ensure JSON is valid
    loaded = json.loads(json_payload)
    assert loaded["name"] == "json_run"
