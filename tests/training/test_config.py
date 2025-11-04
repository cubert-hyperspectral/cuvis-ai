"""Tests for training configuration infrastructure."""

import pytest
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.training.config import (
    OptimizerConfig,
    TrainerConfig,
    TrainingConfig,
    as_dict,
    from_dict_config,
    to_dict_config,
)


def test_trainer_config_defaults():
    """Test TrainerConfig default values."""
    config = TrainerConfig()
    assert config.max_epochs == 1
    assert config.accelerator == "cpu"
    assert config.devices is None
    assert config.precision == "32-true"
    assert config.accumulate_grad_batches == 1
    assert config.enable_progress_bar is True
    assert config.enable_checkpointing is False
    assert config.log_every_n_steps == 50


def test_optimizer_config_defaults():
    """Test OptimizerConfig default values."""
    config = OptimizerConfig()
    assert config.name == "adam"
    assert config.lr == 1e-3
    assert config.weight_decay == 0.0
    assert config.betas is None


def test_training_config_defaults():
    """Test TrainingConfig default values."""
    config = TrainingConfig()
    assert config.seed == 42
    assert isinstance(config.trainer, TrainerConfig)
    assert isinstance(config.optimizer, OptimizerConfig)
    assert config.monitor_plugins == ["loguru"]


def test_training_config_custom():
    """Test TrainingConfig with custom values."""
    trainer = TrainerConfig(max_epochs=10, accelerator="gpu")
    optimizer = OptimizerConfig(name="adamw", lr=0.001)
    config = TrainingConfig(seed=123, trainer=trainer, optimizer=optimizer)
    
    assert config.seed == 123
    assert config.trainer.max_epochs == 10
    assert config.trainer.accelerator == "gpu"
    assert config.optimizer.name == "adamw"
    assert config.optimizer.lr == 0.001


def test_as_dict():
    """Test as_dict conversion."""
    config = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=5),
        optimizer=OptimizerConfig(lr=0.01)
    )
    
    result = as_dict(config)
    
    assert result["seed"] == 42
    assert result["trainer"]["max_epochs"] == 5
    assert result["optimizer"]["lr"] == 0.01
    assert result["monitor_plugins"] == ["loguru"]


def test_to_dict_config():
    """Test to_dict_config conversion."""
    config = TrainingConfig(seed=99)
    dict_config = to_dict_config(config)
    
    assert isinstance(dict_config, DictConfig)
    assert dict_config.seed == 99
    assert dict_config.trainer.max_epochs == 1


def test_from_dict_config():
    """Test from_dict_config conversion."""
    dict_config = OmegaConf.create({
        "seed": 77,
        "trainer": {"max_epochs": 20, "accelerator": "gpu"},
        "optimizer": {"name": "sgd", "lr": 0.1},
        "monitor_plugins": ["wandb"]
    })
    
    config = from_dict_config(dict_config)
    
    assert isinstance(config, TrainingConfig)
    assert config.seed == 77
    assert config.trainer.max_epochs == 20
    assert config.trainer.accelerator == "gpu"
    assert config.optimizer.name == "sgd"
    assert config.optimizer.lr == 0.1
    assert config.monitor_plugins == ["wandb"]


def test_roundtrip_serialization():
    """Test serialization roundtrip: config -> dict -> config."""
    original = TrainingConfig(
        seed=42,
        trainer=TrainerConfig(max_epochs=15, accelerator="gpu", devices=2),
        optimizer=OptimizerConfig(name="adam", lr=0.003, weight_decay=0.01)
    )
    
    # Convert to dict config
    dict_config = to_dict_config(original)
    
    # Convert back to TrainingConfig
    restored = from_dict_config(dict_config)
    
    # Verify all fields match
    assert restored.seed == original.seed
    assert restored.trainer.max_epochs == original.trainer.max_epochs
    assert restored.trainer.accelerator == original.trainer.accelerator
    assert restored.trainer.devices == original.trainer.devices
    assert restored.optimizer.name == original.optimizer.name
    assert restored.optimizer.lr == original.optimizer.lr
    assert restored.optimizer.weight_decay == original.optimizer.weight_decay
