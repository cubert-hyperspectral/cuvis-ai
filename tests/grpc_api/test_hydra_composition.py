from pathlib import Path

import pytest

from cuvis_ai_core.utils.config_helpers import resolve_config_with_hydra


def test_basic_config_resolution(pipeline_factory):
    """Test basic Hydra config resolution using existing fixtures."""
    config_dir = pipeline_factory(
        [
            (
                "training",
                {
                    "seed": 42,
                    "optimizer": {
                        "name": "adamw",
                        "lr": 0.001,
                        "weight_decay": 0.01,
                    },
                    "max_epochs": 100,
                    "batch_size": 32,
                },
                False,
            )
        ]
    )

    config_dict = resolve_config_with_hydra(
        config_type="training",
        config_path="training.yaml",
        search_paths=[str(config_dir)],
    )

    assert config_dict["seed"] == 42
    assert config_dict["optimizer"]["name"] == "adamw"
    assert config_dict["max_epochs"] == 100


def test_config_resolution_with_overrides(pipeline_factory):
    """Test Hydra config resolution with overrides using existing fixtures."""
    config_dir = pipeline_factory(
        [
            (
                "training",
                {
                    "seed": 42,
                    "optimizer": {"name": "adamw", "lr": 0.001},
                    "max_epochs": 100,
                    "batch_size": 32,
                },
                False,
            )
        ]
    )

    config_dict = resolve_config_with_hydra(
        config_type="training",
        config_path="training.yaml",
        search_paths=[str(config_dir)],
        overrides=["optimizer.lr=0.005", "max_epochs=50", "batch_size=64"],
    )

    assert config_dict["optimizer"]["lr"] == 0.005
    assert config_dict["max_epochs"] == 50
    assert config_dict["batch_size"] == 64
    assert config_dict["seed"] == 42
    assert config_dict["optimizer"]["name"] == "adamw"


def test_config_not_found():
    """Test error when config file not found."""
    with pytest.raises(FileNotFoundError) as exc_info:
        resolve_config_with_hydra(
            config_type="training",
            config_path="nonexistent.yaml",
            search_paths=["/tmp"],
        )

    assert "not found" in str(exc_info.value).lower()


def test_multiple_search_paths(pipeline_factory, tmp_path):
    """Test config resolution with multiple search paths using existing fixtures."""
    # Create first config directory
    config_dir1 = tmp_path / "configs1"
    config_dir1.mkdir()
    training_file1 = config_dir1 / "training.yaml"
    training_file1.write_text("""seed: 42
max_epochs: 100
optimizer:
  name: adamw
  lr: 0.001
batch_size: 32""")

    # Create second config directory
    config_dir2 = tmp_path / "configs2"
    config_dir2.mkdir()
    training_file2 = config_dir2 / "training.yaml"
    training_file2.write_text("""seed: 999
max_epochs: 200
optimizer:
  name: adamw
  lr: 0.001
batch_size: 32""")

    config_dict = resolve_config_with_hydra(
        config_type="training",
        config_path="training.yaml",
        search_paths=[str(config_dir1), str(config_dir2)],
    )

    # Should find the first config in the first search path
    assert config_dict["seed"] == 42
    assert config_dict["max_epochs"] == 100


def test_trainrun_resolution_with_config_root():
    """Trainrun config should compose against the config root (defaults across groups)."""
    config_dict = resolve_config_with_hydra(
        config_type="trainrun",
        config_path="trainrun/rx_statistical",
        search_paths=[str(Path("configs").resolve())],
    )

    assert config_dict["pipeline"]["metadata"]["name"] == "RX_Statistical"
    assert config_dict["training"]["trainer"]["max_epochs"] == 10
