"""Test helper functions in cuvis_ai/grpc/helpers.py."""

import pytest
from pydantic import ValidationError

from cuvis_ai_core.grpc.helpers import find_weights_file, resolve_pipeline_path
from cuvis_ai_core.training.config import DataConfig, TrainingConfig, TrainRunConfig
from cuvis_ai_core.utils.config_helpers import (
    apply_config_overrides,
    generate_json_schema,
    get_config_class,
    resolve_config_with_hydra,
    validate_config_dict,
)


class TestPipelinePathResolution:
    """Test pipeline path resolution logic."""

    def test_resolve_pipeline_path_with_short_name(self):
        """Test resolving pipeline by short name (e.g., 'rx_statistical')."""
        resolved = resolve_pipeline_path("rx_statistical")
        assert resolved.exists()
        assert resolved.name == "rx_statistical.yaml"
        assert "pipeline" in str(resolved)

    def test_resolve_pipeline_path_with_extension(self):
        """Test resolving pipeline with explicit .yaml extension."""
        resolved = resolve_pipeline_path("rx_statistical.yaml")
        assert resolved.exists()
        assert resolved.name == "rx_statistical.yaml"

    def test_resolve_pipeline_path_absolute(self, tmp_path):
        """Test pipeline resolution with absolute path."""
        pipeline_file = tmp_path / "custom_pipeline.yaml"
        pipeline_file.write_text("metadata:\n  name: test\nnodes: []\nconnections: []\n")

        resolved = resolve_pipeline_path(str(pipeline_file))
        assert resolved == pipeline_file

    def test_resolve_pipeline_path_not_found(self):
        """Test that FileNotFoundError is raised for missing pipeline."""
        with pytest.raises(FileNotFoundError, match="Pipeline configuration not found"):
            resolve_pipeline_path("nonexistent_pipeline_xyz")


class TestWeightsFileResolution:
    """Test weights file resolution logic."""

    def test_find_weights_file_in_search_paths(self, tmp_path):
        """Test finding weights file across search paths."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(exist_ok=True)
        weights_file = weights_dir / "test_model.pt"
        weights_file.write_bytes(b"fake-weights-data")

        found = find_weights_file("test_model.pt", [str(weights_dir)])
        assert found == weights_file.resolve()

    def test_find_weights_file_without_extension(self, tmp_path):
        """Test finding weights file when extension is not provided."""
        weights_dir = tmp_path / "weights"
        weights_dir.mkdir(exist_ok=True)
        weights_file = weights_dir / "test_model.pt"
        weights_file.write_bytes(b"fake-weights-data")

        # Should auto-add .pt extension
        found = find_weights_file("test_model", [str(weights_dir)])
        assert found == weights_file.resolve()

    def test_find_weights_file_absolute_path(self, tmp_path):
        """Test finding weights with absolute path."""
        weights_file = tmp_path / "absolute_weights.pt"
        weights_file.write_bytes(b"fake-weights-data")

        found = find_weights_file(str(weights_file), [])
        assert found == weights_file

    def test_find_weights_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when weights not found."""
        with pytest.raises(FileNotFoundError, match="Weights file .* not found"):
            find_weights_file("missing_weights.pt", [str(tmp_path)])


class TestConfigRegistry:
    """Test config type registry and schema generation."""

    def test_get_config_class_valid_types(self):
        """Test getting config classes for valid types."""
        assert get_config_class("data") == DataConfig
        assert get_config_class("training") == TrainingConfig
        assert get_config_class("trainrun") == TrainRunConfig

    def test_get_config_class_invalid_type(self):
        """Test that invalid config type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown config type"):
            get_config_class("invalid_type")

    def test_generate_json_schema_structure(self):
        """Test JSON schema generation produces valid schema."""
        schema = generate_json_schema("data")

        assert "title" in schema
        assert schema["title"] == "DataConfig"
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

    def test_generate_json_schema_has_required_fields(self):
        """Test that schema includes required fields."""
        schema = generate_json_schema("data")

        required = set(schema.get("required", []))
        assert "cu3s_file_path" in required
        # batch_size has a default value, so it's not actually required
        # Only cu3s_file_path should be required

    def test_validate_config_dict_valid(self):
        """Test validation with valid config dictionary."""
        config_dict = {
            "cu3s_file_path": "/tmp/data.cu3s",
            "annotation_json_path": "/tmp/annotations.json",
            "train_ids": [1, 2, 3],
            "val_ids": [4],
            "test_ids": [5],
            "batch_size": 4,
            "processing_mode": "Reflectance",
        }

        valid, errors = validate_config_dict("data", config_dict)
        assert valid
        assert len(errors) == 0

    def test_validate_config_dict_invalid(self):
        """Test validation with invalid config dictionary."""
        config_dict = {
            "cu3s_file_path": "/tmp/data.cu3s",
            "batch_size": 0,  # Invalid: must be > 0
        }

        valid, errors = validate_config_dict("data", config_dict)
        assert not valid
        assert len(errors) > 0
        assert any("batch_size" in err for err in errors)


class TestConfigOverrideUtility:
    """Tests for apply_config_overrides helper."""

    @pytest.fixture()
    def base_config(self):
        return {
            "metadata": {"name": "original"},
            "nodes": [{"params": {"lr": 0.1, "path": "/tmp"}}, {"params": {"flag": False}}],
            "connections": [],
        }

    def test_override_with_list_format(self, base_config):
        updated = apply_config_overrides(
            base_config, ["metadata.name=updated", "nodes.0.params.lr=0.2"]
        )
        assert updated["metadata"]["name"] == "updated"
        assert updated["nodes"][0]["params"]["lr"] == 0.2
        # Original config should remain unchanged
        assert base_config["metadata"]["name"] == "original"

    def test_override_with_dict_format(self, base_config):
        overrides = {"metadata": {"name": "dict-name"}, "nodes": [{"params": {"lr": 0.3}}]}
        updated = apply_config_overrides(base_config, overrides)
        assert updated["metadata"]["name"] == "dict-name"
        assert updated["nodes"][0]["params"]["lr"] == 0.3

    def test_override_nested_and_indices(self, base_config):
        updated = apply_config_overrides(
            base_config, ["nodes.1.params.flag=true", "nodes[0].params.path=/data"]
        )
        assert updated["nodes"][1]["params"]["flag"] is True
        assert updated["nodes"][0]["params"]["path"] == "/data"

    def test_invalid_override_format_raises(self, base_config):
        with pytest.raises(ValueError):
            apply_config_overrides(base_config, ["invalid-format"])


class TestHydraConfigResolution:
    """Test Hydra-based config resolution."""

    def test_resolve_config_with_hydra_trainrun(self, tmp_path):
        """Test resolving trainrun config with Hydra composition."""
        config_root = tmp_path / "configs"
        config_root.mkdir()

        # Create minimal trainrun config
        trainrun_yaml = config_root / "test_trainrun.yaml"
        trainrun_yaml.write_text(
            """
name: hydra-test-trainrun
pipeline:
  metadata:
    name: test-pipeline
  nodes: []
  connections: []
data:
  cu3s_file_path: /tmp/data.cu3s
  batch_size: 2
training:
  optimizer:
    name: adamw
    lr: 0.001
  trainer:
    max_epochs: 1
  batch_size: 2
  max_epochs: 1
metric_nodes: []
loss_nodes: []
"""
        )

        resolved = resolve_config_with_hydra(
            "trainrun",
            "test_trainrun.yaml",
            search_paths=[str(config_root)],
            overrides=["data.batch_size=4"],
        )

        assert resolved["name"] == "hydra-test-trainrun"
        assert resolved["data"]["batch_size"] == 4  # Override applied
        assert resolved["training"]["optimizer"]["name"] == "adamw"

    def test_resolve_config_with_hydra_file_not_found(self, tmp_path):
        """Test that FileNotFoundError is raised when config not found."""
        with pytest.raises(FileNotFoundError, match="Config file .* not found"):
            resolve_config_with_hydra(
                "trainrun",
                "nonexistent.yaml",
                search_paths=[str(tmp_path)],
            )

    def test_resolve_config_with_hydra_validation_error(self, tmp_path):
        """Test that validation error is raised for invalid config."""
        config_root = tmp_path / "configs"
        config_root.mkdir()

        # Create invalid trainrun config (missing required fields)
        invalid_yaml = config_root / "invalid.yaml"
        invalid_yaml.write_text(
            """
name: invalid-trainrun
# Missing required pipeline, data, training sections
"""
        )

        with pytest.raises(ValidationError):  # Pydantic ValidationError
            resolve_config_with_hydra(
                "trainrun",
                "invalid.yaml",
                search_paths=[str(config_root)],
            )
