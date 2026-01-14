"""Tests for YAML-based PipelineBuilder from cuvis_ai.pipeline.factory."""

from pathlib import Path

import pytest
import yaml

from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.pipeline.factory import PipelineBuilder
from cuvis_ai.pipeline.pipeline import CuvisPipeline


@pytest.fixture
def minimal_pipeline_yaml():
    """Minimal valid pipeline YAML configuration."""
    return {
        "metadata": {
            "name": "Test Pipeline",
            "description": "A minimal test pipeline",
            "created": "2024-11-27",
            "cuvis_ai_version": "0.1.5",
            "tags": ["test"],
            "author": "test_suite",
        },
        "nodes": [
            {
                "name": "data",
                "class": "cuvis_ai.node.data.LentilsAnomalyDataNode",
                "params": {
                    "normal_class_ids": [0],
                },
            },
            {
                "name": "normalizer",
                "class": "cuvis_ai.node.normalization.MinMaxNormalizer",
                "params": {
                    "eps": 1e-6,
                    "use_running_stats": False,
                },
            },
            {
                "name": "selector",
                "class": "cuvis_ai.node.selector.SoftChannelSelector",
                "params": {
                    "n_select": 3,
                    "input_channels": 10,
                    "init_method": "uniform",
                    "temperature_init": 5.0,
                    "temperature_min": 0.1,
                    "temperature_decay": 0.9,
                    "hard": False,
                },
            },
        ],
        "connections": [
            {
                "from": "data.outputs.cube",
                "to": "normalizer.inputs.data",
            },
            {
                "from": "normalizer.outputs.normalized",
                "to": "selector.inputs.data",
            },
        ],
    }


@pytest.fixture
def pipeline_yaml_file(minimal_pipeline_yaml, tmp_path):
    """Create a temporary YAML file with pipeline configuration."""
    yaml_path = tmp_path / "test_pipeline.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(minimal_pipeline_yaml, f)
    return yaml_path


class TestPipelineBuilderYAML:
    """Tests for YAML-based PipelineBuilder."""

    def test_build_from_dict(self, minimal_pipeline_yaml):
        """Test building pipeline from dictionary configuration."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(minimal_pipeline_yaml)

        assert isinstance(pipeline, CuvisPipeline)
        assert pipeline.name == "Test Pipeline"

        # Check nodes were created
        nodes = list(pipeline.nodes())
        assert len(nodes) == 3

        # Check node types
        node_types = {type(node).__name__ for node in nodes}
        assert "LentilsAnomalyDataNode" in node_types
        assert "MinMaxNormalizer" in node_types
        assert "SoftChannelSelector" in node_types

    def test_build_from_yaml_file(self, pipeline_yaml_file):
        """Test building pipeline from YAML file."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(str(pipeline_yaml_file))

        assert isinstance(pipeline, CuvisPipeline)
        assert pipeline.name == "Test Pipeline"

        nodes = list(pipeline.nodes())
        assert len(nodes) == 3

    def test_build_from_path_object(self, pipeline_yaml_file):
        """Test building pipeline from Path object."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(pipeline_yaml_file)

        assert isinstance(pipeline, CuvisPipeline)
        assert len(list(pipeline.nodes())) == 3

    def test_pipeline_forward_pass(self, minimal_pipeline_yaml, create_test_cube):
        """Test that built pipeline can execute forward pass."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(minimal_pipeline_yaml)

        # Find selector to get input channels
        selector = None
        for node in pipeline.nodes():
            if isinstance(node, SoftChannelSelector):
                selector = node
                break

        assert selector is not None, "SoftChannelSelector not found in pipeline"

        # Create test input
        batch_size = 2
        height = 3
        width = 3
        channels = selector.input_channels
        cube, wavelengths = create_test_cube(
            batch_size=batch_size, height=height, width=width, num_channels=channels
        )

        # Execute forward pass
        outputs = pipeline.forward(batch={"cube": cube, "wavelengths": wavelengths})

        # Check that we got outputs
        assert len(outputs) > 0
        assert (selector.name, "selected") in outputs

    def test_connections_validated(self, minimal_pipeline_yaml, create_test_cube):
        """Test that connections are properly established."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(minimal_pipeline_yaml)

        # Get nodes
        data_node = None
        normalizer = None
        selector = None

        for node in pipeline.nodes():
            if isinstance(node, LentilsAnomalyDataNode):
                data_node = node
            elif isinstance(node, MinMaxNormalizer):
                normalizer = node
            elif isinstance(node, SoftChannelSelector):
                selector = node

        assert data_node is not None
        assert normalizer is not None
        assert selector is not None

        # Verify connections exist by running forward pass
        # (Pipeline doesn't expose a connections() method, but we can verify it works)
        batch_size = 2
        cube, wavelengths = create_test_cube(
            batch_size=batch_size, height=3, width=3, num_channels=selector.input_channels
        )
        outputs = pipeline.forward(batch={"cube": cube, "wavelengths": wavelengths})
        assert len(outputs) > 0  # If connections work, we get outputs

    def test_node_parameters_respected(self, minimal_pipeline_yaml):
        """Test that node parameters from config are respected."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(minimal_pipeline_yaml)

        selector = None
        for node in pipeline.nodes():
            if isinstance(node, SoftChannelSelector):
                selector = node
                break

        assert selector is not None
        assert selector.n_select == 3
        assert selector.input_channels == 10
        assert selector.init_method == "uniform"

    def test_metadata_extraction(self, minimal_pipeline_yaml):
        """Test that metadata is correctly extracted and used."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(minimal_pipeline_yaml)

        assert pipeline.name == "Test Pipeline"

    def test_invalid_node_class_raises_error(self, minimal_pipeline_yaml):
        """Test that invalid node class raises ImportError."""
        config = minimal_pipeline_yaml.copy()
        config["nodes"][0]["class"] = "invalid.node.Class"

        builder = PipelineBuilder()
        with pytest.raises(ImportError):
            builder.build_from_config(config)

    def test_invalid_connection_raises_error(self, minimal_pipeline_yaml):
        """Test that invalid connection specification raises error."""
        config = minimal_pipeline_yaml.copy()
        config["connections"].append(
            {
                "from": "nonexistent.outputs.port",
                "to": "normalizer.inputs.data",
            }
        )

        builder = PipelineBuilder()
        with pytest.raises(ValueError, match="not found"):
            builder.build_from_config(config)

    def test_missing_config_file_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        builder = PipelineBuilder()
        with pytest.raises(FileNotFoundError):
            builder.build_from_config("nonexistent_file.yaml")


class TestPipelineBuilderPathResolution:
    """Tests for pipeline path resolution (short names, etc.)."""

    def test_short_name_resolution(self, tmp_path, minimal_pipeline_yaml):
        """Test that short names are resolved to default pipeline directory."""
        # Create a pipeline directory with a test config
        pipeline_dir = tmp_path / "configs" / "pipeline"
        pipeline_dir.mkdir(parents=True)

        pipeline_file = pipeline_dir / "test_pipeline.yaml"
        with open(pipeline_file, "w") as f:
            yaml.dump(minimal_pipeline_yaml, f)

        # Create builder with custom default directory
        builder = PipelineBuilder(default_pipeline_dir=str(pipeline_dir))

        # Should resolve "test_pipeline" to "configs/pipeline/test_pipeline.yaml"
        pipeline = builder.build_from_config("test_pipeline")
        assert isinstance(pipeline, CuvisPipeline)

    def test_absolute_path_resolution(self, pipeline_yaml_file):
        """Test that absolute paths work."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(str(pipeline_yaml_file.absolute()))
        assert isinstance(pipeline, CuvisPipeline)

    def test_relative_path_resolution(self, pipeline_yaml_file, monkeypatch):
        """Test that relative paths work."""
        # Change to the parent directory
        monkeypatch.chdir(pipeline_yaml_file.parent)

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(pipeline_yaml_file.name)
        assert isinstance(pipeline, CuvisPipeline)

    def test_path_without_yaml_extension(self, tmp_path, minimal_pipeline_yaml):
        """Test that paths without .yaml extension are resolved."""
        pipeline_file = tmp_path / "test_pipeline.yaml"
        with open(pipeline_file, "w") as f:
            yaml.dump(minimal_pipeline_yaml, f)

        builder = PipelineBuilder()
        # Try without extension - should add .yaml automatically
        pipeline = builder.build_from_config(str(tmp_path / "test_pipeline"))
        assert isinstance(pipeline, CuvisPipeline)


class TestPipelineBuilderRealConfigs:
    """Tests using actual config files if they exist."""

    def test_load_channel_selector_if_exists(self):
        """Test loading channel_selector.yaml if it exists."""
        config_path = Path("configs/pipeline/channel_selector.yaml")
        if not config_path.exists():
            pytest.skip("channel_selector.yaml not found")

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(str(config_path))

        assert isinstance(pipeline, CuvisPipeline)
        assert len(list(pipeline.nodes())) > 0

    def test_load_rx_statistical_if_exists(self):
        """Test loading rx_statistical.yaml if it exists."""
        config_path = Path("configs/pipeline/rx_statistical.yaml")
        if not config_path.exists():
            pytest.skip("rx_statistical.yaml not found")

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(str(config_path))

        assert isinstance(pipeline, CuvisPipeline)
        assert len(list(pipeline.nodes())) > 0
