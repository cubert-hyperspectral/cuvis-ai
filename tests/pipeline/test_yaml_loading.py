"""Tests for YAML-based pipeline building."""

import pytest

from cuvis_ai_core.pipeline.factory import PipelineBuilder


@pytest.fixture
def simple_pipeline_config():
    """Simple pipeline configuration for testing."""
    return {
        "metadata": {"name": "test_pipeline"},
        "nodes": [
            {
                "name": "normalizer",
                "class": "MinMaxNormalizer",
                "params": {"eps": 1e-6, "use_running_stats": True},
            }
        ],
        "connections": [],
    }


@pytest.fixture
def pipeline_with_connections():
    """Pipeline with multiple nodes and connections."""
    return {
        "metadata": {"name": "test_pipeline"},
        "nodes": [
            {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-6}},
            {
                "name": "selector",
                "class": "SoftChannelSelector",
                "params": {"n_select": 3, "input_channels": 10, "init_method": "variance"},
            },
        ],
        "connections": [{"from": "normalizer.outputs.normalized", "to": "selector.inputs.data"}],
    }


class TestPipelineBuilder:
    """Test PipelineBuilder functionality."""

    def test_build_simple_pipeline(self, simple_pipeline_config):
        """Test building pipeline with single node."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(simple_pipeline_config)

        assert pipeline is not None
        assert len(list(pipeline.nodes)) == 1
        assert pipeline.name == "test_pipeline"

    def test_build_with_connections(self, pipeline_with_connections):
        """Test building pipeline with connections."""
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(pipeline_with_connections)

        assert len(list(pipeline.nodes)) == 2
        # Verify the graph has an edge (connection)
        assert pipeline._graph.number_of_edges() == 1

    def test_load_from_yaml_file(self, tmp_path):
        """Test loading configuration from YAML file."""
        yaml_content = """metadata:
  name: "test"
nodes:
  - name: "normalizer"
    class: "MinMaxNormalizer"
    params:
      eps: 1.0e-6
connections: []
"""
        config_file = tmp_path / "test_pipeline.yaml"
        config_file.write_text(yaml_content)

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config_file)

        assert pipeline is not None
        assert pipeline.name == "test"

    def test_omegaconf_interpolation(self):
        """Test building pipeline with hardcoded node parameters."""
        pipeline_cfg = {
            "metadata": {"name": "test"},
            "nodes": [
                {
                    "name": "selector",
                    "class": "SoftChannelSelector",
                    "params": {"n_select": 3, "input_channels": 61, "init_method": "variance"},
                }
            ],
            "connections": [],
        }

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(pipeline_cfg)

        # Verify node was created successfully
        assert len(list(pipeline.nodes)) == 1

    def test_missing_node_raises_error(self):
        """Test that missing node class raises error."""
        config = {
            "metadata": {"name": "test"},
            "nodes": [{"name": "missing", "class": "NonExistentNode", "params": {}}],
            "connections": [],
        }

        builder = PipelineBuilder()
        with pytest.raises(KeyError):
            builder.build_from_config(config)

    def test_invalid_connection_format_raises_error(self):
        """Test that invalid connection format raises error."""
        config = {
            "metadata": {"name": "test"},
            "nodes": [{"name": "normalizer", "class": "MinMaxNormalizer", "params": {}}],
            "connections": [
                {
                    "from": "invalid_format",  # Missing .outputs.port
                    "to": "normalizer.inputs.data",
                }
            ],
        }

        builder = PipelineBuilder()
        with pytest.raises(ValueError, match="Invalid 'from' specification"):
            builder.build_from_config(config)

    def test_missing_source_node_raises_error(self):
        """Test that missing source node raises error."""
        config = {
            "metadata": {"name": "test"},
            "nodes": [{"name": "normalizer", "class": "MinMaxNormalizer", "params": {}}],
            "connections": [{"from": "missing_node.outputs.data", "to": "normalizer.inputs.data"}],
        }

        builder = PipelineBuilder()
        with pytest.raises(ValueError, match="Source node not found"):
            builder.build_from_config(config)

    def test_missing_target_node_raises_error(self):
        """Test that missing target node raises error."""
        config = {
            "metadata": {"name": "test"},
            "nodes": [{"name": "normalizer", "class": "MinMaxNormalizer", "params": {}}],
            "connections": [
                {"from": "normalizer.outputs.normalized", "to": "missing_node.inputs.data"}
            ],
        }

        builder = PipelineBuilder()
        with pytest.raises(ValueError, match="Target node not found"):
            builder.build_from_config(config)

    def test_counter_based_naming(self):
        """Test that counter-based naming works correctly."""
        config = {
            "metadata": {"name": "test"},
            "nodes": [
                {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-6}},
                {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-6}},
                {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-6}},
            ],
            "connections": [],
        }

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config)

        # Verify nodes have correct names: normalizer, normalizer-1, normalizer-2
        node_names = [node.name for node in pipeline.nodes]
        assert "normalizer" in node_names
        assert "normalizer-1" in node_names
        assert "normalizer-2" in node_names
        assert len(node_names) == 3

    def test_file_not_found_raises_error(self):
        """Test that non-existent file raises FileNotFoundError."""
        builder = PipelineBuilder()
        with pytest.raises(FileNotFoundError):
            builder.build_from_config("nonexistent_file.yaml")

    def test_pipeline_name_from_metadata(self):
        """Test that pipeline name is extracted from metadata."""
        config = {
            "metadata": {"name": "MyCustomPipeline"},
            "nodes": [],
            "connections": [],
        }

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config)

        assert pipeline.name == "MyCustomPipeline"

    def test_pipeline_default_name(self):
        """Test that default pipeline name is used when metadata missing."""
        config = {"nodes": [], "connections": []}

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config)

        assert pipeline.name == "Pipeline"

    def test_empty_params(self):
        """Test that nodes can be created without params."""
        config = {
            "metadata": {"name": "test"},
            "nodes": [
                {
                    "name": "normalizer",
                    "class": "MinMaxNormalizer",
                    # No params field
                }
            ],
            "connections": [],
        }

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config)

        assert len(list(pipeline.nodes)) == 1

    def test_counter_gap_preservation_on_node_removal(self):
        """Test that counter gaps are preserved when nodes are removed and new ones added.

        This verifies the fix for counter assignment that uses the highest counter + 1
        instead of counting existing nodes, which prevents name collisions.

        Example:
            Initial: normalizer (0), normalizer-1 (1), normalizer-2 (2)
            Remove: normalizer-1 (gap at counter 1)
            Add new: normalizer-3 (3) - gap preserved, no collision
        """
        # Step 1: Create pipeline with 3 nodes of same type
        config = {
            "metadata": {"name": "test"},
            "nodes": [
                {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-6}},
                {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-7}},
                {"name": "normalizer", "class": "MinMaxNormalizer", "params": {"eps": 1e-8}},
            ],
            "connections": [],
        }

        builder = PipelineBuilder()
        pipeline = builder.build_from_config(config)

        # Verify initial state: normalizer, normalizer-1, normalizer-2
        node_names = {node.name for node in pipeline.nodes}
        assert node_names == {"normalizer", "normalizer-1", "normalizer-2"}
        assert len(list(pipeline.nodes)) == 3

        # Step 2: Remove the middle node (normalizer-1) using internal graph API
        node_to_remove = None
        for node in pipeline.nodes:
            if node.name == "normalizer-1":
                node_to_remove = node
                break

        assert node_to_remove is not None, "normalizer-1 should exist"
        pipeline._graph.remove_node(node_to_remove)

        # Verify node was removed and gap exists at counter 1
        node_names_after_removal = {node.name for node in pipeline.nodes}
        assert node_names_after_removal == {"normalizer", "normalizer-2"}
        assert len(list(pipeline.nodes)) == 2

        # Step 3: Add a new node with the same base name using internal method
        from cuvis_ai.node.normalization import MinMaxNormalizer

        new_node = MinMaxNormalizer(eps=1e-9, name="normalizer")
        pipeline._assign_counter_and_add_node(new_node)

        # Step 4: Verify the new node gets counter=3 (gap preserved)
        # NOT counter=1 which would fill the gap and cause collision
        node_names_after_add = {node.name for node in pipeline.nodes}
        assert "normalizer-3" in node_names_after_add, (
            "New node should be normalizer-3, not normalizer-1"
        )
        assert "normalizer-1" not in node_names_after_add, "Gap at counter 1 should be preserved"
        assert node_names_after_add == {"normalizer", "normalizer-2", "normalizer-3"}
        assert len(list(pipeline.nodes)) == 3

        # Verify the new node has the correct counter value
        for node in pipeline.nodes:
            if node.name == "normalizer-3":
                assert node._pipeline_counter == 3, "New node should have counter=3"
