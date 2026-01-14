from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.pipeline.pipeline import CuvisPipeline


class TestPipelineIntrospection:
    """Test Pipeline introspection methods with O(N+E) complexity."""

    def test_get_input_specs_returns_entry_ports(self):
        """Test that get_input_specs returns specs for graph entry points."""
        pipeline = CuvisPipeline("test")
        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        normalizer = MinMaxNormalizer(name="normalizer")

        pipeline.connect(data_node.outputs.cube, normalizer.inputs.data)

        input_specs = pipeline.get_input_specs()

        # Data node takes 'cube' as input (entry point)
        assert "cube" in input_specs
        assert input_specs["cube"]["name"] == "cube"
        assert "dtype" in input_specs["cube"]
        assert "shape" in input_specs["cube"]
        assert "required" in input_specs["cube"]

    def test_get_output_specs_returns_exit_ports(self):
        """Test that get_output_specs returns specs for graph exit points."""
        pipeline = CuvisPipeline("test")
        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        normalizer = MinMaxNormalizer(name="normalizer")

        pipeline.connect(data_node.outputs.cube, normalizer.inputs.data)

        output_specs = pipeline.get_output_specs()

        # Normalizer outputs 'normalized' (exit point)
        assert "normalizer.normalized" in output_specs
        assert output_specs["normalizer.normalized"]["name"] == "normalizer.normalized"
        assert "dtype" in output_specs["normalizer.normalized"]
        assert "shape" in output_specs["normalizer.normalized"]
        assert output_specs["normalizer.normalized"]["required"] is False

    def test_introspection_with_multiple_outputs(self):
        """Test introspection with graph having multiple outputs."""
        pipeline = CuvisPipeline("test")

        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        normalizer = MinMaxNormalizer(name="normalizer")
        selector = SoftChannelSelector(n_select=3, input_channels=10, name="selector")

        pipeline.connect(
            (data_node.outputs.cube, normalizer.inputs.data),
            (normalizer.outputs.normalized, selector.inputs.data),
        )

        output_specs = pipeline.get_output_specs()

        # Both selector outputs should be present (exit points)
        assert "selector.selected" in output_specs
        assert "selector.weights" in output_specs

        # Check structure
        assert output_specs["selector.selected"]["name"] == "selector.selected"
        assert output_specs["selector.weights"]["name"] == "selector.weights"

    def test_input_specs_excludes_connected_ports(self):
        """Test that connected input ports are NOT returned as entry points."""
        pipeline = CuvisPipeline("test")

        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        normalizer = MinMaxNormalizer(name="normalizer")

        pipeline.connect(data_node.outputs.cube, normalizer.inputs.data)

        input_specs = pipeline.get_input_specs()

        # normalizer.inputs.data is connected, so should NOT be in input_specs
        # (it would have been "data" as the port name)
        # Only unconnected ports should appear
        # Since data_node.outputs.cube connects to normalizer.inputs.data,
        # normalizer's "data" input is connected and should not appear as entry

        # Check that normalizer's data port is NOT in the input specs
        # (it's connected from data_node)
        normalizer_data_in_specs = any(
            spec["name"] == "data" and "normalizer" in str(spec) for spec in input_specs.values()
        )
        assert not normalizer_data_in_specs

    def test_output_specs_excludes_connected_ports(self):
        """Test that connected output ports are NOT returned as exit points."""
        pipeline = CuvisPipeline("test")

        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        normalizer = MinMaxNormalizer(name="normalizer")

        pipeline.connect(data_node.outputs.cube, normalizer.inputs.data)

        output_specs = pipeline.get_output_specs()

        # data_node.outputs.cube is connected to normalizer, so NOT an exit point
        assert "data.cube" not in output_specs

        # But normalizer.outputs.normalized is NOT connected, so IS an exit point
        assert "normalizer.normalized" in output_specs

    def test_dtype_to_string_conversion(self):
        """Test that dtypes are correctly converted to strings."""

        pipeline = CuvisPipeline("test")

        # Test with actual node that has torch dtypes
        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        normalizer = MinMaxNormalizer(name="normalizer")

        pipeline.connect(data_node.outputs.cube, normalizer.inputs.data)

        output_specs = pipeline.get_output_specs()

        # Check that dtype is a string
        assert isinstance(output_specs["normalizer.normalized"]["dtype"], str)
        # Common dtypes should be recognized
        assert output_specs["normalizer.normalized"]["dtype"] in [
            "float32",
            "float64",
            "int32",
            "int64",
            "uint8",
            "bool",
        ]

    def test_empty_graph_returns_empty_specs(self):
        """Test that empty pipeline returns empty specs."""
        pipeline = CuvisPipeline("test")

        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        assert input_specs == {}
        assert output_specs == {}

    def test_single_node_all_ports_unconnected(self):
        """Test single node with all ports unconnected."""
        pipeline = CuvisPipeline("test")

        data_node = LentilsAnomalyDataNode(normal_class_ids=[0], name="data")
        pipeline._graph.add_node(data_node)

        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()

        # All inputs should be entry points
        assert "cube" in input_specs

        # All outputs should be exit points
        assert "data.cube" in output_specs

    def test_complexity_with_large_graph(self):
        """Test that introspection works efficiently with larger graphs."""
        import time

        pipeline = CuvisPipeline("test")

        # Create a chain of nodes
        nodes = []
        for i in range(50):  # 50 nodes in a chain
            if i == 0:
                node = LentilsAnomalyDataNode(normal_class_ids=[0], name=f"data_{i}")
            else:
                node = MinMaxNormalizer(name=f"norm_{i}")
            nodes.append(node)

        # Connect them in a chain
        for i in range(len(nodes) - 1):
            if i == 0:
                pipeline.connect(nodes[i].outputs.cube, nodes[i + 1].inputs.data)
            else:
                pipeline.connect(nodes[i].outputs.normalized, nodes[i + 1].inputs.data)

        # Time the introspection (should be fast with O(N+E))
        start = time.time()
        input_specs = pipeline.get_input_specs()
        output_specs = pipeline.get_output_specs()
        elapsed = time.time() - start

        # Should complete quickly (under 100ms for 50 nodes)
        assert elapsed < 0.1

        # Verify correctness
        assert "cube" in input_specs  # Entry point
        assert "norm_49.normalized" in output_specs  # Exit point

        # Intermediate nodes should not appear
        assert "data_0.cube" not in output_specs  # Connected
        assert "norm_1.normalized" not in output_specs  # Connected
