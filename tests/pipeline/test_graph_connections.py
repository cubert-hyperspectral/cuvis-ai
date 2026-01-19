"""
Test suite for Graph connection API with port-based wiring.
"""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.pipeline.ports import PortCompatibilityError, PortSpec


class TestGraphConnectionBasics:
    """Test basic connection operations."""

    def test_connect_two_nodes_single_syntax(self) -> None:
        """Connecting two nodes with single connection syntax should succeed."""

        class ProducerNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "output": PortSpec(dtype=torch.float32, shape=(-1, 10)),
            }

            def forward(self, **inputs):
                return {"output": torch.zeros(1, 10)}

        class ConsumerNode(Node):
            INPUT_SPECS = {
                "input": PortSpec(dtype=torch.float32, shape=(-1, 10)),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        producer = ProducerNode()
        consumer = ConsumerNode()

        # Nodes are automatically added when connecting
        pipeline.connect(producer.outputs.output, consumer.input)

        assert pipeline._graph.has_edge(producer, consumer)
        edge_dict = pipeline._graph.get_edge_data(producer, consumer)
        assert edge_dict is not None
        edge_data = next(iter(edge_dict.values()))
        assert edge_data["from_port"] == "output"
        assert edge_data["to_port"] == "input"

    def test_connect_multiple_connections_batch_syntax(self) -> None:
        """Batch connection syntax should create multiple edges."""

        class SourceNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "out1": PortSpec(dtype=torch.float32, shape=(-1,)),
                "out2": PortSpec(dtype=torch.float32, shape=(-1,)),
            }

            def forward(self, **inputs):
                return {"out1": torch.zeros(10), "out2": torch.ones(10)}

        class TargetNode(Node):
            INPUT_SPECS = {
                "in1": PortSpec(dtype=torch.float32, shape=(-1,)),
                "in2": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        source = SourceNode()
        target = TargetNode()

        # Nodes are automatically added when connecting
        pipeline.connect(
            (source.outputs.out1, target.in1),
            (source.outputs.out2, target.in2),
        )

        edge_dict = pipeline._graph.get_edge_data(source, target)
        assert edge_dict is not None
        assert len(edge_dict) == 2


class TestConnectionValidation:
    """Test connection validation logic."""

    def test_connecting_incompatible_dtypes_raises_error(self) -> None:
        """Connecting different dtypes should raise PortCompatibilityError."""

        class FloatNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=())}

            def forward(self, **inputs):
                return {}

        class IntNode(Node):
            INPUT_SPECS = {"in": PortSpec(dtype=torch.int64, shape=())}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        float_node = FloatNode()
        int_node = IntNode()

        # Connecting incompatible types should raise error
        with pytest.raises(PortCompatibilityError, match=r"(?i)dtype"):
            pipeline.connect(float_node.outputs.out, int_node.inputs.__getattribute__("in"))

    def test_connecting_incompatible_shapes_raises_error(self) -> None:
        """Connecting mismatched shapes should raise PortCompatibilityError."""

        class Shape10Node(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(10,))}

            def forward(self, **inputs):
                return {}

        class Shape20Node(Node):
            INPUT_SPECS = {"in": PortSpec(dtype=torch.float32, shape=(20,))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        node_10 = Shape10Node()
        node_20 = Shape20Node()

        # Connecting incompatible shapes should raise error
        with pytest.raises(PortCompatibilityError, match=r"(?i)dimension"):
            pipeline.connect(node_10.outputs.out, node_20.inputs.__getattribute__("in"))

    def test_connecting_compatible_flexible_shapes_succeeds(self) -> None:
        """Flexible shapes with -1 should be compatible."""

        class FlexNode1(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"out": PortSpec(dtype=torch.float32, shape=(-1, 10))}

            def forward(self, **inputs):
                return {}

        class FlexNode2(Node):
            INPUT_SPECS = {"in": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        node_1 = FlexNode1()
        node_2 = FlexNode2()

        # Flexible shapes should be compatible
        pipeline.connect(
            node_1.outputs.out, node_2.inputs.__getattribute__("in")
        )  # Should not raise


class TestMultiDiGraphIntegration:
    """Test MultiDiGraph structure and edge attributes."""

    def test_graph_uses_multidigraph(self) -> None:
        """Graph.graph should be a NetworkX MultiDiGraph."""

        pipeline = CuvisPipeline("test")
        assert isinstance(pipeline._graph, nx.MultiDiGraph)

    def test_multiple_connections_between_same_nodes(self) -> None:
        """Multiple connections between same node pair should be supported."""

        class MultiOutNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "out1": PortSpec(dtype=torch.float32, shape=()),
                "out2": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **inputs):
                return {}

        class MultiInNode(Node):
            INPUT_SPECS = {
                "in1": PortSpec(dtype=torch.float32, shape=()),
                "in2": PortSpec(dtype=torch.float32, shape=()),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        src = MultiOutNode()
        tgt = MultiInNode()

        # Multiple connections between same nodes
        pipeline.connect(
            (src.outputs.out1, tgt.in1),
            (src.outputs.out2, tgt.in2),
        )

        edge_dict = pipeline._graph.get_edge_data(src, tgt)
        assert edge_dict is not None
        assert len(edge_dict) == 2

    def test_edge_iteration_includes_all_connections(self) -> None:
        """Edge iteration should include all multi-edges with data."""

        class Node1(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "a": PortSpec(dtype=torch.float32, shape=()),
                "b": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **inputs):
                return {}

        class Node2(Node):
            INPUT_SPECS = {
                "x": PortSpec(dtype=torch.float32, shape=()),
                "y": PortSpec(dtype=torch.float32, shape=()),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

        pipeline = CuvisPipeline("test")
        node_1 = Node1()
        node_2 = Node2()

        # Connect nodes
        pipeline.connect((node_1.a, node_2.x), (node_1.b, node_2.y))

        edges = list(pipeline._graph.edges(keys=True, data=True))
        assert len(edges) == 2
        for _, _, _, edge_data in edges:
            assert "from_port" in edge_data and "to_port" in edge_data
