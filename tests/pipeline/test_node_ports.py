"""
Test suite for Node port system integration.
"""

from __future__ import annotations

import torch

from cuvis_ai_core.node import Node
from cuvis_ai_core.pipeline.ports import InputPort, OutputPort, PortSpec


class TestNodePortCreation:
    """Test that nodes create ports from specifications."""

    def test_node_creates_input_ports_from_specs(self) -> None:
        """Node should instantiate InputPort proxies for INPUT_SPECS entries."""

        class TestNode(Node):
            INPUT_SPECS = {
                "data": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
                "mask": PortSpec(dtype=torch.bool, shape=(-1, -1, -1), optional=True),
            }
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                raise NotImplementedError

        node = TestNode()

        assert hasattr(node, "_input_ports")
        assert set(node._input_ports) == {"data", "mask"}
        assert isinstance(node._input_ports["data"], InputPort)
        assert isinstance(node._input_ports["mask"], InputPort)

    def test_node_creates_output_ports_from_specs(self) -> None:
        """Node should instantiate OutputPort proxies for OUTPUT_SPECS entries."""

        class TestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "result": PortSpec(dtype=torch.float32, shape=(-1, 10)),
                "confidence": PortSpec(dtype=torch.float32, shape=(-1,)),
            }

            def forward(self, **inputs):
                raise NotImplementedError

        node = TestNode()

        assert hasattr(node, "_output_ports")
        assert set(node._output_ports) == {"result", "confidence"}
        assert isinstance(node._output_ports["result"], OutputPort)
        assert isinstance(node._output_ports["confidence"], OutputPort)

    def test_empty_specs_create_no_ports(self) -> None:
        """Nodes without specs should not create proxy objects."""

        class EmptyNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                raise NotImplementedError

        node = EmptyNode()
        assert node._input_ports == {}
        assert node._output_ports == {}


class TestNodePortAttributes:
    """Test port proxy attribute access on node instances."""

    def test_input_ports_accessible_as_attributes(self) -> None:
        """Input ports should be exposed as attributes using their names."""

        class TestNode(Node):
            INPUT_SPECS = {"features": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                raise NotImplementedError

        node = TestNode()

        assert hasattr(node, "features")
        assert isinstance(node.features, InputPort)
        assert node.features.name == "features"
        assert node.features.node is node

    def test_output_ports_accessible_as_attributes(self) -> None:
        """Output ports should be exposed as attributes using their names."""

        class TestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {"transformed": PortSpec(dtype=torch.float32, shape=(-1, -1))}

            def forward(self, **inputs):
                raise NotImplementedError

        node = TestNode()

        assert hasattr(node, "transformed")
        assert isinstance(node.transformed, OutputPort)
        assert node.transformed.name == "transformed"
        assert node.transformed.node is node

    def test_port_attributes_do_not_shadow_custom_attributes(self) -> None:
        """Port creation should not remove attributes added by subclasses."""

        class TestNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {}

            def __init__(self) -> None:
                super().__init__()
                self.custom_attr = "value"

            def forward(self, **inputs):
                raise NotImplementedError

        node = TestNode()

        assert isinstance(node.data, InputPort)
        assert node.custom_attr == "value"


class TestNodeForwardSignature:
    """Test new forward() expectations using keyword arguments and dict returns."""

    def test_forward_accepts_kwargs(self) -> None:
        """Nodes should accept keyword arguments matching INPUT_SPECS."""

        class TestNode(Node):
            INPUT_SPECS = {
                "x": PortSpec(dtype=torch.float32, shape=(-1, 10)),
                "y": PortSpec(dtype=torch.float32, shape=(-1,)),
            }
            OUTPUT_SPECS = {
                "result": PortSpec(dtype=torch.float32, shape=(-1, 10)),
            }

            def forward(self, **inputs):
                x = inputs["x"]
                y = inputs["y"].unsqueeze(-1)
                return {"result": x + y}

        node = TestNode()
        x = torch.randn(5, 10)
        y = torch.randn(5)

        output = node.forward(x=x, y=y)
        assert isinstance(output, dict)
        assert "result" in output
        assert output["result"].shape == x.shape

    def test_forward_returns_dict(self) -> None:
        """Nodes should return dictionaries keyed by OUTPUT_SPECS."""

        class TestNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}
            OUTPUT_SPECS = {"doubled": PortSpec(dtype=torch.float32, shape=(-1,))}

            def forward(self, **inputs):
                data = inputs["data"]
                return {"doubled": data * 2}

        node = TestNode()
        data = torch.randn(10)

        output = node.forward(data=data)
        assert isinstance(output, dict)
        assert "doubled" in output
        torch.testing.assert_close(output["doubled"], data * 2)

    def test_forward_keys_match_output_specs(self) -> None:
        """Returned keys should align with OUTPUT_SPECS."""

        class TestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {
                "result1": PortSpec(dtype=torch.float32, shape=()),
                "result2": PortSpec(dtype=torch.float32, shape=()),
            }

            def forward(self, **inputs):
                return {
                    "result1": torch.tensor(1.0),
                    "result2": torch.tensor(2.0),
                }

        node = TestNode()
        output = node.forward()
        assert set(output.keys()) == {"result1", "result2"}


class TestMultiplePortNodes:
    """Test nodes with multiple inputs and outputs."""

    def test_node_with_multiple_inputs(self) -> None:
        """Node should expose all declared input ports."""

        class FusionNode(Node):
            INPUT_SPECS = {
                "spectral": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
                "spatial": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
                "thermal": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
            }
            OUTPUT_SPECS = {
                "fused": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1)),
            }

            def forward(self, **inputs):
                spectral = inputs["spectral"]
                spatial = inputs["spatial"]
                thermal = inputs["thermal"]
                fused = torch.cat([spectral, spatial, thermal], dim=-1)
                return {"fused": fused}

        node = FusionNode()

        assert hasattr(node, "spectral")
        assert hasattr(node, "spatial")
        assert hasattr(node, "thermal")

        b, h, w, c = 2, 10, 10, 5
        spec = torch.randn(b, h, w, c)
        spat = torch.randn(b, h, w, c)
        therm = torch.randn(b, h, w, c)

        output = node.forward(spectral=spec, spatial=spat, thermal=therm)
        assert output["fused"].shape == (b, h, w, c * 3)

    def test_node_with_multiple_outputs(self) -> None:
        """Node should expose all declared output ports."""

        class AnalyzerNode(Node):
            INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1, -1))}
            OUTPUT_SPECS = {
                "mean": PortSpec(dtype=torch.float32, shape=(-1,)),
                "std": PortSpec(dtype=torch.float32, shape=(-1,)),
                "min": PortSpec(dtype=torch.float32, shape=(-1,)),
                "max": PortSpec(dtype=torch.float32, shape=(-1,)),
            }

            def forward(self, **inputs):
                data = inputs["data"]
                return {
                    "mean": data.mean(dim=0),
                    "std": data.std(dim=0),
                    "min": data.min(dim=0).values,
                    "max": data.max(dim=0).values,
                }

        node = AnalyzerNode()

        assert hasattr(node, "mean")
        assert hasattr(node, "std")
        assert hasattr(node, "min")
        assert hasattr(node, "max")

        data = torch.randn(100, 10)
        output = node.forward(data=data)
        assert set(output) == {"mean", "std", "min", "max"}
        assert output["mean"].shape == (10,)
        assert output["std"].shape == (10,)
        assert output["min"].shape == (10,)
        assert output["max"].shape == (10,)
