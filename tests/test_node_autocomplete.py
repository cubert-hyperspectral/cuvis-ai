"""Test that port autocomplete works via __init_subclass__ annotations."""

import pytest
import torch

from cuvis_ai_core.node.node import Node
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai_core.pipeline.ports import InputPort, OutputPort, PortSpec


def test_init_subclass_adds_annotations():
    """Verify __init_subclass__ creates port specs for nodes."""

    class TestNode(Node):
        INPUT_SPECS = {
            "input_data": PortSpec(dtype=torch.float32, shape=(-1, -1)),
            "labels": PortSpec(dtype=torch.int64, shape=(-1,)),
        }
        OUTPUT_SPECS = {
            "output_result": PortSpec(dtype=torch.float32, shape=(-1, -1)),
            "scores": PortSpec(dtype=torch.float32, shape=(-1,)),
        }

        def forward(self, **inputs):
            return {}

    # Check that INPUT_SPECS and OUTPUT_SPECS are defined
    assert hasattr(TestNode, "INPUT_SPECS")
    assert hasattr(TestNode, "OUTPUT_SPECS")
    assert "input_data" in TestNode.INPUT_SPECS
    assert "labels" in TestNode.INPUT_SPECS
    assert "output_result" in TestNode.OUTPUT_SPECS
    assert "scores" in TestNode.OUTPUT_SPECS

    # Create instance to check runtime ports
    node = TestNode()
    assert "input_data" in node._input_ports
    assert "labels" in node._input_ports
    assert "output_result" in node._output_ports
    assert "scores" in node._output_ports


def test_real_node_has_annotations_pca():
    """Test that TrainablePCA has port specs."""

    # Check INPUT_SPECS and OUTPUT_SPECS exist
    assert "data" in TrainablePCA.INPUT_SPECS
    assert "projected" in TrainablePCA.OUTPUT_SPECS

    # Create instance to check runtime ports
    pca = TrainablePCA(n_components=5, input_channels=10)
    assert hasattr(pca.inputs, "data")
    assert hasattr(pca.outputs, "projected")


def test_real_node_has_annotations_selector():
    """Test that SoftChannelSelector has port specs."""

    # Check INPUT_SPECS and OUTPUT_SPECS exist
    assert "data" in SoftChannelSelector.INPUT_SPECS
    assert "selected" in SoftChannelSelector.OUTPUT_SPECS
    assert "weights" in SoftChannelSelector.OUTPUT_SPECS

    # Create instance to check runtime ports
    selector = SoftChannelSelector(n_select=5, input_channels=10)
    assert hasattr(selector.inputs, "data")
    assert hasattr(selector.outputs, "selected")
    assert hasattr(selector.outputs, "weights")


def test_runtime_ports_still_created():
    """Verify runtime behavior unchanged - ports are still created at init."""
    from cuvis_ai_core.pipeline.ports import InputPort, OutputPort

    class TestNode(Node):
        INPUT_SPECS = {"test_input": PortSpec(dtype=torch.float32, shape=(-1,))}
        OUTPUT_SPECS = {"test_output": PortSpec(dtype=torch.float32, shape=(-1,))}

        def forward(self, **inputs):
            return {}

    # Create instance
    node = TestNode()

    # Verify ports were created at runtime
    assert hasattr(node, "test_input")
    assert hasattr(node, "test_output")
    assert isinstance(node.test_input, InputPort)
    assert isinstance(node.test_output, OutputPort)
    assert node.test_input is node.test_input
    assert node.outputs.test_output is node.test_output
    # Verify port properties
    assert node.test_input.name == "test_input"
    assert node.test_output.name == "test_output"
    assert hasattr(node.inputs, "test_input")
    assert hasattr(node.outputs, "test_output")


def test_duplicate_port_names_use_namespaces():
    """When a name exists in inputs and outputs, require explicit namespace."""

    class DupNode(Node):
        INPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}
        OUTPUT_SPECS = {"data": PortSpec(dtype=torch.float32, shape=(-1,))}

        def forward(self, **inputs):
            return {}

    node = DupNode()

    # When port name exists in both inputs and outputs, direct access raises error
    with pytest.raises(AttributeError, match="exists in both inputs and outputs"):
        _ = node.data

    # Must use explicit namespace
    assert isinstance(node.inputs.data, InputPort)
    assert isinstance(node.outputs.data, OutputPort)


def test_annotations_with_inheritance():
    """Test that port specs work correctly with class inheritance."""

    class BaseNode(Node):
        INPUT_SPECS = {"base_input": PortSpec(dtype=torch.float32, shape=(-1,))}
        OUTPUT_SPECS = {"base_output": PortSpec(dtype=torch.float32, shape=(-1,))}

        def forward(self, **inputs):
            return {}

    class DerivedNode(BaseNode):
        INPUT_SPECS = {
            **BaseNode.INPUT_SPECS,
            "derived_input": PortSpec(dtype=torch.float32, shape=(-1,)),
        }
        OUTPUT_SPECS = {
            **BaseNode.OUTPUT_SPECS,
            "derived_output": PortSpec(dtype=torch.float32, shape=(-1,)),
        }

    # Check base node specs
    assert "base_input" in BaseNode.INPUT_SPECS
    assert "base_output" in BaseNode.OUTPUT_SPECS

    # Check derived node specs include both base and derived
    assert "base_input" in DerivedNode.INPUT_SPECS
    assert "base_output" in DerivedNode.OUTPUT_SPECS
    assert "derived_input" in DerivedNode.INPUT_SPECS
    assert "derived_output" in DerivedNode.OUTPUT_SPECS


def test_empty_specs_doesnt_fail():
    """Test that nodes with no INPUT_SPECS/OUTPUT_SPECS work fine."""

    class EmptyNode(Node):
        # No INPUT_SPECS or OUTPUT_SPECS defined

        def forward(self, **inputs):
            return {}

    # Should not raise any errors
    node = EmptyNode()

    # Verify it has empty port dicts
    assert len(node._input_ports) == 0
    assert len(node._output_ports) == 0
