"""
Test suite for port system (PortSpec, InputPort, OutputPort, DimensionResolver).
All tests should FAIL initially until implementation is complete.
"""

import pytest
import torch

from cuvis_ai_core.pipeline.ports import (
    DimensionResolver,
    InputPort,
    OutputPort,
    PortSpec,
)


class TestPortSpec:
    """Test PortSpec dataclass and its methods."""

    def test_portspec_creation_with_torch_dtype(self):
        """Test creating PortSpec with torch dtype."""
        spec = PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 15),
            description="Test port",
        )
        assert spec.dtype == torch.float32
        assert spec.shape == (-1, -1, -1, 15)
        assert spec.description == "Test port"
        assert spec.optional is False

    def test_portspec_creation_with_python_types(self):
        """Test creating PortSpec with Python types."""
        spec = PortSpec(dtype=dict, shape=(), description="Metadata port")
        assert spec.dtype == dict
        assert spec.shape == ()

    def test_portspec_optional_flag(self):
        """Test optional port specification."""
        spec = PortSpec(dtype=torch.float32, shape=(-1,), optional=True)
        assert spec.optional is True

    def test_portspec_symbolic_dimensions(self):
        """Test symbolic dimension in shape."""
        spec = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, "n_select"))
        assert spec.shape[3] == "n_select"
        assert isinstance(spec.shape[3], str)


class TestDimensionResolver:
    """Test dimension resolution logic."""

    def test_resolve_all_flexible_dims(self):
        """Test resolving shape with only flexible dimensions."""
        shape = (-1, -1, -1, -1)
        resolved = DimensionResolver.resolve(shape, node=None)
        assert resolved == (-1, -1, -1, -1)

    def test_resolve_fixed_dims(self):
        """Test resolving shape with fixed dimensions."""
        shape = (-1, -1, -1, 15)
        resolved = DimensionResolver.resolve(shape, node=None)
        assert resolved == (-1, -1, -1, 15)

    def test_resolve_symbolic_dims_from_node_params(self):
        """Test resolving symbolic dimensions using node parameters."""

        class MockNode:
            n_select = 10

        node = MockNode()
        shape = (-1, -1, -1, "n_select")
        resolved = DimensionResolver.resolve(shape, node)
        assert resolved == (-1, -1, -1, 10)

    def test_resolve_multiple_symbolic_dims(self):
        """Test resolving multiple symbolic dimensions."""

        class MockNode:
            n_components = 5
            n_channels = 20

        node = MockNode()
        shape = ("n_components", "n_channels")
        resolved = DimensionResolver.resolve(shape, node)
        assert resolved == (5, 20)

    def test_resolve_missing_node_param_raises_error(self):
        """Test that missing node parameter raises clear error."""

        class MockNode:
            pass

        node = MockNode()
        shape = (-1, "missing_param")

        with pytest.raises(AttributeError, match="missing_param"):
            DimensionResolver.resolve(shape, node)


class TestPortCompatibility:
    """Test port compatibility checking."""

    def test_identical_specs_are_compatible(self):
        """Test that identical specs are compatible."""
        spec1 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 15))
        spec2 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 15))

        compatible, message = spec1.is_compatible_with(spec2, None, None)
        assert compatible is True
        assert message == ""

    def test_dtype_mismatch_incompatible(self):
        """Test that dtype mismatch makes ports incompatible."""
        spec1 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 15))
        spec2 = PortSpec(dtype=torch.int64, shape=(-1, -1, -1, 15))

        compatible, message = spec1.is_compatible_with(spec2, None, None)
        assert compatible is False
        assert "dtype" in message.lower()

    def test_flexible_dims_are_compatible(self):
        """Test that flexible dimensions are compatible with any size."""
        spec1 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 15))
        spec2 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))

        compatible, message = spec1.is_compatible_with(spec2, None, None)
        assert compatible is True

    def test_fixed_dim_mismatch_incompatible(self):
        """Test that fixed dimension mismatch is incompatible."""
        spec1 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 15))
        spec2 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 20))

        compatible, message = spec1.is_compatible_with(spec2, None, None)
        assert compatible is False
        assert "dimension" in message.lower()

    def test_shape_length_mismatch_incompatible(self):
        """Test that different shape lengths are incompatible."""
        spec1 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, 15))
        spec2 = PortSpec(dtype=torch.float32, shape=(-1, -1, 15))

        compatible, message = spec1.is_compatible_with(spec2, None, None)
        assert compatible is False
        assert "rank" in message.lower() or "length" in message.lower()

    def test_symbolic_dims_resolved_before_checking(self):
        """Test that symbolic dimensions are resolved before compatibility check."""

        class MockNode1:
            n_select = 15

        class MockNode2:
            n_input = 15

        node1 = MockNode1()
        node2 = MockNode2()

        spec1 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, "n_select"))
        spec2 = PortSpec(dtype=torch.float32, shape=(-1, -1, -1, "n_input"))

        compatible, message = spec1.is_compatible_with(spec2, node1, node2)
        assert compatible is True


class TestPortProxies:
    """Test InputPort and OutputPort proxy objects."""

    def test_output_port_creation(self):
        """Test creating an OutputPort proxy."""

        class MockNode:
            id = "node1"

        node = MockNode()
        spec = PortSpec(dtype=torch.float32, shape=(-1, 15))

        port = OutputPort(node, "output1", spec)
        assert port.node is node
        assert port.name == "output1"
        assert port.spec is spec

    def test_input_port_creation(self):
        """Test creating an InputPort proxy."""

        class MockNode:
            id = "node2"

        node = MockNode()
        spec = PortSpec(dtype=torch.float32, shape=(-1, 15))

        port = InputPort(node, "input1", spec)
        assert port.node is node
        assert port.name == "input1"
        assert port.spec is spec

    def test_port_repr(self):
        """Test port string representation."""

        class MockNode:
            id = "node1"

        node = MockNode()
        spec = PortSpec(dtype=torch.float32, shape=(-1, 15))

        port = OutputPort(node, "output1", spec)
        repr_str = repr(port)
        assert "node1" in repr_str
        assert "output1" in repr_str
