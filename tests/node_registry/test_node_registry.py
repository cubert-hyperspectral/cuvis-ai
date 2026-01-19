"""Tests for NodeRegistry."""

import pytest

from cuvis_ai_core.node import Node
from cuvis_ai_core.utils.node_registry import NodeRegistry


class TestNodeRegistry:
    """Test NodeRegistry functionality."""

    def setup_method(self):
        """Clear registry before each test."""
        NodeRegistry.clear()

    def test_register_decorator(self):
        """Test @register decorator."""

        @NodeRegistry.register
        class TestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        assert "TestNode" in NodeRegistry.list_builtin_nodes()
        assert NodeRegistry.get("TestNode") is TestNode

    def test_register_duplicate_raises_error(self):
        """Test registering duplicate node raises error."""

        @NodeRegistry.register
        class DuplicateTestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        # Try to register another class with the same name
        with pytest.raises(ValueError, match="already registered"):

            class AnotherDuplicateTestNode(Node):
                INPUT_SPECS = {}
                OUTPUT_SPECS = {}

                def forward(self, **inputs):
                    return {}

                def load(self, params, serial_dir):
                    pass

            # Manually set __name__ to create duplicate
            AnotherDuplicateTestNode.__name__ = "DuplicateTestNode"
            NodeRegistry.register(AnotherDuplicateTestNode)

    def test_get_builtin_node(self):
        """Test getting built-in node by name."""

        @NodeRegistry.register
        class MinMaxNormalizer(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        cls = NodeRegistry.get("MinMaxNormalizer")
        assert cls is MinMaxNormalizer

    def test_get_custom_node_via_importlib(self):
        """Test getting custom node via full import path."""
        # This assumes cuvis_ai.node.normalization.MinMaxNormalizer exists
        cls = NodeRegistry.get("cuvis_ai.node.normalization.MinMaxNormalizer")
        assert cls.__name__ == "MinMaxNormalizer"

    def test_get_missing_node_raises_error(self):
        """Test getting non-existent node raises clear error."""
        with pytest.raises(KeyError, match="not found in registry"):
            NodeRegistry.get("NonExistentNode")

    def test_get_invalid_import_path_raises_error(self):
        """Test invalid import path raises ImportError."""
        with pytest.raises(ImportError):
            NodeRegistry.get("invalid.module.path.FakeNode")

    def test_get_missing_class_in_module_raises_error(self):
        """Test getting non-existent class from valid module raises AttributeError."""
        with pytest.raises(AttributeError, match="has no class"):
            NodeRegistry.get("cuvis_ai.node.normalization.NonExistentClass")

    def test_list_builtin_nodes(self):
        """Test listing all registered nodes."""

        @NodeRegistry.register
        class NodeA(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        @NodeRegistry.register
        class NodeB(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        nodes = NodeRegistry.list_builtin_nodes()
        assert "NodeA" in nodes
        assert "NodeB" in nodes
        assert nodes == sorted(nodes)  # Should be sorted

    def test_auto_register_package(self):
        """Test auto-registering entire package."""
        NodeRegistry.clear()
        count = NodeRegistry.auto_register_package("cuvis_ai.node")
        assert count > 0

        # Verify some expected nodes
        registered_nodes = NodeRegistry.list_builtin_nodes()
        assert "MinMaxNormalizer" in registered_nodes
        assert "SoftChannelSelector" in registered_nodes
        assert "TrainablePCA" in registered_nodes

    def test_clear_registry(self):
        """Test clearing registry."""

        @NodeRegistry.register
        class TestNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        assert len(NodeRegistry.list_builtin_nodes()) > 0
        NodeRegistry.clear()
        assert len(NodeRegistry.list_builtin_nodes()) == 0

    def test_get_after_auto_register(self):
        """Test getting nodes after auto-registration."""
        NodeRegistry.clear()
        NodeRegistry.auto_register_package("cuvis_ai.node")

        # Should be able to get built-in nodes by name
        normalizer_cls = NodeRegistry.get("MinMaxNormalizer")
        assert normalizer_cls.__name__ == "MinMaxNormalizer"

        # Should still be able to get via full path
        normalizer_cls2 = NodeRegistry.get("cuvis_ai.node.normalization.MinMaxNormalizer")
        assert normalizer_cls2 is normalizer_cls

    def test_import_from_path_invalid_format(self):
        """Test _import_from_path with invalid format raises ValueError."""
        with pytest.raises((ValueError, ImportError)):
            NodeRegistry._import_from_path("InvalidPathWithNoDot")

    def test_import_from_path_not_a_class(self):
        """Test importing something that's not a class raises TypeError."""
        # Try to import a function or variable instead of a class
        with pytest.raises(TypeError, match="not a class"):
            # __name__ is a module attribute, not a class
            NodeRegistry._import_from_path("cuvis_ai.node.__name__")

    def test_auto_register_invalid_package(self):
        """Test auto-registering invalid package raises ImportError."""
        with pytest.raises(ImportError, match="Failed to import package"):
            NodeRegistry.auto_register_package("nonexistent.package.name")

    def test_auto_register_not_a_package(self):
        """Test auto-registering a module (not package) raises ValueError."""
        # cuvis_ai.node.normalization is a module, not a package
        with pytest.raises(ValueError, match="not a package"):
            NodeRegistry.auto_register_package("cuvis_ai.node.normalization")

    def test_registry_error_message_includes_available_nodes(self):
        """Test that error message includes list of available nodes."""

        @NodeRegistry.register
        class AvailableNode(Node):
            INPUT_SPECS = {}
            OUTPUT_SPECS = {}

            def forward(self, **inputs):
                return {}

            def load(self, params, serial_dir):
                pass

        with pytest.raises(KeyError) as exc_info:
            NodeRegistry.get("MissingNode")

        error_msg = str(exc_info.value)
        assert "Available built-in nodes:" in error_msg
        assert "AvailableNode" in error_msg


class TestNodeRegistryIntegration:
    """Integration tests with actual cuvis_ai nodes."""

    def setup_method(self):
        """Clear and re-register for each test."""
        NodeRegistry.clear()
        NodeRegistry.auto_register_package("cuvis_ai.node")

    def test_can_instantiate_registered_nodes(self):
        """Test that registered nodes can be instantiated."""
        # Get a node class and instantiate it
        NormalizerClass = NodeRegistry.get("MinMaxNormalizer")
        normalizer = NormalizerClass()
        assert normalizer is not None
        assert hasattr(normalizer, "forward")

    def test_multiple_node_types_registered(self):
        """Test that various node types are registered."""
        registered = NodeRegistry.list_builtin_nodes()

        # Check for various node types
        assert any("Normalizer" in name for name in registered)
        assert any("PCA" in name for name in registered)
        assert any("Selector" in name for name in registered)
        assert any("Loss" in name for name in registered)

    def test_registry_preserves_node_class_identity(self):
        """Test that registry returns the same class object."""
        from cuvis_ai.node.normalization import MinMaxNormalizer

        registered_class = NodeRegistry.get("MinMaxNormalizer")
        assert registered_class is MinMaxNormalizer
