"""Node registry for managing built-in and custom node types."""

import importlib
import inspect
from pathlib import Path


class NodeRegistry:
    """
    Hybrid registry supporting both built-in and custom nodes.

    Built-in nodes are registered via @register decorator for O(1) lookup.
    Custom nodes are loaded via importlib using full import paths.

    Examples:
        # Built-in node (registry lookup)
        normalizer_class = NodeRegistry.get("MinMaxNormalizer")

        # Custom node (importlib)
        custom_class = NodeRegistry.get("my_package.nodes.CustomRXDetector")
    """

    _builtin_registry: dict[str, type] = {}

    @classmethod
    def register(cls, node_class: type) -> type:
        """
        Decorator to register a built-in node class.

        Args:
            node_class: The node class to register

        Returns:
            The same node class (for decorator chaining)

        Example:
            @NodeRegistry.register
            class MinMaxNormalizer(Node):
                pass
        """
        class_name = node_class.__name__

        if class_name in cls._builtin_registry:
            raise ValueError(
                f"Node '{class_name}' is already registered. "
                f"Existing: {cls._builtin_registry[class_name]}, "
                f"New: {node_class}"
            )

        cls._builtin_registry[class_name] = node_class
        return node_class

    @classmethod
    def get(cls, class_identifier: str) -> type:
        """
        Get node class by name or full import path.

        Resolution order:
        1. Check built-in registry (O(1) lookup)
        2. Try importlib for full paths (e.g., "my_package.MyNode")
        3. Raise clear error if not found

        Args:
            class_identifier: Either a simple class name for built-in nodes
                            or full import path for custom nodes

        Returns:
            The node class

        Raises:
            KeyError: If node not found in registry or via import
            ImportError: If custom node path is invalid
            AttributeError: If module doesn't contain the class

        Examples:
            # Built-in node
            cls = NodeRegistry.get("MinMaxNormalizer")

            # Custom node with full path
            cls = NodeRegistry.get("my_company.detectors.AdvancedRXDetector")
        """
        # Try built-in registry first
        if class_identifier in cls._builtin_registry:
            return cls._builtin_registry[class_identifier]

        # Try importlib for custom nodes (must have dot in path)
        if "." in class_identifier:
            return cls._import_from_path(class_identifier)

        # Not found
        available = cls.list_builtin_nodes()
        raise KeyError(
            f"Node '{class_identifier}' not found in registry.\n"
            f"For custom nodes, provide full import path (e.g., 'my_package.MyNode').\n"
            f"Available built-in nodes: {available}"
        )

    @classmethod
    def _import_from_path(cls, import_path: str) -> type:
        """
        Import a class from a full module path.

        Args:
            import_path: Full import path (e.g., "my_package.nodes.CustomNode")

        Returns:
            The imported class

        Raises:
            ImportError: If module cannot be imported
            AttributeError: If class doesn't exist in module
        """
        try:
            # Split into module path and class name
            parts = import_path.rsplit(".", 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid import path: '{import_path}'")

            module_path, class_name = parts

            # Import the module
            module = importlib.import_module(module_path)

            # Get the class
            if not hasattr(module, class_name):
                raise AttributeError(
                    f"Module '{module_path}' has no class '{class_name}'. Available: {dir(module)}"
                )

            node_class = getattr(module, class_name)

            # Verify it's a class
            if not inspect.isclass(node_class):
                raise TypeError(f"'{import_path}' is not a class, got {type(node_class)}")

            return node_class

        except ImportError as e:
            raise ImportError(
                f"Failed to import module for '{import_path}': {e}\n"
                f"Ensure the module is installed and the path is correct."
            ) from e
        except AttributeError as e:
            raise AttributeError(f"Failed to load class '{import_path}': {e}") from e

    @classmethod
    def list_builtin_nodes(cls) -> list[str]:
        """
        List all registered built-in node names.

        Returns:
            Sorted list of node class names
        """
        return sorted(cls._builtin_registry.keys())

    @classmethod
    def auto_register_package(
        cls, package_name: str, base_class_path: str = "cuvis_ai.node.node.Node"
    ) -> int:
        """
        Auto-register all Node classes from a package.

        Searches the package for classes that inherit from Node
        and registers them automatically.

        Args:
            package_name: Full package name (e.g., "cuvis_ai.node")
            base_class_path: Full import path to the base Node class

        Returns:
            Number of classes registered

        Example:
            NodeRegistry.auto_register_package("cuvis_ai.node")
        """
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            raise ImportError(f"Failed to import package '{package_name}': {e}") from e

        # Get the package directory
        if not hasattr(package, "__path__"):
            raise ValueError(f"'{package_name}' is not a package (has no __path__)")

        # Import the base Node class first, outside the loop
        try:
            Node = cls._import_from_path(base_class_path)
        except Exception as e:
            raise ImportError(f"Failed to import base class '{base_class_path}': {e}") from e

        registered_count = 0

        # Import all modules in the package
        package_dir = Path(package.__path__[0])
        for module_file in package_dir.glob("*.py"):
            if module_file.name.startswith("_"):
                continue

            module_name = f"{package_name}.{module_file.stem}"
            try:
                module = importlib.import_module(module_name)

                # Find all Node subclasses
                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if issubclass(obj, Node) and obj is not Node:
                        # Only register if defined in this module
                        if obj.__module__ == module_name:
                            cls.register(obj)
                            registered_count += 1

            except Exception as e:
                # Log warning but continue
                print(f"Warning: Failed to auto-register from {module_name}: {e}")

        return registered_count

    @classmethod
    def clear(cls):
        """Clear all registered nodes (primarily for testing)."""
        cls._builtin_registry.clear()
