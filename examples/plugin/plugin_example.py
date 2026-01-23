"""Example demonstrating the NodeRegistry plugin system."""

from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_config import PluginManifest


def main():
    """Demonstrate plugin loading and usage."""
    print("=== NodeRegistry Plugin System Example ===\n")

    # Example 1: Programmatic plugin loading
    print("1. Loading plugins programmatically...")

    # Load a local plugin (assuming it exists)
    try:
        NodeRegistry.load_plugin("example_plugin", {
            "path": "../examples/test_plugin",
            "provides": ["test_plugin.ExampleNode"]
        })
        print("✓ Plugin loaded successfully")
    except Exception as e:
        print(f"✗ Plugin loading failed: {e}")

    # Example 2: List loaded plugins
    print("\n2. Listing loaded plugins...")
    plugins = NodeRegistry.list_plugins()
    print(f"Loaded plugins: {plugins}")

    # Example 3: List plugin nodes
    print("\n3. Listing plugin nodes...")
    plugin_nodes = NodeRegistry.list_plugin_nodes()
    print(f"Plugin nodes: {plugin_nodes}")

    # Example 4: Get a plugin node
    print("\n4. Getting a plugin node...")
    try:
        if "ExampleNode" in plugin_nodes:
            node_class = NodeRegistry.get("ExampleNode")
            print(f"✓ Found node: {node_class.__name__}")
        else:
            print("✗ No ExampleNode found in plugin nodes")
    except Exception as e:
        print(f"✗ Failed to get node: {e}")

    # Example 5: Plugin manifest example
    print("\n5. Creating a plugin manifest...")

    manifest_data = {
        "plugins": {
            "adaclip": {
                "repo": "git@gitlab.cubert.local:cubert/cuvis-ai-adaclip.git",
                "ref": "v0.1.0",
                "provides": ["cuvis_ai_adaclip.node.AdaCLIPDetector"]
            },
            "local_dev": {
                "path": "../my-plugin",
                "provides": ["my_plugin.MyNode"]
            }
        }
    }

    manifest = PluginManifest.from_dict(manifest_data)
    print(f"✓ Created manifest with {len(manifest.plugins)} plugins")

    # Example 6: Cache management
    print("\n6. Cache management...")
    print(f"Current cache directory: {NodeRegistry._cache_dir}")

    # Example 7: Cleanup
    print("\n7. Cleaning up...")
    NodeRegistry.clear_plugins()
    print("✓ All plugins cleared")

    print("\n=== Example Complete ===")

if __name__ == "__main__":
    main()
