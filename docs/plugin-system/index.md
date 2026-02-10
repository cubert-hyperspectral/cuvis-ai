!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

## Quick Start

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# Load plugin from Git
registry = NodeRegistry()
registry.load_plugin(
    name="adaclip",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.1",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# Use plugin node
AdaCLIPDetector = NodeRegistry.get("AdaCLIPDetector", instance=registry)
detector = AdaCLIPDetector(prompt="plastic wrapper", threshold=0.5)
```

## Documentation Guides

<div class="grid cards" markdown>

-   :material-puzzle: **[Plugin System Overview](overview.md)**

    ---

    Comprehensive guide to plugin architecture, loading mechanisms, caching, and node registration.

    **Topics:**
    - NodeRegistry architecture
    - Git and local plugin loading
    - Plugin manifest format
    - Cache management
    - Version management

-   :material-hammer-wrench: **[Plugin Development Guide](development.md)**

    ---

    Step-by-step guide to creating your own plugins from scratch to publication.

    **Topics:**
    - Plugin project structure
    - Node implementation
    - Testing and validation
    - Packaging and distribution
    - Best practices

-   :material-package-variant: **[Plugin Usage Guide](usage.md)**

    ---

    Practical guide for installing and using plugins in your workflows.

    **Topics:**
    - Finding plugins
    - Installation methods
    - Using in pipelines
    - CLI integration
    - Troubleshooting

</div>

---

## Why Use Plugins?

### Benefits

<div class="grid cards" markdown>

-   :material-puzzle-outline: **Modularity**

    ---

    Keep the core framework lean by separating specialized nodes into plugins

-   :material-share-variant: **Distribution**

    ---

    Share your custom nodes via Git repositories or PyPI packages

-   :material-package-lock: **Isolation**

    ---

    Manage dependencies independently per plugin without conflicts

-   :material-code-tags: **Versioning**

    ---

    Use Git tags for reproducible, deterministic plugin versions

-   :material-flash: **Flexibility**

    ---

    Add domain-specific functionality without modifying core code

-   :material-shield-check: **Safety**

    ---

    Plugin failures don't crash the core framework (session isolation)

</div>

### Use Cases

- **Custom Algorithms**: Implement proprietary anomaly detection methods
- **Research**: Experiment with new techniques without polluting core
- **Domain-Specific**: Add industry-specific preprocessing nodes
- **Integration**: Connect to third-party tools and services
- **Community**: Share and reuse community-contributed algorithms

---

## Plugin Examples

### Official Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection using CLIP embeddings

### Central Plugin Registry

cuvis-ai maintains a central registry of official and community plugins at [`configs/plugins/registry.yaml`](../../configs/plugins/registry.yaml).

**Using the Registry:**
```bash
# Load all registered plugins
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/registry.yaml
```

**Registered Plugins:**
- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection with Fisher band selection and mRMR algorithms

**Want to add your plugin?** See [Contributing Guide](../development/contributing.md#plugin-contribution-workflow) for submission instructions.

### Custom Plugin Manifests

You can also create custom manifests to load specific plugins:

```yaml
plugins:
  # Reference plugin from registry
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  # Local development plugin
  my_plugin:
    path: "../my-plugin"
    provides:
      - my_plugin.nodes.CustomNode
```

---

## Key Features

### Git-Based Distribution

- **Automatic cloning** from Git repositories (GitHub, GitLab, etc.)
- **Tag-based versioning** for reproducibility (no branches)
- **Shallow clones** for efficiency (`depth=1`)
- **Automatic caching** in `~/.cuvis_plugins/`

### Local Development

- **Path-based loading** for development workflows
- **Editable installs** with `pip install -e .`
- **Hot reloading** during development
- **Private plugins** without Git hosting

### Dependency Management

- **PEP 621 compliant** `pyproject.toml`
- **Automatic dependency installation** via `uv pip install`
- **Version constraints** enforcement
- **Isolated environments** per session

### CLI Integration

```bash
# Use plugins with restore-pipeline
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path plugins.yaml

# Use plugins with restore-trainrun
uv run restore-trainrun \
    --trainrun-path outputs/trainrun.yaml \
    --mode train
```

---

## Getting Started

### For Users: Installing Plugins

1. **Find a plugin** (GitHub, community, official)
2. **Create manifest** (`plugins.yaml`)
3. **Load and use**:
   ```python
   registry = NodeRegistry()
   registry.load_plugins("plugins.yaml")
   ```

See [Plugin Usage Guide](usage.md) for details.

### For Developers: Creating Plugins

1. **Set up structure** with `pyproject.toml`
2. **Implement nodes** inheriting from `Node`
3. **Add tests** with pytest
4. **Publish** to Git repository
5. **Tag versions** using semver

See [Plugin Development Guide](development.md) for step-by-step instructions.

---

## Related Documentation

### Core Concepts

- [Node System Deep Dive](../concepts/node-system-deep-dive.md) - Understanding node architecture
- [Port System Deep Dive](../concepts/port-system-deep-dive.md) - Port specifications and connections
- [Pipeline Lifecycle](../concepts/pipeline-lifecycle.md) - Pipeline integration

### How-To Guides

- [Build Pipeline (Python)](../how-to/build-pipeline-python.md) - Using plugins in Python pipelines
- [Build Pipeline (YAML)](../how-to/build-pipeline-yaml.md) - Using plugins in YAML configs
- [Add Built-in Node](../how-to/add-builtin-node.md) - Contributing nodes to core

### Reference

- [Node Catalog](../node-catalog/index.md) - Built-in nodes reference
- [Configuration System](../config/config-groups.md) - Hydra configuration
- [gRPC API](../grpc/api-reference.md) - Remote plugin loading
