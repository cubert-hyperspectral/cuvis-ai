!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Loading Plugins

## Introduction

This guide covers how to find, install, and use plugins in your cuvis-ai workflows. Whether you're using plugins from Git repositories or local development, this guide provides practical examples for integrating plugin nodes into your pipelines.

## Quick Start

### Install and Use a Plugin

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# Create registry
registry = NodeRegistry()

# Load plugin from Git
registry.load_plugin(
    name="adaclip",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.1",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# Get node class
AdaCLIPDetector = NodeRegistry.get("AdaCLIPDetector", instance=registry)

# Use in your workflow
detector = AdaCLIPDetector(prompt="plastic wrapper", threshold=0.5)
outputs = detector(data=your_hyperspectral_data)
```

## Finding Plugins

### Official Plugins

cuvis-ai maintains official plugins:

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection

### Community Plugins

Search GitHub topics:

- **[cuvis-ai-plugin](https://github.com/topics/cuvis-ai-plugin)** topic
- Search for "cuvis-ai" + domain keywords

## Using the Central Plugin Registry

cuvis-ai maintains a [central plugin registry](../../configs/plugins/registry.yaml) containing all officially registered plugins. This provides a convenient way to discover and load community plugins without manually configuring each one.

### Loading from Central Registry

Load all plugins from the central registry:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# Load all registered plugins
registry = NodeRegistry()
registry.load_plugins("configs/plugins/registry.yaml")

# All registered plugins now available
AdaCLIPDetector = NodeRegistry.get("AdaCLIPDetector", instance=registry)
```

**Via CLI:**
```bash
# Use central registry with restore-pipeline
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path configs/plugins/registry.yaml
```

### Checking Registered Plugins

View which plugins are in the registry:

```bash
# View registry contents
cat configs/plugins/registry.yaml

# Or in Python
import yaml

with open("configs/plugins/registry.yaml") as f:
    registry_data = yaml.safe_load(f)
    plugins = registry_data.get("plugins", {})

    print("Registered Plugins:")
    for name, config in plugins.items():
        print(f"  - {name}: {config.get('repo', config.get('path'))}")
        print(f"    Tag: {config.get('tag', 'N/A')}")
        print(f"    Provides: {', '.join(config.get('provides', []))}")
```

### Combining Central Registry with Custom Plugins

You can create a custom manifest that references both registry plugins and your own:

```yaml
# my_plugins.yaml
plugins:
  # Load from central registry by copying entry
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  # Add your custom local plugin
  my_dev_plugin:
    path: "../my-plugin"
    provides:
      - my_plugin.nodes.CustomNode

  # Add your private Git plugin
  private_plugin:
    repo: "git@github.com:myorg/private-plugin.git"
    tag: "v1.0.0"
    provides:
      - private_plugin.nodes.PrivateNode
```

**Load combined manifest:**
```python
registry = NodeRegistry()
registry.load_plugins("my_plugins.yaml")

# Now you have access to:
# - Registry plugins (adaclip)
# - Local development plugins (my_dev_plugin)
# - Private plugins (private_plugin)
```

### Contributing to the Registry

Want to add your plugin to the central registry? See the [Contributing Guide](../development/contributing.md#plugin-contribution-workflow) for the submission process.

**Benefits of registry submission:**

- Easier discovery by the community
- Official "blessed" quality signal
- Automatic inclusion in central manifest
- Better visibility for your work

### Creating Plugin Manifests

For reproducible workflows, create a `plugins.yaml` manifest:

```yaml
plugins:
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  custom_detector:
    path: "../my-custom-plugin"
    provides:
      - custom_plugin.nodes.CustomDetector
```

## Installing Plugins

### From Git Repository

Load plugins directly from Git repositories:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()

# Load from GitHub (HTTPS)
registry.load_plugin(
    name="adaclip",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.1",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# Load from GitHub (SSH)
registry.load_plugin(
    name="private_plugin",
    config={
        "repo": "git@github.com:your-org/private-plugin.git",
        "tag": "v2.0.0",
        "provides": ["private_plugin.nodes.PrivateNode"]
    }
)

# Load from GitLab
registry.load_plugin(
    name="gitlab_plugin",
    config={
        "repo": "https://gitlab.com/your-org/plugin.git",
        "tag": "v1.5.0",
        "provides": ["gitlab_plugin.nodes.CustomNode"]
    }
)
```

**Features:**

- Automatic caching in `~/.cuvis_plugins/`
- Tag verification on subsequent loads
- Dependency installation from `pyproject.toml`

### From Local Path

Load plugins from local filesystem (ideal for development):

```python
registry = NodeRegistry()

# Relative path
registry.load_plugin(
    name="local_dev",
    config={
        "path": "../my-plugin",
        "provides": ["my_plugin.nodes.CustomNode"]
    }
)

# Absolute path
registry.load_plugin(
    name="local_prod",
    config={
        "path": "/absolute/path/to/plugin",
        "provides": ["prod_plugin.nodes.MainNode"]
    }
)
```

**Use Cases:**

- Local development and testing
- Private plugins not on Git
- Enterprise internal plugins

### From Manifest File

Load multiple plugins from a YAML manifest:

```python
registry = NodeRegistry()

# Load all plugins from manifest
registry.load_plugins("plugins.yaml")

# All plugins now available
AdaCLIPDetector = NodeRegistry.get("AdaCLIPDetector", instance=registry)
CustomNode = NodeRegistry.get("CustomNode", instance=registry)
```

**Manifest Example** (`plugins.yaml`):

```yaml
plugins:
  # Production plugin with specific version
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  # Pre-release plugin
  experimental:
    repo: "https://github.com/company/experimental.git"
    tag: "v2.0.0-beta.1"
    provides:
      - experimental.features.NewFeatureNode

  # Local development plugin
  dev_plugin:
    path: "../dev-plugin"
    provides:
      - dev_plugin.nodes.DevNode
```

## See Also

- **[Using Plugin Nodes](using-nodes.md)** - Working with plugin nodes, managing and versioning plugins
- **[Plugin Configuration & Examples](plugin-config.md)** - Configuration, examples, troubleshooting, and best practices
