!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Usage Guide

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

## Using Plugin Nodes

### In Python Code

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import Pipeline
import numpy as np

# Load plugins
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")

# Get node class
AdaCLIPDetector = NodeRegistry.get("AdaCLIPDetector", instance=registry)

# Instantiate node
detector = AdaCLIPDetector(
    prompt="plastic wrapper",
    threshold=0.5
)

# Use directly
data = np.random.randn(100, 100, 200).astype(np.float32)
outputs = detector(data=data)
print(f"Anomaly map shape: {outputs['anomaly_map'].shape}")
```

### In Pipeline (Python)

```python
from cuvis_ai_core.pipeline.pipeline import Pipeline

# Define pipeline with plugin nodes
pipeline_dict = {
    "nodes": [
        {
            "class_name": "MinMaxNormalizer",  # Built-in node
            "name": "normalizer",
            "params": {}
        },
        {
            "class_name": "AdaCLIPDetector",  # Plugin node
            "name": "adaclip",
            "params": {
                "prompt": "plastic wrapper",
                "threshold": 0.5
            }
        }
    ],
    "edges": [
        {
            "source": "normalizer.output",
            "target": "adaclip.data"
        }
    ]
}

# Create pipeline with plugin registry
pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

# Execute
outputs = pipeline(data=input_data)
anomaly_map = outputs["adaclip"]["anomaly_map"]
```

### In Pipeline (YAML)

Create `my_pipeline.yaml`:

```yaml
nodes:
  - name: normalizer
    class_name: MinMaxNormalizer
    params: {}

  - name: adaclip_detector
    class_name: AdaCLIPDetector  # From plugin
    params:
      prompt: "plastic wrapper"
      threshold: 0.5

  - name: threshold_selector
    class_name: ThresholdSelector
    params:
      threshold: 0.9

edges:
  - source: normalizer.output
    target: adaclip_detector.data

  - source: adaclip_detector.anomaly_map
    target: threshold_selector.data
```

Load and use:

```python
import yaml
from cuvis_ai_core.pipeline.pipeline import Pipeline

# Load plugins
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")

# Load pipeline config
with open("my_pipeline.yaml") as f:
    pipeline_dict = yaml.safe_load(f)

# Create pipeline
pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

# Run
outputs = pipeline(data=input_data)
```

### Via CLI: restore-pipeline

Use the `restore-pipeline` CLI with plugins:

```bash
# Basic usage
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path plugins.yaml

# With inference on CU3S file
uv run restore-pipeline \
    --pipeline-path configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml \
    --plugins-path plugins.yaml \
    --cu3s-file-path data/test_sample.cu3s

# With loaded weights
uv run restore-pipeline \
    --pipeline-path configs/pipeline/trained_pipeline.yaml \
    --plugins-path plugins.yaml \
    --weights-path outputs/trained_models/pipeline_weights.pt

# Export visualization
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path plugins.yaml \
    --pipeline-vis-ext png
```

### Via CLI: restore-trainrun

Restore training runs with plugins:

```bash
# Display trainrun info
uv run restore-trainrun \
    --trainrun-path outputs/trained_models/trainrun.yaml

# Re-run training
uv run restore-trainrun \
    --trainrun-path outputs/trained_models/trainrun.yaml \
    --mode train

# Validation with plugins
uv run restore-trainrun \
    --trainrun-path outputs/trained_models/trainrun.yaml \
    --mode validate \
    --override validation.batch_size=8
```

## Managing Plugins

### List Loaded Plugins

```python
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")

# Get loaded plugin names
plugins = registry.list_plugins()
print(f"Loaded plugins: {plugins}")
```

### Check Plugin Details

```python
# Check what nodes are available
from cuvis_ai_core.utils.node_registry import NodeRegistry

# After loading plugins
print(f"Available nodes: {list(registry.plugin_registry.keys())}")

# Get node info
node_class = NodeRegistry.get("AdaCLIPDetector", instance=registry)
print(f"Node class: {node_class}")
print(f"Input specs: {node_class.INPUT_SPECS}")
print(f"Output specs: {node_class.OUTPUT_SPECS}")
```

### Unload Plugins

```python
# Unload specific plugin
registry.unload_plugin("adaclip")

# Clear all plugins
registry.clear_plugins()
```

### Plugin Cache Management

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# Set custom cache directory
NodeRegistry.set_cache_dir("/custom/cache/location")

# Clear all plugin caches
NodeRegistry.clear_plugin_cache()

# Clear specific plugin cache
NodeRegistry.clear_plugin_cache("adaclip")
```

**Manual Cache Management:**

```bash
# View cache contents
ls ~/.cuvis_plugins/

# Output:
# adaclip@v0.1.0/
# adaclip@v0.1.1/
# custom-plugin@v1.0.0/

# Remove specific plugin version
rm -rf ~/.cuvis_plugins/adaclip@v0.1.0/

# Clear entire cache
rm -rf ~/.cuvis_plugins/
```

## Version Management

### Pinning Plugin Versions

For reproducibility, always pin plugin versions in production:

```yaml
# ✅ Good: Specific version pinned
plugins:
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"  # Exact version
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

# ⚠️ Avoid: No version control
# (Don't use branches or latest)
```

### Using Multiple Versions

Multiple versions can coexist in cache:

```python
registry = NodeRegistry()

# Load v0.1.0
registry.load_plugin(
    name="adaclip_v0",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.0",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# Load v0.1.1 (different instance)
registry2 = NodeRegistry()
registry2.load_plugin(
    name="adaclip_v1",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.1",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# Both versions available in cache
# ~/.cuvis_plugins/adaclip@v0.1.0/
# ~/.cuvis_plugins/adaclip@v0.1.1/
```

### Updating Plugins

To update a plugin to a new version:

1. **Update manifest:**
   ```yaml
   plugins:
     adaclip:
       tag: "v0.2.0"  # Changed from v0.1.1
       # ... rest of config
   ```

2. **Clear old cache (optional):**
   ```python
   NodeRegistry.clear_plugin_cache("adaclip")
   ```

3. **Reload:**
   ```python
   registry.load_plugins("plugins.yaml")
   ```

## Configuration with Plugins

### Using Plugin Configurations

If a plugin provides Hydra configs, you can override them:

```bash
# Via CLI
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_pipeline.yaml \
    --plugins-path plugins.yaml \
    --override adaclip.threshold=0.7 \
    --override adaclip.prompt="defective area"
```

**In Python:**

```python
from hydra import compose, initialize
from omegaconf import OmegaConf

# Initialize Hydra
with initialize(config_path="../configs"):
    cfg = compose(config_name="config.yaml")

    # Override plugin configs
    cfg.adaclip.threshold = 0.7
    cfg.adaclip.prompt = "defective area"

    # Use in pipeline
    # ...
```

### Plugin Config Files

If plugin provides config files in `configs/`, they're available in Hydra:

```yaml
# Plugin provides: cuvis_ai_plugin/configs/default.yaml
custom_detector:
  threshold: 0.95
  method: "advanced"
  window_size: 5
```

**Access in your config:**

```yaml
# your_config.yaml
defaults:
  - custom_detector: default  # Load plugin's default config

pipeline:
  nodes:
    - name: detector
      class_name: CustomDetector
      params: ${custom_detector}  # Use plugin config
```

## Examples

### Example 1: AdaCLIP Plugin

Complete workflow using AdaCLIP plugin:

```python
import numpy as np
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import Pipeline

# 1. Load plugin
registry = NodeRegistry()
registry.load_plugin(
    name="adaclip",
    config={
        "repo": "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git",
        "tag": "v0.1.1",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# 2. Create pipeline with AdaCLIP
pipeline_dict = {
    "nodes": [
        {
            "class_name": "MinMaxNormalizer",
            "name": "normalizer",
            "params": {}
        },
        {
            "class_name": "AdaCLIPDetector",
            "name": "adaclip",
            "params": {
                "prompt": "plastic wrapper",
                "threshold": 0.5
            }
        }
    ],
    "edges": [
        {
            "source": "normalizer.output",
            "target": "adaclip.data"
        }
    ]
}

pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

# 3. Run pipeline
hyperspectral_data = np.random.randn(100, 100, 200).astype(np.float32)
outputs = pipeline(data=hyperspectral_data)

# 4. Extract results
anomaly_map = outputs["adaclip"]["anomaly_map"]
print(f"Detected anomalies: {anomaly_map.sum()} pixels")
```

### Example 2: Mixed Built-in and Plugin Nodes

Combine built-in and plugin nodes:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import Pipeline

# Load plugins
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")  # Loads multiple plugins

# Create pipeline mixing built-in and plugin nodes
pipeline_dict = {
    "nodes": [
        # Built-in nodes
        {"class_name": "DataLoaderNode", "name": "loader", "params": {"path": "data/"}},
        {"class_name": "MinMaxNormalizer", "name": "normalizer", "params": {}},

        # Plugin node 1
        {"class_name": "CustomDetector", "name": "custom", "params": {"threshold": 0.9}},

        # Plugin node 2
        {"class_name": "AdaCLIPDetector", "name": "adaclip", "params": {"prompt": "defect"}},

        # Built-in node
        {"class_name": "ThresholdSelector", "name": "selector", "params": {"threshold": 0.8}}
    ],
    "edges": [
        {"source": "loader.data", "target": "normalizer.data"},
        {"source": "normalizer.output", "target": "custom.data"},
        {"source": "normalizer.output", "target": "adaclip.data"},
        {"source": "adaclip.anomaly_map", "target": "selector.data"}
    ]
}

pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)
outputs = pipeline()
```

### Example 3: Local Development Workflow

Develop and test plugin locally:

```bash
# 1. Create plugin structure
mkdir my-plugin && cd my-plugin
# ... create plugin files (see Development Guide)

# 2. Install in editable mode
pip install -e .

# 3. Create test manifest
cat > test_plugins.yaml << EOF
plugins:
  my_dev_plugin:
    path: "."
    provides:
      - my_plugin.nodes.CustomNode
EOF

# 4. Test plugin
uv run python << 'PYTHON'
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("test_plugins.yaml")

CustomNode = registry.get("CustomNode", instance=registry)
print(f"✓ Plugin loaded: {CustomNode}")
PYTHON
```

## Troubleshooting

### Plugin Not Found

**Symptom:**
```
Error: Failed to load plugin 'my-plugin'
```

**Solutions:**

1. **Check Git URL and accessibility:**
   ```bash
   git ls-remote https://github.com/your-org/plugin.git
   ```

2. **Verify tag exists:**
   ```bash
   git ls-remote --tags https://github.com/your-org/plugin.git | grep v0.1.0
   ```

3. **Check local path:**
   ```bash
   ls ../my-plugin/pyproject.toml
   ```

4. **Verify network connectivity:**
   ```bash
   curl -I https://github.com/your-org/plugin.git
   ```

### Node Not Found After Loading

**Symptom:**
```python
NodeNotFoundError: Node 'CustomNode' not found
```

**Solutions:**

1. **Check plugin loaded successfully:**
   ```python
   plugins = registry.list_plugins()
   print(f"Loaded: {plugins}")
   ```

2. **Verify node in plugin registry:**
   ```python
   print(f"Available nodes: {list(registry.plugin_registry.keys())}")
   ```

3. **Try full class path:**
   ```python
   node = NodeRegistry.get(
       "my_plugin.nodes.custom_node.CustomNode",
       instance=registry
   )
   ```

4. **Check manifest `provides` list:**
   ```yaml
   plugins:
     my_plugin:
       provides:
         - my_plugin.nodes.custom_node.CustomNode  # Must match exactly
   ```

### Import Errors

**Symptom:**
```
ImportError: cannot import name 'CustomNode' from 'my_plugin.nodes'
```

**Solutions:**

1. **Check `__init__.py` files exist:**
   ```bash
   find my-plugin -name "__init__.py"
   ```

2. **Verify dependencies installed:**
   ```bash
   pip list | grep -i torch
   pip list | grep -i numpy
   ```

3. **Test import manually:**
   ```python
   python -c "from my_plugin.nodes import CustomNode"
   ```

4. **Check `pyproject.toml` dependencies:**
   ```bash
   cat my-plugin/pyproject.toml | grep dependencies -A 5
   ```

### Cache Issues

**Symptom:**
```
Error: Tag mismatch in cache for plugin 'adaclip'
```

**Solutions:**

1. **Clear plugin cache:**
   ```python
   NodeRegistry.clear_plugin_cache("adaclip")
   ```

2. **Manually remove cache:**
   ```bash
   rm -rf ~/.cuvis_plugins/adaclip@*
   ```

3. **Reload plugin:**
   ```python
   registry.load_plugin(...)
   ```

### Dependency Conflicts

**Symptom:**
```
Error: cuvis-ai-adaclip requires torch>=2.9, but 2.0 is installed
```

**Solutions:**

1. **Update conflicting package:**
   ```bash
   pip install --upgrade torch
   ```

2. **Use compatible plugin version:**
   ```yaml
   plugins:
     adaclip:
       tag: "v0.1.0"  # Older version with compatible deps
   ```

3. **Create isolated environment:**
   ```bash
   python -m venv plugin_env
   source plugin_env/bin/activate
   pip install cuvis-ai torch>=2.9
   ```

4. **Check version constraints:**
   ```bash
   pip show cuvis-ai-adaclip | grep Requires
   ```

### Performance Issues

**Symptom:** Plugin node is very slow.

**Solutions:**

1. **Check if dependencies are GPU-enabled:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   ```

2. **Profile plugin execution:**
   ```python
   import time
   start = time.time()
   outputs = node(data=data)
   print(f"Execution time: {time.time() - start:.2f}s")
   ```

3. **Use smaller data for testing:**
   ```python
   # Instead of full data
   test_data = data[:10, :10, :]  # Small subset
   ```

4. **Check plugin documentation for performance tips**

## Best Practices

### 1. Version Pinning

```yaml
# ✅ Good: Pin versions for reproducibility
plugins:
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"  # Exact version
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

# ⚠️ Avoid: Using branches or latest
```

### 2. Manifest Files

```python
# ✅ Good: Use manifest for multiple plugins
registry.load_plugins("plugins.yaml")

# ⚠️ Avoid: Multiple individual loads (harder to maintain)
registry.load_plugin("adaclip", config={...})
registry.load_plugin("custom", config={...})
registry.load_plugin("another", config={...})
```

### 3. Virtual Environments

```bash
# ✅ Good: Separate environments per project
python -m venv project1_env
source project1_env/bin/activate
pip install cuvis-ai
# Load project1 plugins

python -m venv project2_env
source project2_env/bin/activate
pip install cuvis-ai
# Load project2 plugins
```

### 4. Testing

```python
# ✅ Good: Test plugin in isolation first
def test_plugin_isolation():
    registry = NodeRegistry()
    registry.load_plugin("my_plugin", config={...})

    Node = registry.get("CustomNode", instance=registry)
    node = Node()

    # Test with sample data
    test_data = np.random.randn(10, 10, 50).astype(np.float32)
    outputs = node(data=test_data)

    assert outputs is not None
```

### 5. Error Handling

```python
# ✅ Good: Handle plugin loading errors
registry = NodeRegistry()

try:
    registry.load_plugins("plugins.yaml")
except Exception as e:
    print(f"Failed to load plugins: {e}")
    print("Falling back to built-in nodes only")
    # Continue with built-in nodes
```

## See Also

- **[Plugin System Overview](overview.md)** - Architecture and concepts
- **[Plugin Development Guide](development.md)** - Create your own plugins
- **[Node Catalog](../node-catalog/index.md)** - Built-in nodes reference
- **[Pipeline Lifecycle](../concepts/pipeline-lifecycle.md)** - Pipeline integration
- **[Configuration Guide](../config/config-groups.md)** - Hydra configuration
