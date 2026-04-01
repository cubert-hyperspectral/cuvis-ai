!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Configuration & Examples

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
      hparams: ${custom_detector}  # Use plugin config
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
            "hparams": {}
        },
        {
            "class_name": "AdaCLIPDetector",
            "name": "adaclip",
            "hparams": {
                "prompt": "plastic wrapper",
                "threshold": 0.5
            }
        }
    ],
    "connections": [
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
        {"class_name": "DataLoaderNode", "name": "loader", "hparams": {"path": "data/"}},
        {"class_name": "MinMaxNormalizer", "name": "normalizer", "hparams": {}},

        # Plugin node 1
        {"class_name": "CustomDetector", "name": "custom", "hparams": {"threshold": 0.9}},

        # Plugin node 2
        {"class_name": "AdaCLIPDetector", "name": "adaclip", "hparams": {"prompt": "defect"}},

        # Built-in node
        {"class_name": "ThresholdSelector", "name": "selector", "hparams": {"threshold": 0.8}}
    ],
    "connections": [
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
print(f"Plugin loaded: {CustomNode}")
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

| Practice | Recommendation | Anti-pattern |
|---|---|---|
| **Version pinning** | Always pin exact tags in manifests: `tag: "v0.1.1"` | Using branches or omitting tags |
| **Manifest files** | Use a single `plugins.yaml` with `registry.load_plugins()` | Multiple individual `load_plugin()` calls |
| **Virtual environments** | Create a separate venv per project to isolate plugin deps | Installing everything into the system Python |
| **Testing** | Test each plugin in isolation before integrating into a pipeline | Debugging plugin issues inside a large pipeline |
| **Error handling** | Wrap `load_plugins()` in try/except and fall back to built-in nodes | Letting plugin load failures crash the application |

## See Also

- **[Loading Plugins](loading.md)** - Finding, installing, and loading plugins
- **[Using Plugin Nodes](using-nodes.md)** - Working with plugin nodes, managing and versioning plugins
- **[Plugin System Overview](architecture.md)** - Architecture and concepts
- **[Plugin Development Guide](dev-quickstart.md)** - Create your own plugins
- **[Node Catalog](../node-catalog/index.md)** - Built-in nodes reference
- **[Pipeline Lifecycle](../concepts/pipeline-lifecycle.md)** - Pipeline integration
- **[Configuration Guide](../config/config-groups.md)** - Hydra configuration
