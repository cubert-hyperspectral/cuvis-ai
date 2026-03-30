!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Using Plugin Nodes

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
            "hparams": {}
        },
        {
            "class_name": "AdaCLIPDetector",  # Plugin node
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
    hparams: {}

  - name: adaclip_detector
    class_name: AdaCLIPDetector  # From plugin
    hparams:
      prompt: "plastic wrapper"
      threshold: 0.5

  - name: threshold_selector
    class_name: ThresholdSelector
    hparams:
      threshold: 0.9

connections:
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
# Good: Specific version pinned
plugins:
  adaclip:
    repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
    tag: "v0.1.1"  # Exact version
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

# Avoid: No version control
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

## See Also

- **[Loading Plugins](loading.md)** - Finding, installing, and loading plugins
- **[Plugin Configuration & Examples](plugin-config.md)** - Configuration, examples, troubleshooting, and best practices
