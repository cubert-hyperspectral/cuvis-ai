# AdaCLIP Workflow Contribution Guide

This guide documents the complete workflow for contributing AdaCLIP examples to cuvis.ai, from training in the AdaCLIP-cuvis plugin to using the trained pipelines in cuvis.ai via gRPC and Python API.

## Overview

**Important:** Starting with Phase 5, AdaCLIP examples now use the **plugin system** instead of direct package installation. This aligns with the repository split architecture and eliminates the need for manual package installation.

The AdaCLIP workflow follows a **plugin → train → copy → restore → use** pattern:

1. **Load plugin**: Load AdaCLIP plugin from local repository using the NodeRegistry plugin system
2. **Train**: Run training scripts that use the plugin to generate pipeline artifacts
3. **Copy artifacts**: Copy the generated pipeline YAMLs and weights to `cuvis.ai/configs/pipeline/`
4. **Create trainrun configs**: Create Hydra-based trainrun configs in `cuvis.ai/configs/trainrun/` that compose pipeline + data + training configs
5. **Use in cuvis.ai**: Use `restore_trainrun` for training/validation/test or `restore_pipeline` for inference

This workflow ensures that:
- No manual package installation required (plugin system handles loading)
- Training happens using plugin-loaded nodes
- Trained artifacts are version-controlled in cuvis.ai for reproducibility
- Users can restore and use pipelines without needing the plugin repository
- Hydra configs provide flexible composition and overrides

## Plugin System Usage

### Loading AdaCLIP Plugin

All three AdaCLIP examples now use the plugin system. There are two approaches:

#### Approach 1: Programmatic Loading (Used in Examples)

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# IMPORTANT: Create a NodeRegistry instance first - load_plugin() is an instance method!
registry = NodeRegistry()

# Load AdaCLIP plugin from local development clone
registry.load_plugin(
    name="adaclip",
    config={
        "path": r"D:\code-repos\cuvis-ai-adaclip",
        "provides": ["cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector"]
    }
)

# Get the AdaCLIPDetector class from the registry (get() works as both class and instance method)
AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
```

**Key Point:** `load_plugin()` is an **instance method** (requires creating a `NodeRegistry` instance first), while `get()` works as both a class and instance method. This is by design from Phase 4's hybrid architecture:
- Built-in nodes: accessed via class method `NodeRegistry.get("MinMaxNormalizer")`
- Plugin nodes: require instance-based loading first, then can be accessed via class method `NodeRegistry.get("plugin.node.Class")`

#### Approach 2: Manifest-Based Loading

Create a `plugins.yaml` file (see `examples/adaclip/plugins.yaml`):

```yaml
plugins:
  adaclip:
    path: "D:/code-repos/cuvis-ai-adaclip"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector
```

Then load it in your script:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

# Load plugins from manifest
NodeRegistry.load_plugins("examples/adaclip/plugins.yaml")

# Get the AdaCLIPDetector class
AdaCLIPDetector = NodeRegistry.get("cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector")
```

### Prerequisites

**Before running AdaCLIP examples:**

1. Clone the cuvis-ai-adaclip repository:
   ```bash
   cd D:\code-repos
   git clone https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git
   ```

2. Ensure the path in the examples matches your local clone location. If your clone is elsewhere, update the `path` parameter in the plugin loading code.

3. No manual installation needed - the plugin system handles everything!

## Step-by-Step Workflow

### Step 1: Run Training Examples

Run the training scripts in `cuvis-ai/examples/adaclip/`:

```bash
# Example 1: DRCNN + AdaCLIP gradient training
cd cuvis-ai
uv run python examples/adaclip/drcnn_adaclip_gradient_training.py

# Example 2: Concrete selector + AdaCLIP gradient training
uv run python examples/adaclip/concrete_adaclip_gradient_training.py

# Example 3: PCA + AdaCLIP baseline (no gradient training)
uv run python examples/adaclip/pca_adaclip_baseline.py
```

**Note:** The plugin loading happens automatically at the start of each script. You'll see:
```
Loading AdaCLIP plugin from local repository...
✓ AdaCLIP plugin loaded successfully
```

**What this generates:**
- `outputs/adaclip_baseline/trained_models/adaclip_baseline.yaml` - Complete pipeline configuration
- `outputs/adaclip_baseline/trained_models/adaclip_baseline.pt` - Model weights
- `outputs/adaclip_baseline/trainrun.yaml` - Complete trainrun configuration (pipeline + data + training)

### Step 2: Copy Artifacts to cuvis.ai

Copy the generated files to the appropriate locations in `cuvis.ai`:

```bash
# Copy pipeline YAML and weights
cp AdaCLIP-cuvis/outputs/adaclip_baseline/trained_models/adaclip_baseline.yaml \
   cuvis.ai/configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml

cp AdaCLIP-cuvis/outputs/adaclip_baseline/trained_models/adaclip_baseline.pt \
   cuvis.ai/configs/pipeline/anomaly/adaclip/adaclip_baseline.pt
```

**Important:** The pipeline YAML should be the **complete** `PipelineConfig` (with nodes, connections, metadata) extracted from the trained pipeline, not just parameters.

### Step 3: Create Trainrun Config with Hydra Defaults

Create a trainrun config in `cuvis.ai/configs/trainrun/adaclip_baseline.yaml` that uses Hydra defaults to compose the pipeline, data, and training configs:

```yaml
# @package _global_

name: adaclip_baseline

defaults:
  - /pipeline/anomaly/adaclip@pipeline: adaclip_baseline  # References configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml
  - /data@data: lentils                    # References configs/data/lentils.yaml
  - /training@training: default             # References configs/training/default.yaml
  - _self_

data:
  # Override data splits for this example
  train_ids: [0, 2]
  val_ids: [2, 4]
  test_ids: [1, 3, 5]
  batch_size: 1

training:
  seed: 42
  trainer:
    default_root_dir: ./outputs/adaclip_baseline

output_dir: outputs\adaclip_baseline
```

**Key points:**
- The `defaults` section uses Hydra's composition to merge pipeline, data, and training configs
- The `@pipeline`, `@data`, `@training` syntax creates separate config groups
- You can override any values from the defaults in the main config
- The pipeline config should reference the **complete** pipeline YAML (not parameters)

### Step 4: Use in cuvis.ai

Once the artifacts are in place, you can use them in cuvis.ai via unified CLI commands or Python API:

#### Option A: Restore Trainrun (Training/Validation/Test) - Recommended

Use the `restore-trainrun` CLI command for statistical-only pipelines (like AdaCLIP):

```bash
# Display trainrun info
uv run restore-trainrun --trainrun-path configs/trainrun/adaclip_baseline.yaml

# Run statistical training (evaluation pass)
uv run restore-trainrun --trainrun-path configs/trainrun/adaclip_baseline.yaml --mode train

# Run validation
uv run restore-trainrun --trainrun-path configs/trainrun/adaclip_baseline.yaml --mode validate

# Run test inference
uv run restore-trainrun --trainrun-path configs/trainrun/adaclip_baseline.yaml --mode test
```

**When to use:** When you need the full experiment workflow (training, validation, test) with metrics and TensorBoard logging.

#### Option B: Restore Pipeline (Inference Only) - Recommended

Use the `restore-pipeline` CLI command for inference without requiring the full trainrun configuration:

```bash
# Display pipeline info
uv run restore-pipeline --pipeline-path configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml

# Run inference on CU3S file
uv run restore-pipeline \
  --pipeline-path configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml \
  --cu3s-file-path data/Lentils/Lentils_000.cu3s

# Use custom CU3S file
uv run restore-pipeline \
  --pipeline-path configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml \
  --cu3s-file-path /path/to/your/custom_data.cu3s
```

**When to use:** When you only need inference on arbitrary CU3S files, without metrics or full experiment setup.

#### Option C: Python API

Use the Python API directly for programmatic control:

```python
from cuvis_ai.utils import restore_pipeline, restore_trainrun

# Restore and run trainrun
restore_trainrun(
    trainrun_path="configs/trainrun/adaclip_baseline.yaml",
    mode="train"
)

# Restore and run pipeline inference
restore_pipeline(
    pipeline_path="configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml",
    cu3s_file_path="data/Lentils/Lentils_000.cu3s"
)
```

See `restore_pipeline.md` for detailed Python API documentation.

#### Option D: gRPC API

Use gRPC for remote inference:

```bash
# Start gRPC server (in separate terminal)
uv run python -m cuvis_ai.grpc.production_server

# Run gRPC client (in another terminal)
python examples/grpc/adaclip_client.py
```

## Working with Custom Data

### In AdaCLIP-cuvis (Training)

Pass custom data paths via CLI arguments:

```bash
uv run python cuvis_ai_adaclip/examples_cuvis/statistical_baseline.py \
  --cu3s-file-path /path/to/your/custom_data.cu3s \
  --annotation-json-path /path/to/your/annotations.json \
  --train-ids 0 1 2 \
  --val-ids 3 4 \
  --test-ids 5 6 7
```

### In cuvis.ai (Inference)

#### Using CLI Commands (Recommended):

```bash
# Using restore-pipeline
uv run restore-pipeline \
  --pipeline-path configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml \
  --cu3s-file-path /path/to/your/custom_data.cu3s

# Using restore-trainrun
uv run restore-trainrun \
  --trainrun-path configs/trainrun/adaclip_baseline.yaml \
  --mode test
```

Or modify the trainrun config to point to your custom data:

```yaml
data:
  cu3s_file_path: /path/to/your/custom_data.cu3s
  annotation_json_path: /path/to/your/annotations.json
  train_ids: [0, 1, 2]
  val_ids: [3, 4]
  test_ids: [5, 6, 7]
```

#### Using Python API:

```python
from cuvis_ai.data.lentils_anomaly import SingleCu3sDataModule

datamodule = SingleCu3sDataModule(
    cu3s_file_path="/path/to/your/custom_data.cu3s",
    annotation_json_path="/path/to/your/annotations.json",
    train_ids=[0, 1, 2],
    val_ids=[3, 4],
    test_ids=[5, 6, 7],
    batch_size=1,
)
```

## Key Concepts

### Pipeline Config vs Parameters

**Important distinction:**

- **Parameters** (for programmatic building): When building a pipeline in code, you pass parameters like:
  ```python
  pipeline_config = {
      "bandpass": {"min_wavelength_nm": 700.0},
      "encoder": {"num_channels": 27},
      # ... other parameters
  }
  ```

- **Full PipelineConfig** (for restoration): When restoring a pipeline, you need the complete structure:
  ```yaml
  name: adaclip_baseline
  nodes:
    - name: data_node
      type: LentilsAnomalyDataNode
      # ... complete node config
    - name: band_selector
      type: BaselineFalseRGBSelector
      # ... complete node config
  connections:
    - from: data_node
      to: band_selector
  # ... metadata, etc.
  ```

The pipeline YAMLs in `configs/pipeline/` should be **full PipelineConfig** structures extracted from trained pipelines (via `pipeline.serialize()`), not just parameters.

### Hydra Config Composition

The trainrun configs use Hydra's `defaults` to compose multiple configs:

```yaml
defaults:
  - /pipeline/anomaly/adaclip@pipeline: adaclip_baseline  # Loads configs/pipeline/anomaly/adaclip/adaclip_baseline.yaml
  - /data@data: lentils                    # Loads configs/data/lentils.yaml
  - /training@training: default            # Loads configs/training/default.yaml
  - _self_                                  # Merges with current config
```

This allows:
- **Separation of concerns**: Pipeline, data, and training configs are separate
- **Reusability**: Same pipeline config can be used with different data/training configs
- **Override flexibility**: Override any value from defaults in the main config
- **CLI overrides**: Hydra allows CLI overrides like `data.batch_size=2`

### Statistical vs Gradient Training

AdaCLIP uses **statistical training only** (no gradient-based training):
- Statistical nodes (like `SupervisedCIRBandSelector`) require an initial `fit()` call
- No trainable parameters that need gradient optimization
- Use `StatisticalTrainer` instead of `GradientTrainer`
- Use `restore_trainrun_statistical.py` instead of `restore_trainrun.py`

## Usage Examples

For practical examples, see:

- **Root-level `restore_pipeline.md`**: Comprehensive guide with CLI commands and Python API examples for both pipeline restoration and trainrun management
- **CLI Commands**: 
  - `uv run restore-pipeline` - For pipeline inference
  - `uv run restore-trainrun` - For trainrun management (training, validation, testing)

## Best Practices

1. **Always extract complete PipelineConfig**: When copying pipeline YAMLs from AdaCLIP-cuvis, ensure they contain the full structure (nodes, connections, metadata), not just parameters.

2. **Use Hydra defaults**: Compose trainrun configs using Hydra defaults rather than embedding everything inline.

3. **Version control artifacts**: Commit both pipeline YAMLs and weights (if not too big) to `configs/pipeline/` for reproducibility.

4. **Document data paths**: Use absolute paths or clearly document relative path assumptions in trainrun configs.

5. **Test both workflows**: Test both `restore_trainrun` and `restore_pipeline` workflows to ensure they work correctly.

6. **Handle device placement**: Ensure pipelines are moved to the correct device (CPU/GPU) when restoring.

## Troubleshooting

### Issue: "Pipeline configuration is missing in trainrun config"

**Solution**: Ensure the trainrun config has a `defaults` entry that references the pipeline config, or explicitly includes the pipeline config.

### Issue: "Failed to load AdaCLIP plugin"

**Solution**: Ensure the cuvis-ai-adaclip repository is cloned at the correct path. Update the `path` parameter in the plugin loading code if your clone is in a different location.

### Issue: "SupervisedCIRBandSelector not fitted"

**Solution**: For statistical-only pipelines, ensure `stat_trainer.fit()` is called before validation/test. The `restore_trainrun_statistical.py` script handles this automatically.

### Issue: "Extra inputs are not permitted" when loading trainrun config

**Solution**: Ensure the trainrun config uses Hydra defaults correctly. Custom fields should be removed or properly structured according to `TrainRunConfig` schema.

### Issue: "Expected all tensors to be on the same device"

**Solution**: Ensure the pipeline is moved to the correct device before inference. The restore scripts handle this automatically, but if using the Python API directly, call `pipeline.to(device)`.

## Related Documentation

- [cuvis.ai Training Documentation](../../docs/api/training.md)
- [Pipeline API Reference](../../docs/api/pipeline.md)
- [gRPC Deployment Guide](../../docs/deployment/grpc_deployment.md)
- [AdaCLIP-cuvis Repository](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)

## Summary

The AdaCLIP workflow follows a clear separation of concerns:

1. **Plugin Loading**: AdaCLIP is loaded dynamically via the plugin system (no manual installation)
2. **Training**: Examples use plugin-loaded nodes to train pipelines
3. **Artifacts**: Pipeline YAMLs + weights are copied to `cuvis.ai/configs/pipeline/`
4. **Trainrun configs**: In `cuvis.ai/configs/trainrun/` use Hydra to compose pipeline + data + training
5. **Usage**: In cuvis.ai via `restore_trainrun` (full workflow) or `restore_pipeline` (inference only)

This workflow ensures reproducibility, maintainability, and clear separation between plugin-specific code and the core cuvis.ai framework, while demonstrating proper plugin system usage established in Phase 2.
