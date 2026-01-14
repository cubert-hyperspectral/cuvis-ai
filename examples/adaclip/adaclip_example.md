# AdaCLIP Workflow Contribution Guide

This guide documents the complete workflow for contributing AdaCLIP examples to cuvis.ai, from training in the AdaCLIP-cuvis plugin to using the trained pipelines in cuvis.ai via gRPC and Python API.

## Overview

The AdaCLIP workflow follows a **train → copy → restore → use** pattern:

1. **Train in AdaCLIP-cuvis**: Run training scripts in the `AdaCLIP-cuvis` repository to generate pipeline artifacts
2. **Copy artifacts**: Copy the generated pipeline YAMLs and weights to `cuvis.ai/configs/pipeline/`
3. **Create trainrun configs**: Create Hydra-based trainrun configs in `cuvis.ai/configs/trainrun/` that compose pipeline + data + training configs
4. **Use in cuvis.ai**: Use `restore_trainrun` for training/validation/test or `restore_pipeline` for inference

This workflow ensures that:
- Training happens in the plugin repository (AdaCLIP-cuvis) where the plugin-specific code lives
- Trained artifacts are version-controlled in cuvis.ai for reproducibility
- Users can restore and use pipelines without needing the plugin repository
- Hydra configs provide flexible composition and overrides

## Step-by-Step Workflow

### Step 1: Train in AdaCLIP-cuvis

Run the training scripts in `AdaCLIP-cuvis/cuvis_ai_adaclip/examples_cuvis/`:

```bash
# Example: Run baseline training
cd AdaCLIP-cuvis
uv run python cuvis_ai_adaclip/examples_cuvis/statistical_baseline.py \
  --cu3s-file-path /path/to/data/Lentils/Lentils_000.cu3s \
  --annotation-json-path /path/to/data/Lentils/annotations.json \
  --train-ids 0 2 \
  --val-ids 2 4 \
  --test-ids 1 3 5 \
  --batch-size 1 \
  --output-dir ./outputs/adaclip_baseline
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
   cuvis.ai/configs/pipeline/adaclip_baseline.yaml

cp AdaCLIP-cuvis/outputs/adaclip_baseline/trained_models/adaclip_baseline.pt \
   cuvis.ai/configs/pipeline/adaclip_baseline.pt
```

**Important:** The pipeline YAML should be the **complete** `PipelineConfig` (with nodes, connections, metadata) extracted from the trained pipeline, not just parameters.

### Step 3: Create Trainrun Config with Hydra Defaults

Create a trainrun config in `cuvis.ai/configs/trainrun/adaclip_baseline.yaml` that uses Hydra defaults to compose the pipeline, data, and training configs:

```yaml
# @package _global_

name: adaclip_baseline

defaults:
  - /pipeline@pipeline: adaclip_baseline  # References configs/pipeline/adaclip_baseline.yaml
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
uv run restore-pipeline --pipeline-path configs/pipeline/adaclip_baseline.yaml

# Run inference on CU3S file
uv run restore-pipeline \
  --pipeline-path configs/pipeline/adaclip_baseline.yaml \
  --cu3s-file-path data/Lentils/Lentils_000.cu3s

# Use custom CU3S file
uv run restore-pipeline \
  --pipeline-path configs/pipeline/adaclip_baseline.yaml \
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
    pipeline_path="configs/pipeline/adaclip_baseline.yaml",
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
  --pipeline-path configs/pipeline/adaclip_baseline.yaml \
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
  - /pipeline@pipeline: adaclip_baseline  # Loads configs/pipeline/adaclip_baseline.yaml
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

1. **Training** happens in `AdaCLIP-cuvis` (plugin repository)
2. **Artifacts** (pipeline YAMLs + weights) are copied to `cuvis.ai/configs/pipeline/`
3. **Trainrun configs** in `cuvis.ai/configs/trainrun/` use Hydra to compose pipeline + data + training
4. **Usage** in cuvis.ai via `restore_trainrun` (full workflow) or `restore_pipeline` (inference only)

This workflow ensures reproducibility, maintainability, and clear separation between plugin-specific code and the core cuvis.ai framework.
