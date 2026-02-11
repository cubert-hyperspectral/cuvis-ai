!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# How-To: Restore Pipelines from TrainRuns

## Overview
Learn how to restore trained pipelines from TrainRun experiments for inference, continued training, or validation. TrainRuns capture complete experiment state including pipeline configuration, data setup, training parameters, and model weights.

## Prerequisites
- cuvis-ai installed
- Completed training run with saved TrainRun configuration
- Basic understanding of [Pipeline Lifecycle](../concepts/pipeline-lifecycle.md)
- Familiarity with [TrainRun Schema](../config/trainrun-schema.md)

## What is a TrainRun?

A TrainRun is a complete experiment specification that includes:

```python
TrainRunConfig(
    name="my_experiment",           # Experiment identifier
    pipeline={...},                  # Full pipeline structure
    data={...},                      # Data loading configuration
    training={...},                  # Training settings (optimizer, callbacks)
    output_dir="outputs/my_exp/",   # Results directory
    loss_nodes=["bce_loss"],        # Gradient training loss nodes
    metric_nodes=["metrics"],       # Evaluation metrics
    freeze_nodes=[],                # Initially frozen nodes
    unfreeze_nodes=["selector"]     # Nodes to train
)
```

## TrainRun Output Structure

After training, your experiment directory contains:

```
outputs/my_experiment/
├── trained_models/
│   ├── My_Pipeline.yaml              # Pipeline configuration
│   ├── My_Pipeline.pt                # Model weights (trainable nodes)
│   └── my_experiment_trainrun.yaml   # Complete TrainRun config
├── checkpoints/
│   ├── epoch=00.ckpt                 # Periodic checkpoints
│   ├── epoch=05.ckpt
│   ├── last.ckpt                     # Latest epoch
│   └── best.ckpt                     # Best metric checkpoint
└── tensorboard/                      # Training logs
    └── events.out.tfevents...
```

## Restoration Modes

### Mode 1: Info (Display Only)

View experiment details without running:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode info
```

**Output:**
```
TrainRun: my_experiment
Pipeline: My_Pipeline (5 nodes, 7 connections)
Loss nodes: bce_loss
Metric nodes: metrics
Unfreeze nodes: selector
Data: LentilsAnomalyDataNode (train/val/test: [0]/[3,4]/[1,5])
Training: AdamW (lr=0.001, max_epochs=50)
Output directory: outputs/my_experiment/
```

### Mode 2: Train (Run Training)

Re-run or continue training:

```bash
# Train from scratch
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train

# Continue from checkpoint
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train \
  --checkpoint-path outputs/my_experiment/checkpoints/epoch=05.ckpt
```

### Mode 3: Validate (Validation Set)

Run evaluation on validation set:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode validate
```

**Output:** Validation metrics (IoU, AUC, F1, etc.)

### Mode 4: Test (Test Set)

Run evaluation on test set:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode test
```

## Quick Inference (Pipeline Only)

For simple inference without full TrainRun context:

```bash
# Display pipeline structure
uv run restore-pipeline \
  --pipeline-path outputs/my_experiment/trained_models/My_Pipeline.yaml

# Run inference on .cu3s file
uv run restore-pipeline \
  --pipeline-path outputs/my_experiment/trained_models/My_Pipeline.yaml \
  --cu3s-file-path data/test_sample.cu3s \
  --processing-mode Reflectance
```

## Configuration Overrides

Modify experiment settings during restoration without editing YAML files:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train \
  --override output_dir=outputs/my_experiment_v2 \
  --override data.batch_size=32 \
  --override training.optimizer.lr=0.0001 \
  --override training.trainer.max_epochs=100 \
  --override nodes.2.params.threshold=0.8
```

**Common overrides:**
- `output_dir` - Change save location
- `data.batch_size` - Adjust batch size
- `data.train_ids`, `data.val_ids`, `data.test_ids` - Change data splits
- `training.optimizer.lr` - Modify learning rate
- `training.trainer.max_epochs` - Adjust training duration
- `nodes.N.params.*` - Override node parameters (N = node index)

## Python API Usage

### Basic Restoration

```python
from cuvis_ai.utils import restore_trainrun

# Display info
restore_trainrun(
    trainrun_path="outputs/my_experiment/trained_models/my_experiment_trainrun.yaml",
    mode="info"
)

# Run training
restore_trainrun(
    trainrun_path="outputs/my_experiment/trained_models/my_experiment_trainrun.yaml",
    mode="train"
)

# Validate
restore_trainrun(
    trainrun_path="outputs/my_experiment/trained_models/my_experiment_trainrun.yaml",
    mode="validate"
)
```

### Restoration with Overrides

```python
from cuvis_ai.utils import restore_trainrun
from omegaconf import OmegaConf

# Load and modify config
config_overrides = OmegaConf.create({
    "output_dir": "outputs/my_experiment_v2",
    "training": {
        "optimizer": {"lr": 0.0001},
        "trainer": {"max_epochs": 100}
    },
    "data": {
        "batch_size": 32
    }
})

restore_trainrun(
    trainrun_path="outputs/my_experiment/trained_models/my_experiment_trainrun.yaml",
    mode="train",
    config_overrides=config_overrides
)
```

### Manual Pipeline Loading

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import TrainRunConfig

# Load TrainRun config
trainrun_config = TrainRunConfig.load_from_file(
    "outputs/my_experiment/trained_models/my_experiment_trainrun.yaml"
)

# Extract and build pipeline
pipeline_config = trainrun_config.pipeline
pipeline = CuvisPipeline.from_config(pipeline_config)

# Load weights
pipeline.load_weights(
    "outputs/my_experiment/trained_models/My_Pipeline.pt",
    strict=True
)

# Run inference
pipeline.validate()
result = pipeline.execute()
```

## Statistical vs Gradient Training

### Statistical-Only Pipeline

Pipelines without gradient training (no loss nodes):

```bash
# Statistical training automatically detected
uv run restore-trainrun \
  --trainrun-path outputs/rx_statistical/trained_models/rx_statistical_trainrun.yaml \
  --mode train
```

**Behavior:**
1. Runs statistical initialization (mean, covariance, running stats)
2. No gradient descent applied
3. Fast execution (no backpropagation)

### Two-Phase Training

Pipelines with both statistical and gradient components:

```bash
# Automatically runs both phases
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train
```

**Execution flow:**
1. **Phase 1: Statistical initialization**
   - Calibrate statistical parameters
   - Fit normalizers on training data
   - Initialize detector components
2. **Phase 2: Gradient training**
   - Unfreeze specified nodes
   - Apply gradient descent with optimizer
   - Monitor metrics and early stopping
   - Save best checkpoint

## Resume Training from Checkpoint

### Continue Interrupted Training

```bash
# Resume from last checkpoint
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train \
  --checkpoint-path outputs/my_experiment/checkpoints/last.ckpt

# Resume from best checkpoint
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train \
  --checkpoint-path outputs/my_experiment/checkpoints/best.ckpt
```

### Fine-tune with Different Settings

```bash
# Continue training with reduced learning rate
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train \
  --checkpoint-path outputs/my_experiment/checkpoints/epoch=05.ckpt \
  --override training.optimizer.lr=0.00001 \
  --override training.trainer.max_epochs=100 \
  --override output_dir=outputs/my_experiment_finetuned
```

## gRPC Remote Restoration

For remote server-based restoration:

```python
import grpc
from cuvis_ai_core.grpc import cuvis_ai_pb2, cuvis_ai_pb2_grpc

# Connect to server
channel = grpc.insecure_channel("localhost:50051")
stub = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

# Restore TrainRun
restore_response = stub.RestoreTrainRun(
    cuvis_ai_pb2.RestoreTrainRunRequest(
        trainrun_path="outputs/my_experiment/trained_models/my_experiment_trainrun.yaml",
        weights_path="outputs/my_experiment/trained_models/My_Pipeline.pt",
        strict=True
    )
)

session_id = restore_response.session_id

# Run training
for progress in stub.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_GRADIENT
    )
):
    print(f"Epoch {progress.current_epoch}/{progress.total_epochs} - "
          f"Loss: {progress.train_loss:.4f}")

# Run validation
val_response = stub.Validate(
    cuvis_ai_pb2.ValidateRequest(session_id=session_id)
)
print(f"Validation metrics: {val_response.metrics}")
```

## Weight Loading Behavior

### With Weights File (.pt)

```bash
# Weights automatically loaded if present
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode validate
```

**Behavior:**
- Trainable nodes restored from `.pt` file
- Statistical nodes skip initialization (already trained)
- Ready for immediate inference/validation

### Without Weights File

```bash
# Statistical initialization required
uv run restore-trainrun \
  --trainrun-path outputs/my_experiment/trained_models/my_experiment_trainrun.yaml \
  --mode train
```

**Behavior:**
- Statistical nodes must run initialization first
- Trainable nodes start from random initialization
- Full training required before inference

### Explicit Weight Loading

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

# Load pipeline
pipeline = CuvisPipeline.load_pipeline(
    "outputs/my_experiment/trained_models/My_Pipeline.yaml"
)

# Load weights explicitly
pipeline.load_weights(
    "outputs/my_experiment/trained_models/My_Pipeline.pt",
    strict=True  # Require exact parameter match
)

# Or load from checkpoint
pipeline.load_from_checkpoint(
    "outputs/my_experiment/checkpoints/epoch=09.ckpt",
    strict=False  # Allow partial loading
)
```

## Checkpoint Configuration

Configure checkpoint saving during training:

```python
from cuvis_ai_core.training.config import TrainingConfig, ModelCheckpointConfig

training_cfg = TrainingConfig(
    trainer={
        "max_epochs": 50,
        "callbacks": {
            "checkpoint": ModelCheckpointConfig(
                dirpath="outputs/my_experiment/checkpoints",
                monitor="metrics_anomaly/iou",  # Metric to track
                mode="max",                      # Maximize IoU
                save_top_k=3,                    # Keep best 3 checkpoints
                save_last=True,                  # Always save last epoch
                filename="{epoch:02d}",          # Naming pattern
                verbose=True
            )
        }
    }
)
```

**Checkpoint strategies:**
- `save_top_k=1` - Save only best checkpoint (minimal storage)
- `save_top_k=3` - Save top 3 checkpoints (moderate storage)
- `save_top_k=-1` - Save all checkpoints (maximum storage)
- `save_last=True` - Always save most recent epoch
- `monitor` - Metric to optimize (`metrics_anomaly/iou`, `metrics_anomaly/auc`, etc.)
- `mode` - `"max"` (maximize metric) or `"min"` (minimize loss)

## Version Compatibility

### Automatic Training Type Detection

The restoration system automatically detects training type:

```python
# Detects gradient training (has loss_nodes)
trainrun_config = TrainRunConfig.load_from_file(
    "outputs/channel_selector/trained_models/channel_selector_trainrun.yaml"
)

if trainrun_config.loss_nodes:
    print("Gradient training detected")
    # Uses GradientTrainer
else:
    print("Statistical training detected")
    # Uses StatisticalTrainer
```

### Handling Missing Weights

```python
from pathlib import Path

trainrun_path = Path("outputs/my_experiment/trained_models/my_experiment_trainrun.yaml")
weights_path = trainrun_path.parent / f"{trainrun_path.stem.replace('_trainrun', '')}.pt"

if weights_path.exists():
    print("Loading with trained weights")
    pipeline.load_weights(str(weights_path), strict=True)
else:
    print("No weights found - statistical initialization required")
    # Must run training first
```

## Complete Examples

### Example 1: Reproduce Training Results

```bash
# Display original experiment
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode info

# Re-run training with same settings
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --override output_dir=outputs/channel_selector_reproduced

# Validate on original validation set
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode validate

# Test on original test set
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode test
```

### Example 2: Transfer Learning

```bash
# Start from pre-trained weights, train on new data
uv run restore-trainrun \
  --trainrun-path outputs/lentils_model/trained_models/lentils_trainrun.yaml \
  --mode train \
  --override output_dir=outputs/beans_model \
  --override data.train_ids=[10,11,12] \
  --override data.val_ids=[13] \
  --override data.test_ids=[14,15] \
  --override training.optimizer.lr=0.0001 \
  --override training.trainer.max_epochs=30
```

### Example 3: Hyperparameter Search

```bash
# Try different learning rates
for lr in 0.001 0.0001 0.00001; do
  uv run restore-trainrun \
    --trainrun-path outputs/base_model/trained_models/base_trainrun.yaml \
    --mode train \
    --override output_dir=outputs/hp_search_lr_${lr} \
    --override training.optimizer.lr=${lr} \
    --override training.trainer.max_epochs=50
done

# Compare results
uv run compare-trainruns \
  outputs/hp_search_lr_0.001 \
  outputs/hp_search_lr_0.0001 \
  outputs/hp_search_lr_0.00001
```

### Example 4: Production Inference

```python
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
import torch

# Load trained pipeline for inference
pipeline = CuvisPipeline.load_pipeline(
    "outputs/production_model/trained_models/Production_Pipeline.yaml",
    weights_path="outputs/production_model/trained_models/Production_Pipeline.pt",
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Validate pipeline
pipeline.validate()

# Run inference on new data
import cuvis

# Load hyperspectral measurement
measurement = cuvis.Measurement("data/new_sample.cu3s")
cube = measurement.data["Reflectance"]  # Shape: (H, W, C)

# Prepare input
import torch
cube_tensor = torch.from_numpy(cube).float()
cube_tensor = cube_tensor.permute(2, 0, 1)  # (C, H, W)
cube_tensor = cube_tensor.unsqueeze(0)      # (1, C, H, W)

# Run pipeline
with torch.no_grad():
    result = pipeline.execute(cube=cube_tensor)

# Extract outputs
anomaly_scores = result["scores"]     # Anomaly heatmap
decisions = result["decisions"]       # Binary mask
metrics = result.get("metrics", {})   # Optional metrics

print(f"Detected anomalies: {decisions.sum().item()} pixels")
```

---

## Loading External Plugins

The cuvis.ai framework supports loading external plugin nodes that extend pipeline capabilities. When loading plugins, **dependencies are automatically installed** to ensure plugins work out of the box.

### Plugin Configuration

Plugins are specified in a YAML manifest file (e.g., `plugins.yaml`):

```yaml
plugins:
  adaclip:
    # For local development
    path: "D:/code-repos/cuvis-ai-adaclip"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector

  # Or for production (Git repository)
  # adaclip:
  #   repo: "https://github.com/cubert-hyperspectral/cuvis-ai-adaclip.git"
  #   ref: "v1.2.3"
  #   provides:
  #     - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector
```

### Automatic Dependency Management

When a plugin is loaded:

1. **Dependency Detection**: Reads `pyproject.toml` from plugin directory (PEP 621 compliant)
2. **Automatic Installation**: Installs missing dependencies via `uv pip install`
3. **Conflict Resolution**: Delegates version conflict resolution to `uv`
4. **Transparent Process**: Logs what dependencies are being installed

**Requirements:**
- Plugin **must** have a `pyproject.toml` file following [PEP 621](https://peps.python.org/pep-0621/)
- Plugin dependencies specified in `project.dependencies` section

### Using Plugins with CLI

Load external plugins when restoring pipelines:

```bash
uv run restore-pipeline \
  --pipeline-path configs/pipeline/adaclip_baseline.yaml \
  --plugins-path examples/adaclip/plugins_local.yaml
```

With inference:

```bash
uv run restore-pipeline \
  --pipeline-path configs/pipeline/adaclip_baseline.yaml \
  --plugins-path examples/adaclip/plugins.yaml \
  --cu3s-file-path data/Lentils/Lentils_000.cu3s
```

### Using Plugins with Python API

```python
from cuvis_ai_core.utils import restore_pipeline

# Load pipeline with plugins
pipeline = restore_pipeline(
    pipeline_path="configs/pipeline/adaclip_baseline.yaml",
    plugins_path="examples/adaclip/plugins.yaml"
)

# Or with inference
pipeline = restore_pipeline(
    pipeline_path="configs/pipeline/adaclip_baseline.yaml",
    plugins_path="examples/adaclip/plugins.yaml",
    cu3s_file_path="data/Lentils/Lentils_000.cu3s"
)
```

### Manual Plugin Loading (Advanced)

For more control, load plugins manually with NodeRegistry:

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

# Create registry instance
registry = NodeRegistry()

# Load plugins (automatically installs dependencies)
registry.load_plugins("examples/adaclip/plugins.yaml")

# Load pipeline with plugin-aware registry
pipeline = CuvisPipeline.load_pipeline(
    "configs/pipeline/adaclip_baseline.yaml",
    node_registry=registry
)
```

### Example Output

When loading a plugin with dependencies:

```
INFO | Using local plugin 'adaclip' at D:\code-repos\cuvis-ai-adaclip
DEBUG | Extracted 6 dependencies from pyproject.toml
INFO | Installing 6 dependencies for plugin 'adaclip'...
INFO | Dependencies to install: cuvis==3.5.0, cuvis-ai, cuvis-ai-core, ftfy, seaborn, click
INFO | ✓ Plugin 'adaclip' dependencies installed successfully
DEBUG | Registered plugin node 'AdaCLIPDetector' from 'adaclip'
INFO | Loaded plugin 'adaclip' with 1 nodes
```

### Plugin Development Guidelines

For your plugin to work with automatic dependency management:

1. **Include `pyproject.toml`** in plugin root directory
2. **Specify dependencies** in `project.dependencies`:
   ```toml
   [project]
   name = "my-plugin"
   dependencies = [
       "numpy>=1.20.0",
       "pandas>=1.5.0",
       "scikit-learn>=1.0.0",
   ]
   ```
3. **Follow PEP 621** standard for project metadata
4. **Test locally** before deploying to Git

For complete plugin development documentation, see [Plugin Development Guide](../plugin-system/development.md).

---

## Troubleshooting

### Issue: Missing Weights File
```
FileNotFoundError: [Errno 2] No such file or directory: 'My_Pipeline.pt'
```
**Solution:** Train the pipeline first or ensure weights were saved during training:
```bash
# Re-train to generate weights
uv run restore-trainrun --trainrun-path ... --mode train
```

### Issue: Weight Shape Mismatch
```
RuntimeError: Error(s) in loading state_dict: size mismatch for selector.weights
```
**Solution:** Pipeline structure changed. Options:
1. Use `strict=False` to load partial weights
2. Retrain from scratch
3. Manually adapt weights

```python
# Load with strict=False
pipeline.load_weights("My_Pipeline.pt", strict=False)
```

### Issue: Checkpoint Not Found
```
FileNotFoundError: Checkpoint 'epoch=10.ckpt' not found
```
**Solution:** Check available checkpoints:
```bash
ls outputs/my_experiment/checkpoints/
# Use last.ckpt or adjust epoch number
```

### Issue: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:** Reduce batch size or use CPU:
```bash
uv run restore-trainrun \
  --trainrun-path ... \
  --override data.batch_size=1 \
  --override training.trainer.accelerator=cpu
```

### Issue: Statistical Node Not Initialized
```
RuntimeError: RXGlobal requires statistical initialization before use
```
**Solution:** Run training mode first (not test/validate):
```bash
# Train to initialize statistical nodes
uv run restore-trainrun --trainrun-path ... --mode train

# Then validate/test
uv run restore-trainrun --trainrun-path ... --mode validate
```

### Issue: Missing pyproject.toml in Plugin
```
FileNotFoundError: Plugin 'my-plugin' must have a pyproject.toml file.
PEP 621 (https://peps.python.org/pep-0621/) specifies pyproject.toml
as the standard for Python project metadata and dependencies.
```
**Solution:** Add a `pyproject.toml` file to your plugin root directory

### Issue: Plugin Dependency Conflicts
```
RuntimeError: Failed to install dependencies for plugin 'my-plugin'.
This may indicate version conflicts or missing packages.
uv could not resolve the dependency tree.
```
**Solution:**
- Review dependency version constraints in your `pyproject.toml`
- Check for conflicts with main environment dependencies

### Issue: Plugin Import Errors
```
ImportError: Failed to import module for 'my_plugin.node.MyNode': No module named 'some_package'
```
**Solution:**
- Ensure the package is listed in your plugin's `pyproject.toml` dependencies
- Check that `uv pip install` completed successfully

---

## Best Practices

### 1. Always Save TrainRun Configs

```python
# At end of training script
trainrun_config.save_to_file(
    str(output_dir / "trained_models" / f"{experiment_name}_trainrun.yaml")
)

# Save weights separately
pipeline.save_weights(
    str(output_dir / "trained_models" / f"{pipeline_name}.pt")
)
```

### 2. Use Descriptive Experiment Names

```python
# Good: descriptive, unique names
trainrun_config = TrainRunConfig(
    name="channel_selector_lentils_3bands_v2",
    ...
)

# Avoid: generic names
trainrun_config = TrainRunConfig(
    name="experiment1",  # Not descriptive
    ...
)
```

### 3. Version Control YAML Files

```bash
# Track TrainRun configs in git
git add outputs/*/trained_models/*_trainrun.yaml
git add outputs/*/trained_models/*.yaml  # Pipeline YAMLs
git commit -m "Add trained model configs for channel selector experiment"

# Use .gitignore for large binary files
echo "*.pt" >> .gitignore
echo "*.ckpt" >> .gitignore
```

### 4. Document Experiment Purpose

```yaml
metadata:
  name: Channel_Selector_Lentils
  description: >
    Channel selection for lentil anomaly detection.
    Selects optimal 3 spectral bands from 61 channels using
    gradient-based optimization with entropy regularization.
    Trained on lentils dataset with stone anomalies.
  tags:
    - channel-selection
    - lentils
    - production-ready
  author: your_name
  created: 2026-02-04
  version: 2.1.0
```

### 5. Test Restoration Before Deployment

```bash
# Always verify restored pipeline works
uv run restore-trainrun --trainrun-path ... --mode info
uv run restore-trainrun --trainrun-path ... --mode validate

# Check metrics match original training
```

## See Also
- [Build Pipelines in Python](build-pipeline-python.md)
- [Build Pipelines in YAML](build-pipeline-yaml.md)
- [TrainRun Schema Reference](../config/trainrun-schema.md)
- [Pipeline Schema Reference](../config/pipeline-schema.md)
- [Remote gRPC Usage](remote-grpc.md)
- [Monitoring and Visualization](monitoring-and-viz.md)
