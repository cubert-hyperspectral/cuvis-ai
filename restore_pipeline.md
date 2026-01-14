# Pipeline and TrainRun Restoration Guide

This guide covers restoring and running trained pipelines and trainruns for inference and continued training.

## Overview

The cuvis.ai framework provides utilities for:

1. **Restoring Pipeline** - Quick inference with trained models
2. **Restoring TrainRun** - Full experiment reproduction, validation, and testing

All functionality is available through:
- **Python API**: `from cuvis_ai.utils import restore_pipeline, restore_trainrun`
- **CLI Commands**: `uv run restore-pipeline` and `uv run restore-trainrun`

## Table of Contents

- [Running Experiments](#running-experiments)
  - [Channel Selector (Gradient Training)](#channel-selector-gradient-training)
  - [RX Statistical (Statistical Training)](#rx-statistical-statistical-training)
  - [Output Structure](#output-structure)
- [Restoring Pipeline for Inference](#restoring-pipeline-for-inference)
- [Restoring Complete TrainRuns](#restoring-complete-trainruns)
- [Configuration Overrides](#configuration-overrides)
- [Practical Examples](#practical-examples)

---

## Running Experiments

### Channel Selector (Gradient Training)

The channel selector example demonstrates a complete gradient-based training pipeline with statistical initialization, channel selection optimization, and checkpointing.

**Basic Usage:**

```bash
uv run python examples/channel_selector.py
```

**With Custom Configuration:**

```bash
uv run python examples/channel_selector.py output_dir=outputs/my_experiment data.batch_size=16
```

**What It Does:**

1. Creates a pipeline with channel selection, RX anomaly detection, and visualization nodes
2. Initializes statistics (mean, covariance) using statistical training
3. Unfreezes selector, RX, and logit head nodes for gradient optimization
4. Trains using gradient descent with early stopping and checkpointing
5. Saves:
   - Trained pipeline: `outputs/channel_selector/trained_models/Channel_Selector.yaml` + `.pt`
   - TrainRun config: `outputs/channel_selector/trained_models/channel_selector_trainrun.yaml`
   - Checkpoints: `outputs/channel_selector/checkpoints/epoch=XX.ckpt`
   - Visualizations and logs

### RX Statistical (Statistical Training)

The RX statistical example demonstrates pure statistical training without gradient optimization.

**Basic Usage:**

```bash
uv run python examples/rx_statistical.py
```

**With Custom Configuration:**

```bash
uv run python examples/rx_statistical.py output_dir=outputs/rx_test
```

**What It Does:**

1. Creates a pipeline with RX anomaly detection and metrics
2. Fits statistical parameters (mean, covariance) from training data
3. Runs validation and test evaluations
4. Saves:
   - Trained pipeline: `outputs/rx_statistical/trained_models/RX_Statistical.yaml` + `.pt`
   - TrainRun config: `outputs/rx_statistical/trained_models/rx_statistical_trainrun.yaml`
   - Visualizations and logs

### Output Structure

After running an experiment, the output directory contains:

```
outputs/channel_selector/
├── pipeline/
│   ├── Channel_Selector.png      # GraphViz visualization
│   └── Channel_Selector.md        # Mermaid diagram
├── checkpoints/
│   ├── epoch=00.ckpt
│   ├── epoch=01.ckpt
│   ├── last.ckpt
│   └── ...
└── trained_models/
    ├── Channel_Selector.yaml      # Pipeline configuration
    ├── Channel_Selector.pt        # Trained weights
    └── channel_selector_trainrun.yaml  # Complete trainrun config
```

---

## Restoring Pipeline for Inference

Use `restore_pipeline` for quick inference without recreating the full trainrun setup.

### Using the CLI

**Display Pipeline Info:**

```bash
uv run restore-pipeline --pipeline-path outputs/channel_selector/trained_models/Channel_Selector.yaml
```

**Run Inference on a .cu3s File:**

```bash
uv run restore-pipeline \
  --pipeline-path outputs/channel_selector/trained_models/Channel_Selector.yaml \
  --cu3s-file-path data/lentils/test_measurements.cu3s \
  --processing-mode Reflectance
```

**With Custom Device:**

```bash
uv run restore-pipeline \
  --pipeline-path outputs/channel_selector/trained_models/Channel_Selector.yaml \
  --device cuda
```

### Using the Python API

```python
from cuvis_ai.utils import restore_pipeline

# Load pipeline and display specs
pipeline = restore_pipeline(
    pipeline_path="outputs/channel_selector/trained_models/Channel_Selector.yaml"
)

# Load pipeline and run inference
pipeline = restore_pipeline(
    pipeline_path="outputs/channel_selector/trained_models/Channel_Selector.yaml",
    cu3s_file_path="data/lentils/test_measurements.cu3s",
    processing_mode="Reflectance"
)
```

### Parameters

- `--pipeline-path` (required): Path to pipeline YAML configuration
- `--weights-path` (optional): Path to weights file (defaults to `.pt` file next to pipeline YAML)
- `--device` (optional): Device to use (`auto`, `cpu`, `cuda`)
- `--cu3s-file-path` (optional): Path to .cu3s file for inference
- `--processing-mode` (optional): Processing mode (`Raw`, `Reflectance`)
- `--override` (optional): Override config values (can be used multiple times)

### Config Overrides Example

Override TensorBoard output directory:

```bash
uv run restore-pipeline \
  --pipeline-path outputs/channel_selector/trained_models/Channel_Selector.yaml \
  --override nodes.10.params.output_dir=outputs/custom_tb
```

---

## Restoring Complete TrainRuns

Use `restore_trainrun` to reproduce entire trainruns, validate models, or run tests.

The `restore_trainrun` function intelligently detects whether to use gradient-based or statistical-only training based on the trainrun configuration.

### Using the CLI

#### 1. Info Mode (Default)

Display trainrun information and pipeline specifications:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml
```

#### 2. Train Mode

Re-run or continue training:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train
```

**Resume from Checkpoint (Gradient Training):**

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --checkpoint-path outputs/channel_selector/checkpoints/epoch=09.ckpt
```

#### 3. Validate Mode

Run validation only:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode validate
```

#### 4. Test Mode

Run test evaluation:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode test
```

### Using the Python API

```python
from cuvis_ai.utils import restore_trainrun

# Display info
restore_trainrun(
    trainrun_path="outputs/channel_selector/trained_models/channel_selector_trainrun.yaml",
    mode="info"
)

# Run training
restore_trainrun(
    trainrun_path="outputs/channel_selector/trained_models/channel_selector_trainrun.yaml",
    mode="train"
)

# Run validation
restore_trainrun(
    trainrun_path="outputs/channel_selector/trained_models/channel_selector_trainrun.yaml",
    mode="validate"
)
```

### CLI Parameters

- `--trainrun-path` (required): Path to trainrun YAML file
- `--mode` (optional): Execution mode (`info`, `train`, `validate`, `test`)
- `--checkpoint-path` (optional): Checkpoint to resume training from
- `--device` (optional): Device to use (`auto`, `cpu`, `cuda`)
- `--override` (optional): Override config values (can be used multiple times)

---

## Configuration Overrides

The `--override` parameter allows you to modify configuration values without editing files. Use dot notation to specify nested values.

### Single Override

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --override output_dir=outputs/custom_experiment
```

### Multiple Overrides

Specify `--override` multiple times:

```bash
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --override output_dir=outputs/custom_experiment \
  --override data.batch_size=32 \
  --override training.optimizer.lr=0.0005
```

### Common Override Patterns

**Change Output Directory:**
```bash
--override output_dir=outputs/new_location
```

**Modify Data Configuration:**
```bash
--override data.batch_size=16
--override data.train_ids=[0,1,2,3]
```

**Adjust Training Parameters (Gradient Training):**
```bash
--override training.optimizer.lr=0.001
--override training.trainer.max_epochs=50
```

**Override Node Parameters:**
```bash
--override nodes.5.params.temperature_init=10.0
```

---

## Practical Examples

### Example 1: Complete Training Workflow

```bash
# Step 1: Run initial training
uv run python examples/channel_selector.py

# Step 2: View trainrun info
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml

# Step 3: Continue training with more epochs
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --override training.trainer.max_epochs=100
```

### Example 2: TrainRun Variations

```bash
# Run original training
uv run python examples/channel_selector.py

# Try with different hyperparameters
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --override output_dir=outputs/channel_selector_v2 \
  --override training.optimizer.lr=0.0001 \
  --override data.batch_size=8
```

### Example 3: Quick Inference

```bash
# After training, run inference on new data
uv run restore-pipeline \
  --pipeline-path outputs/channel_selector/trained_models/Channel_Selector.yaml \
  --cu3s-file-path data/production/sample_001.cu3s
```

### Example 4: Evaluation on Different Splits

```bash
# Validate on validation set
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode validate

# Evaluate on test set
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode test
```

### Example 5: Cross-Device Training

```bash
# Train on GPU
uv run restore-trainrun \
  --trainrun-path outputs/channel_selector/trained_models/channel_selector_trainrun.yaml \
  --mode train \
  --device cuda

# Run inference on CPU
uv run restore-pipeline \
  --pipeline-path outputs/channel_selector/trained_models/Channel_Selector.yaml \
  --device cpu \
  --cu3s-file-path data/test.cu3s
```

### Example 6: Statistical-Only TrainRun

```bash
# Train a statistical-only pipeline
uv run python examples/rx_statistical.py

# View info
uv run restore-trainrun \
  --trainrun-path outputs/rx_statistical/trained_models/rx_statistical_trainrun.yaml

# Run validation
uv run restore-trainrun \
  --trainrun-path outputs/rx_statistical/trained_models/rx_statistical_trainrun.yaml \
  --mode validate
```

---

## Troubleshooting

### Missing Checkpoint Files

If checkpoint files are missing, training will start from scratch. Ensure the checkpoint path is correct and the file exists.

### Configuration Override Syntax

- Use dot notation: `parent.child.param=value`
- No spaces around `=`
- For lists: `param=[1,2,3]`
- For strings with spaces: `param="value with spaces"`

### Device Availability

If CUDA is not available and you specify `--device cuda`, the script will fail. Use `--device auto` to automatically select the best available device.

### Output Directory Conflicts

When using `--override output_dir=...`, ensure the directory doesn't conflict with existing experiments or use unique names.

### Auto-Detection of Training Type

The `restore_trainrun` function automatically detects whether the trainrun uses:
- **Gradient Training**: If training config with trainer is present → uses `GradientTrainer`
- **Statistical Training**: If only statistical config is present → uses `StatisticalTrainer`

No manual configuration is needed; the function handles both cases intelligently.

---

## Summary

- **Run experiments**: Use example scripts like `channel_selector.py` or `rx_statistical.py`
- **Quick inference**: Use `uv run restore-pipeline` with pipeline YAML files
- **Reproduce trainruns**: Use `uv run restore-trainrun` with trainrun YAML files
- **Python API**: `from cuvis_ai.utils import restore_pipeline, restore_trainrun`
- **Customize behavior**: Use `--override` for configuration changes
- **Resume training**: Combine `--mode train` with `--checkpoint-path` (gradient training only)

For more information, see the API documentation and other examples in the `examples/` directory.
