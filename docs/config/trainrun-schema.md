!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# TrainRun Configuration Schema

## Overview
Complete reference for TrainRun configuration structure, fields, validation rules, and examples.

## TrainRunConfig Structure

```python
TrainRunConfig(
    name: str,                      # Experiment identifier
    pipeline: Dict,                 # Full pipeline configuration
    data: Dict,                     # Data loading configuration
    training: TrainingConfig,       # Training settings
    output_dir: str,                # Results directory path
    loss_nodes: List[str] = [],     # Loss computation nodes
    metric_nodes: List[str] = [],   # Metric computation nodes
    freeze_nodes: List[str] = [],   # Initially frozen nodes
    unfreeze_nodes: List[str] = [], # Nodes to unfreeze for training
    tags: Dict = {}                 # Optional metadata
)
```

## Required Fields

### name (string)
Unique experiment identifier used for organizing outputs and logging.

**Example:**
```yaml
name: channel_selector_experiment
```

### pipeline (dict)
Complete pipeline specification with nodes and connections. Can be inline or composed via Hydra.

**Inline format:**
```yaml
pipeline:
  name: My_Pipeline
  nodes:
    - name: data_loader
      class: cuvis_ai.node.data.LentilsAnomalyDataNode
      params:
        normal_class_ids: [0, 1]
  connections:
    - from: data_loader.outputs.cube
      to: normalizer.inputs.data
```

**Composition format:**
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical  # Load from configs/pipeline/
```

### data (dict)
Data loading and splitting configuration.

**Required fields:**
- `cu3s_file_path` (string): Path to .cu3s data file
- `train_ids`, `val_ids`, `test_ids` (list of int): Sample indices for splits
- `batch_size` (int): Batch size for training

**Example:**
```yaml
data:
  cu3s_file_path: data/Lentils/Lentils_000.cu3s
  annotation_json_path: data/Lentils/Lentils_000.json
  train_ids: [0, 2, 3]
  val_ids: [1, 5]
  test_ids: [1, 5]
  batch_size: 2
  processing_mode: Reflectance
  shuffle: true
```

### training (dict)
Training configuration including optimizer, scheduler, and callbacks.

**Required fields:**
- `trainer` (dict): PyTorch Lightning trainer settings
- `optimizer` (dict): Optimizer configuration
- `scheduler` (dict, optional): Learning rate scheduler

**Example:**
```yaml
training:
  seed: 42
  trainer:
    max_epochs: 50
    accelerator: auto
    devices: 1
    precision: "32-true"
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    name: reduce_on_plateau
    monitor: metrics_anomaly/iou
    mode: max
```

### output_dir (string)
Directory path for saving results, checkpoints, and logs.

**Example:**
```yaml
output_dir: outputs/my_experiment
# Or with variable interpolation:
output_dir: ./outputs/${name}
```

## Optional Fields

### loss_nodes (list of string)
Names of loss nodes for gradient training. Empty for statistical-only training.

**Example:**
```yaml
loss_nodes:
  - bce_loss
  - entropy_loss
```

### metric_nodes (list of string)
Names of metric nodes for evaluation.

**Example:**
```yaml
metric_nodes:
  - metrics_anomaly
```

### freeze_nodes (list of string)
Nodes to keep frozen (non-trainable) throughout training.

**Example:**
```yaml
freeze_nodes:
  - data_loader
  - normalizer
```

### unfreeze_nodes (list of string)
Nodes to unfreeze for gradient training after statistical initialization.

**Example:**
```yaml
unfreeze_nodes:
  - channel_selector
  - rx_detector
  - logit_head
```

### tags (dict)
Optional metadata for experiment organization.

**Example:**
```yaml
tags:
  dataset: lentils
  method: channel_selection
  version: v2
```

## Complete Examples

### Example 1: Statistical Training (RX Detector)

```yaml
# @package _global_

name: rx_statistical_experiment

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

data:
  train_ids: [0, 2, 3]
  val_ids: [1, 5]
  test_ids: [1, 5]
  batch_size: 1

training:
  seed: 42
  trainer:
    max_epochs: 1  # Only statistical initialization
    accelerator: auto

output_dir: outputs/rx_statistical

# No loss nodes = statistical training only
loss_nodes: []
metric_nodes: [metrics]
freeze_nodes: []
unfreeze_nodes: []
```

### Example 2: Gradient Training (Channel Selector)

```yaml
# @package _global_

name: channel_selector_experiment

defaults:
  - /pipeline/anomaly/rx@pipeline: channel_selector
  - /data@data: lentils
  - /training@training: default
  - _self_

data:
  train_ids: [0]
  val_ids: [3, 4]
  test_ids: [1, 5]
  batch_size: 1

training:
  seed: 42
  trainer:
    max_epochs: 50
    accelerator: auto
    devices: 1
    precision: "32-true"

    callbacks:
      model_checkpoint:
        dirpath: outputs/channel_selector/checkpoints
        monitor: metrics_anomaly/iou
        mode: max
        save_top_k: 3
        save_last: true

  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01

  scheduler:
    name: reduce_on_plateau
    monitor: metrics_anomaly/iou
    mode: max
    patience: 5

output_dir: outputs/channel_selector

# Gradient training configuration
loss_nodes:
  - bce_loss
  - entropy_loss
metric_nodes:
  - metrics_anomaly
unfreeze_nodes:
  - selector
  - rx_global
  - logit_head

tags:
  method: channel_selection
  dataset: lentils
```

### Example 3: Two-Phase Training (Deep SVDD)

```yaml
# @package _global_

name: deep_svdd_experiment

defaults:
  - /pipeline/anomaly/deep_svdd@pipeline: deep_svdd
  - /data@data: lentils
  - /training@training: default
  - _self_

data:
  train_ids: [0, 2, 3]
  val_ids: [1]
  test_ids: [5]
  batch_size: 2

training:
  seed: 42
  trainer:
    max_epochs: 100
    accelerator: gpu
    devices: 1
    precision: "16-mixed"  # Use mixed precision for speed
    gradient_clip_val: 1.0

    callbacks:
      early_stopping:
        monitor: metrics_anomaly/auc
        patience: 10
        mode: max
        min_delta: 0.001

  optimizer:
    name: adam
    lr: 0.0001
    weight_decay: 0.0001

  scheduler:
    name: cosine_annealing
    T_max: 100
    eta_min: 1e-6

output_dir: outputs/deep_svdd

loss_nodes:
  - deepsvdd_loss
metric_nodes:
  - metrics_anomaly
unfreeze_nodes:
  - normalizer
  - projection
freeze_nodes:
  - data_loader

tags:
  method: deep_svdd
  dataset: lentils
  gpu: true
```

## Validation Rules

### 1. File Structure Requirements

**Required Hydra directive:**
```yaml
# @package _global_
```
Must be the first line of every trainrun configuration file.

**Required sections:**
- `defaults` (list)
- `name` (string)
- `output_dir` (string)

### 2. Composition Directives

**defaults list must include:**
```yaml
defaults:
  - /pipeline/anomaly/<category>@pipeline: <pipeline_name>
  - /data@data: <data_name>
  - /training@training: <training_name>
  - _self_  # Must be last for overrides to work
```

### 3. Node Name Validation

- `loss_nodes`, `metric_nodes`, `freeze_nodes`, `unfreeze_nodes` must reference valid node names defined in pipeline
- Names are case-sensitive and must match exactly

**Example validation:**
```python
# Pipeline defines these nodes:
nodes = ["data_loader", "normalizer", "selector", "rx", "bce_loss", "metrics"]

# Valid unfreeze_nodes:
unfreeze_nodes = ["selector", "rx"]  # ✓

# Invalid:
unfreeze_nodes = ["Selector", "RX"]  # ✗ (wrong case)
unfreeze_nodes = ["unknown_node"]     # ✗ (not in pipeline)
```

### 4. Training Type Detection

**Statistical training:**
```yaml
loss_nodes: []  # Empty or omitted
# System uses StatisticalTrainer
```

**Gradient training:**
```yaml
loss_nodes: [loss1, loss2]  # Non-empty
# System uses GradientTrainer
```

### 5. Output Directory Rules

- Must be writable path
- Supports variable interpolation: `${name}`, `${oc.env:USER}`
- Automatically creates subdirectories:
  - `trained_models/`
  - `checkpoints/`
  - `tensorboard/`

## Common Patterns

### Pattern 1: Quick Experiment

Minimal configuration for rapid iteration:

```yaml
# @package _global_

name: quick_test

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

output_dir: outputs/${name}
```

### Pattern 2: Hyperparameter Sweep

Base configuration for multi-run experiments:

```yaml
# @package _global_

name: hp_sweep_lr_${training.optimizer.lr}

defaults:
  - /pipeline/anomaly/rx@pipeline: channel_selector
  - /data@data: lentils
  - /training@training: default
  - _self_

training:
  optimizer:
    lr: 0.001  # Override from command line

output_dir: outputs/${name}
```

Usage:
```bash
python train.py -m training.optimizer.lr=0.001,0.0001,0.00001
```

### Pattern 3: Production Configuration

Comprehensive configuration with all settings explicit:

```yaml
# @package _global_

name: production_model_v1

defaults:
  - /pipeline/anomaly/rx@pipeline: channel_selector
  - /data@data: lentils
  - /training@training: default
  - _self_

data:
  cu3s_file_path: data/production/train.cu3s
  train_ids: [0, 1, 2, 3, 4]
  val_ids: [5, 6]
  test_ids: [7, 8, 9]
  batch_size: 4
  shuffle: true

training:
  seed: 42
  trainer:
    max_epochs: 200
    accelerator: gpu
    devices: 1
    precision: "16-mixed"
    deterministic: true
    gradient_clip_val: 1.0

    callbacks:
      model_checkpoint:
        dirpath: outputs/${name}/checkpoints
        monitor: metrics_anomaly/iou
        mode: max
        save_top_k: 1
        save_last: true

      early_stopping:
        monitor: metrics_anomaly/iou
        patience: 20
        mode: max
        min_delta: 0.001

  optimizer:
    name: adamw
    lr: 0.0005
    weight_decay: 0.01

  scheduler:
    name: reduce_on_plateau
    monitor: metrics_anomaly/iou
    mode: max
    factor: 0.5
    patience: 10

output_dir: outputs/${name}

loss_nodes:
  - bce_loss
  - entropy_loss
metric_nodes:
  - metrics_anomaly
unfreeze_nodes:
  - selector
  - rx_global
  - logit_head

tags:
  version: v1
  purpose: production
  dataset: production_dataset
  created: 2026-02-04
```

## See Also
- [Build Pipelines in YAML](../how-to/build-pipeline-yaml.md)
- [Restore Pipeline from TrainRun](../how-to/restore-pipeline-trainrun.md)
- [Hydra Composition](hydra-composition.md)
- [Pipeline Schema](pipeline-schema.md)
- [Config Groups](config-groups.md)
