!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Configuration Groups

Organize and manage CUVIS.AI configurations using Hydra config groups for modular, composable experiment setups.

## Overview

Configuration groups provide modular organization of experiment parameters:

- **pipeline**: Pipeline architecture definitions (nodes + connections)
- **data**: Data loading and splitting configurations
- **training**: Trainer, optimizer, and scheduler settings
- **trainrun**: Composed experiments combining all groups

**Benefits:**
- Reusable configurations across experiments
- Easy parameter sweeps and comparisons
- Clear separation of concerns
- Version control friendly

---

## Quick Start

### Directory Structure

```
configs/
├── data/
│   └── lentils.yaml
├── pipeline/
│   ├── rx_statistical.yaml
│   ├── channel_selector.yaml
│   ├── deep_svdd.yaml
│   └── ...
├── training/
│   └── default.yaml
└── trainrun/
    ├── rx_statistical.yaml
    ├── channel_selector.yaml
    └── ...
```

### Basic Usage

**Select configs in trainrun:**
```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

name: my_experiment
```

**Override from command line:**
```bash
uv run python train.py \
    training.trainer.max_epochs=100 \
    data.batch_size=4
```

---

## Config Groups

### 1. Pipeline Group

**Location:** `configs/pipeline/`

Pipeline group defines the computational graph (nodes + connections).

**Available pipelines:**
- `rx_statistical.yaml` - RX anomaly detector (statistical)
- `channel_selector.yaml` - Channel selection + RX (gradient)
- `deep_svdd.yaml` - Deep SVDD anomaly detection
- `drcnn_adaclip.yaml` - DRCNN + AdaClip integration
- `concrete_adaclip.yaml` - Concrete band selector + AdaClip
- `adaclip_baseline.yaml` - AdaClip baseline

**Selection pattern:**
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
```

**Example: rx_statistical.yaml**
```yaml
metadata:
  name: RX_Statistical
  description: RX anomaly detector with statistical initialization
  tags:
    - statistical
    - rx
  author: cuvis.ai

nodes:
  - name: LentilsAnomalyDataNode
    class: cuvis_ai.node.data.LentilsAnomalyDataNode
    params:
      normal_class_ids: [0, 1]

  - name: MinMaxNormalizer
    class: cuvis_ai.node.normalization.MinMaxNormalizer
    params:
      eps: 1.0e-06
      use_running_stats: true

  - name: RXGlobal
    class: cuvis_ai.anomaly.rx_detector.RXGlobal
    params:
      num_channels: 61
      eps: 1.0e-06

connections:
  - from: LentilsAnomalyDataNode.outputs.cube
    to: MinMaxNormalizer.inputs.data
  - from: MinMaxNormalizer.outputs.normalized
    to: RXGlobal.inputs.data
```

**Override pipeline parameters:**
```yaml
# In trainrun config
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - _self_

# Override specific node params
pipeline:
  nodes:
    - name: RXGlobal
      params:
        eps: 1.0e-08  # Override from 1.0e-06
```

### 2. Data Group

**Location:** `configs/data/`

Data group defines data loading, splitting, and preprocessing.

**Available configs:**
- `lentils.yaml` - Lentils anomaly detection dataset

**Selection pattern:**
```yaml
defaults:
  - /data@data: lentils
```

**Example: lentils.yaml**
```yaml
cu3s_file_path: data/Lentils/Lentils_000.cu3s
annotation_json_path: data/Lentils/Lentils_000.json

train_ids: [0, 2, 3]
val_ids: [1, 5]
test_ids: [1, 5]

batch_size: 2
shuffle: true
num_workers: 0
processing_mode: Reflectance
```

**Override data parameters:**
```yaml
# In trainrun config
defaults:
  - /data@data: lentils
  - _self_

data:
  train_ids: [0, 1, 2]    # Override
  val_ids: [3, 4]         # Override
  batch_size: 4           # Override
```

**Command-line overrides:**
```bash
uv run python train.py \
    data.train_ids=[0,1,2] \
    data.batch_size=8
```

### 3. Training Group

**Location:** `configs/training/`

Training group defines trainer, optimizer, scheduler, and callback settings.

**Available configs:**
- `default.yaml` - Default training configuration

**Selection pattern:**
```yaml
defaults:
  - /training@training: default
```

**Example: default.yaml**
```yaml
seed: 42

trainer:
  max_epochs: 5
  accelerator: auto
  devices: 1
  precision: "32-true"
  log_every_n_steps: 10
  val_check_interval: 1.0
  enable_checkpointing: true
  gradient_clip_val: 1.0

  callbacks:
    learning_rate_monitor:
      logging_interval: epoch
      log_momentum: false

optimizer:
  name: adamw
  lr: 0.001
  weight_decay: 0.01
  betas: [0.9, 0.999]
```

**Override training parameters:**
```yaml
# In trainrun config
defaults:
  - /training@training: default
  - _self_

training:
  trainer:
    max_epochs: 100           # Override
    accelerator: gpu          # Override

  optimizer:
    lr: 0.0001                # Override

  scheduler:                  # Add new field
    name: reduce_on_plateau
    monitor: metrics_anomaly/iou
    mode: max
```

**Command-line overrides:**
```bash
uv run python train.py \
    training.trainer.max_epochs=100 \
    training.optimizer.lr=0.0001 \
    training.scheduler.patience=10
```

### 4. TrainRun Group

**Location:** `configs/trainrun/`

TrainRun group composes pipeline, data, and training configs into complete experiments.

**Available trainruns:**
- `rx_statistical.yaml` - Statistical RX training
- `channel_selector.yaml` - Channel selector gradient training
- `deep_svdd.yaml` - Deep SVDD training
- `drcnn_adaclip.yaml` - DRCNN + AdaClip training
- `concrete_adaclip.yaml` - Concrete band selector training

**Selection pattern:**
```python
@hydra.main(config_path="../configs", config_name="trainrun/rx_statistical", version_base=None)
def main(cfg: DictConfig):
    ...
```

**Example: rx_statistical.yaml**
```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

name: rx_statistical
output_dir: ./outputs/${name}

# Statistical training (no losses)
loss_nodes: []
metric_nodes: [metrics_anomaly]
freeze_nodes: []
unfreeze_nodes: []

tags:
  method: rx
  training: statistical
  dataset: lentils
```

**Example: drcnn_adaclip.yaml**
```yaml
# @package _global_

defaults:
  - /data@data: lentils
  - /training@training: default
  - _self_

name: drcnn_adaclip
output_dir: ./outputs/${name}

training:
  seed: 42
  trainer:
    max_epochs: 20
    precision: "32-true"
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    name: reduce_on_plateau
    monitor: metrics_anomaly/iou
    mode: max
    factor: 0.5
    patience: 5

# Gradient training configuration
loss_nodes: [iou_loss]
metric_nodes: [metrics_anomaly]
unfreeze_nodes: [channel_mixer]

tags:
  method: drcnn_adaclip
  training: gradient
  dataset: lentils
```

---

## Group Selection Patterns

### Pattern 1: Standard Composition

Most common pattern for complete experiments:

```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: channel_selector
  - /data@data: lentils
  - /training@training: default
  - _self_

name: my_experiment
output_dir: ./outputs/${name}
```

**Execution:**
```bash
uv run python train.py
```

### Pattern 2: Selective Overrides

Override specific sections without full replacement:

```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

# Override only data splits
data:
  train_ids: [0, 2, 3]
  val_ids: [1]
  test_ids: [5]

# Override only optimizer learning rate
training:
  optimizer:
    lr: 0.0001
```

### Pattern 3: Command-Line Selection

Select config groups at runtime:

```bash
# Select different pipeline
uv run python train.py pipeline=deep_svdd

# Select different data config
uv run python train.py data=custom_dataset

# Combine selections
uv run python train.py \
    pipeline=channel_selector \
    data=lentils \
    training=default \
    training.optimizer.lr=0.0001
```

### Pattern 4: Multi-Run Sweeps

Sweep over multiple config combinations:

```bash
# Sweep over pipelines
uv run python train.py -m pipeline=rx_statistical,channel_selector,deep_svdd

# Sweep over hyperparameters
uv run python train.py -m training.optimizer.lr=0.001,0.0001,0.00001

# Combined sweep
uv run python train.py -m \
    pipeline=rx_statistical,channel_selector \
    training.optimizer.lr=0.001,0.0001
```

**Hydra creates separate output directories:**
```
outputs/
├── pipeline=rx_statistical,lr=0.001/
├── pipeline=rx_statistical,lr=0.0001/
├── pipeline=channel_selector,lr=0.001/
└── pipeline=channel_selector,lr=0.0001/
```

---

## Override Syntax

### Dot Notation

Access nested fields using dot notation:

```bash
training.trainer.max_epochs=100
training.optimizer.lr=0.001
data.batch_size=16
pipeline.nodes.RXGlobal.params.eps=1e-08
```

### List Assignment

Assign lists using bracket notation:

```bash
data.train_ids=[0,1,2]
data.val_ids=[3,4]
loss_nodes=[bce_loss,entropy_loss]
```

### Dictionary Assignment

Assign dictionaries using brace notation:

```bash
training.scheduler={name:reduce_on_plateau,monitor:val/loss,mode:min}
```

### Multiple Overrides

Chain multiple overrides separated by spaces:

```bash
uv run python train.py \
    training.trainer.max_epochs=100 \
    training.optimizer.lr=0.0001 \
    training.optimizer.weight_decay=0.01 \
    data.batch_size=8 \
    data.train_ids=[0,1,2] \
    output_dir=outputs/custom_experiment
```

---

## Creating Custom Config Groups

### Step 1: Create Group Directory

```bash
mkdir configs/data/my_dataset
```

### Step 2: Create Config File

**File:** `configs/data/my_dataset/default.yaml`

```yaml
cu3s_file_path: data/MyDataset/dataset.cu3s
annotation_json_path: data/MyDataset/annotations.json

train_ids: [0, 1, 2, 3]
val_ids: [4, 5]
test_ids: [6, 7, 8]

batch_size: 4
shuffle: true
num_workers: 4
processing_mode: Reflectance
```

### Step 3: Use in TrainRun

```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: my_dataset/default
  - /training@training: default
  - _self_

name: my_experiment
```

### Step 4: Create Variants

**File:** `configs/data/my_dataset/augmented.yaml`

```yaml
defaults:
  - default  # Inherit from default.yaml
  - _self_

# Override augmentation settings
use_augmentation: true
augmentation:
  horizontal_flip: true
  vertical_flip: true
  rotation_degrees: 15
```

**Usage:**
```yaml
defaults:
  - /data@data: my_dataset/augmented
```

---

## Best Practices

### 1. Config Organization

**Keep related configs together:**
```
configs/
├── data/
│   ├── lentils/
│   │   ├── default.yaml
│   │   ├── augmented.yaml
│   │   └── small_subset.yaml
│   └── tomatoes/
│       └── default.yaml
```

**Use descriptive names:**
```yaml
# Good
configs/pipeline/anomaly/rx/channel_selector_with_rx.yaml

# Avoid
configs/pipeline/pipeline1.yaml
```

### 2. Defaults Ordering

**Always put `_self_` last:**
```yaml
# Correct
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - _self_  # ← LAST

# Wrong
defaults:
  - _self_
  - /pipeline/anomaly/rx@pipeline: rx_statistical  # Won't override _self_
```

### 3. Minimal Overrides

**Only override what you need:**
```yaml
# Good: Minimal changes
defaults:
  - /training@training: default
  - _self_

training:
  optimizer:
    lr: 0.0001  # Only override LR

# Avoid: Duplicating entire config
training:
  seed: 42
  trainer:
    max_epochs: 5
    accelerator: auto
    devices: 1
    # ... (duplicates default.yaml)
```

### 4. Variable Interpolation

**Use variable references:**
```yaml
name: my_experiment
output_dir: ./outputs/${name}

training:
  trainer:
    default_root_dir: ${output_dir}
```

**Environment variables:**
```yaml
data:
  cu3s_file_path: ${oc.env:DATA_ROOT}/Lentils_000.cu3s
  # Falls back if DATA_ROOT not set:
  cu3s_file_path: ${oc.env:DATA_ROOT,./data/Lentils}/Lentils_000.cu3s
```

### 5. Documentation

**Add comments to configs:**
```yaml
# RX Statistical Training
# This trainrun performs statistical initialization of RX detector
# without gradient-based training.

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - _self_

# Experiment identifier
name: rx_statistical

# Empty loss nodes = statistical training only
loss_nodes: []
```

---

## Common Patterns

### Pattern 1: Hyperparameter Sweep

**Base config:** `configs/trainrun/sweep_base.yaml`

```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: channel_selector
  - /data@data: lentils
  - /training@training: default
  - _self_

name: sweep_lr_${training.optimizer.lr}
output_dir: ./outputs/sweep/${name}
```

**Execution:**
```bash
uv run python train.py -m \
    --config-name=trainrun/sweep_base \
    training.optimizer.lr=0.001,0.0001,0.00001
```

### Pattern 2: Dataset Comparison

```bash
uv run python train.py -m \
    data=lentils,tomatoes,cucumbers \
    name=comparison_${data}
```

### Pattern 3: Pipeline Comparison

```bash
uv run python train.py -m \
    pipeline=rx_statistical,channel_selector,deep_svdd \
    name=${pipeline}_comparison
```

### Pattern 4: Configuration Inheritance

**Base config:** `configs/training/base_optimizer.yaml`

```yaml
optimizer:
  name: adamw
  betas: [0.9, 0.999]
```

**Variant:** `configs/training/high_lr.yaml`

```yaml
defaults:
  - base_optimizer
  - _self_

optimizer:
  lr: 0.001
  weight_decay: 0.01
```

**Variant:** `configs/training/low_lr.yaml`

```yaml
defaults:
  - base_optimizer
  - _self_

optimizer:
  lr: 0.00001
  weight_decay: 0.001
```

---

## Troubleshooting

### Config Not Found

**Problem:** `ConfigAttributeError: Key 'pipeline' is not in struct`

**Solution:** Check config group path and filename:
```bash
# Check available configs
ls configs/pipeline/

# Verify path in defaults
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical  # ← Must match filename
```

### Override Not Applied

**Problem:** Override in trainrun doesn't apply.

**Solution:** Ensure `_self_` is last in defaults:
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - _self_  # ← MUST BE LAST

# Overrides below
data:
  batch_size: 16  # ← Now works
```

### Missing Required Field

**Problem:** `MissingMandatoryValue: Missing mandatory value: data.cu3s_file_path`

**Solution:** Verify data config has required fields:
```yaml
# configs/data/lentils.yaml
cu3s_file_path: data/Lentils/Lentils_000.cu3s  # ← Required
annotation_json_path: data/Lentils/Lentils_000.json
```

### Package Directive Errors

**Problem:** Configs merged at wrong level.

**Solution:** Use `@package _global_` for trainrun configs:
```yaml
# @package _global_  ← Required for trainruns

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
```

---

## See Also

- **Configuration Guides**:
  - [Hydra Composition](hydra-composition.md) - Composition patterns and inheritance
  - [TrainRun Schema](trainrun-schema.md) - Complete trainrun configuration reference
  - [Pipeline Schema](pipeline-schema.md) - Pipeline YAML structure
- **How-To Guides**:
  - [Build Pipelines in YAML](../how-to/build-pipeline-yaml.md) - Create pipeline configs
  - [Configuration Guide](../user-guide/configuration.md) - Configuration overview
- **Examples**:
  - `configs/trainrun/` - Example trainrun configurations
  - `configs/pipeline/` - Example pipeline configurations
  - `examples/rx_statistical.py` - Using config groups in code
