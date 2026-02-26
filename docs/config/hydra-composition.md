!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Composition Patterns

Master Hydra composition for flexible, reusable, and modular configuration management in CUVIS.AI.

## Overview

Hydra enables powerful configuration composition:

- **Defaults List**: Compose multiple configs into one
- **Package Directives**: Control config placement in hierarchy
- **Inheritance**: Reuse and extend base configurations
- **Variable Interpolation**: Reference and compute values dynamically
- **Multi-Run Sweeps**: Hyperparameter optimization and grid search
- **Command-Line Overrides**: Runtime configuration changes

**Benefits:**
- Eliminate configuration duplication
- Compose experiments from reusable pieces
- Easy hyperparameter sweeps
- Clear configuration hierarchy

---

## Quick Start

### Basic Composition

**Trainrun config:**
```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_

name: my_experiment
output_dir: ./outputs/${name}
```

**Usage:**
```python
@hydra.main(config_path="../configs", config_name="trainrun/my_experiment", version_base=None)
def main(cfg: DictConfig):
    print(cfg.name)  # Access composed config
```

---

## Package Directives

### @package _global_

Merges config at the root level.

**Most common for trainrun configs:**
```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
name: experiment_1
```

**Result:**
```yaml
pipeline:
  metadata:
    name: RX_Statistical
  nodes: [...]
name: experiment_1  # ← Merged at root
```

### @package _group_

Merges config under the group name.

**Example:**
```yaml
# @package training

defaults:
  - /optimizer: adamw
  - /scheduler: reduce_on_plateau

seed: 42
```

**Result:**
```yaml
training:
  seed: 42
  optimizer: {...}
  scheduler: {...}
```

### Explicit Package Paths

Control exactly where configs are placed:

```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical       # → pipeline: {...}
  - /data@data: lentils                      # → data: {...}
  - /training@training: default              # → training: {...}
```

**Path format:** `/source_group@target_key: config_name`

---

## Defaults List

### Structure and Ordering

The `defaults` list determines config composition order:

```yaml
# @package _global_

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical   # Load first
  - /data@data: lentils                  # Load second
  - /training@training: default          # Load third
  - _self_                               # THIS CONFIG (must be last)

# Overrides below (only applied because _self_ is last)
data:
  batch_size: 16
```

**Critical rule:** `_self_` must be last to allow overrides in the current file.

### Absolute vs Relative Paths

**Absolute paths** (start with `/`):
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  # Searches: configs/pipeline/anomaly/rx/rx_statistical.yaml
```

**Relative paths** (no leading `/`):
```yaml
# In configs/training/high_lr.yaml
defaults:
  - base_optimizer  # Searches: configs/training/base_optimizer.yaml
  - _self_
```

### Conditional Defaults

Exclude defaults using `optional`:

```yaml
defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /training/scheduler@training.scheduler: ${scheduler_name}
  - optional /augmentation@data.augmentation: ${augmentation}
```

---

## Config Inheritance

### Simple Inheritance

**Base config:** `configs/training/base.yaml`
```yaml
seed: 42

trainer:
  max_epochs: 5
  accelerator: auto
  devices: 1

optimizer:
  name: adamw
  betas: [0.9, 0.999]
```

**Variant:** `configs/training/high_lr.yaml`
```yaml
defaults:
  - base
  - _self_

optimizer:
  lr: 0.001           # Add new field
  weight_decay: 0.01  # Add new field
```

**Result:**
```yaml
seed: 42                        # From base
trainer:
  max_epochs: 5                 # From base
  accelerator: auto             # From base
  devices: 1                    # From base
optimizer:
  name: adamw                   # From base
  betas: [0.9, 0.999]          # From base
  lr: 0.001                     # From high_lr
  weight_decay: 0.01            # From high_lr
```

### Multi-Level Inheritance

**Level 1:** `configs/training/base_optimizer.yaml`
```yaml
optimizer:
  name: adamw
  betas: [0.9, 0.999]
```

**Level 2:** `configs/training/base_training.yaml`
```yaml
defaults:
  - base_optimizer
  - _self_

seed: 42
trainer:
  max_epochs: 5
```

**Level 3:** `configs/training/custom_training.yaml`
```yaml
defaults:
  - base_training
  - _self_

optimizer:
  lr: 0.001
trainer:
  max_epochs: 100  # Override base
```

**Result:** Combines all three levels with later configs overriding earlier ones.

### Override Behavior

Hydra uses **merge** strategy by default:

**Base:**
```yaml
data:
  train_ids: [0, 2, 3]
  val_ids: [1, 5]
  batch_size: 2
```

**Override:**
```yaml
defaults:
  - base
  - _self_

data:
  batch_size: 16  # Only override batch_size
```

**Result:**
```yaml
data:
  train_ids: [0, 2, 3]    # From base
  val_ids: [1, 5]         # From base
  batch_size: 16          # Overridden
```

---

## Variable Interpolation

### Simple Interpolation

Reference other values in the config:

```yaml
name: my_experiment
output_dir: ./outputs/${name}
# Resolves to: ./outputs/my_experiment

training:
  trainer:
    default_root_dir: ${output_dir}
    # Resolves to: ./outputs/my_experiment
```

### Environment Variables

Access environment variables with `oc.env`:

```yaml
data:
  cu3s_file_path: ${oc.env:DATA_ROOT}/Lentils_000.cu3s
```

**With fallback:**
```yaml
data:
  cu3s_file_path: ${oc.env:DATA_ROOT,./data/Lentils}/Lentils_000.cu3s
  # Use $DATA_ROOT if set, otherwise use ./data/Lentils
```

### Computed Values

**Conditional values:**
```yaml
training:
  accelerator: ${oc.env:ACCELERATOR,auto}
  devices: ${oc.decode:"1 if '${training.accelerator}' == 'cpu' else -1"}
```

**Path manipulation:**
```yaml
name: experiment_01
checkpoint_dir: ${output_dir}/checkpoints
latest_checkpoint: ${checkpoint_dir}/last.ckpt
```

### OmegaConf Resolvers

**Built-in resolvers:**

`oc.env` - Environment variable:
```yaml
data_root: ${oc.env:DATA_ROOT}
```

`oc.decode` - Python expression:
```yaml
use_gpu: ${oc.decode:"'cuda' if torch.cuda.is_available() else 'cpu'"}
```

`oc.create` - Create object:
```yaml
timestamp: ${oc.create:datetime.datetime.now}
```

### Cross-Group References

Reference values from other config groups:

```yaml
# In trainrun config
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /training@training: default
  - _self_

# Reference pipeline name
output_dir: ./outputs/${pipeline.metadata.name}

# Reference training seed
experiment_seed: ${training.seed}
```

---

## Override Mechanisms

### 1. Config-Level Overrides

**In trainrun config file:**
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_  # ← Must be last

# Override specific fields
data:
  train_ids: [0, 1, 2]
  batch_size: 16

training:
  optimizer:
    lr: 0.0001
  trainer:
    max_epochs: 100
```

### 2. Command-Line Overrides

**Dot notation:**
```bash
python train.py training.optimizer.lr=0.001
```

**Nested overrides:**
```bash
python train.py \
    training.trainer.max_epochs=100 \
    training.optimizer.lr=0.001 \
    training.optimizer.weight_decay=0.01 \
    data.batch_size=16
```

**List assignment:**
```bash
python train.py data.train_ids=[0,1,2,3]
```

**Dictionary assignment:**
```bash
python train.py training.scheduler={name:reduce_on_plateau,patience:10}
```

### 3. Config Group Selection

**Switch entire config groups:**
```bash
python train.py pipeline=channel_selector
python train.py data=custom_dataset
python train.py training=high_lr
```

### 4. Programmatic Overrides

**In Python code:**
```python
from omegaconf import OmegaConf

@hydra.main(config_path="../configs", config_name="trainrun/default_gradient")
def main(cfg: DictConfig):
    # Override via OmegaConf
    cfg.training.optimizer.lr = 0.0001
    cfg.data.batch_size = 32

    # Or use merge
    overrides = OmegaConf.create({
        "training": {
            "optimizer": {"lr": 0.0001},
            "trainer": {"max_epochs": 100}
        }
    })
    cfg = OmegaConf.merge(cfg, overrides)

    # Convert to dict for usage
    config_dict = OmegaConf.to_container(cfg, resolve=True)
```

---

## Multi-Run Sweeps

### Basic Sweep Syntax

Use `-m` flag to enable multi-run mode:

```bash
python train.py -m training.optimizer.lr=0.001,0.0001,0.00001
```

Hydra creates separate runs:
```
outputs/
├── multirun/
│   └── 2026-02-04/
│       ├── 10-30-00/
│       │   ├── 0/  # lr=0.001
│       │   ├── 1/  # lr=0.0001
│       │   └── 2/  # lr=0.00001
```

### Sweep Multiple Parameters

**Cartesian product:**
```bash
python train.py -m \
    training.optimizer.lr=0.001,0.0001 \
    training.optimizer.weight_decay=0.01,0.001
```

**Creates 4 runs:**
1. lr=0.001, weight_decay=0.01
2. lr=0.001, weight_decay=0.001
3. lr=0.0001, weight_decay=0.01
4. lr=0.0001, weight_decay=0.001

### Sweep Config Groups

```bash
python train.py -m pipeline=rx_statistical,channel_selector,deep_svdd
```

**Or combine:**
```bash
python train.py -m \
    pipeline=rx_statistical,channel_selector \
    training.optimizer.lr=0.001,0.0001
```

### Custom Sweep Configurations

**Base config:** `configs/trainrun/sweep_base.yaml`
```yaml
# @package _global_

defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /data@data: lentils
  - /training@training: default
  - _self_

name: sweep_${pipeline_name}_lr_${training.optimizer.lr}
output_dir: ./outputs/sweeps/${name}
```

**Execute sweep:**
```bash
python train.py \
    --config-name=trainrun/sweep_base \
    -m \
    pipeline_name=rx_statistical,channel_selector \
    training.optimizer.lr=0.001,0.0001,0.00001
```

### Sweep Output Organization

Hydra creates hierarchical directories:

```
outputs/
└── multirun/
    └── 2026-02-04/
        └── 10-30-00/
            ├── .hydra/
            │   ├── config.yaml
            │   ├── hydra.yaml
            │   └── overrides.yaml
            ├── 0/  # First combination
            │   ├── .hydra/
            │   ├── pipeline/
            │   └── trained_models/
            ├── 1/  # Second combination
            └── 2/  # Third combination
```

---

## Advanced Composition Patterns

### Pattern 1: Base + Variants

**Base config:** `configs/trainrun/base_experiment.yaml`
```yaml
# @package _global_

defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /data@data: lentils
  - /training@training: default
  - _self_

name: ${pipeline_name}_experiment
output_dir: ./outputs/${name}

tags:
  dataset: lentils
  method: ${pipeline_name}
```

**Variant configs:**
- `configs/trainrun/rx_experiment.yaml`
- `configs/trainrun/channel_selector_experiment.yaml`

```yaml
# Each variant
defaults:
  - base_experiment
  - _self_

pipeline_name: rx_statistical
```

### Pattern 2: Conditional Composition

**Based on mode:**
```yaml
defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /training/optimizer@training.optimizer: ${optimizer_type}
  - /training/scheduler@training.scheduler: ${scheduler_type}
  - optional /training/callbacks@training.callbacks: ${callbacks_preset}
  - _self_

pipeline_name: rx_statistical
optimizer_type: adamw
scheduler_type: reduce_on_plateau
callbacks_preset: null  # Optional
```

### Pattern 3: Hierarchical Configs

**Directory structure:**
```
configs/
├── pipeline/
│   ├── statistical/
│   │   ├── rx.yaml
│   │   └── lad.yaml
│   └── gradient/
│       ├── channel_selector.yaml
│       └── deep_svdd.yaml
```

**Usage:**
```yaml
defaults:
  - /pipeline@pipeline: statistical/rx
  # or
  - /pipeline@pipeline: gradient/channel_selector
```

### Pattern 4: Config Recipes

**Recipe:** `configs/recipes/fast_prototype.yaml`
```yaml
# @package _global_

defaults:
  - /trainrun@_here_: default
  - _self_

training:
  trainer:
    max_epochs: 3
    fast_dev_run: false

data:
  batch_size: 1
  num_workers: 0

output_dir: ./outputs/quick_test
```

**Usage:**
```bash
python train.py --config-name=recipes/fast_prototype
```

### Pattern 5: Mixin Configs

**Mixin:** `configs/mixins/debug.yaml`
```yaml
# @package training

trainer:
  fast_dev_run: true
  limit_train_batches: 10
  limit_val_batches: 5
  enable_progress_bar: true

optimizer:
  lr: 0.01  # Higher LR for fast debugging
```

**Usage:**
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - /mixins/debug@training:_here_  # Merge debug settings
  - _self_
```

### Pattern 6: Dynamic Experiment Generation

**Generator config:** `configs/experiments/generate.yaml`
```yaml
# @package _global_

defaults:
  - /pipeline@pipeline: ${experiment.pipeline}
  - /data@data: ${experiment.dataset}
  - /training@training: ${experiment.training_preset}
  - _self_

experiment:
  pipeline: rx_statistical
  dataset: lentils
  training_preset: default

name: ${experiment.pipeline}_on_${experiment.dataset}
output_dir: ./outputs/${name}
```

**Sweep different experiments:**
```bash
python train.py \
    --config-name=experiments/generate \
    -m \
    experiment.pipeline=rx_statistical,channel_selector \
    experiment.dataset=lentils,tomatoes
```

---

## Best Practices

### 1. Defaults Ordering

**Always put `_self_` last:**
```yaml
# ✓ Correct
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - _self_

data:
  batch_size: 16  # Overrides work

# ✗ Wrong
defaults:
  - _self_
  - /pipeline/anomaly/rx@pipeline: rx_statistical

data:
  batch_size: 16  # Doesn't override!
```

### 2. Package Directives

**Use `@package _global_` for trainrun configs:**
```yaml
# configs/trainrun/my_experiment.yaml
# @package _global_  # ← Always add this

defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
```

**Avoid mixing package directives** in the same config.

### 3. Clear Variable Names

**Good:**
```yaml
name: ${pipeline.metadata.name}_experiment
output_dir: ./outputs/${name}
checkpoint_path: ${output_dir}/checkpoints
```

**Avoid:**
```yaml
name: ${a}
output_dir: ${b}/${c}
```

### 4. Minimal Overrides

**Only override what you need:**
```yaml
# ✓ Good
defaults:
  - /training@training: default
  - _self_

training:
  optimizer:
    lr: 0.0001  # Only override LR

# ✗ Bad (duplicates entire config)
training:
  seed: 42
  trainer:
    max_epochs: 5
    accelerator: auto
    # ... (all fields repeated)
```

### 5. Document Complex Compositions

```yaml
# Base Experiment Template
#
# This config composes:
# - Pipeline: Specified via pipeline_name variable
# - Data: Lentils dataset with custom splits
# - Training: Default settings with overrideable LR
#
# Usage:
#   python train.py --config-name=base_experiment pipeline_name=rx_statistical

# @package _global_

defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /data@data: lentils
  - /training@training: default
  - _self_
```

### 6. Validate Interpolations

**Check for typos:**
```yaml
# ✓ Correct
output_dir: ./outputs/${name}

# ✗ Typo
output_dir: ./outputs/${nmae}  # Will error at runtime
```

**Use resolve=True when converting:**
```python
config_dict = OmegaConf.to_container(cfg, resolve=True)
```

---

## Troubleshooting

### Missing Key Error

**Problem:** `KeyError: 'pipeline'`

**Solution:** Check defaults list and package directives:
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical  # ← Ensure @pipeline part
```

### Interpolation Error

**Problem:** `InterpolationResolutionError: Could not resolve ${name}`

**Solution:** Ensure referenced key exists:
```yaml
name: my_experiment  # ← Must be defined
output_dir: ./outputs/${name}
```

### Override Not Applied

**Problem:** Override in config doesn't work.

**Solution:** Ensure `_self_` is last:
```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - _self_  # ← MUST BE LAST

# Overrides below
pipeline:
  nodes: [...]
```

### Package Directive Confusion

**Problem:** Config appears at wrong level in hierarchy.

**Solution:** Check package directive:
```yaml
# For trainrun configs, use:
# @package _global_

# For group-specific configs, use:
# @package training
# or
# @package data
```

### Circular Dependency

**Problem:** `CircularReferenceError`

**Solution:** Avoid circular references:
```yaml
# ✗ Circular
a: ${b}
b: ${a}

# ✓ Fixed
a: base_value
b: ${a}_extended
```

---

## See Also

- **Configuration Guides**:
  - [Config Groups](config-groups.md) - Organizing configuration groups
  - [TrainRun Schema](trainrun-schema.md) - Complete trainrun reference
  - [Pipeline Schema](pipeline-schema.md) - Pipeline YAML structure
- **User Guide**:
  - [Configuration Overview](../user-guide/configuration.md) - Configuration system overview
- **External Resources**:
  - [Hydra Documentation](https://hydra.cc/) - Official Hydra docs
  - [OmegaConf Documentation](https://omegaconf.readthedocs.io/) - OmegaConf reference
- **Examples**:
  - `configs/trainrun/` - Example trainrun compositions
  - `examples/rx_statistical.py` - Using Hydra in code
