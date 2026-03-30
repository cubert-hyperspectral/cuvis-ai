!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Advanced Patterns

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
  - [Hydra Composition Patterns](hydra-composition.md) - Core composition patterns
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
