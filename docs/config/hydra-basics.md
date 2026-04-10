!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Composition Basics

Hydra enables powerful configuration composition for CUVIS.AI.

## Overview

Key capabilities:

- **Defaults List**: Compose multiple configs into one
- **Package Directives**: Control config placement in hierarchy
- **Inheritance**: Reuse and extend base configurations
- **Variable Interpolation**: Reference and compute values dynamically
- **Multi-Run Sweeps**: Hyperparameter optimization and grid search
- **Command-Line Overrides**: Runtime configuration changes

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

Merges config at the root level. **Most common for trainrun configs:**
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

Merges config under the group name:
```yaml
# @package training

defaults:
  - /optimizer: adamw
  - /scheduler: reduce_on_plateau

seed: 42
```

### Explicit Package Paths

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

```yaml
defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /training/scheduler@training.scheduler: ${scheduler_name}
  - optional /augmentation@data.augmentation: ${augmentation}
```

---

## See Also

- [Hydra Inheritance & Overrides](hydra-inheritance.md) — config inheritance, interpolation, overrides
- [Hydra Sweeps & Advanced](hydra-sweeps.md) — multi-run sweeps, composition patterns
