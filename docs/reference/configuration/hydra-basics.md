!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Composition Basics

Hydra enables powerful configuration composition for Cuvis.AI.

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
  - /data@data: lentils
  - /training@training: default
  - _self_

name: my_experiment
# Pipeline reference, resolved relative to this trainrun's directory.
pipeline: ../pipeline/anomaly/rx/channel_selector.yaml
output_dir: ./outputs/${name}
```

The data and training settings are config groups; the pipeline is not. A trainrun names its
pipeline yaml by path, and `TrainRunConfig` refuses a pipeline composed into the config.

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
  - /data@data: lentils
  - _self_

name: experiment_1
```

**Result:**
```yaml
data:
  data_module: cu3s
  splits: {...}
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
  - /data@data: lentils                      # → data: {...}
  - /training@training: default              # → training: {...}
```

**Path format:** `/source_group@target_key: config_name`

The pipeline does not go through the defaults list: `pipeline:` is the path of the pipeline yaml.

---

## Defaults List

### Structure and Ordering

```yaml
# @package _global_

defaults:
  - /data@data: lentils                  # Load first
  - /training@training: default          # Load second
  - _self_                               # THIS CONFIG (must be last)

name: my_experiment
pipeline: ../pipeline/anomaly/rx/channel_selector.yaml

# Overrides below (only applied because _self_ is last)
data:
  batch_size: 16
```

**Critical rule:** `_self_` must be last to allow overrides in the current file.

### Absolute vs Relative Paths

**Absolute paths** (start with `/`):
```yaml
defaults:
  - /data@data: lentils
  # Searches: cuvis_ai/configs/data/lentils.yaml
```

**Relative paths** (no leading `/`):
```yaml
# In cuvis_ai/configs/training/high_lr.yaml
defaults:
  - base_optimizer  # Searches: cuvis_ai/configs/training/base_optimizer.yaml
  - _self_
```

### Conditional Defaults

```yaml
defaults:
  - /training/scheduler@training.scheduler: ${scheduler_name}
  - optional /augmentation@data.augmentation: ${augmentation}
```

The pipeline cannot be chosen this way: it is a path, not a config group. Give each pipeline its
own trainrun, as the packaged ones in `cuvis_ai/configs/trainrun/` do.

---

## See Also

- [Hydra Inheritance & Overrides](hydra-inheritance.md) — config inheritance, interpolation, overrides
- [Hydra Sweeps & Advanced](hydra-sweeps.md) — multi-run sweeps, composition patterns
