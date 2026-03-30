!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Sweeps & Advanced Patterns

Multi-run sweeps for hyperparameter search and advanced composition patterns.

## Multi-Run Sweeps

### Basic Sweep Syntax

Use `-m` flag to enable multi-run mode:

```bash
python train.py -m training.optimizer.lr=0.001,0.0001,0.00001
```

Hydra creates separate runs:
```
outputs/multirun/2026-02-04/10-30-00/
├── 0/  # lr=0.001
├── 1/  # lr=0.0001
└── 2/  # lr=0.00001
```

### Sweep Multiple Parameters

**Cartesian product:**
```bash
python train.py -m \
    training.optimizer.lr=0.001,0.0001 \
    training.optimizer.weight_decay=0.01,0.001
```

Creates 4 runs (2 x 2).

### Sweep Config Groups

```bash
python train.py -m pipeline=rx_statistical,channel_selector,deep_svdd

# Combine with parameters:
python train.py -m \
    pipeline=rx_statistical,channel_selector \
    training.optimizer.lr=0.001,0.0001
```

### Custom Sweep Configurations

```yaml
# configs/trainrun/sweep_base.yaml
# @package _global_

defaults:
  - /pipeline@pipeline: ${pipeline_name}
  - /data@data: lentils
  - /training@training: default
  - _self_

name: sweep_${pipeline_name}_lr_${training.optimizer.lr}
output_dir: ./outputs/sweeps/${name}
```

```bash
python train.py \
    --config-name=trainrun/sweep_base \
    -m \
    pipeline_name=rx_statistical,channel_selector \
    training.optimizer.lr=0.001,0.0001,0.00001
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
```

**Variant configs** inherit and set `pipeline_name`:
```yaml
defaults:
  - base_experiment
  - _self_

pipeline_name: rx_statistical
```

### Pattern 2: Conditional Composition

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

### Pattern 3: Config Recipes

```yaml
# configs/recipes/fast_prototype.yaml
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

```bash
python train.py --config-name=recipes/fast_prototype
```

### Pattern 4: Mixin Configs

```yaml
# configs/mixins/debug.yaml
# @package training

trainer:
  fast_dev_run: true
  limit_train_batches: 10
  enable_progress_bar: true
optimizer:
  lr: 0.01  # Higher LR for fast debugging
```

```yaml
# Usage in trainrun
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - /mixins/debug@training:_here_  # Merge debug settings
  - _self_
```

### Pattern 5: Dynamic Experiment Generation

```yaml
# configs/experiments/generate.yaml
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
```

```bash
python train.py \
    --config-name=experiments/generate \
    -m \
    experiment.pipeline=rx_statistical,channel_selector \
    experiment.dataset=lentils,tomatoes
```

---

## Best Practices

1. **Always put `_self_` last** in defaults list
2. **Use `@package _global_`** for trainrun configs
3. **Use clear variable names** — `${pipeline.metadata.name}` not `${a}`
4. **Only override what you need** — don't duplicate entire configs
5. **Document complex compositions** with header comments
6. **Use `resolve=True`** when converting to dict: `OmegaConf.to_container(cfg, resolve=True)`

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `KeyError: 'pipeline'` | Missing `@pipeline` in defaults | Add `@pipeline` after path |
| `InterpolationResolutionError` | Referenced key doesn't exist | Define the key before referencing |
| Override not applied | `_self_` not last | Move `_self_` to end of defaults |
| Config at wrong level | Wrong package directive | Use `@package _global_` for trainruns |
| `CircularReferenceError` | Circular interpolation | Break the cycle: `a: ${b}` + `b: ${a}` |

---

## See Also

- [Hydra Basics](hydra-basics.md) — package directives, defaults list
- [Hydra Inheritance & Overrides](hydra-inheritance.md) — config inheritance, interpolation
- [Hydra CLI & Debugging](hydra-advanced.md) — CLI tips, structured configs
- [Hydra Documentation](https://hydra.cc/) — official Hydra docs
- [OmegaConf Documentation](https://omegaconf.readthedocs.io/) — OmegaConf reference
