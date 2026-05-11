!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Hydra Inheritance & Overrides

Config inheritance, variable interpolation, and override mechanisms.

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

```yaml
# Level 1: configs/training/base_optimizer.yaml
optimizer:
  name: adamw
  betas: [0.9, 0.999]

# Level 2: configs/training/base_training.yaml
defaults:
  - base_optimizer
  - _self_
seed: 42
trainer:
  max_epochs: 5

# Level 3: configs/training/custom_training.yaml
defaults:
  - base_training
  - _self_
optimizer:
  lr: 0.001
trainer:
  max_epochs: 100  # Override base
```

### Override Behavior

Hydra uses **merge** strategy by default — only explicitly set fields are overridden:

```yaml
# Base
data:
  train_ids: [0, 2, 3]
  val_ids: [1, 5]
  batch_size: 2

# Override (only batch_size changes)
defaults:
  - base
  - _self_
data:
  batch_size: 16  # train_ids and val_ids unchanged
```

---

## Variable Interpolation

### Simple Interpolation

```yaml
name: my_experiment
output_dir: ./outputs/${name}
# Resolves to: ./outputs/my_experiment

training:
  trainer:
    default_root_dir: ${output_dir}
```

### Environment Variables

```yaml
data:
  cu3s_file_path: ${oc.env:DATA_ROOT}/Lentils_000.cu3s

  # With fallback:
  cu3s_file_path: ${oc.env:DATA_ROOT,./data/Lentils}/Lentils_000.cu3s
```

### OmegaConf Resolvers

| Resolver | Purpose | Example |
|----------|---------|---------|
| `oc.env` | Environment variable | `${oc.env:DATA_ROOT}` |
| `oc.decode` | Python expression | `${oc.decode:"1 if x else 0"}` |
| `oc.create` | Create object | `${oc.create:datetime.datetime.now}` |

### Cross-Group References

```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /training@training: default
  - _self_

output_dir: ./outputs/${pipeline.metadata.name}
experiment_seed: ${training.seed}
```

---

## Override Mechanisms

### 1. Config-Level Overrides

```yaml
defaults:
  - /pipeline/anomaly/rx@pipeline: rx_statistical
  - /data@data: lentils
  - /training@training: default
  - _self_  # ← Must be last

data:
  train_ids: [0, 1, 2]
  batch_size: 16
training:
  optimizer:
    lr: 0.0001
```

### 2. Command-Line Overrides

```bash
python train.py training.optimizer.lr=0.001

python train.py \
    training.trainer.max_epochs=100 \
    training.optimizer.lr=0.001 \
    data.batch_size=16

# List/dict assignment
python train.py data.train_ids=[0,1,2,3]
python train.py training.scheduler={name:reduce_on_plateau,patience:10}
```

### 3. Config Group Selection

```bash
python train.py pipeline=channel_selector
python train.py data=custom_dataset
python train.py training=high_lr
```

### 4. Programmatic Overrides

```python
from omegaconf import OmegaConf

@hydra.main(config_path="../configs", config_name="trainrun/default_gradient")
def main(cfg: DictConfig):
    cfg.training.optimizer.lr = 0.0001

    overrides = OmegaConf.create({
        "training": {"optimizer": {"lr": 0.0001}, "trainer": {"max_epochs": 100}}
    })
    cfg = OmegaConf.merge(cfg, overrides)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
```

---

## See Also

- [Hydra Basics](hydra-basics.md) — package directives, defaults list
- [Hydra Sweeps & Advanced](hydra-sweeps.md) — multi-run sweeps, composition patterns
