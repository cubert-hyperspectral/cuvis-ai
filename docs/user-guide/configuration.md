# Configuration

CUVIS.AI uses [Hydra](https://hydra.cc/) for flexible, reproducible configuration management. This guide covers configuration patterns and best practices.

## Configuration Structure

Configuration files are located in `cuvis_ai/conf/` and use YAML format with Hydra's composition system.

### Basic Structure

```yaml
# Defaults - compose other configs
defaults:
  - general
  - _self_

# Graph configuration
graph:
  name: my_pipeline

# Node configurations
nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
    eps: 1.0e-6

# Datamodule configuration  
datamodule:
  _target_: cuvis_ai.data.lentils_anomaly.LentilsAnomaly
  data_dir: ${oc.env:DATA_ROOT,./data/Lentils}
  batch_size: 4

# Training configuration
training:
  seed: 42
  trainer:
    max_epochs: 10
    accelerator: auto
  optimizer:
    name: adam
    lr: 0.001
```

## Configuration Files

### Available Configs

| File | Purpose |
|------|---------|
| `general.yaml` | Dataloader and processing settings |
| `anomaly_rx.yaml` | RX detector configuration with composition |
| `wandb.yaml` | WandB monitoring configuration |
| `train_phase1.yaml` | Phase 1 (statistical) training |
| `train_phase2.yaml` | Phase 2 (visualization) training |
| `train_phase3.yaml` | Phase 3 (gradient) training |
| `train_phase4.yaml` | Phase 4 (selector) training |

### Composition with `defaults`

Hydra allows composing multiple configs:

```yaml
defaults:
  - general          # Load general.yaml
  - wandb            # Load wandb.yaml  
  - _self_           # This config overrides above
```

## CLI Overrides

Override any configuration parameter from the command line:

```bash
# Override single parameter
python train.py training.trainer.max_epochs=20

# Override multiple parameters
python train.py \
    training.trainer.max_epochs=20 \
    training.optimizer.lr=0.0001 \
    datamodule.batch_size=8

# Override nested parameters
python train.py nodes.pca.n_components=5
```

## Environment Variables

Use environment variables in configs:

```yaml
datamodule:
  data_dir: ${oc.env:DATA_ROOT,./data/Lentils}  # $DATA_ROOT or default
  
wandb:
  api_key: ${oc.env:WANDB_API_KEY}  # Required env var
```

## TrainingConfig

The `TrainingConfig` dataclass wraps all training parameters:

```python
from cuvis_ai.training.config import TrainingConfig, TrainerConfig, OptimizerConfig

config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(
        max_epochs=10,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=10,
    ),
    optimizer=OptimizerConfig(
        name="adam",
        lr=0.001,
        weight_decay=0.0,
        betas=(0.9, 0.999),
    ),
    monitor_plugins=[]
)
```

### Trainer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_epochs` | int | 10 | Number of training epochs |
| `accelerator` | str | "auto" | Device: "auto", "cpu", "gpu", "cuda" |
| `devices` | int | 1 | Number of devices to use |
| `precision` | str | "32" | Precision: "32", "16-mixed", "bf16-mixed" |
| `log_every_n_steps` | int | 50 | Logging frequency |
| `gradient_clip_val` | float\|None | None | Gradient clipping threshold |

### Optimizer Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | str | "adam" | Optimizer: "adam", "sgd", "adamw" |
| `lr` | float | 0.001 | Learning rate |
| `weight_decay` | float | 0.0 | L2 regularization |
| `betas` | tuple | (0.9, 0.999) | Adam betas |
| `momentum` | float | 0.9 | SGD momentum |

## Configuration Recipes

### Development (Fast Iteration)

```yaml
training:
  seed: 42
  trainer:
    max_epochs: 2
    accelerator: cpu
    devices: 1
    fast_dev_run: false
    limit_train_batches: 0.1
    limit_val_batches: 0.1

datamodule:
  batch_size: 2
  num_workers: 0
```

### Production (Full Training)

```yaml
training:
  seed: 42
  trainer:
    max_epochs: 50
    accelerator: gpu
    devices: 1
    precision: "16-mixed"
    log_every_n_steps: 10
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01

datamodule:
  batch_size: 16
  num_workers: 4
```

### Multi-GPU Training

```yaml
training:
  trainer:
    accelerator: gpu
    devices: 4
    strategy: ddp
    precision: "16-mixed"
```

## Configuration Validation

Hydra validates configurations at runtime:

```python
from omegaconf import OmegaConf
from cuvis_ai.training.config import TrainingConfig

# Load and validate
cfg = OmegaConf.load("config.yaml")
training_cfg = TrainingConfig.from_dict_config(cfg.training)
```

## Best Practices

### 1. Use Composition

Break large configs into reusable pieces:

```yaml
# base.yaml
defaults:
  - general
  - monitoring/wandb

# experiment.yaml  
defaults:
  - base
  - _self_
  
training:
  trainer:
    max_epochs: 100
```

### 2. Version Control Configs

Commit configuration files for reproducibility:

```bash
git add cuvis_ai/conf/
git commit -m "Add experiment config"
```

### 3. Use Structured Configs

Define configs as dataclasses for type safety:

```python
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

@dataclass
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 3

cs = ConfigStore.instance()
cs.store(name="model_config", node=ModelConfig)
```

### 4. Document Custom Configs

Add comments explaining parameters:

```yaml
nodes:
  pca:
    n_components: 3  # Number of principal components to retain
    trainable: true  # Enable gradient-based fine-tuning
```

## Troubleshooting

### Config Not Found

```
ConfigFileNotFoundError: Cannot find 'my_config.yaml'
```

**Solution**: Ensure config file is in correct directory and path is specified correctly.

### Override Parse Error

```
OverrideParseException: Expected '=' in override 'training.trainer.max_epochs:10'
```

**Solution**: Use `=` not `:` for overrides: `training.trainer.max_epochs=10`

### Type Mismatch

```
ValidationError: Field 'max_epochs' expected type 'int', got 'str'
```

**Solution**: Ensure correct type in override: `max_epochs=10` not `max_epochs="10"`

## Next Steps

- **[Quickstart](quickstart.md)**: See configuration in action
- **[Tutorials](../tutorials/phase1_statistical.md)**: Phase-specific configurations
- **[API Reference](../api/training.md)**: TrainingConfig API details
