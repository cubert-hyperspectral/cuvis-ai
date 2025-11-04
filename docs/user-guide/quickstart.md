# Quickstart

Get started with CUVIS.AI in 5 minutes! This guide walks you through building and training your first hyperspectral analysis pipeline.

## Prerequisites

- CUVIS.AI installed (see [Installation](installation.md))
- Python 3.10+
- Basic familiarity with PyTorch

## Your First Pipeline

### 1. Import Required Components

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.training.config import TrainingConfig, TrainerConfig
```

### 2. Build the Graph

Create a simple pipeline with normalization and anomaly detection:

```python
# Create graph
graph = Graph("rx_quickstart")

# Add normalization node
normalizer = MinMaxNormalizer(eps=1e-6)
graph.add_node(normalizer)

# Add RX anomaly detector
rx = RXGlobal(eps=1e-6, trainable_stats=False)
graph.add_node(rx, parent=normalizer)
```

### 3. Prepare Data

Load the example dataset:

```python
# Instantiate datamodule
datamodule = LentilsAnomaly(
    data_dir="./data/Lentils",
    batch_size=4,
    num_workers=0
)
```

!!! tip "Downloading Data"
    The LentilsAnomaly datamodule will automatically download the dataset on first use (~200MB).

### 4. Configure Training

Set up training parameters:

```python
# Configure training (statistical initialization only)
config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(
        max_epochs=0,  # 0 = statistical initialization only
        accelerator="auto",
        devices=1,
    )
)
```

### 5. Train the Model

Run statistical initialization:

```python
# Train - Phase 1 (statistical initialization)
trainer = graph.train(
    datamodule=datamodule,
    training_config=config
)
```

Expected output:
```
[INFO] Initializing statistical nodes...
[INFO] Initializing MinMaxNormalizer...
[INFO] Initialized with running_min and running_max
[INFO] Initializing RXGlobal...
[INFO] Initialized with mu (mean) and cov (covariance)
[INFO] Statistical initialization complete
```

### 6. Use the Model

Forward pass through the trained pipeline:

```python
import torch

# Get a batch from the datamodule
datamodule.setup("test")
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

# Forward pass
with torch.no_grad():
    output, _, _ = graph(batch["cube"])

print(f"RX anomaly scores shape: {output.shape}")
print(f"Mean score: {output.mean():.4f}")
print(f"Anomalies (score > threshold): {(output > 5.0).float().mean():.2%}")
```

## Complete Example

Here's the complete script:

```python
"""Quickstart example for CUVIS.AI"""

from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.training.config import TrainingConfig, TrainerConfig
import torch

# Build graph
graph = Graph("rx_quickstart")
normalizer = MinMaxNormalizer(eps=1e-6)
rx = RXGlobal(eps=1e-6, trainable_stats=False)
graph.add_node(normalizer)
graph.add_node(rx, parent=normalizer)

# Prepare data
datamodule = LentilsAnomaly(
    data_dir="./data/Lentils",
    batch_size=4,
    num_workers=0
)

# Configure training
config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(max_epochs=0, accelerator="auto", devices=1)
)

# Train
trainer = graph.train(datamodule=datamodule, training_config=config)

# Test forward pass
datamodule.setup("test")
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

with torch.no_grad():
    output, _, _ = graph(batch["cube"])

print(f"✓ Pipeline trained successfully!")
print(f"✓ RX scores shape: {output.shape}")
print(f"✓ Detected {(output > 5.0).float().mean():.2%} anomalies")
```

## Running with Hydra Configuration

For reproducible experiments, use Hydra configuration:

1. **Create config file** (`config.yaml`):
```yaml
graph:
  name: rx_quickstart

nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
    eps: 1.0e-6
  rx:
    _target_: cuvis_ai.anomaly.rx_detector.RXGlobal
    eps: 1.0e-6
    trainable_stats: false

datamodule:
  _target_: cuvis_ai.data.lentils_anomaly.LentilsAnomaly
  data_dir: ./data/Lentils
  batch_size: 4
  num_workers: 0

training:
  seed: 42
  trainer:
    max_epochs: 0
    accelerator: auto
    devices: 1
```

2. **Create training script**:
```python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import TrainingConfig

@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Build graph from config
    graph = Graph(cfg.graph.name)
    normalizer = instantiate(cfg.nodes.normalizer)
    rx = instantiate(cfg.nodes.rx)
    graph.add_node(normalizer)
    graph.add_node(rx, parent=normalizer)
    
    # Instantiate datamodule
    datamodule = instantiate(cfg.datamodule)
    
    # Parse training config
    training_cfg = TrainingConfig.from_dict_config(cfg.training)
    
    # Train
    graph.train(datamodule=datamodule, training_config=training_cfg)

if __name__ == "__main__":
    main()
```

3. **Run with CLI overrides**:
```bash
python train.py training.trainer.devices=2 datamodule.batch_size=8
```

## Next Steps

Now that you have a working pipeline:

- **[Phase 1 Tutorial](../tutorials/phase1_statistical.md)**: Learn about statistical initialization
- **[Phase 3 Tutorial](../tutorials/phase3_gradient_training.md)**: Add gradient-based training
- **[Configuration Guide](configuration.md)**: Deep dive into Hydra configuration
- **[API Reference](../api/pipeline.md)**: Explore all available components

## Common Patterns

### Adding Visualization

```python
from cuvis_ai.training.visualizations import AnomalyHeatmap
from cuvis_ai.training.monitors import DummyMonitor

# Add visualization leaf
heatmap = AnomalyHeatmap(log_frequency=1)
graph.add_leaf_node(heatmap, parent=rx)

# Register monitor
monitor = DummyMonitor(output_dir="./outputs/artifacts")
graph.register_monitor(monitor)
```

### Enabling GPU Training

```python
config = TrainingConfig(
    trainer=TrainerConfig(
        max_epochs=10,
        accelerator="gpu",  # or "cuda"
        devices=1,
        precision="16-mixed"  # Mixed precision for faster training
    )
)
```

### Saving and Loading Models

```python
# Save
graph.save_to_file("model.zip")

# Load
from cuvis_ai.pipeline.graph import Graph
loaded_graph = Graph.load_from_file("model.zip")
```

## Troubleshooting

### Dataset Download Issues

If automatic download fails, manually download from the Lentils dataset repository and extract to `./data/Lentils/`.

### Out of Memory

Reduce batch size:
```python
datamodule = LentilsAnomaly(batch_size=2)  # Reduce from 4 to 2
```

### Slow Training

Enable multiple workers:
```python
datamodule = LentilsAnomaly(num_workers=4)  # Use 4 CPU cores
```

---

!!! success "Congratulations!"
    You've successfully built and trained your first CUVIS.AI pipeline! Explore the tutorials to learn more advanced features.
