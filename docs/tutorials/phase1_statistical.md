# Phase 1: Statistical Training

This tutorial covers Phase 1 of the CUVIS.AI training pipeline: statistical initialization. You'll learn how to bootstrap models using efficient non-parametric methods that provide a strong baseline without any gradient computation.

## Overview

Phase 1 focuses on initializing models using statistical methods:

- **RX Detector**: Computes mean vector (μ) and covariance matrix (Σ) from training data
- **MinMaxNormalizer**: Tracks global min/max statistics for normalization
- **No Gradients**: All initialization is done via closed-form solutions
- **Fast**: Typically completes in seconds to minutes

This phase is essential because it:

1. Provides excellent initialization for gradient training
2. Often produces good results without any gradient updates
3. Ensures numerical stability (normalized inputs, well-conditioned covariance)
4. Enables freezing statistical nodes while training downstream layers

## What You'll Learn

- How statistical initialization works
- Building a graph with statistical nodes
- Running Phase 1 training (max_epochs=0)
- Understanding initialization outputs
- Model serialization and loading
- Using Hydra configuration for reproducibility

## Prerequisites

- CUVIS.AI installed ([Installation Guide](../user-guide/installation.md))
- Basic understanding of hyperspectral data
- Familiarity with the [Quickstart](../user-guide/quickstart.md)

## Concepts

### Statistical Initialization

Statistical nodes implement two key methods:

```python
class RXGlobal(Node):
    @property
    def requires_initial_fit(self) -> bool:
        return True
    
    def initialize_from_data(self, data_iterator):
        """Accumulate statistics from data stream"""
        for x, y, m in data_iterator:
            self.update(x)  # Accumulate statistics
        self.finalize()     # Compute final parameters
```

When `graph.train()` is called with `max_epochs=0`, the following happens:

1. Graph identifies nodes with `requires_initial_fit=True`
2. Nodes are topologically sorted (respecting dependencies)
3. Each node receives a transformed data stream from its parents
4. Node computes statistics (e.g., mean, covariance) in one pass
5. Node stores results as buffers (non-trainable) or parameters (trainable)

### Two-Phase Training

CUVIS.AI supports two training phases:

- **Phase 1 (Statistical)**: `max_epochs=0` → Initialize with statistics only
- **Phase 2 (Gradient)**: `max_epochs > 0` → Optional gradient-based fine-tuning

This tutorial focuses on Phase 1. Phase 3 covers gradient training.

## Tutorial: Building an RX Anomaly Detector

### Step 1: Imports

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.training.config import TrainingConfig, TrainerConfig
import torch
```

### Step 2: Build the Graph

Create a pipeline with normalization → RX detector:

```python
# Create graph
graph = Graph("rx_statistical_baseline")

# Add normalization (will compute running min/max)
normalizer = MinMaxNormalizer(
    eps=1e-6,                    # Small constant for numerical stability
    use_running_stats=True       # Enable statistical initialization
)
graph.add_node(normalizer)

# Add RX anomaly detector (will compute mean and covariance)
rx = RXGlobal(
    eps=1e-6,                    # Small constant for covariance regularization
    trainable_stats=False        # Keep statistics frozen (not trainable)
)
graph.add_node(rx, parent=normalizer)
```

**Key Points:**

- `normalizer` is the root node (no parent)
- `rx` receives normalized data from `normalizer` (parent=normalizer)
- Both nodes have `requires_initial_fit=True`
- Graph will initialize them in topological order: normalizer → rx

### Step 3: Prepare Data

Instantiate the datamodule:

```python
# Create datamodule
datamodule = LentilsAnomaly(
    data_dir="./data/Lentils",   # Dataset directory
    batch_size=4,                # Batch size for statistics computation
    num_workers=0,               # CPU workers for data loading
)
```

!!! tip "Dataset Download"
    The LentilsAnomaly dataset (~200MB) will be automatically downloaded on first use. You can specify a different directory with the `data_dir` parameter.

### Step 4: Configure Training

Configure Phase 1 (statistical initialization only):

```python
# Training configuration
config = TrainingConfig(
    seed=42,                     # Random seed for reproducibility
    trainer=TrainerConfig(
        max_epochs=0,            # 0 = statistical init only (no gradient training)
        accelerator="auto",      # Use GPU if available, else CPU
        devices=1,               # Number of devices
    )
)
```

**Important:** `max_epochs=0` tells the graph to run statistical initialization only, without any gradient-based training.

### Step 5: Run Training

Execute Phase 1:

```python
# Train - Phase 1 (statistical initialization)
trainer = graph.train(
    datamodule=datamodule,
    training_config=config
)
```

**Expected Output:**

```
[INFO] Starting training with config: TrainingConfig(seed=42, ...)
[INFO] Initializing statistical nodes...
[INFO] Found 2 nodes requiring initialization: MinMaxNormalizer, RXGlobal
[INFO] Initializing MinMaxNormalizer...
[INFO]   Processing 270 batches...
[INFO]   Initialized with running_min: shape torch.Size([61])
[INFO]   Initialized with running_max: shape torch.Size([61])
[INFO] Initializing RXGlobal...
[INFO]   Processing 270 batches...
[INFO]   Initialized with mu (mean): shape torch.Size([61])
[INFO]   Initialized with cov (covariance): shape torch.Size([61, 61])
[INFO] Statistical initialization complete in 3.2 seconds
```

### Step 6: Inspect Results

Examine the initialized parameters:

```python
# MinMaxNormalizer statistics
print(f"Normalizer min: {normalizer.running_min.shape}")  # torch.Size([61])
print(f"Normalizer max: {normalizer.running_max.shape}")  # torch.Size([61])
print(f"Min values: {normalizer.running_min[:5]}")         # First 5 channels

# RX detector statistics
print(f"RX mean (μ): {rx.mu.shape}")                      # torch.Size([61])
print(f"RX covariance (Σ): {rx.cov.shape}")               # torch.Size([61, 61])
print(f"Covariance diagonal: {torch.diag(rx.cov)[:5]}")   # First 5 variances
```

### Step 7: Forward Pass

Use the initialized model for inference:

```python
# Get a test batch
datamodule.setup("test")
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

# Forward pass
with torch.no_grad():
    scores, _, _ = graph(batch["cube"])

# Analyze results
print(f"Anomaly scores shape: {scores.shape}")            # [B, H, W]
print(f"Mean score: {scores.mean():.4f}")                 # ~3-5 typical
print(f"Score range: [{scores.min():.2f}, {scores.max():.2f}]")

# Apply threshold (mean + 2σ)
threshold = scores.mean() + 2 * scores.std()
anomalies = (scores > threshold).float()
anomaly_rate = anomalies.mean().item()
print(f"Anomaly rate: {anomaly_rate:.2%}")                # ~0.5-2% typical
```

### Step 8: Save the Model

Serialize the trained graph:

```python
# Save entire graph (structure + hyperparameters + trained weights)
graph.save_to_file("rx_statistical.zip")

# Later: Load the graph
from cuvis_ai.pipeline.graph import Graph
loaded_graph = Graph.load_from_file("rx_statistical.zip")

# Verify it works
with torch.no_grad():
    loaded_scores, _, _ = loaded_graph(batch["cube"])
assert torch.allclose(scores, loaded_scores), "Loaded model mismatch!"
```

## Hydra Configuration Approach

For reproducible experiments, use Hydra configuration instead of programmatic setup.

### Configuration File

Create `my_phase1_config.yaml`:

```yaml
defaults:
  - general           # Load general.yaml (dataloaders, etc.)
  - _self_            # This config overrides above

graph:
  name: rx_statistical_baseline

nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
    eps: 1.0e-6
    use_running_stats: true
  
  rx:
    _target_: cuvis_ai.anomaly.rx_detector.RXGlobal
    eps: 1.0e-6
    trainable_stats: false

datamodule:
  _target_: cuvis_ai.data.lentils_anomaly.LentilsAnomaly
  data_dir: ${oc.env:DATA_ROOT,./data/Lentils}
  batch_size: 4
  num_workers: 0

training:
  seed: 42
  trainer:
    max_epochs: 0          # Phase 1: statistical only
    accelerator: auto
    devices: 1
  optimizer:
    name: adam
    lr: 0.001
```

### Training Script

Create `train_phase1.py`:

```python
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.training.config import TrainingConfig

@hydra.main(config_path=".", config_name="my_phase1_config", version_base=None)
def main(cfg: DictConfig):
    # Build graph from config
    graph = Graph(cfg.graph.name)
    
    # Instantiate nodes
    normalizer = instantiate(cfg.nodes.normalizer)
    rx = instantiate(cfg.nodes.rx)
    
    # Add nodes to graph
    graph.add_node(normalizer)
    graph.add_node(rx, parent=normalizer)
    
    # Instantiate datamodule
    datamodule = instantiate(cfg.datamodule)
    
    # Parse training config
    training_cfg = TrainingConfig.from_dict_config(cfg.training)
    
    # Train
    trainer = graph.train(
        datamodule=datamodule,
        training_config=training_cfg
    )
    
    print("✓ Training complete!")

if __name__ == "__main__":
    main()
```

### Run with CLI Overrides

```bash
# Use defaults
python train_phase1.py

# Override batch size and device
python train_phase1.py \
    datamodule.batch_size=8 \
    training.trainer.devices=2

# Change dataset directory
python train_phase1.py \
    datamodule.data_dir=/path/to/custom/dataset
```

## Understanding the Outputs

### MinMaxNormalizer

After initialization, the normalizer has:

- `running_min`: Global minimum for each channel (shape: [C])
- `running_max`: Global maximum for each channel (shape: [C])

Normalization formula:

```
x_normalized = (x - running_min) / (running_max - running_min + eps)
```

This scales all channels to [0, 1] range.

### RXGlobal

After initialization, the RX detector has:

- `mu`: Mean vector (shape: [C])
- `cov`: Covariance matrix (shape: [C, C])

RX anomaly score formula:

```
score(x) = (x - μ)ᵀ Σ⁻¹ (x - μ)
```

This is the Mahalanobis distance from the mean, measuring how "unusual" a pixel is.

**Typical Values:**

- Mean score: 3-5 (pixels close to training distribution)
- High scores: >10 (potential anomalies)
- Threshold: Often mean + 2σ or mean + 3σ

## Common Patterns

### Adding PCA After RX

```python
from cuvis_ai.node.pca import TrainablePCA

# Add PCA for dimensionality reduction
pca = TrainablePCA(
    n_components=3,
    trainable=False  # Keep statistical initialization
)
graph.add_node(pca, parent=normalizer)  # Before RX

# Now RX receives 3 principal components instead of 61 channels
```

### Trainable vs Frozen Statistics

```python
# Frozen: Statistics remain fixed after initialization
rx_frozen = RXGlobal(trainable_stats=False)

# Trainable: Statistics converted to nn.Parameter for gradient training
rx_trainable = RXGlobal(trainable_stats=True)
```

If `trainable_stats=True`, you can fine-tune the mean and covariance with gradients in Phase 2 (covered in Phase 3 tutorial).

### Multiple Statistical Nodes

```python
# Complex pipeline with multiple statistical nodes
graph.add_node(normalizer)                      # Level 0
graph.add_node(pca, parent=normalizer)          # Level 1  
graph.add_node(rx, parent=pca)                  # Level 2

# Initialization order: normalizer → pca → rx (topologically sorted)
```

## Troubleshooting

### Singular Covariance Matrix

**Problem:** RX detector fails with "Matrix is singular" error.

**Solution:**

1. Increase `eps` parameter: `RXGlobal(eps=1e-4)`
2. Add PCA before RX to reduce dimensions
3. Ensure sufficient training samples (need > C samples for C channels)

### Normalization Produces NaN

**Problem:** Forward pass produces NaN values.

**Solution:**

1. Check for constant channels: `running_max == running_min`
2. Increase `eps` parameter: `MinMaxNormalizer(eps=1e-5)`
3. Remove or fix problematic channels in preprocessing

### Out of Memory

**Problem:** Covariance computation runs out of memory.

**Solution:**

1. Reduce batch size: `datamodule = LentilsAnomaly(batch_size=2)`
2. Use incremental covariance updates (already implemented in RXGlobal)
3. Reduce number of channels with PCA or selector

### Wrong Initialization Order

**Problem:** Nodes initialized in wrong order, causing errors.

**Solution:**

The graph automatically handles topological sorting. Ensure parent relationships are correct:

```python
graph.add_node(normalizer)               # Root node
graph.add_node(pca, parent=normalizer)   # Specify parent
graph.add_node(rx, parent=pca)           # Specify parent
```

## Performance Tips

### Batch Size

- **Larger batches**: Faster initialization but more memory
- **Smaller batches**: Slower but works with limited memory
- **Optimal**: 4-16 for most datasets

### Number of Workers

- **num_workers=0**: Single-threaded (simple, good for debugging)
- **num_workers=4**: Parallel data loading (faster)
- **Optimal**: Set to number of CPU cores (but watch memory)

### Device Selection

```python
config = TrainingConfig(
    trainer=TrainerConfig(
        accelerator="cpu",   # Force CPU
        accelerator="gpu",   # Force GPU
        accelerator="auto",  # Auto-detect (recommended)
    )
)
```

Statistical initialization is typically CPU-bound (data loading) rather than compute-bound, so GPU may not help much in Phase 1.

## Complete Example

See the full working example in `examples_torch/phase1_statistical_training.py`.

Run it with:

```bash
# Programmatic approach
python examples_torch/phase1_statistical_training.py

# Or use the pre-configured version
python examples_torch/phase1_statistical_training.py \
    --config-name train_phase1
```

## Next Steps

Now that you understand statistical initialization:

- **[Phase 2: Visualization](phase2_visualization.md)**: Add visualization leaves and monitoring
- **[Phase 3: Gradient Training](phase3_gradient_training.md)**: Enable gradient-based fine-tuning
- **[Configuration Guide](../user-guide/configuration.md)**: Master Hydra configuration
- **[API Reference](../api/pipeline.md)**: Explore all available nodes

## Key Takeaways

✓ **Statistical initialization is fast** - No gradients, closed-form solutions  
✓ **Often sufficient** - Good results without any gradient training  
✓ **Strong baseline** - Excellent starting point for Phase 2  
✓ **Topologically sorted** - Graph handles initialization order automatically  
✓ **Serializable** - Save and load complete trained models  
✓ **Reproducible** - Hydra configuration ensures experiment reproducibility
