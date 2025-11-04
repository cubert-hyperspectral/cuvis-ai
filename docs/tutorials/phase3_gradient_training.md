# Phase 3: Gradient Training

This tutorial covers Phase 3 of the CUVIS.AI training pipeline: gradient-based training with trainable nodes, loss leaves, and metric leaves. You'll learn how to fine-tune statistically initialized models using backpropagation.

## Overview

Phase 3 extends the pipeline with:

- **Trainable PCA**: Gradient-based fine-tuning of principal components
- **Loss Leaves**: Orthogonality loss, reconstruction loss, multi-loss aggregation
- **Metric Leaves**: Explained variance, orthogonality metrics, score statistics
- **Production Monitoring**: Full WandB and TensorBoard integration
- **Two-Phase Training**: Statistical initialization → Gradient fine-tuning

This phase is powerful because it:

1. Improves upon statistical initialization through optimization
2. Maintains orthogonality constraints during training
3. Tracks training progress with rich metrics
4. Enables end-to-end differentiable pipelines
5. Supports early stopping and checkpointing

## What You'll Learn

- Converting statistical nodes to trainable parameters
- Adding loss and metric leaves
- Configuring gradient-based training
- Monitoring training with metrics and losses
- Understanding orthogonality regularization
- Tracking explained variance during training
- Using early stopping and checkpoints

## Prerequisites

- Completed [Phase 1 Tutorial](phase1_statistical.md) and [Phase 2 Tutorial](phase2_visualization.md)
- Understanding of statistical initialization and visualization
- Basic knowledge of gradient descent and backpropagation

## Concepts

### Two-Phase Training Workflow

Phase 3 introduces gradient training after statistical initialization:

```python
# Phase 1: Statistical initialization (max_epochs=0)
# - RX computes mean and covariance
# - PCA computes principal components via SVD
# - Fast, closed-form solutions

# Phase 2: Gradient training (max_epochs>0)
# - Convert buffers to nn.Parameters
# - Backpropagate through losses
# - Fine-tune with gradient descent
```

When `graph.train()` is called with `max_epochs > 0`:

1. **Statistical Init**: Initialize nodes with `requires_initial_fit=True`
2. **Prepare for Training**: Call `prepare_for_train()` on trainable nodes
3. **Lightning Training**: Run gradient-based optimization
4. **Monitor**: Track losses, metrics, visualizations

### Trainable vs Frozen Nodes

Nodes can be:

- **Frozen**: Parameters remain fixed after initialization (default)
- **Trainable**: Parameters converted to `nn.Parameter` for gradient updates

```python
# Frozen PCA (keep SVD initialization)
pca_frozen = TrainablePCA(n_components=3, trainable=False)

# Trainable PCA (fine-tune with gradients)
pca_trainable = TrainablePCA(n_components=3, trainable=True)
```

### Loss Aggregation

The Lightning module automatically aggregates losses from all loss leaves:

```python
# Multiple loss leaves
orth_loss = OrthogonalityLoss(weight=1.0)     # Weight: 1.0
recon_loss = MSEReconstructionLoss(weight=0.5) # Weight: 0.5

# Aggregated loss
total_loss = 1.0 * orth_loss + 0.5 * recon_loss
# Backprop through total_loss
```

### Metric Tracking

Metric leaves compute evaluation metrics without affecting gradients:

```python
# Metrics tracked but don't contribute to loss
var_metric = ExplainedVarianceMetric()      # Track PC variance
orth_metric = ComponentOrthogonalityMetric() # Monitor orthogonality

# Logged to monitoring backends
# No gradient computation
```

## Tutorial: Training Trainable PCA

### Step 1: Imports

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.training.config import TrainingConfig, TrainerConfig, OptimizerConfig

# Loss leaves
from cuvis_ai.training.losses import OrthogonalityLoss

# Metric leaves
from cuvis_ai.training.metrics import (
    ExplainedVarianceMetric,
    ComponentOrthogonalityMetric,
)

# Monitoring
from cuvis_ai.training.monitors import DummyMonitor
```

### Step 2: Build Graph with Trainable PCA

```python
# Create graph
graph = Graph("trainable_pca")

# Add normalization (frozen)
normalizer = MinMaxNormalizer(eps=1e-6)
graph.add_node(normalizer)

# Add TRAINABLE PCA
pca = TrainablePCA(
    n_components=3,          # Number of principal components
    trainable=True,          # Enable gradient training
    whiten=False,            # Don't whiten (scale by eigenvalues)
)
graph.add_node(pca, parent=normalizer)
```

**Key Point:** `trainable=True` means PCA components will be fine-tuned with gradients after SVD initialization.

### Step 3: Add Loss Leaves

```python
# Orthogonality regularization
# Penalizes deviation from orthonormal components
orth_loss = OrthogonalityLoss(
    weight=1.0,              # Loss weight in aggregation
)
graph.add_leaf_node(orth_loss, parent=pca)
```

**Why Orthogonality Loss?**

During gradient training, PCA components may drift from orthogonality. This loss maintains the constraint:

```
L_orth = ||W^T W - I||_F^2
```

Where W is the component matrix and I is identity. This ensures components remain orthonormal.

### Step 4: Add Metric Leaves

```python
# Track explained variance per component
var_metric = ExplainedVarianceMetric()
graph.add_leaf_node(var_metric, parent=pca)

# Monitor orthogonality drift
orth_metric = ComponentOrthogonalityMetric()
graph.add_leaf_node(orth_metric, parent=pca)
```

**Metrics vs Losses:**

- **Losses**: Backpropagated, affect training
- **Metrics**: Monitoring only, no gradients

### Step 5: Register Monitoring

```python
# Filesystem monitor for artifacts
monitor = DummyMonitor(output_dir="./outputs/artifacts")
graph.register_monitor(monitor)

# Optional: WandB for rich tracking
# from cuvis_ai.training.monitors import WandBMonitor
# wandb_monitor = WandBMonitor(project="cuvis_ai", name="trainable_pca")
# graph.register_monitor(wandb_monitor)
```

### Step 6: Configure Gradient Training

```python
# Training configuration
config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(
        max_epochs=10,           # 10 epochs of gradient training
        accelerator="auto",      # Use GPU if available
        devices=1,
        precision="32",          # Mixed precision: "16-mixed" for speed
        log_every_n_steps=10,    # Log frequency
    ),
    optimizer=OptimizerConfig(
        name="adam",             # Optimizer: adam, sgd, adamw
        lr=0.001,                # Learning rate
        weight_decay=0.0,        # L2 regularization
        betas=(0.9, 0.999),      # Adam betas
    )
)
```

**Important:** `max_epochs > 0` triggers gradient training after statistical initialization.

### Step 7: Train the Model

```python
# Two-phase training
trainer = graph.train(
    datamodule=LentilsAnomaly(
        data_dir="./data/Lentils",
        batch_size=4,
        num_workers=0,
    ),
    training_config=config
)
```

**Expected Output:**

```
[INFO] Starting training with config: TrainingConfig(seed=42, ...)
[INFO] Phase 1: Initializing statistical nodes...
[INFO]   Initialized MinMaxNormalizer
[INFO]   Initialized TrainablePCA (SVD initialization)
[INFO]   Explained variance: [85.2%, 10.3%, 3.1%] = 98.6% total
[INFO] Phase 1 complete in 2.1 seconds

[INFO] Phase 2: Preparing trainable nodes...
[INFO]   Converting PCA components to nn.Parameters
[INFO] Starting gradient training...

Epoch 1/10:  100%|████████| 270/270 [00:15<00:00, 17.2it/s, loss=0.045]
[INFO]   train/loss/orthogonality: 0.045
[INFO]   val/metric/explained_variance_total: 98.7%
[INFO]   val/metric/orthogonality_error: 0.003

Epoch 2/10:  100%|████████| 270/270 [00:14<00:00, 18.1it/s, loss=0.012]
[INFO]   train/loss/orthogonality: 0.012
[INFO]   val/metric/explained_variance_total: 98.8%
[INFO]   val/metric/orthogonality_error: 0.001

...

[INFO] Training complete in 2m 45s
[INFO] Best validation loss: 0.008 at epoch 7
```

### Step 8: Analyze Results

Examine training metrics:

```python
# Metrics are logged to monitoring backends
# For DummyMonitor, check JSONL files:

import json

with open("outputs/artifacts/metrics.jsonl") as f:
    metrics = [json.loads(line) for line in f]

# Filter orthogonality metrics
orth_metrics = [m for m in metrics if "orthogonality" in m["name"]]
for m in orth_metrics[:5]:
    print(f"Epoch {m['step']}: {m['name']} = {m['value']:.4f}")
```

Expected: Orthogonality error decreases over epochs, staying < 0.01.

### Step 9: Verify Trained Model

Test forward pass and check orthogonality:

```python
import torch

# Get test batch
datamodule.setup("test")
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

# Forward pass through trained PCA
with torch.no_grad():
    pc_scores, _, _ = graph(batch["cube"])

print(f"PC scores shape: {pc_scores.shape}")  # [B, 3, H, W]

# Check orthogonality of trained components
components = pca.components  # [3, 61]
gram = components @ components.T
print(f"Components gram matrix:\n{gram}")
# Should be close to identity:
# [[1.000, 0.001, 0.002],
#  [0.001, 1.000, 0.001],
#  [0.002, 0.001, 1.000]]
```

### Step 10: Save Trained Model

```python
# Save complete trained graph
graph.save_to_file("trainable_pca_trained.zip")

# Later: Load and use
loaded_graph = Graph.load_from_file("trainable_pca_trained.zip")
with torch.no_grad():
    loaded_scores, _, _ = loaded_graph(batch["cube"])
```

## Loss Types

### OrthogonalityLoss

Maintains orthonormality of PCA components during training.

**Formula:**
```
L = ||W^T W - I||_F^2
```

**Configuration:**

```python
orth_loss = OrthogonalityLoss(
    weight=1.0,              # Loss weight
)
graph.add_leaf_node(orth_loss, parent=pca)
```

**When to Use:**
- Always with trainable PCA
- Prevents component collapse
- Maintains interpretability

**Typical Values:**
- Fresh initialization: 0.001-0.01
- Well-trained: < 0.001
- Problematic: > 0.1 (components drifting)

### MSEReconstructionLoss

Measures reconstruction quality for autoencoders or compressed representations.

**Formula:**
```
L = ||X - X_reconstructed||^2
```

**Configuration:**

```python
from cuvis_ai.training.losses import MSEReconstructionLoss

recon_loss = MSEReconstructionLoss(
    weight=0.5,              # Loss weight
    use_labels_as_target=True,  # Use batch["labels"] as target
)
graph.add_leaf_node(recon_loss, parent=decoder_node)
```

**When to Use:**
- Training autoencoders
- Dimensionality reduction with reconstruction
- Image/signal reconstruction tasks

### AnomalyBCEWithLogits

Binary cross-entropy for anomaly detection with RX logit head (Phase 4).

**Configuration:**

```python
from cuvis_ai.training.losses import AnomalyBCEWithLogits
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead

# Add RX detector + logit head
rx = RXGlobal(trainable_stats=False)
logit_head = RXLogitHead(trainable=True)
graph.add_node(rx, parent=pca)
graph.add_node(logit_head, parent=rx)

# Add BCE loss
bce_loss = AnomalyBCEWithLogits(
    weight=1.0,
    pos_weight=None,         # Weight for positive class (optional)
)
graph.add_leaf_node(bce_loss, parent=logit_head)
```

**When to Use:**
- Training anomaly detectors end-to-end
- Requires labeled anomaly masks
- Phase 4: RX logit head training

### WeightedMultiLoss

Combines multiple losses with learnable or fixed weights.

**Configuration:**

```python
from cuvis_ai.training.losses import WeightedMultiLoss

# Parent losses
orth_loss = OrthogonalityLoss(weight=1.0)
recon_loss = MSEReconstructionLoss(weight=0.5)

# Multi-loss aggregator
multi_loss = WeightedMultiLoss(
    child_losses=[orth_loss, recon_loss],
    weights=[1.0, 0.5],      # Fixed weights
    learnable=False,         # Set True for learnable weights
)
# Note: Typically you just add individual losses and Lightning aggregates them
```

## Metric Types

### ExplainedVarianceMetric

Tracks explained variance per component and cumulatively.

**Metrics Logged:**
- `explained_variance_pc1`, `pc2`, `pc3`, ... (per component)
- `explained_variance_total` (cumulative)

**Configuration:**

```python
var_metric = ExplainedVarianceMetric()
graph.add_leaf_node(var_metric, parent=pca)
```

**Interpretation:**
- PC1: 80-90% typical (most informative)
- PC2: 5-15% typical
- PC3: 2-5% typical
- Total: Sum should be high (>95% for 3 components good)

**Monitoring:**
- Should remain stable or slightly increase during training
- Sudden drops indicate component collapse

### ComponentOrthogonalityMetric

Monitors orthogonality error (Frobenius norm of W^T W - I).

**Metrics Logged:**
- `component_orthogonality_error` (scalar)

**Configuration:**

```python
orth_metric = ComponentOrthogonalityMetric()
graph.add_leaf_node(orth_metric, parent=pca)
```

**Interpretation:**
- < 0.001: Excellent orthogonality
- 0.001-0.01: Good (acceptable for most applications)
- 0.01-0.1: Fair (may need stronger regularization)
- > 0.1: Poor (increase orthogonality loss weight)

### AnomalyDetectionMetrics

Computes precision, recall, F1, IoU for anomaly detection.

**Requires:** Ground truth anomaly masks in `batch["mask"]`.

**Configuration:**

```python
from cuvis_ai.training.metrics import AnomalyDetectionMetrics

anom_metrics = AnomalyDetectionMetrics(
    threshold_multiplier=2.0,    # Threshold = mean + 2*std
)
graph.add_leaf_node(anom_metrics, parent=rx)
```

**Metrics Logged:**
- `anomaly_precision` - % of predicted anomalies that are true anomalies
- `anomaly_recall` - % of true anomalies detected
- `anomaly_f1` - Harmonic mean of precision and recall
- `anomaly_iou` - Intersection over Union

**Interpretation:**
- High precision + low recall: Conservative detector (few false positives)
- Low precision + high recall: Aggressive detector (many false positives)
- High F1: Balanced performance
- IoU > 0.5: Good spatial overlap

### ScoreStatisticsMetric

Tracks distribution statistics of scores/outputs.

**Metrics Logged:**
- `score_mean`, `score_std`, `score_min`, `score_max`
- `score_q25`, `score_q50`, `score_q75` (quartiles)

**Configuration:**

```python
from cuvis_ai.training.metrics import ScoreStatisticsMetric

stats_metric = ScoreStatisticsMetric()
graph.add_leaf_node(stats_metric, parent=rx)
```

**Interpretation:**
- Stable mean/std: Consistent detector
- Increasing std: More variability (possibly more anomalies)
- Extreme values: Check for outliers or errors

## Production Monitoring Integration

### WandB Full Integration

```python
from cuvis_ai.training.monitors import WandBMonitor

wandb_monitor = WandBMonitor(
    project="cuvis_ai_production",
    entity="my_team",
    name="phase3_trainable_pca",
    tags=["phase3", "pca", "gradient"],
    notes="Training PCA with orthogonality regularization",
    mode="online",           # or "offline" for local logging
)
graph.register_monitor(wandb_monitor)

# Automatic tracking:
# - All losses and metrics
# - Hyperparameters (from TrainingConfig)
# - Visualizations (from Phase 2)
# - System metrics (GPU, CPU, memory)
```

**View on WandB Dashboard:**
- Metric plots over time
- Hyperparameter comparison
- Artifact browser
- Run comparison

### TensorBoard Full Integration

```python
from cuvis_ai.training.monitors import TensorBoardMonitor

tb_monitor = TensorBoardMonitor(
    log_dir="./outputs/tensorboard",
    flush_secs=30,
)
graph.register_monitor(tb_monitor)

# Launch TensorBoard:
# tensorboard --logdir=./outputs/tensorboard
```

**Features:**
- Scalar plots (losses, metrics)
- Histogram of gradients
- Graph visualization
- Image logging (visualizations)

## Hydra Configuration

### Configuration File

Create `phase3_config.yaml`:

```yaml
defaults:
  - general
  - _self_

graph:
  name: trainable_pca

nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
    eps: 1.0e-6
  
  pca:
    _target_: cuvis_ai.node.pca.TrainablePCA
    n_components: 3
    trainable: true
    whiten: false

loss_leaves:
  orthogonality_loss:
    _target_: cuvis_ai.training.losses.OrthogonalityLoss
    weight: 1.0

metric_leaves:
  explained_variance:
    _target_: cuvis_ai.training.metrics.ExplainedVarianceMetric
  
  component_orthogonality:
    _target_: cuvis_ai.training.metrics.ComponentOrthogonalityMetric

monitoring:
  dummy:
    _target_: cuvis_ai.training.monitors.DummyMonitor
    output_dir: ./outputs/artifacts
  
  wandb:
    enabled: false
    _target_: cuvis_ai.training.monitors.WandBMonitor
    project: cuvis_ai_phase3
    name: trainable_pca_experiment

datamodule:
  _target_: cuvis_ai.data.lentils_anomaly.LentilsAnomaly
  data_dir: ${oc.env:DATA_ROOT,./data/Lentils}
  batch_size: 4
  num_workers: 0

training:
  seed: 42
  trainer:
    max_epochs: 10
    accelerator: auto
    devices: 1
    precision: "32"
    log_every_n_steps: 10
  optimizer:
    name: adam
    lr: 0.001
    weight_decay: 0.0
```

### Training Script

See `examples_torch/phase3_gradient_training.py` for complete implementation.

### Run with CLI Overrides

```bash
# Use defaults
python examples_torch/phase3_gradient_training.py

# More epochs, higher learning rate
python examples_torch/phase3_gradient_training.py \
    training.trainer.max_epochs=20 \
    training.optimizer.lr=0.01

# Enable mixed precision for speed
python examples_torch/phase3_gradient_training.py \
    training.trainer.precision="16-mixed"

# Enable WandB
python examples_torch/phase3_gradient_training.py \
    monitoring.wandb.enabled=true
```

## Advanced Techniques

### Learning Rate Scheduling

Lightning automatically uses ReduceLROnPlateau scheduler:

```python
# Automatically configured in TrainingConfig
# Reduces LR when validation loss plateaus
# Factor: 0.1 (divide by 10)
# Patience: 10 epochs
```

### Early Stopping

Add early stopping callback:

```python
from lightning.pytorch.callbacks import EarlyStopping

config = TrainingConfig(
    trainer=TrainerConfig(
        max_epochs=100,
        callbacks=[
            EarlyStopping(
                monitor="val/loss/total",
                patience=10,
                mode="min",
            )
        ]
    )
)
```

### Model Checkpointing

Save best models during training:

```python
from lightning.pytorch.callbacks import ModelCheckpoint

config = TrainingConfig(
    trainer=TrainerConfig(
        max_epochs=50,
        callbacks=[
            ModelCheckpoint(
                dirpath="./checkpoints",
                filename="best-{epoch:02d}-{val_loss:.2f}",
                monitor="val/loss/total",
                mode="min",
                save_top_k=3,
            )
        ]
    )
)
```

### Gradient Clipping

Prevent exploding gradients:

```python
config = TrainingConfig(
    trainer=TrainerConfig(
        gradient_clip_val=1.0,   # Clip gradients to max norm 1.0
    )
)
```

## Troubleshooting

### Component Collapse

**Problem:** All PCA components become similar during training.

**Solution:**

1. Increase orthogonality loss weight: `OrthogonalityLoss(weight=10.0)`
2. Reduce learning rate: `lr=0.0001`
3. Monitor orthogonality metric - stop if error > 0.1

### NaN Losses

**Problem:** Losses become NaN during training.

**Solution:**

1. Reduce learning rate: `lr=0.0001`
2. Add gradient clipping: `gradient_clip_val=1.0`
3. Check for numerical instability (divide by zero, log of negative)
4. Use mixed precision: `precision="16-mixed"`

### Slow Training

**Problem:** Training is too slow.

**Solution:**

1. Enable mixed precision: `precision="16-mixed"` (2-3x speedup on GPU)
2. Increase batch size: `batch_size=16`
3. Use multiple workers: `num_workers=4`
4. Reduce visualization frequency: `log_frequency=10`

### Poor Convergence

**Problem:** Loss doesn't decrease or decreases very slowly.

**Solution:**

1. Try different learning rates: `lr=0.01` or `lr=0.0001`
2. Change optimizer: `name="adamw"` with `weight_decay=0.01`
3. Increase training epochs: `max_epochs=50`
4. Check statistical initialization quality

## Performance Tips

### Mixed Precision Training

Use `"16-mixed"` for 2-3x speedup:

```python
config = TrainingConfig(
    trainer=TrainerConfig(
        precision="16-mixed",    # Automatic mixed precision
    )
)
```

Benefits:
- Faster training (2-3x on modern GPUs)
- Lower memory usage
- Minimal accuracy loss

### Batch Size Tuning

Larger batches = faster but more memory:

```python
# Small (debugging)
batch_size = 2

# Medium (typical)
batch_size = 8

# Large (production, requires GPU)
batch_size = 32
```

### Worker Parallelism

Use multiple CPU cores for data loading:

```python
datamodule = LentilsAnomaly(
    batch_size=8,
    num_workers=4,           # 4 parallel workers
)
```

Rule of thumb: `num_workers = min(4, num_cpus)`

## Complete Example

See `examples_torch/phase3_gradient_training.py` for the complete working example.

Run it:

```bash
python examples_torch/phase3_gradient_training.py
```

## Next Steps

Now that you understand gradient training:

- **[Phase 4: Channel Selection](phase4_channel_selection.md)**: Add soft channel selector for feature selection
- **[Configuration Guide](../user-guide/configuration.md)**: Advanced Hydra configuration
- **[API Reference](../api/training.md)**: Explore all loss and metric options

## Key Takeaways

✓ **Two-phase training** - Statistical init then gradient fine-tuning  
✓ **Trainable nodes** - Convert buffers to parameters with `trainable=True`  
✓ **Loss aggregation** - Multiple losses combined automatically  
✓ **Orthogonality regularization** - Maintains PCA component constraints  
✓ **Rich metrics** - Track variance, orthogonality, statistics  
✓ **Production monitoring** - Full WandB/TensorBoard integration  
✓ **Flexible optimization** - Adam, SGD, learning rate scheduling  
✓ **Lightning features** - Early stopping, checkpointing, gradient clipping
