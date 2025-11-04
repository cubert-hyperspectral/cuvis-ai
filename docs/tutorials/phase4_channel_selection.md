# Phase 4: Channel Selection

This tutorial covers Phase 4 of the CUVIS.AI training pipeline: soft channel selection with temperature-based Gumbel-Softmax, entropy/diversity regularization, and end-to-end anomaly detection. You'll learn how to train networks that automatically select the most informative spectral channels.

## Overview

Phase 4 introduces advanced selection and detection capabilities:

- **Soft Channel Selector**: Learnable channel selection with temperature annealing
- **Entropy Regularization**: Encourage exploration of channel combinations
- **Diversity Regularization**: Prevent concentration on few channels
- **RX Logit Head**: Trainable anomaly detection threshold
- **End-to-End Training**: Anomaly detection with binary cross-entropy
- **Selector Visualizations**: Track temperature, selection masks, stability

This phase is powerful because it:

1. Automatically discovers informative wavelengths
2. Reduces computational cost by selecting fewer channels
3. Improves interpretability (which channels matter?)
4. Enables end-to-end differentiable anomaly detection
5. Supports domain knowledge injection through initialization

## What You'll Learn

- Implementing soft channel selection
- Temperature annealing strategies
- Entropy and diversity regularization
- Training anomaly detectors end-to-end
- Monitoring channel selection over time
- Interpreting selection results
- Dynamic graph restructuring

## Prerequisites

- Completed [Phase 1](phase1_statistical.md), [Phase 2](phase2_visualization.md), and [Phase 3](phase3_gradient_training.md) tutorials
- Understanding of statistical initialization, visualization, and gradient training
- Basic familiarity with attention mechanisms (helpful but not required)

## Concepts

### Soft Channel Selection

Traditional hard selection (top-k channels) is non-differentiable. Soft selection uses Gumbel-Softmax:

```python
# Soft weights (differentiable)
weights = softmax(logits / temperature)
selected = input * weights  # Weighted combination

# Hard selection (inference)
top_k_indices = topk(logits, k=n_select)
selected = input[:, top_k_indices]
```

Benefits:
- **Differentiable**: Can backpropagate through selection
- **Smooth**: Gradual transition from exploration to exploitation
- **Temperature control**: High T = explore, Low T = exploit

### Temperature Annealing

Temperature controls selection sharpness:

```python
# High temperature (T=5.0): Soft, exploratory
# weights ≈ [0.15, 0.18, 0.16, 0.17, 0.14, ...]  # Spread out

# Low temperature (T=0.1): Sharp, exploitative  
# weights ≈ [0.001, 0.98, 0.002, 0.015, 0.001, ...]  # Concentrated
```

Annealing schedule:

```python
T_epoch = max(T_min, T_init * decay^epoch)
```

### Entropy and Diversity

**Entropy**: Encourages exploration of different channels

```python
H = -sum(p * log(p))  # High entropy = uniform weights
```

**Diversity**: Encourages selecting spread-out channels

```python
D = variance(selected_indices)  # High diversity = spread selection
```

### End-to-End Anomaly Detection

With RXLogitHead, threshold is learned:

```python
# Statistical RX (Phase 1-3)
score = (x - μ)^T Σ^(-1) (x - μ)
anomaly = score > threshold  # Fixed threshold

# Trainable RX (Phase 4)
logit = scale * (score - bias)  # Learnable scale, bias
anomaly = sigmoid(logit) > 0.5  # Learned threshold
```

Training with BCE loss:

```python
loss = BCE(sigmoid(logit), ground_truth_mask)
```

## Tutorial: Building a Channel Selector Pipeline

### Step 1: Imports

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.anomaly.rx_detector import RXGlobal
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.data.lentils_anomaly import LentilsAnomaly
from cuvis_ai.training.config import TrainingConfig, TrainerConfig, OptimizerConfig

# Losses
from cuvis_ai.training.losses import (
    SelectorEntropyRegularizer,
    SelectorDiversityRegularizer,
    OrthogonalityLoss,
    AnomalyBCEWithLogits,
)

# Metrics
from cuvis_ai.training.metrics import (
    ExplainedVarianceMetric,
    AnomalyDetectionMetrics,
)

# Selector visualizations
from cuvis_ai.training.special_visualization.selector_visualizations import (
    SelectorTemperaturePlot,
    SelectorChannelMaskPlot,
    SelectorStabilityPlot,
)

# Monitoring
from cuvis_ai.training.monitors import DummyMonitor
```

### Step 2: Build Graph with Selector

```python
# Create graph
graph = Graph("channel_selector_pipeline")

# Normalization (frozen)
normalizer = MinMaxNormalizer(eps=1e-6)
graph.add_node(normalizer)

# CHANNEL SELECTOR (trainable)
selector = SoftChannelSelector(
    n_select=15,                 # Select 15 out of 61 channels
    init_method="variance",      # Initialize by channel variance
    temperature_init=5.0,        # Start with high temperature (exploratory)
    temperature_min=0.1,         # Minimum temperature (exploitative)
    temperature_decay=0.9,       # Decay per epoch: T *= 0.9
    trainable=True,              # Enable gradient training
)
graph.add_node(selector, parent=normalizer)

# PCA on selected channels (trainable)
pca = TrainablePCA(
    n_components=3,
    trainable=True,
)
graph.add_node(pca, parent=selector)

# RX detector on PCA scores (frozen statistics)
rx = RXGlobal(
    eps=1e-6,
    trainable_stats=False,
)
graph.add_node(rx, parent=pca)

# Trainable threshold via logit head
logit_head = RXLogitHead(
    init_scale=1.0,             # Initial scale parameter
    init_bias=5.0,              # Initial bias parameter
    trainable=True,             # Learn optimal threshold
)
graph.add_node(logit_head, parent=rx)
```

**Pipeline Flow:**

```
Input (61 channels) 
  → Normalizer
  → Selector (15 channels selected)
  → PCA (3 principal components)
  → RX (anomaly scores)
  → LogitHead (binary predictions)
```

### Step 3: Add Selector Regularizers

```python
# Entropy regularizer (encourage exploration)
entropy_reg = SelectorEntropyRegularizer(
    weight=0.01,                # Loss weight (tune for your task)
)
graph.add_leaf_node(entropy_reg, parent=selector)

# Diversity regularizer (prevent concentration)
diversity_reg = SelectorDiversityRegularizer(
    weight=0.01,                # Loss weight
)
graph.add_leaf_node(diversity_reg, parent=selector)
```

**Why These Regularizers?**

- **Entropy**: Without it, selector may concentrate on single best channel
- **Diversity**: Encourages selecting channels spread across spectrum
- **Balance**: Weights should be small (0.001-0.1) to not dominate task loss

### Step 4: Add PCA Regularizer

```python
# Orthogonality loss for PCA
orth_loss = OrthogonalityLoss(weight=1.0)
graph.add_leaf_node(orth_loss, parent=pca)
```

### Step 5: Add Anomaly Detection Loss

```python
# Binary cross-entropy for anomaly detection
# Requires ground truth masks in batch["mask"]
bce_loss = AnomalyBCEWithLogits(
    weight=1.0,                 # Task loss weight
    pos_weight=None,            # Optional: weight positive class more
)
graph.add_leaf_node(bce_loss, parent=logit_head)
```

**Total Loss:**

```
L_total = 1.0 * L_bce + 
          1.0 * L_orth + 
          0.01 * L_entropy + 
          0.01 * L_diversity
```

### Step 6: Add Metrics

```python
# PCA explained variance
var_metric = ExplainedVarianceMetric()
graph.add_leaf_node(var_metric, parent=pca)

# Anomaly detection metrics (precision, recall, F1, IoU)
anom_metrics = AnomalyDetectionMetrics(
    threshold=0.5,              # Sigmoid threshold for binary prediction
)
graph.add_leaf_node(anom_metrics, parent=logit_head)
```

### Step 7: Add Selector Visualizations

```python
# Temperature annealing visualization
temp_viz = SelectorTemperaturePlot(
    log_frequency=1,            # Log every epoch
)
graph.add_leaf_node(temp_viz, parent=selector)

# Channel selection mask visualization
mask_viz = SelectorChannelMaskPlot(
    log_frequency=5,            # Log every 5 epochs
)
graph.add_leaf_node(mask_viz, parent=selector)

# Selection stability tracking
stability_viz = SelectorStabilityPlot(
    log_frequency=1,            # Log every epoch
)
graph.add_leaf_node(stability_viz, parent=selector)
```

### Step 8: Register Monitoring

```python
# Filesystem monitoring
monitor = DummyMonitor(output_dir="./outputs/artifacts")
graph.register_monitor(monitor)
```

### Step 9: Configure Training

```python
# Training configuration
config = TrainingConfig(
    seed=42,
    trainer=TrainerConfig(
        max_epochs=20,           # More epochs for selector convergence
        accelerator="auto",
        devices=1,
        precision="32",
        log_every_n_steps=10,
    ),
    optimizer=OptimizerConfig(
        name="adam",
        lr=0.001,
        weight_decay=0.0,
    )
)
```

### Step 10: Train the Model

```python
# Two-phase training: statistical init + gradient training
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
[INFO] Phase 1: Statistical initialization...
[INFO]   Initializing MinMaxNormalizer
[INFO]   Initializing SoftChannelSelector (variance-based)
[INFO]     Selected initial channels: [45, 46, 47, ..., 59] (high variance)
[INFO]   Initializing TrainablePCA on 15 channels
[INFO]     Explained variance: [92.3%, 4.8%, 2.9%] = 100.0%
[INFO]   Initializing RXGlobal on 3 PCs
[INFO]   Initializing RXLogitHead (bias=5.12 from score distribution)

[INFO] Phase 2: Gradient training...

Epoch 1/20:  100%|████| 270/270 [00:18<00:00, 14.5it/s]
[INFO]   train/loss/total: 0.523
[INFO]   train/loss/bce: 0.489
[INFO]   train/loss/orthogonality: 0.021
[INFO]   train/loss/entropy: 0.012
[INFO]   train/loss/diversity: 0.001
[INFO]   val/metric/anomaly_f1: 0.67
[INFO]   val/metric/explained_variance_total: 100.0%
[INFO]   Selector temperature: 5.000
[INFO]   Selected channels: [47, 48, 46, 49, 57, 50, 56, ...]

Epoch 10/20:  100%|████| 270/270 [00:17<00:00, 15.2it/s]
[INFO]   train/loss/total: 0.312
[INFO]   train/loss/bce: 0.298
[INFO]   train/loss/orthogonality: 0.003
[INFO]   train/loss/entropy: 0.010
[INFO]   train/loss/diversity: 0.001
[INFO]   val/metric/anomaly_f1: 0.84
[INFO]   Selector temperature: 1.937
[INFO]   Selected channels: [48, 49, 47, 50, 51, 57, 56, ...]

Epoch 20/20:  100%|████| 270/270 [00:17<00:00, 15.8it/s]
[INFO]   train/loss/total: 0.245
[INFO]   train/loss/bce: 0.234
[INFO]   train/loss/orthogonality: 0.001
[INFO]   train/loss/entropy: 0.009
[INFO]   train/loss/diversity: 0.001
[INFO]   val/metric/anomaly_f1: 0.89
[INFO]   Selector temperature: 0.352

[INFO] Training complete in 6m 12s
[INFO] Final selected channels: [48, 49, 50, 51, 47, 52, 57, 56, 46, 58, 53, 59, 45, 55, 54]
[INFO] Best F1: 0.89 at epoch 18
```

### Step 11: Analyze Selection Results

```python
# Examine final channel selection
final_channels = selector.get_selected_channels()
print(f"Selected {len(final_channels)} channels: {final_channels}")

# Get selection probabilities
probs = selector.get_selection_probabilities()
print("\nTop 5 channels by probability:")
top_5 = torch.argsort(probs, descending=True)[:5]
for idx in top_5:
    print(f"  Channel {idx}: {probs[idx]:.4f}")

# Check if selection is stable (low entropy)
from torch.distributions import Categorical
dist = Categorical(probs=probs)
entropy = dist.entropy()
print(f"\nFinal selection entropy: {entropy:.4f}")
print("  (Low entropy = confident selection)")
```

**Typical Output:**

```
Selected 15 channels: [48, 49, 50, 51, 47, 52, 57, 56, 46, 58, 53, 59, 45, 55, 54]

Top 5 channels by probability:
  Channel 48: 0.0724
  Channel 49: 0.0698
  Channel 50: 0.0681
  Channel 51: 0.0676
  Channel 47: 0.0671

Final selection entropy: 2.64
  (Low entropy = confident selection)
```

### Step 12: Test Anomaly Detection

```python
# Get test batch
datamodule.setup("test")
test_loader = datamodule.test_dataloader()
batch = next(iter(test_loader))

# Forward pass
with torch.no_grad():
    logits, _, _ = graph(batch["cube"])
    predictions = torch.sigmoid(logits) > 0.5

# Compute metrics
if "mask" in batch:
    gt_mask = batch["mask"]
    
    # Precision: Of predicted anomalies, how many are correct?
    precision = (predictions & gt_mask).sum() / predictions.sum()
    
    # Recall: Of actual anomalies, how many did we detect?
    recall = (predictions & gt_mask).sum() / gt_mask.sum()
    
    # F1 score
    f1 = 2 * precision * recall / (precision + recall)
    
    print(f"Test set performance:")
    print(f"  Precision: {precision:.2%}")
    print(f"  Recall: {recall:.2%}")
    print(f"  F1 Score: {f1:.2%}")
```

### Step 13: Interpret Selection

Analyze which wavelengths were selected:

```python
# Assuming wavelengths for Lentils dataset
wavelengths = np.linspace(400, 1000, 61)  # Example: 400-1000nm
selected_wavelengths = wavelengths[final_channels]

print("\nSelected wavelengths (nm):")
for ch, wl in zip(final_channels, selected_wavelengths):
    print(f"  Channel {ch}: {wl:.1f} nm")

# Visualize selection
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 4))
plt.plot(wavelengths, probs.cpu().numpy(), 'b-', alpha=0.6, label='Selection probability')
plt.scatter(selected_wavelengths, probs[final_channels].cpu().numpy(), 
            c='red', s=100, zorder=5, label='Selected channels')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Selection Probability')
plt.title('Learned Channel Selection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('channel_selection.png')
```

## Selector Configuration Options

### Initialization Methods

**Variance-based** (recommended):
```python
selector = SoftChannelSelector(
    n_select=15,
    init_method="variance",      # Select high-variance channels
)
# Good for: Automatic discovery of informative channels
```

**Uniform**:
```python
selector = SoftChannelSelector(
    n_select=15,
    init_method="uniform",       # Equal weights for all channels
)
# Good for: Exploration-heavy tasks, no prior knowledge
```

**Custom** (domain knowledge):
```python
# Initialize with specific channels (e.g., known absorption bands)
custom_logits = torch.zeros(61)
custom_logits[[20, 25, 30, 40, 45]] = 5.0  # Boost specific channels
selector = SoftChannelSelector(n_select=15, init_method="uniform")
selector.logits.data = custom_logits
# Good for: Incorporating domain expertise
```

### Temperature Schedules

**Standard annealing** (recommended):
```python
selector = SoftChannelSelector(
    temperature_init=5.0,        # Start exploratory
    temperature_min=0.1,         # End exploitative
    temperature_decay=0.9,       # T *= 0.9 per epoch
)
```

**Fast annealing**:
```python
selector = SoftChannelSelector(
    temperature_init=10.0,
    temperature_min=0.05,
    temperature_decay=0.8,       # Faster convergence
)
```

**Slow annealing**:
```python
selector = SoftChannelSelector(
    temperature_init=3.0,
    temperature_min=0.5,
    temperature_decay=0.95,      # More exploration
)
```

## RX Logit Head Configuration

### Statistical Initialization

Automatically initializes from RX score distribution:

```python
logit_head = RXLogitHead(
    init_scale=1.0,              # Typical: 0.5-2.0
    init_bias=None,              # Auto: mean + 2*std from data
    trainable=True,
)
```

During initialization:
1. Computes RX scores on training data
2. Estimates threshold: `bias = mean(scores) + 2*std(scores)`
3. Sets scale to control sensitivity

### Manual Configuration

For specific threshold behavior:

```python
logit_head = RXLogitHead(
    init_scale=2.0,              # Higher = more sensitive
    init_bias=3.0,               # Lower = detect more anomalies
    trainable=True,
)
```

**Effect of parameters:**
- **High scale**: Sharper decision boundary
- **Low bias**: More pixels classified as anomalous
- **High bias**: Fewer pixels classified as anomalous

## Dynamic Graph Restructuring

Rearrange graph structure at runtime:

```python
# Initial: Normalizer → PCA → RX
graph.add_node(normalizer)
graph.add_node(pca, parent=normalizer)
graph.add_node(rx, parent=pca)

# Insert selector between normalizer and PCA
selector = SoftChannelSelector(n_select=15)
graph.add_node(selector, parent=normalizer)
graph.set_parent(pca, selector)  # Move PCA to receive from selector

# New structure: Normalizer → Selector → PCA → RX
```

**Use cases:**
- A/B testing different pipelines
- Progressive architecture search
- Runtime optimization

## Selector Visualizations

### Temperature Plot

Shows temperature decay over training:

```python
temp_viz = SelectorTemperaturePlot(log_frequency=1)
```

**Output**: Line plot of temperature vs epoch

**Interpretation:**
- Smooth decay: Normal annealing
- Plateaus: At temperature_min
- Abrupt changes: Check configuration

### Channel Mask Plot

Shows soft weights and hard selection:

```python
mask_viz = SelectorChannelMaskPlot(log_frequency=5)
```

**Output**: 2-panel figure
- Panel 1: Soft weights (selection probabilities) over channels
- Panel 2: Hard mask (binary: selected/not selected)

**Interpretation:**
- Concentrated weights: Confident selection
- Spread weights: Uncertain or exploring
- Changing selection: Still learning

### Stability Plot

Tracks selection changes over time:

```python
stability_viz = SelectorStabilityPlot(log_frequency=1)
```

**Output**: Line plot of Jaccard similarity vs epoch

**Interpretation:**
- High similarity (>0.8): Stable selection
- Low similarity (<0.5): Selection still changing
- Increasing trend: Converging to final selection

## Hydra Configuration

Create `phase4_config.yaml`:

```yaml
defaults:
  - general
  - _self_

graph:
  name: channel_selector_pipeline

nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
    eps: 1.0e-6
  
  selector:
    _target_: cuvis_ai.node.selector.SoftChannelSelector
    n_select: 15
    init_method: variance
    temperature_init: 5.0
    temperature_min: 0.1
    temperature_decay: 0.9
    trainable: true
  
  pca:
    _target_: cuvis_ai.node.pca.TrainablePCA
    n_components: 3
    trainable: true
  
  rx:
    _target_: cuvis_ai.anomaly.rx_detector.RXGlobal
    eps: 1.0e-6
    trainable_stats: false
  
  logit_head:
    _target_: cuvis_ai.anomaly.rx_logit_head.RXLogitHead
    init_scale: 1.0
    init_bias: null
    trainable: true

loss_leaves:
  entropy_reg:
    _target_: cuvis_ai.training.losses.SelectorEntropyRegularizer
    weight: 0.01
  
  diversity_reg:
    _target_: cuvis_ai.training.losses.SelectorDiversityRegularizer
    weight: 0.01
  
  orthogonality_loss:
    _target_: cuvis_ai.training.losses.OrthogonalityLoss
    weight: 1.0
  
  bce_loss:
    _target_: cuvis_ai.training.losses.AnomalyBCEWithLogits
    weight: 1.0

metric_leaves:
  explained_variance:
    _target_: cuvis_ai.training.metrics.ExplainedVarianceMetric
  
  anomaly_metrics:
    _target_: cuvis_ai.training.metrics.AnomalyDetectionMetrics

visualization_leaves:
  temp_viz:
    _target_: cuvis_ai.training.special_visualization.selector_visualizations.SelectorTemperaturePlot
    log_frequency: 1
  
  mask_viz:
    _target_: cuvis_ai.training.special_visualization.selector_visualizations.SelectorChannelMaskPlot
    log_frequency: 5
  
  stability_viz:
    _target_: cuvis_ai.training.special_visualization.selector_visualizations.SelectorStabilityPlot
    log_frequency: 1

monitoring:
  dummy:
    _target_: cuvis_ai.training.monitors.DummyMonitor
    output_dir: ./outputs/artifacts

datamodule:
  _target_: cuvis_ai.data.lentils_anomaly.LentilsAnomaly
  data_dir: ${oc.env:DATA_ROOT,./data/Lentils}
  batch_size: 4
  num_workers: 0

training:
  seed: 42
  trainer:
    max_epochs: 20
    accelerator: auto
    devices: 1
    precision: "32"
  optimizer:
    name: adam
    lr: 0.001
```

Run with:

```bash
python examples_torch/phase4_soft_channel_selector.py
```

## Troubleshooting

### Selector Concentrates on Few Channels

**Problem:** Selector selects only 2-3 channels instead of desired 15.

**Solution:**

1. Increase diversity regularizer weight: `weight=0.1`
2. Increase entropy regularizer weight: `weight=0.05`
3. Slow down temperature decay: `decay=0.95`
4. Start with higher temperature: `temperature_init=10.0`

### Selection Unstable (Keeps Changing)

**Problem:** Selected channels change drastically between epochs.

**Solution:**

1. Decrease learning rate: `lr=0.0001`
2. Increase temperature_min: `temperature_min=0.5`
3. Add gradient clipping: `gradient_clip_val=1.0`
4. Check if BCE loss is converging

### Poor Anomaly Detection Performance

**Problem:** F1 score remains low (<0.5).

**Solution:**

1. Verify ground truth masks are correct
2. Try different logit head initialization
3. Increase BCE loss weight: `bce_loss = AnomalyBCEWithLogits(weight=5.0)`
4. Check if statistical initialization works (Phase 1)
5. Simplify pipeline: Remove selector, train RX+LogitHead first

### NaN Losses

**Problem:** Training produces NaN losses.

**Solution:**

1. Reduce learning rate: `lr=0.0001`
2. Add gradient clipping: `gradient_clip_val=0.5`
3. Reduce temperature_init: `temperature_init=3.0`
4. Check for divide-by-zero in entropy computation

## Performance Tips

### Selector Overhead

Channel selection adds computational cost:

```python
# Minimal overhead (recommended)
n_select = 15  # Select ~25% of channels

# Significant overhead
n_select = 50  # Select ~80% (defeats purpose)
```

Balance: Select enough for task, but not too many.

### Training Speed

```python
# Fast (debugging)
max_epochs = 5
log_frequency = 10  # Visualizations every 10 epochs

# Normal (experimentation)
max_epochs = 20
log_frequency = 5

# Production (final model)
max_epochs = 50
log_frequency = 1
```

### Memory Usage

```python
# Low memory
batch_size = 2
precision = "16-mixed"

# High memory (faster)
batch_size = 16
precision = "32"
```

## Complete Example

See `examples_torch/phase4_soft_channel_selector.py` for the complete implementation.

Run it:

```bash
# Use defaults
python examples_torch/phase4_soft_channel_selector.py

# Override settings
python examples_torch/phase4_soft_channel_selector.py \
    training.trainer.max_epochs=10 \
    nodes.selector.n_select=10 \
    nodes.selector.temperature_init=3.0
```

## Next Steps

Congratulations! You've completed all four training phases. Now you can:

- **Experiment**: Try different selector configurations
- **Analyze**: Examine which channels were selected and why
- **Optimize**: Tune hyperparameters for your specific task
- **Deploy**: Export trained models for production use
- **Extend**: Create custom nodes and leaves for your domain

## Key Takeaways

✓ **Soft selection** - Differentiable channel selection with Gumbel-Softmax  
✓ **Temperature annealing** - Smooth transition from exploration to exploitation  
✓ **Regularization** - Entropy and diversity prevent degenerate solutions  
✓ **End-to-end training** - Anomaly detection with learned thresholds  
✓ **Interpretability** - Discover which channels matter for your task  
✓ **Selector visualizations** - Monitor temperature, selection, stability  
✓ **Dynamic restructuring** - Modify graph architecture at runtime  
✓ **Production ready** - Complete pipeline from raw data to predictions
