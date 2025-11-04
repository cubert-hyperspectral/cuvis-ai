# Feature Reference

Comprehensive capability matrix for CUVIS.AI's training pipeline.

## Overview

CUVIS.AI provides a modular, graph-based training pipeline for hyperspectral data analysis. This page documents all available components, their capabilities, and integration points.

---

## Node Types

### Preprocessing Nodes

#### MinMaxNormalizer
**Purpose**: Min-max normalization to [0, 1] range

**Features**:
- Running statistics tracking across dataset
- Statistical initialization from data
- Optional trainable parameters
- Numerical stability via epsilon parameter

**Configuration**:
```yaml
normalizer:
  _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
  eps: 1e-6
  use_running_stats: true
```

**Input/Output**: `[B, H, W, C]` → `[B, H, W, C]`

#### StandardNormalizer
**Purpose**: Z-score normalization (mean=0, std=1)

**Features**:
- Per-channel or global normalization
- Running mean/std tracking
- Statistical initialization

**Configuration**:
```yaml
normalizer:
  _target_: cuvis_ai.normalization.normalization.StandardNormalizer
  eps: 1e-6
  per_channel: true
```

**Input/Output**: `[B, H, W, C]` → `[B, H, W, C]`

---

### Feature Extraction

#### TrainablePCA
**Purpose**: Principal Component Analysis with gradient-based fine-tuning

**Capabilities**:
- **Statistical Initialization**: SVD decomposition for initial components
- **Gradient Training**: Fine-tune components via backpropagation
- **Orthogonality Constraints**: Maintain component orthonormality
- **Variance Tracking**: Monitor explained variance per component

**Features**:
- Configurable number of components
- Optional whitening transformation
- Center data before projection
- Explained variance computation

**Configuration**:
```yaml
pca:
  _target_: cuvis_ai.node.pca.TrainablePCA
  n_components: 3
  trainable: true
  whiten: false
  center: true
```

**Input/Output**: `[B, H, W, C]` → `[B, H, W, K]` where K = n_components

**Training Workflow**:
1. **Phase 1**: SVD initialization from training data
2. **Phase 2**: Gradient-based fine-tuning with OrthogonalityLoss
3. **Inference**: Fixed projection using learned components

#### SoftChannelSelector
**Purpose**: Learnable channel selection with temperature annealing

**Capabilities**:
- **Soft Selection**: Differentiable channel weighting during training
- **Hard Selection**: Top-k selection at inference time
- **Temperature Scheduling**: Gumbel-Softmax with configurable decay
- **Initialization Methods**: Uniform or variance-based importance

**Features**:
- Configurable number of channels to select
- Temperature annealing: exponential decay per epoch
- Entropy regularization for exploration
- Diversity regularization for spread
- Selection stability tracking

**Configuration**:
```yaml
selector:
  _target_: cuvis_ai.node.selector.SoftChannelSelector
  n_select: 15
  init_method: variance  # or "uniform"
  temperature_init: 5.0
  temperature_min: 0.1
  temperature_decay: 0.9
  trainable: true
```

**Input/Output**: `[B, H, W, C]` → `[B, H, W, C]` (reweighted)

**Temperature Schedule**: `T(epoch) = max(T_min, T_init * decay^epoch)`

---

### Anomaly Detection

#### RXGlobal
**Purpose**: Reed-Xiaoli (RX) global anomaly detector

**Capabilities**:
- **Statistical Fitting**: Estimate global mean and covariance
- **Mahalanobis Distance**: Compute anomaly scores
- **Optional Training**: Fine-tune statistics with gradients

**Features**:
- Efficient covariance computation
- Numerical stability via epsilon
- Configurable trainable statistics
- Batch processing support

**Configuration**:
```yaml
rx:
  _target_: cuvis_ai.anomaly.rx_detector.RXGlobal
  eps: 1e-6
  trainable_stats: false
```

**Input/Output**: `[B, H, W, C]` → `[B, H, W, 1]` (anomaly scores)

**Formula**: `score = (x - μ)ᵀ Σ⁻¹ (x - μ)`

#### RXLogitHead
**Purpose**: Trainable anomaly threshold with learnable parameters

**Capabilities**:
- **Statistical Initialization**: Threshold from score distribution (mean + 2σ)
- **Gradient Training**: Learn optimal threshold via BCE loss
- **Affine Transformation**: `logit = scale * (score - bias)`

**Features**:
- Learnable scale and bias parameters
- End-to-end training with labels
- Compatible with BCE loss
- Sigmoid activation for binary prediction

**Configuration**:
```yaml
logit_head:
  _target_: cuvis_ai.anomaly.rx_logit_head.RXLogitHead
  init_scale: 1.0
  init_bias: 5.0
  trainable: true
```

**Input/Output**: `[B, H, W, 1]` → `[B, H, W, 1]` (logits)

---

## Training Infrastructure

### Training Modes

| Mode | max_epochs | GPU Required | Use Case |
|------|-----------|--------------|----------|
| **Statistical Only** | 0 | No | Fast baseline, initialization-only |
| **Gradient Training** | >0 | Recommended | Fine-tune initialized models |
| **Mixed (Recommended)** | >0 with statistical init | Recommended | Best performance |

### Training Phases

#### Phase 1: Statistical Initialization
- Automatically identifies nodes with `requires_initial_fit=True`
- Sorts nodes topologically (respects dependencies)
- Passes transformed data through parent nodes
- Calls `initialize_from_data()` on each node
- Converts to trainable parameters or freezes

**Nodes with Statistical Init**:
- MinMaxNormalizer (running_min, running_max)
- RXGlobal (mean μ, covariance Σ)
- TrainablePCA (SVD components)
- SoftChannelSelector (channel importance)

#### Phase 2: Gradient Training
- Creates PyTorch Lightning module
- Configures optimizer and scheduler
- Runs training loop with backpropagation
- Generates visualizations and metrics
- Logs to monitoring backends

---

## Loss Functions

### OrthogonalityLoss
**Purpose**: Maintain PCA component orthonormality

**Formula**: `L = weight * ||W @ Wᵀ - I||²_F`

**Compatible Parents**: TrainablePCA

**Configuration**:
```yaml
loss_leaves:
  orthogonality:
    _target_: cuvis_ai.training.losses.OrthogonalityLoss
    weight: 1.0
```

**Usage**: Regularization during PCA fine-tuning

### AnomalyBCEWithLogits
**Purpose**: Binary cross-entropy for anomaly detection

**Features**:
- Numerically stable (uses logits)
- Class imbalance handling via `pos_weight`
- Automatic accuracy computation

**Compatible Parents**: Any node outputting logits

**Configuration**:
```yaml
loss_leaves:
  anomaly_bce:
    _target_: cuvis_ai.training.losses.AnomalyBCEWithLogits
    weight: 1.0
    pos_weight: 10.0  # For imbalanced datasets
    reduction: mean
```

### MSEReconstructionLoss
**Purpose**: Mean squared error for reconstruction tasks

**Features**:
- Flexible target specification (labels or metadata)
- SNR computation
- Configurable reduction

**Compatible Parents**: Any reconstruction node

**Configuration**:
```yaml
loss_leaves:
  reconstruction:
    _target_: cuvis_ai.training.losses.MSEReconstructionLoss
    reduction: mean
```

### SelectorEntropyRegularizer
**Purpose**: Encourage exploration in channel selection

**Formula**: `L = -weight * entropy` where `entropy = -Σ(p log p)`

**Features**:
- Positive weight → maximize entropy (exploration)
- Negative weight → minimize entropy (exploitation)
- Optional target entropy

**Compatible Parents**: SoftChannelSelector

**Configuration**:
```yaml
loss_leaves:
  entropy_reg:
    _target_: cuvis_ai.training.losses.SelectorEntropyRegularizer
    weight: 0.01  # Positive for exploration
```

**Typical Schedule**: High weight early → low weight late

### SelectorDiversityRegularizer
**Purpose**: Prevent concentration on few channels

**Formula**: `L = weight * (-variance(weights))`

**Compatible Parents**: SoftChannelSelector

**Configuration**:
```yaml
loss_leaves:
  diversity_reg:
    _target_: cuvis_ai.training.losses.SelectorDiversityRegularizer
    weight: 0.01
```

### WeightedMultiLoss
**Purpose**: Aggregate multiple losses with configurable weights

**Features**:
- Dynamic loss combination
- Per-loss weight configuration
- Hierarchical loss organization

**Configuration**: Programmatic only (see API docs)

---

## Metrics

### ExplainedVarianceMetric
**Purpose**: Track PCA explained variance

**Features**:
- Per-component variance ratios
- Cumulative variance (sum to 1.0)
- Monotonic increase verification

**Compatible Parents**: TrainablePCA

**Configuration**:
```yaml
metric_leaves:
  explained_var:
    _target_: cuvis_ai.training.metrics.ExplainedVarianceMetric
```

**Output**: `{comp_0: 0.85, comp_1: 0.10, comp_2: 0.05, cumulative: 1.0}`

### ComponentOrthogonalityMetric
**Purpose**: Monitor PCA orthogonality during training

**Features**:
- Frobenius norm of `WᵀW - I`
- Tracks degradation over training
- Validates orthogonality constraints

**Compatible Parents**: TrainablePCA

**Configuration**:
```yaml
metric_leaves:
  orthogonality:
    _target_: cuvis_ai.training.metrics.ComponentOrthogonalityMetric
```

**Output**: `{orthogonality_error: 0.001}`

### AnomalyDetectionMetrics
**Purpose**: Comprehensive anomaly detection evaluation

**Features**:
- Precision, recall, F1 score
- IoU (Intersection over Union)
- Confusion matrix (TP, TN, FP, FN)
- Configurable threshold

**Compatible Parents**: Any anomaly detection node

**Configuration**:
```yaml
metric_leaves:
  anomaly_metrics:
    _target_: cuvis_ai.training.metrics.AnomalyDetectionMetrics
    threshold: 0.5
```

**Output**: `{precision: 0.9, recall: 0.85, f1: 0.87, iou: 0.77, ...}`

### ScoreStatisticsMetric
**Purpose**: Distribution statistics for anomaly scores

**Features**:
- Mean, std, min, max
- Quantiles (25%, 50%, 75%, 95%, 99%)
- Outlier detection

**Compatible Parents**: Any node outputting scores

**Configuration**:
```yaml
metric_leaves:
  score_stats:
    _target_: cuvis_ai.training.metrics.ScoreStatisticsMetric
```

---

## Visualizations

### PCAVisualization
**Purpose**: 2D/3D scatter plots of PCA projections

**Features**:
- Automatic 2D/3D mode based on n_components
- Color by labels or scores
- Subsampling for large datasets
- Explained variance in axis labels

**Configuration**:
```yaml
visualization_leaves:
  pca_viz:
    _target_: cuvis_ai.training.visualizations.PCAVisualization
    log_frequency: 5
    subsample: 1000
```

**Output**: Matplotlib figure with scatter plot

### AnomalyHeatmap
**Purpose**: Spatial heatmap of anomaly scores

**Features**:
- 3-panel layout: raw heatmap, overlay, binary mask
- Configurable colormap and range
- Threshold-based binary prediction
- Batch indexing for visualization

**Configuration**:
```yaml
visualization_leaves:
  heatmap:
    _target_: cuvis_ai.training.visualizations.AnomalyHeatmap
    log_frequency: 10
    batch_idx: 0
    colormap: 'hot'
```

**Output**: 3-subplot PNG with heatmap, overlay, and mask

### ScoreHistogram
**Purpose**: Distribution histogram with statistics

**Features**:
- Histogram with KDE overlay
- Summary statistics (mean, std, quantiles)
- Anomaly threshold line
- Log-scale option

**Configuration**:
```yaml
visualization_leaves:
  histogram:
    _target_: cuvis_ai.training.visualizations.ScoreHistogram
    log_frequency: 10
    n_bins: 50
```

**Output**: Histogram plot with annotations

### SelectorTemperaturePlot
**Purpose**: Temperature annealing curve over training

**Features**:
- Temperature vs epoch/step
- Exponential decay visualization
- Min temperature threshold line

**Configuration**:
```yaml
visualization_leaves:
  temp_plot:
    _target_: cuvis_ai.training.special_visualization.selector_visualizations.SelectorTemperaturePlot
    log_frequency: 1
```

**Output**: Line plot of temperature schedule

### SelectorChannelMaskPlot
**Purpose**: Channel selection probabilities and hard mask

**Features**:
- 2-panel: soft weights + hard selection
- Channel index labeling
- Selection threshold visualization
- Wavelength mapping (if available)

**Configuration**:
```yaml
visualization_leaves:
  channel_mask:
    _target_: cuvis_ai.training.special_visualization.selector_visualizations.SelectorChannelMaskPlot
    log_frequency: 5
```

**Output**: Bar plots showing selection weights and mask

### SelectorStabilityPlot
**Purpose**: Selection stability tracking via Jaccard similarity

**Features**:
- Jaccard similarity between consecutive selections
- Change tracking over training
- Stability convergence visualization

**Configuration**:
```yaml
visualization_leaves:
  stability:
    _target_: cuvis_ai.training.special_visualization.selector_visualizations.SelectorStabilityPlot
    log_frequency: 1
```

**Output**: Line plot of selection stability

---

## Monitoring Backends

### DummyMonitor
**Purpose**: Filesystem-based monitoring (no external dependencies)

**Features**:
- JSONL metric logging (one line per log)
- PKL artifact persistence
- PNG thumbnail generation
- Configurable output directory
- Filename sanitization

**Configuration**:
```yaml
monitoring:
  dummy:
    _target_: cuvis_ai.training.monitors.DummyMonitor
    output_dir: ./outputs/artifacts
    enabled: true
```

**Output Structure**:
```
outputs/artifacts/
├── metrics_train.jsonl
├── metrics_val.jsonl
├── step_000000/
│   ├── train/viz/parent_PCAVisualization_0.pkl
│   ├── train/viz/parent_PCAVisualization_0.png
│   └── ...
└── step_000010/
    └── ...
```

**Metrics Format** (JSONL):
```json
{"step": 0, "epoch": 0, "loss": 0.5, "accuracy": 0.9}
{"step": 1, "epoch": 0, "loss": 0.4, "accuracy": 0.92}
```

### WandBMonitor
**Purpose**: Weights & Biases experiment tracking

**Features**:
- Web UI dashboard
- Hyperparameter sweeps
- Artifact versioning
- Team collaboration
- Online/offline modes
- Graceful degradation (works without wandb installed)

**Configuration**:
```yaml
monitoring:
  wandb:
    _target_: cuvis_ai.training.monitors.WandBMonitor
    project: cuvis-ai-experiments
    entity: my-team
    tags: [pca, anomaly-detection]
    mode: online  # or "offline", "disabled"
    enabled: true
```

**Requirements**: `wandb` package + API key

**Environment**:
```bash
export WANDB_API_KEY=your_key_here
```

### TensorBoardMonitor
**Purpose**: Local TensorBoard logging

**Features**:
- Local web UI (`tensorboard --logdir=...`)
- Scalar metrics
- Image/figure logging
- Histogram tracking
- Configurable flush intervals
- Graceful degradation (works without tensorboard installed)

**Configuration**:
```yaml
monitoring:
  tensorboard:
    _target_: cuvis_ai.training.monitors.TensorBoardMonitor
    log_dir: ./outputs/tensorboard
    flush_secs: 30
    enabled: true
```

**Requirements**: `tensorboard` package

**Usage**:
```bash
tensorboard --logdir=./outputs/tensorboard
```

---

## Configuration System

### Hydra Integration

**Features**:
- **Composition**: Combine multiple config files via `defaults:`
- **CLI Overrides**: Modify any parameter from command line
- **Structured Configs**: Type-safe dataclass-based configs
- **Instantiation**: `hydra.utils.instantiate()` for object creation
- **Config Groups**: Organize related configs

**Example Config** (`conf/train.yaml`):
```yaml
defaults:
  - general
  - wandb
  - _self_

graph:
  name: my_experiment

nodes:
  normalizer:
    _target_: cuvis_ai.normalization.normalization.MinMaxNormalizer
  pca:
    _target_: cuvis_ai.node.pca.TrainablePCA
    n_components: 3

training:
  seed: 42
  trainer:
    max_epochs: 10
    accelerator: auto
```

**CLI Overrides**:
```bash
python train.py training.trainer.max_epochs=20 nodes.pca.n_components=5
```

### Environment Variables

**Syntax**: `${oc.env:VAR_NAME,default_value}`

**Example**:
```yaml
datamodule:
  data_dir: ${oc.env:DATA_ROOT,./data}
  
monitoring:
  wandb:
    api_key: ${oc.env:WANDB_API_KEY}
```

### Type Safety

All configs use dataclasses with type hints:

```python
@dataclass
class TrainerConfig:
    max_epochs: int = 10
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32"
```

**Benefits**:
- IDE autocompletion
- Type checking
- Validation at instantiation
- Self-documenting code

---

## Advanced Features

### Dynamic Graph Restructuring

**Feature**: Runtime graph modification via `graph.set_parent()`

**Use Cases**:
- Insert selector nodes into existing pipelines
- Swap preprocessing steps
- A/B testing different configurations

**Example**:
```python
# Original: normalizer → pca
graph.add_node(normalizer)
graph.add_node(pca, parent=normalizer)

# Insert selector: normalizer → selector → pca
selector = SoftChannelSelector(n_select=15)
graph.add_node(selector, parent=normalizer)
graph.set_parent(pca, selector)
```

**Validation**:
- Dimension constraint checking
- Cycle detection
- Graph integrity verification
- Automatic rollback on failure

### Leaf Node Validation

**Feature**: Automatic parent compatibility checking

**Protocol**:
```python
class MyLossNode(LossNode):
    compatible_parent_types = (TrainablePCA,)
    required_parent_attributes = ("compute_loss",)
```

**Benefits**:
- Prevent configuration errors
- Self-documenting requirements
- Clear error messages

### Serialization

**Features**:
- Complete graph structure saved
- Node hyperparameters preserved
- Trained weights included
- Version tracking
- Reproducibility

**Format**: ZIP archive with YAML + binary data

**Usage**:
```python
# Save
graph.save_to_file("model.zip")

# Load
graph = Graph.load_from_file("model.zip")
```

### Mixed Precision Training

**Feature**: FP16/BF16 for faster training

**Configuration**:
```yaml
training:
  trainer:
    precision: 16-mixed  # or "bf16-mixed"
```

**Benefits**:
- 2-3x speedup on modern GPUs
- Reduced memory usage
- Automatic loss scaling

### Multi-GPU Training

**Feature**: Data parallel training (DDP)

**Configuration**:
```yaml
training:
  trainer:
    accelerator: gpu
    devices: 4  # Use 4 GPUs
    strategy: ddp
```

**Scaling**: Linear speedup up to 4 GPUs (tested)

---

## Integration Patterns

### Example 1: Anomaly Detection Pipeline
```yaml
# Normalizer → Selector → RX → LogitHead
nodes:
  normalizer: {_target_: ...MinMaxNormalizer}
  selector: {_target_: ...SoftChannelSelector, n_select: 15}
  rx: {_target_: ...RXGlobal}
  logit_head: {_target_: ...RXLogitHead}

loss_leaves:
  bce: {_target_: ...AnomalyBCEWithLogits}
  entropy: {_target_: ...SelectorEntropyRegularizer}

metric_leaves:
  anomaly_metrics: {_target_: ...AnomalyDetectionMetrics}
```

### Example 2: PCA Analysis Pipeline
```yaml
# Normalizer → PCA
nodes:
  normalizer: {_target_: ...MinMaxNormalizer}
  pca: {_target_: ...TrainablePCA, n_components: 3}

loss_leaves:
  orthogonality: {_target_: ...OrthogonalityLoss}

metric_leaves:
  explained_var: {_target_: ...ExplainedVarianceMetric}
  
visualization_leaves:
  pca_viz: {_target_: ...PCAVisualization}
```

---

## Best Practices

### 1. Statistical Initialization
✓**Do**: Always use statistical initialization for better convergence
```yaml
training:
  trainer:
    max_epochs: 10  # Not 0
```

❌ **Don't**: Skip statistical init and start from random
```python
# This loses valuable initialization info
pca = TrainablePCA()  # Random init, no data
```

### 2. Regularization
✓**Do**: Balance exploration and exploitation
```yaml
loss_leaves:
  entropy: {weight: 0.01}  # Early training
  diversity: {weight: 0.01}
```

❌ **Don't**: Use conflicting regularizers
```yaml
# Don't minimize AND maximize entropy
entropy: {weight: -0.01}  # Minimize (exploit)
entropy2: {weight: 0.01}  # Maximize (explore)
```

### 3. Monitoring
✓**Do**: Use multiple backends for different purposes
```yaml
monitoring:
  dummy: {enabled: true}  # Always on
  wandb: {enabled: true}   # For experiments
```

❌ **Don't**: Rely solely on external services
```yaml
monitoring:
  wandb: {enabled: true}  # What if API is down?
```

### 4. Logging Frequency
✓**Do**: Balance detail vs performance
```yaml
visualization_leaves:
  pca_viz: {log_frequency: 5}     # Every 5 steps
  heatmap: {log_frequency: 10}    # Every 10 steps
```

❌ **Don't**: Log every step for expensive operations
```yaml
heatmap: {log_frequency: 1}  # 20% overhead!
```

---

## Performance Characteristics

### Computational Complexity

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| RX Covariance | O(C²N) | C=channels, N=samples |
| PCA SVD | O(C²N + C³) | Dominated by SVD |
| Selector Forward | O(HWC) | Linear in spatial dims |
| Orthogonality Loss | O(K²) | K=components |
| Visualization | O(N) or O(N log N) | Depends on subsample |

### Memory Requirements

| Component | Memory | Scaling |
|-----------|--------|---------|
| RX Covariance | O(C²) | 200 channels → 320KB (float32) |
| PCA Components | O(KC) | 3 comps × 200 channels → 2.4KB |
| Batch Data | O(BHWC) | 4×256×256×100 → 100MB |
| Gradients | 2× parameters | Same as parameters |

### Bottlenecks

1. **Statistical Init**: RX covariance for C>200
   - **Solution**: Use variance-based selector first
2. **Visualization**: Generation at every step
   - **Solution**: Increase log_frequency
3. **Disk I/O**: Saving large artifacts
   - **Solution**: Reduce batch_idx count in heatmaps

---

## Capability Matrix

| Feature | Statistical Init | Gradient Training | Visualization | Monitoring |
|---------|-----------------|-------------------|---------------|------------|
| **MinMaxNormalizer** | ✓| ⚠️ Optional | ➖ | ➖ |
| **TrainablePCA** | ✓SVD | ✓+ Orth Loss | ✓2D/3D | ✓Variance |
| **SoftChannelSelector** | ✓Variance | ✓+ Entropy/Div | ✓Masks | ✓Stability |
| **RXGlobal** | ✓Cov | ⚠️ Optional | ✓Heatmap | ✓Stats |
| **RXLogitHead** | ✓Threshold | ✓+ BCE | ✓Heatmap | ✓Metrics |

Legend: ✓Supported | ⚠️ Optional | ➖ N/A

---

## Version History

| Version | Date | Key Features |
|---------|------|-------------|
| 0.1.0 | 2025-10 | Initial release with all Phase 1-5 features |

---

## Next Steps

- **[API Reference](../api/pipeline.md)**: Detailed API documentation
- **[Tutorials](../tutorials/phase1_statistical.md)**: Step-by-step guides
- **[Limitations](../user-guide/limitations.md)**: Known constraints
- **[Stress Testing](stress_testing.md)**: Performance benchmarks
