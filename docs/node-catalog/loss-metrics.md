!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Loss & Metrics Nodes

## Overview

Loss and metric nodes enable training supervision and performance evaluation. These nodes:

- **Loss Nodes**: Compute differentiable objectives for gradient-based optimization
- **Metric Nodes**: Track non-differentiable performance indicators for monitoring

**Key characteristics:**
- Execute only during training/validation/test (not inference)
- Support multi-loss training (combine multiple objectives)
- Provide real-time feedback for model tuning

---

## Loss Nodes

Loss nodes compute differentiable objectives for backpropagation. All inherit from `LossNode` base class.

### AnomalyBCEWithLogits

**Description:** Binary cross-entropy loss for anomaly detection with numerical stability

**Perfect for:**
- Pixel-wise anomaly detection supervision
- Handling class imbalance with `pos_weight`
- Standard binary classification loss

**Training Paradigm:** Requires labeled anomaly masks

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| predictions | float32 | (B,H,W,1) | Predicted logits | No |
| targets | bool | (B,H,W,1) | Ground truth binary masks | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| loss | float32 | () | Scalar BCE loss |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight | float | 1.0 | Overall loss weight |
| pos_weight | float | None | Positive class weight (for imbalance) |
| reduction | str | "mean" | Reduction: "mean", "sum", or "none" |

#### Example Usage (Python)

```python
from cuvis_ai.node.losses import AnomalyBCEWithLogits

# Create loss with class imbalance handling
loss = AnomalyBCEWithLogits(
    weight=1.0,
    pos_weight=10.0,  # 10x weight for anomaly pixels
    reduction="mean"
)

# Use in pipeline
pipeline.add_nodes(bce_loss=loss)
pipeline.connect(
    (logit_head.logits, bce_loss.predictions),
    (data.mask, bce_loss.targets)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  bce_loss:
    type: AnomalyBCEWithLogits
    config:
      weight: 1.0
      pos_weight: 10.0  # Handle class imbalance
      reduction: "mean"

connections:
  - [model.logits, bce_loss.predictions]
  - [data.mask, bce_loss.targets]
```

#### See Also

- [Tutorial 2: Channel Selector](../tutorials/channel-selector.md#loss-composition)
- API Reference: ::: cuvis_ai.node.losses.AnomalyBCEWithLogits

---

### DeepSVDDSoftBoundaryLoss

**Description:** Soft-boundary Deep SVDD objective with learnable radius

**Perfect for:**
- One-class anomaly detection
- Deep SVDD training
- Hypersphere boundary learning

**Training Paradigm:** Unsupervised (no labels required)

**Algorithm:**

$$
\mathcal{L} = R^2 + \frac{1}{\nu} \cdot \frac{1}{N} \sum_{i=1}^{N} \max(0, \|z_i - c\|^2 - R^2)
$$

where $R$ is the learnable hypersphere radius, $\nu \in (0,1)$ controls outlier tolerance.

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| embeddings | float32 | (B,H,W,D) | Deep SVDD embeddings | No |
| center | float32 | (D,) | Center vector from tracker | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| loss | float32 | () | Deep SVDD soft boundary loss |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| nu | float | 0.05 | Outlier fraction (0 < nu < 1) |
| weight | float | 1.0 | Overall loss weight |

#### Example Usage (Python)

```python
from cuvis_ai.node.losses import DeepSVDDSoftBoundaryLoss

# Create Deep SVDD loss
loss = DeepSVDDSoftBoundaryLoss(nu=0.05, weight=1.0)

# Use in pipeline
pipeline.add_nodes(deep_svdd_loss=loss)
pipeline.connect(
    (projection.embeddings, deep_svdd_loss.embeddings),
    (center_tracker.center, deep_svdd_loss.center)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  deep_svdd_loss:
    type: DeepSVDDSoftBoundaryLoss
    config:
      nu: 0.05  # Expect 5% outliers
      weight: 1.0

connections:
  - [projection.embeddings, deep_svdd_loss.embeddings]
  - [center_tracker.center, deep_svdd_loss.center]
```

#### Nu Parameter Guide

| nu | Behavior | Use Case |
|----|----------|----------|
| 0.01 | Tight boundary | Clean data, few outliers |
| **0.05** | **Balanced** (recommended) | **General use** |
| 0.1 | Loose boundary | Noisy data, many outliers |

#### See Also

- [Tutorial 3: Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md#training-loss)
- [DeepSVDDCenterTracker](deep-learning.md#deepsvddcentertracker)
- API Reference: ::: cuvis_ai.node.losses.DeepSVDDSoftBoundaryLoss

---

### OrthogonalityLoss

**Description:** Regularization enforcing orthonormality of PCA components

**Perfect for:**
- Trainable PCA gradient training
- Preventing component collapse
- Maintaining PCA properties during optimization

**Formula:**

$$
\mathcal{L}_{\text{orth}} = \|W W^T - I\|_F^2
$$

where $W$ is the components matrix, $I$ is identity.

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| components | float32 | (K,C) | PCA components matrix | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| loss | float32 | () | Weighted orthogonality loss |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight | float | 1.0 | Regularization strength |

#### Example Usage (Python)

```python
from cuvis_ai.node.losses import OrthogonalityLoss

# Create orthogonality regularizer
orth_loss = OrthogonalityLoss(weight=0.01)

# Use in pipeline
pipeline.add_nodes(orth_loss=orth_loss)
pipeline.connect(
    (pca.components, orth_loss.components)
)
```

#### See Also

- [TrainablePCA](deep-learning.md#trainablepca)
- [Tutorial 4: AdaCLIP Workflow](../tutorials/adaclip-workflow.md#variant-1-pca-baseline)
- API Reference: ::: cuvis_ai.node.losses.OrthogonalityLoss

---

### IoULoss

**Description:** Differentiable IoU (Intersection over Union) loss for segmentation

**Perfect for:**
- End-to-end AdaCLIP training
- Segmentation-based anomaly detection
- Direct IoU optimization

**Formula:**

$$
\mathcal{L}_{\text{IoU}} = 1 - \frac{|\text{pred} \cap \text{target}|}{|\text{pred} \cup \text{target}|}
$$

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| predictions | float32 | (B,H,W,1) | Continuous anomaly scores | No |
| targets | bool | (B,H,W,1) | Ground truth binary masks | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| loss | float32 | () | IoU loss (1 - IoU) |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight | float | 1.0 | Overall loss weight |
| smooth | float | 1e-6 | Numerical stability constant |
| normalize_method | str | "sigmoid" | Score normalization: "sigmoid", "clamp", "minmax" |

#### Example Usage (Python)

```python
from cuvis_ai.node.losses import IoULoss

# Create IoU loss
iou_loss = IoULoss(
    weight=1.0,
    smooth=1e-6,
    normalize_method="sigmoid"  # For logit inputs
)

# Use in pipeline
pipeline.add_nodes(iou_loss=iou_loss)
pipeline.connect(
    (adaclip.scores, iou_loss.predictions),
    (data.mask, iou_loss.targets)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  iou_loss:
    type: IoULoss
    config:
      weight: 1.0
      smooth: 1e-6
      normalize_method: "sigmoid"

connections:
  - [adaclip.scores, iou_loss.predictions]
  - [data.mask, iou_loss.targets]
```

#### Normalize Method Guide

| Method | Use Case | Input Range |
|--------|----------|-------------|
| **sigmoid** | Logits from model | Unbounded |
| **clamp** | Scores already in [0,1] | Near [0,1] |
| **minmax** | Varying score ranges | Any |

#### See Also

- [Tutorial 4: AdaCLIP Workflow](../tutorials/adaclip-workflow.md#variant-2-drcnn-mixer)
- API Reference: ::: cuvis_ai.node.losses.IoULoss

---

### DistinctnessLoss

**Description:** Repulsion loss encouraging selector diversity (prevents band collapse)

**Perfect for:**
- ConcreteBandSelector training
- Preventing channel collapse to same band
- Encouraging diverse band selection

**Formula:**

$$
\mathcal{L}_{\text{distinct}} = \frac{1}{N_{\text{pairs}}} \sum_{i < j} \cos(w_i, w_j)
$$

Minimizing this encourages low cosine similarity between selector vectors.

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| selection_weights | float32 | (K,C) | Selector weight matrix | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| loss | float32 | () | Repulsion loss |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight | float | 0.1 | Regularization strength |
| eps | float | 1e-6 | Stability constant |

#### Example Usage (Python)

```python
from cuvis_ai.node.losses import DistinctnessLoss

# Create distinctness regularizer
distinct_loss = DistinctnessLoss(weight=0.1)

# Use in pipeline
pipeline.add_nodes(distinct_loss=distinct_loss)
pipeline.connect(
    (selector.selection_weights, distinct_loss.selection_weights)
)
```

#### See Also

- [Tutorial 4: AdaCLIP Workflow](../tutorials/adaclip-workflow.md#variant-3-concrete-selector)
- [ConcreteBandSelector](deep-learning.md#concretebandselector)
- API Reference: ::: cuvis_ai.node.losses.DistinctnessLoss

---

### SelectorEntropyRegularizer

**Description:** Entropy regularization encouraging exploration in channel selection

**Perfect for:**
- SoftChannelSelector training
- Preventing premature selection convergence
- Balancing exploration vs exploitation

**Formula:**

$$
\mathcal{L}_{\text{entropy}} = -\sum_{i=1}^{C} p_i \log(p_i)
$$

Positive weight maximizes entropy (exploration), negative minimizes (exploitation).

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight | float | 0.01 | Regularization strength |
| target_entropy | float | None | Target entropy (if set, uses squared error) |
| eps | float | 1e-6 | Stability constant |

#### See Also

- [Tutorial 2: Channel Selector](../tutorials/channel-selector.md#entropy-regularization)
- [SoftChannelSelector](selectors.md#softchannelselector)
- API Reference: ::: cuvis_ai.node.losses.SelectorEntropyRegularizer

---

### SelectorDiversityRegularizer

**Description:** Diversity regularization via negative variance maximization

**Perfect for:**
- SoftChannelSelector training
- Encouraging spread across channels
- Preventing concentration on few channels

**Formula:**

$$
\mathcal{L}_{\text{diversity}} = -\text{Var}(w) = -\frac{1}{C} \sum_{i=1}^{C} (w_i - \bar{w})^2
$$

Minimizing loss = maximizing variance = maximizing diversity.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| weight | float | 0.01 | Regularization strength |

#### See Also

- [Tutorial 2: Channel Selector](../tutorials/channel-selector.md#diversity-regularization)
- [SoftChannelSelector](selectors.md#softchannelselector)
- API Reference: ::: cuvis_ai.node.losses.SelectorDiversityRegularizer

---

## Metrics Nodes

Metric nodes compute non-differentiable performance indicators for monitoring. Execute only during validation/test.

### AnomalyDetectionMetrics

**Description:** Comprehensive anomaly detection metrics (precision, recall, F1, IoU, AP)

**Perfect for:**
- Evaluating anomaly detection performance
- Model selection and hyperparameter tuning
- Performance reporting

**Metrics Computed:**
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1 Score**: Harmonic mean of precision and recall
- **IoU**: Intersection over Union (Jaccard index)
- **Average Precision**: Area under precision-recall curve (if logits provided)

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| decisions | bool | (B,H,W,1) | Binary anomaly decisions | No |
| targets | bool | (B,H,W,1) | Ground truth binary masks | No |
| logits | float32 | (B,H,W,1) | Optional logits for AP | Yes |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| metrics | list | () | List of Metric objects |

#### Example Usage (Python)

```python
from cuvis_ai.node.metrics import AnomalyDetectionMetrics

# Create metrics node
metrics = AnomalyDetectionMetrics()

# Use in pipeline
pipeline.add_nodes(metrics=metrics)
pipeline.connect(
    (decider.decisions, metrics.decisions),
    (data.mask, metrics.targets),
    (logit_head.logits, metrics.logits)  # Optional for AP
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  metrics:
    type: AnomalyDetectionMetrics

connections:
  - [decider.decisions, metrics.decisions]
  - [data.mask, metrics.targets]
  - [logit_head.logits, metrics.logits]  # Optional
```

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md#metrics-tracking)
- [BinaryDecider](utility.md#binarydecider)
- API Reference: ::: cuvis_ai.node.metrics.AnomalyDetectionMetrics

---

### ExplainedVarianceMetric

**Description:** Tracks PCA explained variance ratios and cumulative variance

**Perfect for:**
- Monitoring PCA component quality
- Determining optimal number of components
- Validating dimensionality reduction

**Metrics Computed:**
- Per-component variance: `explained_variance_pc1`, `explained_variance_pc2`, ...
- Total variance: Sum of all components
- Cumulative variance: Running sum of components

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| explained_variance_ratio | float32 | (K,) | Variance ratios from PCA | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| metrics | list | () | List of Metric objects |

#### Example Usage (Python)

```python
from cuvis_ai.node.metrics import ExplainedVarianceMetric

# Create variance metric
var_metric = ExplainedVarianceMetric()

# Use in pipeline
pipeline.add_nodes(var_metric=var_metric)
pipeline.connect(
    (pca.explained_variance_ratio, var_metric.explained_variance_ratio)
)
```

#### See Also

- [TrainablePCA](deep-learning.md#trainablepca)
- [Tutorial 4: AdaCLIP Workflow](../tutorials/adaclip-workflow.md#variant-1-pca-baseline)
- API Reference: ::: cuvis_ai.node.metrics.ExplainedVarianceMetric

---

### AnomalyPixelStatisticsMetric

**Description:** Computes pixel-level statistics for anomaly detection results

**Perfect for:**
- Monitoring anomaly detection behavior
- Tracking proportion of detected anomalies per batch
- Quick sanity checks on detection outputs

**Metrics Computed:**
- **Total Pixels**: `anomaly/total_pixels` - Total number of pixels in the batch
- **Anomalous Pixels**: `anomaly/anomalous_pixels` - Count of pixels classified as anomalies
- **Anomaly Percentage**: `anomaly/anomaly_percentage` - Percentage of anomalous pixels

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| decisions | bool | (B,H,W,1) | Binary anomaly decisions | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| metrics | list | () | List of Metric objects |

#### Example Usage (Python)

```python
from cuvis_ai.node.metrics import AnomalyPixelStatisticsMetric

# Create pixel statistics metric
pixel_stats = AnomalyPixelStatisticsMetric()

# Use in pipeline
pipeline.add_nodes(pixel_stats=pixel_stats)
pipeline.connect(
    (decider.decisions, pixel_stats.decisions),
    (pixel_stats.metrics, monitor.metrics)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  pixel_stats:
    type: AnomalyPixelStatisticsMetric

connections:
  - [decider.decisions, pixel_stats.decisions]
  - [pixel_stats.metrics, monitor.metrics]
```

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md#metrics-tracking)
- [BinaryDecider](utility.md#binarydecider)
- [AnomalyDetectionMetrics](#anomalydetectionmetrics)
- API Reference: ::: cuvis_ai.node.metrics.AnomalyPixelStatisticsMetric

---

## Multi-Loss Training

Combine multiple loss objectives for complex training scenarios:

### Example: Channel Selector with Regularization

```yaml
nodes:
  # Primary supervision
  bce_loss:
    type: AnomalyBCEWithLogits
    config:
      weight: 1.0
      pos_weight: 10.0

  # Selector regularizers
  entropy_reg:
    type: SelectorEntropyRegularizer
    config:
      weight: 0.01  # Encourage exploration

  diversity_reg:
    type: SelectorDiversityRegularizer
    config:
      weight: 0.01  # Encourage spread

connections:
  - [logit_head.logits, bce_loss.predictions]
  - [data.mask, bce_loss.targets]
  - [selector.weights, entropy_reg.weights]
  - [selector.weights, diversity_reg.weights]

# Combined loss = 1.0*BCE + 0.01*entropy + 0.01*diversity
```

### Example: AdaCLIP with IoU + Distinctness

```yaml
nodes:
  # Primary loss
  iou_loss:
    type: IoULoss
    config:
      weight: 1.0

  # Regularizer for band diversity
  distinct_loss:
    type: DistinctnessLoss
    config:
      weight: 0.1

connections:
  - [adaclip.scores, iou_loss.predictions]
  - [data.mask, iou_loss.targets]
  - [selector.selection_weights, distinct_loss.selection_weights]

# Combined loss = 1.0*IoU + 0.1*distinctness
```

---

## Loss Weight Tuning Guide

| Loss Component | Typical Range | Effect |
|----------------|---------------|--------|
| **Primary loss** (BCE, IoU) | 1.0 | Main supervision signal |
| **Orthogonality** | 0.001 - 0.1 | Prevent PCA collapse |
| **Entropy reg** | 0.001 - 0.05 | Balance exploration |
| **Diversity reg** | 0.001 - 0.05 | Encourage spread |
| **Distinctness** | 0.01 - 0.5 | Prevent band collapse |

**Tuning strategy:**
1. Start with primary loss only
2. Add regularizers with low weights (0.01)
3. Increase regularizer weights if needed
4. Monitor metrics to validate improvement

---

## Additional Resources

- **Tutorial:** [Channel Selector](../tutorials/channel-selector.md) - Multi-loss training
- **Tutorial:** [Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md) - Deep SVDD loss
- **Tutorial:** [AdaCLIP Workflow](../tutorials/adaclip-workflow.md) - IoU + distinctness
- **Concepts:** [Two-Phase Training](../concepts/two-phase-training.md)
- **API Reference:** [cuvis_ai.node.losses](../../api/node/#losses)
- **API Reference:** [cuvis_ai.node.metrics](../../api/node/#metrics)
