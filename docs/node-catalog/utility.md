!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Utility Nodes

## Overview

Utility nodes provide essential supporting functions like decision thresholding, score transformation, and label mapping. These nodes:

- Enable flexible decision boundaries (fixed threshold vs adaptive quantile)
- Transform raw scores to logit space for loss functions
- Convert multi-class labels to binary anomaly masks
- Support two-stage decision strategies

---

## Nodes in This Category

### BinaryDecider

**Description:** Simple fixed-threshold decision node with sigmoid transformation

**Perfect for:**
- Basic binary classification
- Fixed threshold experiments
- Production inference with known threshold

**Training Paradigm:** None (stateless transform)

**Algorithm:**

1. Apply sigmoid: $p = \sigma(\text{logits})$
2. Threshold: $\text{decision} = (p \geq \text{threshold})$

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| logits | float32 | (B,H,W,C) | Input logits | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| decisions | bool | (B,H,W,1) | Binary decision mask |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| threshold | float | 0.5 | Decision threshold (after sigmoid) |

#### Example Usage (Python)

```python
from cuvis_ai.deciders.binary_decider import BinaryDecider

# Create decider
decider = BinaryDecider(threshold=0.5)

# Use in pipeline
pipeline.add_nodes(decider=decider)
pipeline.connect(
    (logit_head.logits, decider.logits),
    (decider.decisions, metrics.decisions)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  decider:
    type: BinaryDecider
    config:
      threshold: 0.5  # Adjust based on precision/recall trade-off

connections:
  - [logit_head.logits, decider.logits]
  - [decider.decisions, metrics.decisions]
```

#### Threshold Selection Guide

| Threshold | Behavior | Use Case |
|-----------|----------|----------|
| 0.3 | High recall, low precision | Minimize false negatives |
| **0.5** | **Balanced** (default) | **General use** |
| 0.7 | Low recall, high precision | Minimize false positives |

**Recommendation:** Tune threshold on validation set to optimize target metric (F1, IoU, etc.).

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md#decision-thresholding)
- [QuantileBinaryDecider](#quantilebinarydecider) - Adaptive threshold
- API Reference: ::: cuvis_ai.deciders.binary_decider.BinaryDecider

---

### QuantileBinaryDecider

**Description:** Adaptive quantile-based thresholding computed per batch

**Perfect for:**
- Adaptive anomaly detection
- Deep SVDD with variable score distributions
- Unknown optimal threshold scenarios

**Training Paradigm:** None (stateless transform)

**Algorithm:**

1. Compute quantile threshold per batch: $t = \text{quantile}(\text{logits}, q)$
2. Threshold: $\text{decision} = (\text{logits} \geq t)$

Default: Reduce over (H, W, C) dimensions for (B, H, W, C) input.

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| logits | float32 | (B,H,W,C) | Input logits/scores | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| decisions | bool | (B,H,W,1) | Binary decision mask |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| quantile | float | 0.995 | Quantile threshold in [0,1] |
| reduce_dims | Sequence[int] | None | Dims to reduce (None = all non-batch) |

#### Example Usage (Python)

```python
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider

# Create quantile decider
decider = QuantileBinaryDecider(
    quantile=0.995,  # Top 0.5% marked as anomalies
    reduce_dims=None  # Default: reduce over (H, W, C)
)

# Use in pipeline
pipeline.add_nodes(decider=decider)
pipeline.connect(
    (scorer.scores, decider.logits),
    (decider.decisions, metrics.decisions)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  decider:
    type: QuantileBinaryDecider
    config:
      quantile: 0.995  # Top 0.5% pixels
      reduce_dims: null  # Reduce over all non-batch dims

connections:
  - [deep_svdd_scores.scores, decider.logits]
  - [decider.decisions, metrics.decisions]
```

#### Quantile Selection Guide

| Quantile | % Anomalies | Use Case |
|----------|-------------|----------|
| 0.99 | ~1% | High anomaly ratio |
| **0.995** | **~0.5%** (recommended) | **Balanced** |
| 0.999 | ~0.1% | Low anomaly ratio |

**Recommendation:** Match quantile to expected anomaly frequency in your data.

#### See Also

- [Tutorial 3: Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md#quantile-thresholding)
- [DeepSVDDScores](deep-learning.md#deepsvddscores)
- API Reference: ::: cuvis_ai.deciders.binary_decider.QuantileBinaryDecider

---

### TwoStageBinaryDecider

**Description:** Two-stage decision: image-level gate → pixel-level quantile threshold

**Perfect for:**
- Reducing false positives on normal images
- Hierarchical anomaly detection
- High-precision applications

**Training Paradigm:** None (stateless transform)

**Algorithm:**

**Stage 1 (Image-level gate):**
1. Compute image score: Mean of top-k% pixels
2. If image score < threshold → Return blank mask (no anomalies)

**Stage 2 (Pixel-level):**
3. If gate passed → Apply quantile thresholding per image

This prevents false positives on normal images while allowing fine-grained detection on anomalous images.

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| logits | float32 | (B,H,W,C) | Input logits/scores | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| decisions | bool | (B,H,W,1) | Binary decision mask |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| image_threshold | float | 0.5 | Stage 1 gate threshold |
| top_k_fraction | float | 0.001 | Fraction of pixels for image score |
| quantile | float | 0.995 | Stage 2 quantile threshold |
| reduce_dims | Sequence[int] | None | Dims to reduce (Stage 2) |

#### Example Usage (Python)

```python
from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider

# Create two-stage decider
decider = TwoStageBinaryDecider(
    image_threshold=0.5,  # Gate: requires high max score
    top_k_fraction=0.001,  # Use top 0.1% for image score
    quantile=0.995  # Pixel threshold if gate passes
)

# Use in pipeline
pipeline.add_nodes(decider=decider)
pipeline.connect(
    (scorer.scores, decider.logits),
    (decider.decisions, metrics.decisions)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  two_stage_decider:
    type: TwoStageBinaryDecider
    config:
      image_threshold: 0.5
      top_k_fraction: 0.001
      quantile: 0.995

connections:
  - [scorer.scores, two_stage_decider.logits]
  - [two_stage_decider.decisions, metrics.decisions]
```

#### When to Use

**Use TwoStageBinaryDecider when:**
- Need high precision (minimize false positives)
- Most images are normal (sparse anomalies)
- Can tolerate missing subtle anomalies

**Use QuantileBinaryDecider when:**
- Need balanced precision/recall
- Anomalies present in most images
- Want simpler single-stage thresholding

#### See Also

- [QuantileBinaryDecider](#quantilebinarydecider)
- API Reference: ::: cuvis_ai.deciders.two_stage_decider.TwoStageBinaryDecider

---

### ScoreToLogit

**Description:** Trainable affine transformation converting RX scores to logits

**Perfect for:**
- Two-phase RX training (statistical init → gradient fine-tuning)
- Calibrating RX scores for BCE loss
- Learning optimal score scaling

**Training Paradigm:** Two-phase (statistical init → unfreeze → gradient training)

**Algorithm:**

$$
\text{logit} = \text{scale} \cdot (\text{score} - \text{bias})
$$

**Statistical initialization:**
- `bias = mean(scores) + 2*std(scores)` (anomaly threshold estimate)
- `scale = 1.0` (initial scaling)

**Gradient training:** Unfreeze to optimize `scale` and `bias` via backprop.

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| scores | float32 | (B,H,W,K) | RX anomaly scores | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| logits | float32 | (B,H,W,K) | Calibrated logits |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| init_scale | float | 1.0 | Initial value for the scale parameter |
| init_bias | float | 0.0 | Initial value for the bias parameter (threshold) |

#### Example Usage (Python)

```python
from cuvis_ai.node.conversion import ScoreToLogit

# Create logit head
from cuvis_ai_core.training import StatisticalTrainer

logit_head = ScoreToLogit(init_scale=1.0, init_bias=0.0)
pipeline.add_node(logit_head)

# Phase 1: Statistical initialization
trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)
trainer.fit()  # Automatically initializes logit_head

# Phase 2 (optional): Enable gradient training
logit_head.unfreeze()  # Convert buffers to nn.Parameters

# Use in pipeline
pipeline.connect(
    (rx.scores, logit_head.scores),
    (logit_head.logits, bce_loss.predictions)
)
```

#### Example Configuration (YAML)

```yaml
nodes:
  rx_detector:
    type: RXGlobal
    config:
      num_channels: 61

  logit_head:
    type: ScoreToLogit
    config:
      init_scale: 1.0
      init_bias: 0.0

  bce_loss:
    type: AnomalyBCEWithLogits

connections:
  - [rx_detector.scores, logit_head.scores]
  - [logit_head.logits, bce_loss.predictions]
```

#### See Also

- [Tutorial 1: RX Statistical](../tutorials/rx-statistical.md)
- [ScoreToLogit](utility.md#scoretologit)
- [Two-Phase Training](../concepts/two-phase-training.md)
- API Reference: ::: cuvis_ai.node.conversion.ScoreToLogit

---

### BinaryAnomalyLabelMapper

**Description:** Converts multi-class segmentation masks to binary anomaly masks

**Perfect for:**
- Preprocessing multi-class datasets
- Mapping specific classes to "anomaly" category
- Flexible normal/anomaly definition

**Training Paradigm:** None (stateless mapping)

**Mapping Modes:**

1. **Explicit anomaly IDs** (recommended):
   ```python
   mapper = BinaryAnomalyLabelMapper(
       normal_ids=[0],      # Class 0 = normal
       anomaly_ids=[1,2,3]  # Classes 1,2,3 = anomaly
   )
   ```

2. **Implicit (all non-normal are anomalies)**:
   ```python
   mapper = BinaryAnomalyLabelMapper(
       normal_ids=[0],
       anomaly_ids=None  # Everything except 0 is anomaly
   )
   ```

#### Port Specifications

**Input Ports:**

| Port | Type | Shape | Description | Optional |
|------|------|-------|-------------|----------|
| cube | float32 | (B,H,W,C) | Input cube (passed through) | No |
| mask | any | (B,H,W,1) | Multi-class mask | No |

**Output Ports:**

| Port | Type | Shape | Description |
|------|------|-------|-------------|
| cube | float32 | (B,H,W,C) | Cube (unchanged) |
| mask | bool | (B,H,W,1) | Binary anomaly mask |

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| normal_ids | list[int] | [] | Class IDs for normal pixels |
| anomaly_ids | list[int] | None | Class IDs for anomalies (None = implicit) |

#### Example Usage (Python)

```python
from cuvis_ai.node.labels import BinaryAnomalyLabelMapper

# Create mapper
mapper = BinaryAnomalyLabelMapper(
    normal_ids=[0],        # Background class
    anomaly_ids=[1, 2, 3]  # Defect classes
)

# Used internally by LentilsAnomalyDataNode
```

#### Example Configuration (YAML)

```yaml
nodes:
  data:
    type: LentilsAnomalyDataNode
    config:
      normal_ids: [0]      # Class 0 = normal
      anomaly_ids: [1, 2, 3]  # Classes 1,2,3 = anomaly
```

#### Common Issues

**1. Overlapping normal and anomaly IDs**

```python
# Problem: Same class in both lists
mapper = BinaryAnomalyLabelMapper(
    normal_ids=[0],
    anomaly_ids=[0, 1, 2]  # 0 appears in both!
)
# Raises ValueError

# Solution: Ensure disjoint sets
mapper = BinaryAnomalyLabelMapper(
    normal_ids=[0],
    anomaly_ids=[1, 2]  # No overlap
)
```

**2. Unassigned classes**

```python
# Data has classes: 0, 1, 2, 3, 4
mapper = BinaryAnomalyLabelMapper(
    normal_ids=[0],
    anomaly_ids=[1, 2]  # Classes 3, 4 unassigned!
)
# Warning logged, classes 3, 4 treated as anomalies

# Solution: Explicitly assign all classes
mapper = BinaryAnomalyLabelMapper(
    normal_ids=[0, 3, 4],  # All non-anomalies
    anomaly_ids=[1, 2]     # Actual anomalies
)
```

#### See Also

- [LentilsAnomalyDataNode](data-nodes.md#lentilsanomalydatanode)
- API Reference: ::: cuvis_ai.node.labels.BinaryAnomalyLabelMapper

---

## Decision Strategy Comparison

| Decider | Threshold | Adaptivity | Use Case |
|---------|-----------|------------|----------|
| **BinaryDecider** | Fixed | None | Known threshold, production |
| **QuantileBinaryDecider** | Adaptive (per-batch) | Medium | Variable distributions |
| **TwoStageBinaryDecider** | Adaptive (hierarchical) | High | High-precision applications |

---

## Complete Pipeline Example

Typical anomaly detection workflow with utilities:

```yaml
nodes:
  # Data
  data:
    type: LentilsAnomalyDataNode
    config:
      normal_ids: [0]
      anomaly_ids: [1, 2, 3]

  # Preprocessing
  normalizer:
    type: MinMaxNormalizer

  # Detection
  rx_detector:
    type: RXGlobal
    config:
      num_channels: 61

  # Utilities
  logit_head:
    type: ScoreToLogit
    config:
      init_scale: 1.0
      init_bias: 0.0

  decider:
    type: QuantileBinaryDecider  # or BinaryDecider, TwoStageBinaryDecider
    config:
      quantile: 0.995

  # Evaluation
  metrics:
    type: AnomalyDetectionMetrics

connections:
  # Data flow
  - [data.cube, normalizer.data]
  - [normalizer.output, rx_detector.data]

  # Score → Logit → Decision
  - [rx_detector.scores, logit_head.scores]
  - [logit_head.logits, decider.logits]

  # Evaluation
  - [decider.decisions, metrics.decisions]
  - [data.mask, metrics.targets]  # Binary mask from label mapper
```

---

## Additional Resources

- **Tutorial:** [RX Statistical Detection](../tutorials/rx-statistical.md) - BinaryDecider + ScoreToLogit
- **Tutorial:** [Deep SVDD Gradient](../tutorials/deep-svdd-gradient.md) - QuantileBinaryDecider
- **Concepts:** [Two-Phase Training](../concepts/two-phase-training.md)
- **API Reference:** [cuvis_ai.deciders](../../api/deciders/)
- **API Reference:** [cuvis_ai.node.conversion](../../api/node/#conversion)
- **API Reference:** [cuvis_ai.node.labels](../../api/node/#labels)
