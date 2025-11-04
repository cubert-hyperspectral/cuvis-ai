# Nodes

Complete API reference for all node implementations in CUVIS.AI.

## Overview

Nodes are the building blocks of CUVIS.AI pipelines. Each node transforms input data and can be chained with other nodes to create complex processing pipelines.

### Node Types

| Category | Nodes | Purpose |
|----------|-------|---------|
| **Normalization** | MinMaxNormalizer, StandardNormalizer | Scale input data to standard ranges |
| **Feature Extraction** | TrainablePCA | Dimensionality reduction with gradient optimization |
| **Channel Selection** | SoftChannelSelector | Learnable spectral channel selection |
| **Anomaly Detection** | RXGlobal, RXLogitHead | Reed-Xiaoli detector with trainable threshold |
| **PyTorch Integration** | TorchNode, TorchVisionNode | Wrap PyTorch models as nodes |

### Node Lifecycle

All nodes follow a consistent lifecycle:

1. **Initialization**: Create node with hyperparameters
2. **Statistical Fitting** (optional): `initialize_from_data()` called during Phase 1
3. **Training Preparation** (optional): `prepare_for_train()` converts buffers to parameters
4. **Gradient Training** (optional): Backpropagation through trainable parameters
5. **Inference**: Forward pass with learned/fitted parameters

## Quick Example

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA
from cuvis_ai.node.selector import SoftChannelSelector

# Build graph with multiple node types
graph = Graph("feature_extraction")

# Normalization node
normalizer = MinMaxNormalizer(use_running_stats=True)
graph.add_node(normalizer)

# Channel selection node (learnable)
selector = SoftChannelSelector(n_select=15, trainable=True)
graph.add_node(selector, parent=normalizer)

# PCA node (learnable)
pca = TrainablePCA(n_components=3, trainable=True)
graph.add_node(pca, parent=selector)

# Train - statistical init + gradient optimization
trainer = graph.train(datamodule=datamodule, training_config=config)
```

## Base Node

The `Node` base class defines the common interface for all nodes.

**Key Properties:**
- `requires_initial_fit`: Whether node needs statistical initialization
- `is_trainable`: Whether node has trainable parameters
- `is_frozen`: Whether node parameters are frozen

**Key Methods:**
- `initialize_from_data()`: Statistical initialization from dataset
- `prepare_for_train()`: Convert buffers to trainable parameters
- `freeze()`: Freeze all parameters (disable gradients)

::: cuvis_ai.node.node

## Feature Extraction

### TrainablePCA

Principal Component Analysis with gradient-based fine-tuning.

**Features:**
- SVD initialization from training data
- Gradient optimization with orthogonality constraints
- Explained variance tracking
- Optional whitening transformation

**Example:**
```python
pca = TrainablePCA(
    n_components=3,
    trainable=True,
    whiten=False,
    center=True
)
```

::: cuvis_ai.node.pca

## Channel Selection

### SoftChannelSelector

Temperature-based learnable channel selection using Gumbel-Softmax.

**Features:**
- Soft selection during training (differentiable)
- Hard selection at inference (top-k)
- Temperature annealing schedule
- Variance-based or uniform initialization

**Example:**
```python
selector = SoftChannelSelector(
    n_select=15,
    init_method="variance",
    temperature_init=5.0,
    temperature_decay=0.9,
    trainable=True
)
```

::: cuvis_ai.node.selector

## Anomaly Detection

### RXGlobal

Reed-Xiaoli (RX) global anomaly detector using Mahalanobis distance.

**Features:**
- Statistical initialization (global mean and covariance)
- Optional trainable statistics
- Efficient covariance computation
- Numerical stability guarantees

**Example:**
```python
rx = RXGlobal(
    eps=1e-6,
    trainable_stats=False
)
```

::: cuvis_ai.anomaly.rx_detector

### RXLogitHead

Trainable anomaly threshold with learnable scale and bias parameters.

**Features:**
- Statistical initialization from score distribution
- End-to-end training with BCE loss
- Affine transformation: `logit = scale * (score - bias)`

**Example:**
```python
logit_head = RXLogitHead(
    init_scale=1.0,
    init_bias=5.0,
    trainable=True
)
```

::: cuvis_ai.anomaly.rx_logit_head

## Normalization

### MinMaxNormalizer

Min-max normalization to [0, 1] range with running statistics.

**Features:**
- Statistical initialization from training data
- Per-channel or global normalization
- Running statistics for consistent normalization

**Example:**
```python
normalizer = MinMaxNormalizer(
    eps=1e-6,
    use_running_stats=True
)
```

### StandardNormalizer

Z-score normalization (mean=0, std=1) with running statistics.

**Features:**
- Per-channel or global normalization
- Statistical initialization
- Running mean/std tracking

**Example:**
```python
normalizer = StandardNormalizer(
    eps=1e-6,
    per_channel=True
)
```

::: cuvis_ai.normalization.normalization

## PyTorch Integration

### TorchNode

Wrap any PyTorch `nn.Module` as a graph node.

**Features:**
- Zero-copy integration with PyTorch models
- Automatic gradient flow
- Compatible with pretrained models

::: cuvis_ai.node.torch

### TorchVisionNode

Specialized node for TorchVision models.

::: cuvis_ai.node.torchvision

## See Also

- **[Graph API](pipeline.md)**: Build graphs from nodes
- **[Training API](training.md)**: Train node parameters
- **[Configuration Guide](../user-guide/configuration.md)**: Configure nodes with Hydra
- **[Tutorials](../tutorials/phase1_statistical.md)**: Step-by-step node usage examples
