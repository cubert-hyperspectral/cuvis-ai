# Graph & Pipeline

Comprehensive API reference for graph construction, pipeline execution, and validation.

## Overview

The `cuvis_ai.pipeline` module provides the core infrastructure for building and executing computational graphs. The graph-based architecture allows:

- **Flexible Node Composition**: Chain preprocessing, feature extraction, and decision nodes
- **Dynamic Restructuring**: Modify graph structure at runtime with validation
- **Training Orchestration**: Two-phase training (statistical initialization + gradient optimization)
- **Leaf Node Integration**: Attach losses, metrics, and visualizations to any node

## Quick Example

```python
from cuvis_ai.pipeline.graph import Graph
from cuvis_ai.normalization.normalization import MinMaxNormalizer
from cuvis_ai.node.pca import TrainablePCA

# Build graph
graph = Graph("my_pipeline")
normalizer = MinMaxNormalizer()
pca = TrainablePCA(n_components=3)

graph.add_node(normalizer)
graph.add_node(pca, parent=normalizer)

# Execute forward pass
output = graph(input_data)

# Train with statistical initialization + gradient training
trainer = graph.train(datamodule=datamodule, training_config=config)
```

## Core Components

### Graph

The `Graph` class is the central orchestrator for building and training pipelines.

**Key Features:**
- Node management with parent-child relationships
- Leaf node attachment (losses, metrics, visualizations)
- Monitoring plugin registration
- Two-phase training orchestration
- Serialization/deserialization support

::: cuvis_ai.pipeline.graph

### Executor

The executor handles forward pass computation through the graph.

**Key Features:**
- Topological execution order
- Intermediate result caching
- Batch processing support
- Error handling and recovery

::: cuvis_ai.pipeline.executor

### Validator

The validator ensures graph integrity and compatibility.

**Key Features:**
- Cycle detection
- Dimension constraint checking
- Parent-child compatibility validation
- Leaf node protocol verification

::: cuvis_ai.pipeline.validator

## See Also

- **[Node API](nodes.md)**: Available node types and implementations
- **[Training API](training.md)**: Training infrastructure and leaf nodes
- **[Configuration Guide](../user-guide/configuration.md)**: Graph configuration with Hydra
- **[Quickstart Tutorial](../user-guide/quickstart.md)**: Getting started with graphs
