# Architecture

This page documents the CUVIS.AI architecture with detailed diagrams showing the training lifecycle, data flow, and system components, including the new **Typed I/O system**.

## Port-Based Architecture Overview

CUVIS.AI now uses a port-based architecture where nodes communicate through typed input/output ports. This provides type safety, better error messages, and more flexible pipeline construction.

```mermaid
graph TB
    A[Input Batch] --> B[CuvisPipeline]
    B --> C[Port-Based Routing]
    C --> D[Node Execution]
    D --> E[Port Output Collection]
    E --> F[Result Dictionary]
    
    C --> G[Port Validation]
    G --> H[Type Checking]
    G --> I[Shape Validation]
    G --> J[Stage Filtering]
    
    style A fill:#e1f5ff
    style F fill:#d4edda
    style G fill:#fff3cd
```

## Complete Training Lifecycle

The training process flows through three main phases with port-based execution:

```mermaid
graph TB
    A[pipeline.train called] --> B{Check Statistical Nodes}
    B -->|Has Statistical Nodes| C[Phase 1: Initialize Statistical Nodes]
    B -->|No Statistical Nodes| D[Phase 2: Setup Lightning]
    C --> C1[Find nodes with requires_initial_fit=True]
    C1 --> C2[Sort nodes by port connections]
    C2 --> C3[For each node]
    C3 --> C4[Get transformed data via port routing]
    C4 --> C5[Call node.initialize_from_data]
    C5 --> C6[Call node.prepare_for_gradient_train]
    C6 --> C7{More nodes?}
    C7 -->|Yes| C3
    C7 -->|No| D
    D --> D1[Create CuvisLightningModule]
    D1 --> D2[Register loss/metric/viz nodes via ports]
    D2 --> D3[Create pl.Trainer]
    D3 --> E[Phase 3: Training Loop]
    E --> E1[training_step]
    E1 --> E2[Port-based forward pass]
    E2 --> E3[validation_step]
    E3 --> E4{Callbacks?}
    E4 -->|Yes| E5[Execute callbacks]
    E5 --> E6{Continue?}
    E4 -->|No| E6
    E6 -->|Yes| E1
    E6 -->|No| F[Training Complete]
```

## Port-Based Data Flow

This diagram shows how data flows through port connections and how leaf nodes attach via ports:

```mermaid
graph LR
    A[Input Batch] --> B[Port Distribution]
    B --> C[MinMaxNormalizer.data]
    C --> D[MinMaxNormalizer.normalized]
    D --> E[SoftChannelSelector.data]
    E --> F[SoftChannelSelector.selected]
    F --> G[TrainablePCA.features]
    G --> H[TrainablePCA.projected]
    H --> I[RXGlobal.data]
    I --> J[RXGlobal.scores]
    
    E --> K[SelectorEntropyReg.weights]
    F --> L[SelectorDiversityReg.weights]
    J --> M[AnomalyBCEWithLogits.predictions]
    J --> N[AnomalyHeatmap.data]
    
    K --> O[SelectorEntropyReg.loss]
    L --> P[SelectorDiversityReg.loss]
    M --> Q[AnomalyBCEWithLogits.loss]
    
    O --> R[GradientTrainer loss_nodes]
    P --> R
    Q --> R
    
    N --> S[VisualizationManager.inputs]
    R --> T[Backward Pass]
    S --> U[Monitor Output]
    
    style A fill:#e1f5ff
    style T fill:#ffc107
    style U fill:#2196f3
```

## Port-Based Forward Pass

The new forward pass uses port-based routing and returns a dictionary with port keys:

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 20, 'rankSpacing': 30}} }%%
flowchart TD
    A["pipeline.forward(batch=...)"] --> B[Port-Based Batch Distribution]
    B --> C[Resolve Input Ports]
    C --> D[Topological Sort by Port Connections]
    D --> E[For each node in order]
    E --> F[Collect inputs from connected output ports]
    F --> G[Execute node.forward(**inputs)]
    G --> H[Store outputs in port dictionary]
    H --> I{More nodes?}
    I -->|Yes| E
    I -->|No| J[Return port output dictionary]
    
    B --> K[Port Validation]
    K --> L[Type Checking]
    K --> M[Shape Compatibility]
    K --> N[Stage Filtering]
    
    style A fill:#e1f5ff
    style J fill:#d4edda
    style K fill:#fff3cd
```

## Typed I/O System Benefits

### Type Safety
- Runtime validation of port compatibility
- Clear error messages for type mismatches
- Prevention of invalid connections

### Flexible Pipeline Construction
- Explicit port-based connections: `pipeline.connect(source.port, target.port)`
- Variadic ports for fan-in/out (e.g., monitoring nodes that consume multiple inputs)
- Stage-aware execution filtering

### Better Debugging
- Port-specific error messages
- Connection graph visualization
- Batch distribution tracking

## Two-Phase Training Strategy with Ports

CUVIS.AI uses a two-phase training approach with port-based execution:

### Phase 1: Statistical Initialization
- **Purpose**: Bootstrap models with efficient statistical methods
- **Nodes**: RX detector (mean/covariance), PCA (SVD), MinMaxNormalizer
- **Port Flow**: Data flows through port connections for statistical computation
- **Speed**: Fast, typically seconds to minutes
- **Hardware**: Can run on CPU

### Phase 2: Gradient Training
- **Purpose**: Fine-tune initialized models with backpropagation
- **Nodes**: All trainable nodes receive gradients via port connections
- **Port Flow**: Gradients flow backward through port connections
- **Speed**: Slower, depends on model size and data
- **Hardware**: GPU recommended

## Node Types with Port Specifications

### Processing Nodes
- **MinMaxNormalizer**:
  - Input: `data` (raw cube)
  - Output: `normalized` (normalized cube)
- **StandardNormalizer**:
  - Input: `data` (raw cube)
  - Output: `normalized` (standardized cube)
- **SoftChannelSelector**:
  - Input: `data` (cube)
  - Output: `selected` (selected channels)

### Feature Extraction
- **TrainablePCA**:
  - Input: `features` (input features)
  - Output: `projected` (PCA features), `explained_variance`

### Anomaly Detection
- **RXGlobal**:
  - Input: `data` (features)
  - Output: `scores` (anomaly scores), `logits`
- **RXLogitHead**:
  - Input: `scores` (anomaly scores)
  - Output: `logits` (transformed scores)

## Port-Based Leaf Node Attachment

### Loss Nodes
- Attach via port connections: `pipeline.connect(node.output, loss.input)`
- Register each node directly with `GradientTrainer(loss_nodes=[...])`; no aggregator needed
- Examples: OrthogonalityLoss, AnomalyBCEWithLogits, SelectorEntropyRegularizer

### Metric Nodes
- Connect to output ports for metric computation
- Examples: AccuracyMetric, AUCMetric

### Visualization Nodes
- Attach to output ports for visualization generation
- Examples: AnomalyHeatmap, PCAVisualization

## Migration from Legacy API

The port-based system replaces the legacy tuple-based API:

**Before (Legacy):**
```python
graph.add_node(normalizer)
graph.add_node(selector, parent=normalizer)
output, _, _ = graph(input_data)
```

**After (Port-based):**
```python
pipeline.connect(normalizer.normalized, selector.data)
outputs = pipeline.forward(batch={f"{normalizer.id}.data": input_data})
```

## Next Steps

- **[Ports API](../api/ports.md)**: Detailed port system documentation
- **[Migration Guide](../user-guide/typed-io-migration.md)**: Transition guide from legacy API
- **[API Reference](../api/pipeline.md)**: Complete API documentation
- **[Tutorials](../tutorials/phase1_statistical.md)**: Step-by-step guides with port examples
