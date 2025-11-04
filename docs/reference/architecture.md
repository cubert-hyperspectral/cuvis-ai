# Architecture

This page documents the CUVIS.AI architecture with detailed diagrams showing the training lifecycle, data flow, and system components.

## Complete Training Lifecycle

The training process flows through three main phases:

```mermaid
graph TB
    A[graph.train called] --> B{Check Statistical Nodes}
    B -->|Has Statistical Nodes| C[Phase 1: Initialize Statistical Nodes]
    B -->|No Statistical Nodes| D[Phase 2: Setup Lightning]
    C --> C1[Find nodes with requires_initial_fit=True]
    C1 --> C2[Sort nodes topologically]
    C2 --> C3[For each node]
    C3 --> C4[Get transformed data stream]
    C4 --> C5[Call node.initialize_from_data]
    C5 --> C6[Call node.prepare_for_train]
    C6 --> C7{More nodes?}
    C7 -->|Yes| C3
    C7 -->|No| D
    D --> D1[Create CuvisLightningModule]
    D1 --> D2[Register loss/metric/viz nodes]
    D2 --> D3[Create pl.Trainer]
    D3 --> E[Phase 3: Training Loop]
    E --> E1[training_step]
    E1 --> E2[validation_step]
    E2 --> E3{Callbacks?}
    E3 -->|Yes| E4[Execute callbacks]
    E4 --> E5{Continue?}
    E3 -->|No| E5
    E5 -->|Yes| E1
    E5 -->|No| F[Training Complete]
```

## Statistical Initialization Phase

Phase 1 initializes statistical nodes before gradient-based training:

```mermaid
graph TD
    A[LightningGraph.on_fit_start] --> C[Sort nodes topologically]
    C --> D{For each statistical node}
    D --> E[Get parent nodes]
    E --> F[Iterate training data]
    F --> G[Transform through parents]
    G --> H[node.update statistics]
    H --> I{More batches?}
    I -->|Yes| F
    I -->|No| J[node.finalize]
    J --> K{is_trainable?}
    K -->|Yes| L[node.prepare_for_train]
    K -->|No| M[node.freeze]
    L --> N[Register loss function]
    N --> O{More nodes?}
    M --> O
    O -->|Yes| D
    O -->|No| P[Training begins]
    
    style A fill:#e1f5ff
    style P fill:#d4edda
    style N fill:#fff3cd
```

## Training Loop (Detailed)

The training loop handles forward pass, loss aggregation, and optimization:

```mermaid
%%{init: {'flowchart': {'nodeSpacing': 20, 'rankSpacing': 30}} }%%
flowchart TD
    A["Lightning Trainer.fit"] --> B["Get batch (DataLoader)"]
    B --> C[training_step]
    C --> D[Graph.forward]
    D --> D1{Forward mode?}
    D1 -- Eager/JIT --> D2[Path A]
    D1 -- Compiled/Other --> D3[Path B]
    D2 --> E["Process through nodes"]
    D3 --> E
    E --> E1["Node pipeline transform"]
    E1 --> F{Leaf type?}
    F -- Loss --> G["Collect losses"]
    F -- Metric --> H["Collect metrics"]
    F -- Viz --> I["Queue visualizations"]
    G --> J["Aggregate total_loss"]
    H --> K["Log metrics"]
    I --> L["Log (W&B / TensorBoard)"]
    J --> M["Backward pass"]
    K --> M
    L --> M
    M --> N["Optimizer step"]
    N --> O{Validation epoch?}
    O -- Yes --> P[validation_step]
    O -- No  --> Q{More batches?}
    P --> P1["Compute val_loss"]
    P1 --> P2["Compute val_metrics"]
    P2 --> P3{Callbacks?}
    P3 -- EarlyStopping --> P4["Check patience"]
    P3 -- Checkpoint --> P5["Save best model"]
    P3 -- LR Scheduler --> P6["Adjust LR"]
    P4 --> Q
    P5 --> Q
    P6 --> Q
    Q -- Yes --> B
    Q -- No  --> R["Epoch complete"]

    %% Styles
    style D fill:#e1f5ff
    style J fill:#fff3cd
    style M fill:#ffc107
    style R fill:#d4edda
```

## Data Flow with Leaf Nodes

This diagram shows how data flows through nodes and how leaf nodes attach for losses, metrics, and visualizations:

```mermaid
graph LR
    A[Input Batch] --> S[SoftChannelSelector]
    S --> C[RXGlobal]
    C --> D[MinMax]
    
    S --> S1[Selector Entropy Reg]
    S --> S2[Selector Diversity Reg]
    
    C --> C1[RX Cov Reg]
    C --> C2[Anomaly Heatmap]
    
    D --> D1[Score Histogram Viz]
    
    S1 --> F[Loss Aggregator]
    S2 --> F
    C1 --> F
    
    G[Visualization Manager]
    C2 --> G
    D1 --> G
    
    H[Metrics Logger]
    
    F --> I[Backward]
    G --> J[Monitors WB / TensorBoard]
    H --> J
    
    style A fill:#e1f5ff
    style F fill:#ffc107
    style G fill:#9c27b0
    style H fill:#4caf50
    style I fill:#ff5722
    style J fill:#2196f3
```

## Two-Phase Training Strategy

CUVIS.AI uses a two-phase training approach:

### Phase 1: Statistical Initialization
- **Purpose**: Bootstrap models with efficient statistical methods
- **Nodes**: RX detector (mean/covariance), PCA (SVD), MinMaxNormalizer
- **Speed**: Fast, typically seconds to minutes
- **Hardware**: Can run on CPU

### Phase 2: Gradient Training
- **Purpose**: Fine-tune initialized models with backpropagation
- **Nodes**: All trainable nodes receive gradients
- **Speed**: Slower, depends on model size and data
- **Hardware**: GPU recommended

## Node Types

### Processing Nodes
- **MinMaxNormalizer**: Min-max normalization with running statistics
- **StandardNormalizer**: Z-score normalization
- **SoftChannelSelector**: Learnable channel selection

### Feature Extraction
- **TrainablePCA**: SVD-based PCA with gradient fine-tuning
- **ConvBlock**: Convolutional feature extraction

### Anomaly Detection
- **RXGlobal**: Reed-Xiaoli global anomaly detector
- **RXLogitHead**: Trainable anomaly threshold

## Leaf Node Types

### Loss Nodes
- Attach to parent nodes to provide training signals
- Examples: OrthogonalityLoss, AnomalyBCEWithLogits, SelectorEntropyRegularizer

### Metric Nodes
- Compute and track metrics during training
- Examples: ExplainedVarianceMetric, AnomalyDetectionMetrics

### Visualization Nodes
- Generate visualizations at specified intervals
- Examples: PCAVisualization, AnomalyHeatmap, ScoreHistogram

### Monitoring Nodes
- Forward metrics and artifacts to external backends
- Examples: WandBMonitor, TensorBoardMonitor

## Next Steps

- **[Features](features.md)**: Detailed feature matrix
- **[API Reference](../api/pipeline.md)**: Complete API documentation
- **[Tutorials](../tutorials/phase1_statistical.md)**: Step-by-step guides
