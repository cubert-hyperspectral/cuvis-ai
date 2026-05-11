# Core Concepts Overview

Cuvis.AI is built on five interconnected concepts that work together to create flexible, type-safe ML pipelines:

```mermaid
graph TB
    N[Nodes<br/>Processing units with typed I/O]
    P[Ports<br/>Type-safe connections]
    PL[Pipelines<br/>DAG orchestration]
    TPT[Two-Phase Training<br/>Statistical + Gradient]
    ES[Execution Stages<br/>Conditional execution]

    N -->|connected via| P
    P -->|orchestrated by| PL
    PL -->|trained with| TPT
    PL -->|filtered by| ES

    classDef plain fill:transparent,stroke:currentColor,color:currentColor;
    class N,P,PL,TPT,ES plain;
    linkStyle default stroke:currentColor,color:currentColor;

    click N "node/" "Node deep dive"
    click P "port/" "Port deep dive"
    click PL "pipeline/" "Pipeline deep dive"
    click TPT "training/" "Two-phase training"
    click ES "execution-stages/" "Execution stages"
```

Each concept below has a dedicated deep-dive page with comprehensive diagrams and examples.
