# Core Concepts Overview

CUVIS.AI is built on five interconnected concepts that work together to create flexible, type-safe ML pipelines:

```mermaid
graph TB
    subgraph " "
        N[Nodes<br/>Processing units with typed I/O]
        P[Ports<br/>Type-safe connections]
        PL[Pipelines<br/>DAG orchestration]
        TPT[Two-Phase Training<br/>Statistical + Gradient]
        ES[Execution Stages<br/>Conditional execution]
    end

    N -->|connected via| P
    P -->|orchestrated by| PL
    PL -->|trained with| TPT
    PL -->|filtered by| ES

    style N fill:#e1f5ff
    style P fill:#fff3cd
    style PL fill:#ffe66d
    style TPT fill:#d4edda
    style ES fill:#f3e5f5

    click N "../node-system-deep-dive/"
    click P "../port-system-deep-dive/"
    click PL "../pipeline-lifecycle/"
    click TPT "../two-phase-training/"
    click ES "../execution-stages/"
```

Each concept below has a dedicated deep-dive page with comprehensive diagrams and examples.
