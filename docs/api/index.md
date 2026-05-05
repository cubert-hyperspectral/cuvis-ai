# API Reference

Framework-level API documentation generated from current docstrings.

## API Modules

<div class="grid cards" markdown>

-   :material-pipe: **[Pipeline & Graph](pipeline.md)**

    ---

    Pipeline and graph construction APIs

-   :material-transit-connection-variant: **[Ports](ports.md)**

    ---

    Port system and data contracts

-   :material-tools: **[Utilities](utilities.md)**

    ---

    Helper functions and utilities

-   :material-api: **[gRPC API](../grpc/api-session.md)**

    ---

    Current CuvisAIService API surface

</div>

---

## Node API

Node implementations are documented in the **[Node Catalog](../node-catalog/index.md)** —
each catalog page renders the live module docstrings for the nodes in that category:

- [Data nodes](../node-catalog/data-nodes.md) — readers, video sources, JSON/NumPy loaders
- [Preprocessing](../node-catalog/preprocessing.md) — normalization, occlusion, conversion
- [Selectors](../node-catalog/selectors.md) — channel selectors, channel mixers, dimensionality reduction
- [Statistical](../node-catalog/statistical.md) — RX/LAD anomaly detectors, spectral angle/extractor
- [Loss & Metrics](../node-catalog/loss-metrics.md) — losses, metrics, monitoring
- [Visualization](../node-catalog/visualization.md) — anomaly viz, pipeline viz
- [Output](../node-catalog/output.md) — JSON/NumPy/video writers
- [Utility](../node-catalog/utility.md) — deciders, conversion, labels, prompts
