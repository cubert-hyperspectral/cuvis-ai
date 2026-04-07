!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin System

The cuvis-ai plugin system enables extending the framework with custom nodes and functionality without modifying the core codebase. Distribute your algorithms via Git, share with the community, and maintain independent versioning.

## Quick Start

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugins("configs/plugins/trackeval.yaml")

HOTAMetricNode = registry.get("HOTAMetricNode", instance=registry)
```

## Guides

<div class="grid cards" markdown>

-   :material-puzzle: **[Plugin System Overview](overview.md)**

    ---

    Architecture, manifest rules, path resolution, and loading behavior.

-   :material-hammer-wrench: **[Plugin Development Guide](development.md)**

    ---

    Create, test, package, and tag a cuvis-ai plugin for local or Git-based use.

-   :material-package-variant: **[Plugin Usage Guide](usage.md)**

    ---

    Load released and local plugins through selective manifests such as `trackeval.yaml` and `sam3.yaml`.

</div>

## Official Plugin Manifests

- [`configs/plugins/adaclip.yaml`](../../configs/plugins/adaclip.yaml): released AdaCLIP plugin manifest
- [`configs/plugins/trackeval.yaml`](../../configs/plugins/trackeval.yaml): released TrackEval plugin manifest pinned to `v0.1.0`
- [`configs/plugins/sam3.yaml`](../../configs/plugins/sam3.yaml): local SAM3 plugin manifest for a checkout at `D:\code-repos\cuvis-ai-sam3\sam3-init`

## Official Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection
- **[cuvis-ai-trackeval](https://github.com/cubert-hyperspectral/cuvis-ai-trackeval)** - HOTA, CLEAR, and Identity tracking metrics
- **[cuvis-ai-sam3](https://github.com/cubert-hyperspectral/cuvis-ai-sam3)** - local SAM3 tracking workflows and prompt propagation nodes

See [Plugin Usage Guide](usage.md) for manifest-based loading examples.
