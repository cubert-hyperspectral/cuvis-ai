!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Pipeline & Graph API

## Overview

Pipeline and graph construction in CUVIS.AI is handled through configuration files and the PyTorch Lightning framework. The cuvis_ai package provides nodes that can be composed into pipelines.

For information on building and configuring pipelines, see the guides below.

---

## Pipeline Construction

Pipeline construction is done through:

- **YAML Configuration**: Define pipelines declaratively using YAML config files
- **Python API**: Build pipelines programmatically using PyTorch Lightning modules
- **Node Composition**: Compose nodes from the cuvis_ai package into processing graphs

---

## Related Pages

- [Build Pipeline (Python)](../how-to/build-pipeline-python.md) - Python API for pipeline construction
- [Build Pipeline (YAML)](../how-to/build-pipeline-yaml.md) - YAML configuration approach
- [Pipeline Lifecycle](../concepts/pipeline-lifecycle.md) - Understanding pipeline execution
- [Node System Deep Dive](../concepts/node-system-deep-dive.md) - Node composition patterns
- [Configuration Basics](../user-guide/configuration.md) - Configuration system overview

---

## Node API Reference

For the complete API of available nodes that can be used in pipelines, see:

- [Nodes API](nodes.md) - All available node implementations
