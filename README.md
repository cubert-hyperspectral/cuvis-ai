![Cuvis.AI banner](docs/images/banner.png)

# Cuvis.AI

[![PyPI version](https://img.shields.io/pypi/v/cuvis-ai.svg)](https://pypi.org/project/cuvis-ai/)
[![CI Status](https://github.com/cubert-hyperspectral/cuvis-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://docs.cuvis.ai/latest/)

Cuvis.AI is an opensource and extensible framework for building AI powered processing pipelines for hyperspectral video data.
It allows you to process and structure spectral data, train and apply machine learning models, visualize and interpret results, and deploy applications in real time environments.
Pipelines are built from reusable modular nodes and can be extended with custom plugins or external integrations.
Cuvis.AI bridges the gap between hyperspectral hardware and real world applications and enables faster development, testing, and deployment of new solutions.


## Platform

Cuvis.AI is split across three repositories:

| Repository | Role |
|---|---|
| [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core) | Framework — base `Node` class, pipeline orchestration, two-phase training, gRPC services, plugin system |
| [cuvis-ai-schemas](https://github.com/cubert-hyperspectral/cuvis-ai-schemas) | Shared Protobuf / gRPC schema definitions and generated types |
| **cuvis-ai** (this repo) | Catalog — 40+ domain-specific nodes for anomaly detection, preprocessing, band selection, and more |

## Quick Start

**As a library** (in your own project):

```bash
uv add cuvis-ai
```

> **GPU support**: For PyTorch with CUDA, see the [Installation Guide](https://docs.cuvis.ai/latest/user-guide/installation/) for setup instructions.

**For development** (within this repo):

```bash
uv sync
```

See the [Installation Guide](https://docs.cuvis.ai/latest/user-guide/installation/) for prerequisites and detailed setup.

## Documentation

Full documentation is available at **https://docs.cuvis.ai/latest/**.

- [Quick Start](https://docs.cuvis.ai/latest/user-guide/quickstart/)
- [Core Concepts](https://docs.cuvis.ai/latest/concepts/overview/)
- [Node Catalog](https://docs.cuvis.ai/latest/node-catalog/)
- [Plugin System](https://docs.cuvis.ai/latest/plugin-system/)
- [API Reference](https://docs.cuvis.ai/latest/api/)
- [Contributing](https://docs.cuvis.ai/latest/development/contributing/)

## Links

- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

---

Apache License 2.0 — see [LICENSE](LICENSE).
