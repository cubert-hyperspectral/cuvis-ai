![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# CUVIS.AI

[![PyPI version](https://img.shields.io/pypi/v/cuvis-ai.svg)](https://pypi.org/project/cuvis-ai/)
[![CI Status](https://github.com/cubert-hyperspectral/cuvis-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-gh--pages-blue)](https://cubert-hyperspectral.github.io/cuvis-ai/)

A modular toolkit for building graph-based ML pipelines for hyperspectral imaging — from preprocessing through training to deployment.

## Platform

cuvis.ai is split across three repositories:

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

**For development** (within this repo):

```bash
uv sync
```

See the [Installation Guide](https://cubert-hyperspectral.github.io/cuvis-ai/user-guide/installation/) for prerequisites and detailed setup.

## Documentation

Full documentation is available at **https://cubert-hyperspectral.github.io/cuvis-ai/**.

- [Quick Start](https://cubert-hyperspectral.github.io/cuvis-ai/user-guide/quickstart/)
- [Core Concepts](https://cubert-hyperspectral.github.io/cuvis-ai/concepts/overview/)
- [Node Catalog](https://cubert-hyperspectral.github.io/cuvis-ai/node-catalog/)
- [Plugin System](https://cubert-hyperspectral.github.io/cuvis-ai/plugin-system/)
- [API Reference](https://cubert-hyperspectral.github.io/cuvis-ai/api/)
- [Contributing](https://cubert-hyperspectral.github.io/cuvis-ai/development/contributing/)

## Links

- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

---

Apache License 2.0 — see [LICENSE](LICENSE).
