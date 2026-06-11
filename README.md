![image](https://github.com/cubert-hyperspectral/cuvis.sdk/blob/main/branding/logo/banner.png?raw=true)

# Cuvis.AI

[![PyPI][pypi-badge]][pypi-link]
[![CI][ci-badge]][ci-link]
[![codecov][cov-badge]][cov-link]
[![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat-square)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-cuvis.ai-8CA1AF?style=flat-square)](https://docs.cuvis.ai/latest/)

[pypi-badge]: https://img.shields.io/pypi/v/cuvis-ai?style=flat-square&logo=pypi&logoColor=white
[pypi-link]: https://pypi.org/project/cuvis-ai/
[ci-badge]: https://img.shields.io/github/actions/workflow/status/cubert-hyperspectral/cuvis-ai/ci.yml?style=flat-square&logo=githubactions&logoColor=white&label=CI
[ci-link]: https://github.com/cubert-hyperspectral/cuvis-ai/actions/workflows/ci.yml
[cov-badge]: https://img.shields.io/codecov/c/github/cubert-hyperspectral/cuvis-ai?style=flat-square&logo=codecov&logoColor=white
[cov-link]: https://codecov.io/gh/cubert-hyperspectral/cuvis-ai

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

Companion repo: [cuvis-ai-agentic-skills](https://github.com/cubert-hyperspectral/cuvis-ai-agentic-skills) — agentic skills for authoring nodes, plugins, pipelines, and training runs against this platform.

## Quick Start

**As a library** (in your own project):

```bash
uv add cuvis-ai
```

> **GPU support**: For PyTorch with CUDA, see the [Installation Guide](https://docs.cuvis.ai/latest/get-started/installation/) for setup instructions.

**For development** (within this repo):

```bash
uv sync
```

See the [Installation Guide](https://docs.cuvis.ai/latest/get-started/installation/) for prerequisites and detailed setup.

## Documentation

Full documentation is available at **https://docs.cuvis.ai/latest/**.

- [Quick Start](https://docs.cuvis.ai/latest/get-started/quickstart/)
- [Core Concepts](https://docs.cuvis.ai/latest/concepts/)
- [Node Catalog](https://docs.cuvis.ai/latest/catalogs/nodes/)
- [Plugin System](https://docs.cuvis.ai/latest/reference/plugin-development/overview/)
- [API Reference](https://docs.cuvis.ai/latest/reference/python-api/)
- [Contributing](https://docs.cuvis.ai/latest/reference/contributing/)

## Links

- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)

---

Apache License 2.0 — see [LICENSE](LICENSE).
