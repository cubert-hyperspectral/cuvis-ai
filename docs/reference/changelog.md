!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Changelog

This page provides a high-level overview of cuvis-ai releases. For the complete changelog with all technical details, see the [main CHANGELOG.md](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/CHANGELOG.md).

---

## Latest Release

### Version 0.2.4 (February 5, 2026)

**Documentation Complete!** This release marks the completion of comprehensive documentation for cuvis-ai, making the framework more accessible and easier to use for everyone.

**Highlights:**

- **70+ Documentation Pages** covering all aspects of the framework
- **6 Step-by-Step Tutorials** for common workflows (RX Statistical, Channel Selector, Deep SVDD, AdaCLIP, gRPC, Remote Deployment)
- **Complete Node Catalog** documenting 50+ built-in nodes across 11 categories
- **Plugin System Guides** for extending the framework with custom nodes
- **gRPC API Documentation** with sequence diagrams and client patterns
- **Configuration References** for Hydra composition and schema definitions
- **95%+ Docstring Coverage** across all public APIs
- **Central Plugin Registry** for community plugin discovery

**New Documentation:**

- API reference with auto-generated documentation from docstrings
- 7 how-to guides for building pipelines, restoration, monitoring, and more
- Development guides for contributing and maintaining code quality
- 20+ diagrams (Mermaid + Graphviz) for visual understanding

**Improvements:**

- README refactored to eliminate duplicate content
- Contributing guide enhanced with complete plugin contribution workflow
- All code examples updated to use `uv run` pattern
- MkDocs configuration with Material theme and advanced search

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.2.4)**

---

## Previous Releases

### Version 0.2.3 (January 29, 2026)

**Architecture Evolution:** Major refactoring to split cuvis-ai into core framework (cuvis-ai-core) and algorithm catalog (cuvis-ai).

**Key Changes:**

- Plugin system with dynamic loading from Git repositories and local filesystem
- 426 tests migrated to cuvis-ai-core with independent CI/CD
- Session-scoped plugin isolation for gRPC services
- Import pattern changes to `cuvis_ai_core.*`
- New gRPC RPCs for plugin management

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.2.3)**

---

### Version 0.2.2 (January 15, 2026)

**Restoration Utilities:** Simplified pipeline and trainrun restoration with auto-detection.

**Key Changes:**

- New CLI commands: `restore-pipeline`, `restore-trainrun`
- Auto-detection of statistical vs gradient training workflows
- Consolidated restoration utilities in single module
- Standardized Python API surface

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.2.2)**

---

### Version 0.2.1 (January 8, 2026)

**Configuration System:** Pydantic v2 models and server-side Hydra composition.

**Key Changes:**

- Pydantic v2 config models as single source of truth
- Session-scoped configuration management
- Terminology change: Experiment → TrainRun
- Explicit 4-step workflow for gRPC API
- Standardized config transport with `config_bytes`

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.2.1)**

---

### Version 0.2.0 (December 20, 2025)

**YAML-Driven Pipelines:** Introduced configuration-based pipeline building.

**Key Changes:**

- YAML-driven pipeline configuration with OmegaConf
- Hybrid NodeRegistry for built-ins and custom nodes
- End-to-end serialization (YAML + .pt weights)
- Version compatibility guards
- Simplified node state management

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.2.0)**

---

### Version 0.1.5 (December 1, 2025)

**gRPC Services:** Introduced remote API for pipeline execution.

**Key Changes:**

- gRPC service stack with proto definitions
- Buf Schema Registry integration
- Session management and PipelineBuilder
- Two-phase training support
- Pipeline introspection RPCs

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.1.5)**

---

### Version 0.1.3 (November 6, 2025)

**Port-Based I/O:** Major refactor introducing typed input/output system.

**Key Changes:**

- Port-based typed I/O with PortSpec
- Graph connection API with auto-validation
- Multi-input/output support
- PyTorch Lightning integration
- Executor refactored for port-based routing

**[View Full Release Notes →](https://github.com/cubert-hyperspectral/cuvis-ai/releases/tag/v0.1.3)**

---

## Versioning

cuvis-ai follows [Semantic Versioning](https://semver.org/) (MAJOR.MINOR.PATCH):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards-compatible)
- **PATCH**: Bug fixes (backwards-compatible)

---

## Release Channels

- **Stable**: Use tagged releases for production deployments
- **Latest**: Track the `main` branch for latest stable code
- **Development**: Track feature branches for experimental features

---

## See Also

- [Installation Guide](../user-guide/installation.md) - Get started with cuvis-ai
- [GitHub Releases](https://github.com/cubert-hyperspectral/cuvis-ai/releases) - Download specific versions
- [CHANGELOG.md](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/CHANGELOG.md) - Complete technical changelog
- [Contributing Guide](../development/contributing.md) - How to contribute to cuvis-ai
