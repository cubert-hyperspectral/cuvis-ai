# CUVIS.AI Documentation

CUVIS.AI is a modular, low-code/no-code framework for building reproducible machine-learning pipelines for hyperspectral data analysis. It provides a thin abstraction over PyTorch, PyTorch Lightning, and Hydra, with reusable nodes you can compose into graph-based HSI workflows.

## What you can do

- **Typed I/O System**: Port-based connections with type safety and validation 
- **Statistical Initialization**: Bootstrap models with non-parametric methods (RX detector, PCA) 
- **Gradient-Based Training**: Fine-tune models with PyTorch Lightning 
- **Flexible Node Architecture**: Composable preprocessing, feature extraction, and decision modules 
- **Comprehensive Monitoring**: Integrated TensorBoard support and extendible to other frameworks
- **Configuration Management**: Hydra-based configuration with CLI overrides

## Quick Links

- **[Installation](user-guide/installation.md)**
- **[Quickstart](user-guide/quickstart.md)**
- **[Core Concepts](concepts/overview.md)**
- **[API Reference](api/pipeline.md)**
- **[Tutorials](tutorials/index.md)**


!!! tip "Ready to get started?"
    - Start with the [Installation Guide](user-guide/installation.md), then follow the [Quickstart](user-guide/quickstart.md).
    - Want to contribute? See the [Contributing Guide](development/contributing.md).
    - Found an issue? [Report bugs / request features](https://github.com/cubert-hyperspectral/cuvis-ai/issues)

---
Apache License 2.0 — see [LICENSE](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/LICENSE).
