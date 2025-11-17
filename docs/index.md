# CUVIS.AI Documentation

Welcome to the CUVIS.AI documentation! This project provides a PyTorch Lightning-based training pipeline for hyperspectral data analysis, with a focus on anomaly detection and dimensionality reduction.

## Overview

CUVIS.AI is a modular, graph-based framework for processing hyperspectral data with support for:

- **Typed I/O System**: Port-based connections with type safety and validation
- **Statistical Initialization**: Bootstrap models with non-parametric methods (RX detector, PCA)
- **Gradient-Based Training**: Fine-tune models with PyTorch Lightning
- **Flexible Node Architecture**: Composable preprocessing, feature extraction, and decision modules
- **Comprehensive Monitoring**: Integrated WandB and TensorBoard support
- **Visualization Tools**: Built-in visualizations for PCA, anomaly detection, and channel selection
- **Configuration Management**: Hydra-based configuration with CLI overrides

## Key Features

### 🔧 Typed I/O System
Build pipelines with port-based connections that provide type safety, better error messages, and flexible pipeline construction. Connect nodes using `canvas.connect(source.port, target.port)` syntax.

### 🔧 Modular Architecture
Build custom pipelines by composing nodes (normalizers, PCA, RX detector, channel selectors) with automatic dependency resolution and port-based connections.

### 📊 Two-Phase Training
1. **Statistical Phase**: Initialize models using efficient statistical methods
2. **Gradient Phase**: Fine-tune with backpropagation for optimal performance

### 📈 Rich Monitoring
Track experiments with built-in support for:

- Loss and metric tracking
- Visualization generation (PCA plots, heatmaps, histograms)
- Artifact persistence (models, figures, logs)
- Multiple backends (DummyMonitor, WandB, TensorBoard)

### ⚙️ Flexible Configuration
Use Hydra for reproducible experiments:

```bash
python train.py training.trainer.max_epochs=10 nodes.pca.n_components=5
```

## Quick Links

- **[Installation](user-guide/installation.md)**: Get started with CUVIS.AI
- **[Quickstart](user-guide/quickstart.md)**: 5-minute tutorial with new Typed I/O system
- **[Tutorials](tutorials/phase1_statistical.md)**: Step-by-step guides
- **[API Reference](api/pipeline.md)**: Detailed API documentation
- **[Migration Guide](user-guide/typed-io-migration.md)**: Transition from legacy API

## Project Phases

CUVIS.AI development follows a structured implementation plan:

| Phase | Status | Description |
|-------|--------|-------------|
| Phase 0 | Complete | Environment & baseline skeleton |
| Phase 1 | Complete | Statistical pipeline orchestration |
| Phase 2 | Complete | Visualization & monitoring nodes |
| Phase 3 | Complete | Trainable PCA, loss/metric leaves, monitoring |
| Phase 4 | Complete | Soft channel selector & advanced heads |
| Phase 5 | Complete | Typed I/O system implementation |
| Phase 6 | Complete | Documentation & validation |

## Getting Help

- **GitHub Issues**: [Report bugs or request features](https://github.com/cubert-hyperspectral/cuvis.ai/issues)
- **Documentation**: Browse the user guide and tutorials
- **Examples**: Check the `examples_torch/` directory for working code

## License

CUVIS.AI is licensed under the Apache License 2.0. See [LICENSE](https://github.com/cubert-hyperspectral/cuvis.ai/blob/main/LICENSE) for details.

---

!!! tip "Ready to get started?"
    Head over to the [Installation Guide](user-guide/installation.md) to set up CUVIS.AI, then try the [Quickstart Tutorial](user-guide/quickstart.md) for a hands-on introduction.
