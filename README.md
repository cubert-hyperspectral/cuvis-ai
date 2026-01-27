![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# cuvis.ai

cuvis.ai is a software toolkit designed to facilitate the development of artificial intelligence (AI) and machine 
learning applications for hyperspectral measurements.

- **Website:** https://www.cubert-hyperspectral.com/
- **Source code:** https://github.com/cubert-hyperspectral/
- **Support:** http://support.cubert-hyperspectral.com/

This toolkit enables the creation of a graph from a set of different preexisting supervised and unsupervised nodes. 
Furthermore, it provides data preprocessing and output postprocessing, thus offering a comprehensive package for the 
development of AI capabilities for hyperspectral images. 

This repository is aimed at companies, universities and private enthusiasts alike. Its objective is to provide a 
foundation for the development of cutting-edge hyperspectral AI applications.

## Architecture

cuvis.ai has been architected as a modular system comprising a core framework and domain-specific catalog:

- **[cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core)**: Framework repository providing base `Node` class, pipeline orchestration, training infrastructure, gRPC services, and `NodeRegistry` with plugin loading capabilities
- **cuvis-ai** (this repository): Catalog repository with domain-specific nodes for anomaly detection, preprocessing, band selection, and hyperspectral-specific algorithms

The plugin system enables external nodes to be loaded dynamically from Git repositories or local filesystem paths via `NodeRegistry.load_plugins()`. This allows teams to develop custom nodes independently without modifying the catalog repository.

## Installation

### Prerequisites

If you want to directly work with cubert session files (.cu3s), you need to install cuvis C SDK from 
[here](https://cloud.cubert-gmbh.de/s/qpxkyWkycrmBK9m).

Local development now relies on [uv](https://docs.astral.sh/uv/) for Python and dependency management.  
If `uv` is not already available on your system you can install it following their installation instructions.

### Local development with uv

Create or refresh a development environment at the repository root with:

```bash
uv sync --all-extras --dev
```

This installs the runtime dependencies declared in `pyproject.toml`. `uv` automatically provisions the Python version declared in the project metadata, so no manual interpreter management is required.

#### Enable Git Hooks (Required)

After cloning the repository, enable the git hooks for code quality enforcement:

```bash
git config core.hooksPath .githooks
```

This configures Git to use the version-controlled hooks in `.githooks/` which automatically enforce code formatting, linting, and testing standards before commits and pushes. See [docs/development/git-hooks.md](docs/development/git-hooks.md) for details.

#### Advanced environment setup

When you need the reproducible development toolchain (JupyterLab, TensorBoard, etc.) from the lock file, run:

```bash
uv sync --locked --extra dev
```

Use `uv run` to execute project tooling without manually activating virtual environments, for example:

```bash
uv run pytest
```

Collect coverage details (the `dev` extra installs `pytest-cov`) with:

```bash
uv run pytest --cov=cuvis_ai --cov-report=term-missing
```

Ruff handles both formatting and linting. Format sources and check style with:

```bash
uv run ruff format .
uv run ruff check .
```

The configuration enforces import ordering, newline hygiene, modern string formatting, safe exception chaining, and practical return type annotations while avoiding noisy `Any` policing.

Validate packaging metadata and build artifacts before publishing:

```bash
uv build
```


To build the documentation, add the `docs` extra:

```bash
uv sync --locked --extra docs
```

Combine extras as needed (e.g. `uv sync --locked --extra dev --extra docs`). Whenever the `pyproject.toml` or `uv.lock` changes, rerun `uv sync --locked` with the extras you need to stay up to date.

## Available Built-in Nodes

<details>
<summary>Click to expand the full list of built-in nodes in cuvis-ai</summary>

### Data Nodes
- `LentilsAnomalyDataNode` - Lentils dataset data loader

### Preprocessing Nodes
- `BandpassByWavelength` - Spectral band filtering by wavelength range

### Band Selection
- `BaselineFalseRGBSelector` - RGB composite from predefined bands
- `HighContrastBandSelector` - High-contrast band selection
- `CIRFalseColorSelector` - Color infrared (CIR) false color composite
- `SupervisedCIRBandSelector` - Supervised CIR band selection using Fisher scores and mRMR
- `SupervisedWindowedFalseRGBSelector` - Supervised RGB selection with windowed band constraints
- `SupervisedFullSpectrumBandSelector` - Supervised full-spectrum band selection

### Channel Processing
- `LearnableChannelMixer` - Trainable channel mixing with statistical initialization
- `ConcreteBandSelector` - Differentiable band selection using Gumbel-Softmax (Concrete distribution)
- `SoftChannelSelector` - Soft attention-based channel selection with temperature annealing
- `TopKIndices` - Select top-k channel indices from weights

### Dimensionality Reduction
- `TrainablePCA` - Trainable PCA with statistical initialization and gradient-based updates

### Anomaly Detection
- `AdaCLIPLocalNode` - Local AdaCLIP model for anomaly detection
- `AdaCLIPAPINode` - HuggingFace API-based AdaCLIP

### Labels & Targets
- `BinaryAnomalyLabelMapper` - Map class IDs to binary anomaly labels

### Loss Nodes
- `AnomalyBCEWithLogits` - Binary cross-entropy loss for anomaly detection
- `MSEReconstructionLoss` - Mean squared error reconstruction loss
- `OrthogonalityLoss` - Component orthogonality regularization
- `DistinctnessLoss` - Band selection distinctness regularization
- `SelectorEntropyRegularizer` - Entropy regularization for soft selectors
- `SelectorDiversityRegularizer` - Diversity regularization for band selection
- `DeepSVDDSoftBoundaryLoss` - Deep SVDD soft-boundary loss
- `IoULoss` - Intersection over Union loss

### Normalization Nodes
- `IdentityNormalizer` - Pass-through (no normalization)
- `MinMaxNormalizer` - Min-max normalization with optional running statistics
- `SigmoidNormalizer` - Sigmoid-based normalization
- `ZScoreNormalizer` - Z-score (standardization) normalization
- `SigmoidTransform` - Sigmoid activation transform
- `PerPixelUnitNorm` - Per-pixel L2 normalization

### Metrics Nodes
- `ExplainedVarianceMetric` - Track explained variance ratio (for PCA)
- `AnomalyDetectionMetrics` - Precision, Recall, F1, IoU, AUC-ROC for anomaly detection
- `ScoreStatisticsMetric` - Mean, std, min, max of anomaly scores
- `ComponentOrthogonalityMetric` - Measure orthogonality of learned components
- `SelectorEntropyMetric` - Entropy of channel selection weights
- `SelectorDiversityMetric` - Diversity of selected channels

### Visualization Nodes
- `CubeRGBVisualizer` - RGB visualization from hyperspectral cube
- `PCAVisualization` - Visualize PCA components
- `AnomalyMask` - Binary anomaly mask visualization
- `ScoreHeatmapVisualizer` - Anomaly score heatmap
- `RGBAnomalyMask` - RGB overlay with predicted/ground-truth anomaly masks
- `DRCNNTensorBoardViz` - DRCNN-specific TensorBoard visualizations

### Monitoring
- `TensorBoardMonitorNode` - Log metrics and artifacts to TensorBoard

</details>

## Contributed Nodes

<details>
<summary>Click to expand community-contributed plugin nodes</summary>

### cuvis-ai-adaclip
**Repository**: [cubert-hyperspectral/cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)

AdaCLIP-based anomaly detection nodes with advanced band selection strategies for hyperspectral imaging. Provides nodes for baseline detection, supervised band selection using Fisher scores and mRMR, and various false-color composite generators (CIR, RGB, high-contrast).

</details>

## Contributing Custom Nodes via Plugins

External teams can develop custom nodes without modifying this catalog repository:

1. **Create a plugin repository** with your custom nodes (inherit from `cuvis_ai_core.node.Node`)
2. **Configure plugin manifest** (`configs/plugins.yaml`) specifying Git repo/local path and provided node classes
3. **Load plugins** via `NodeRegistry.load_plugins("plugins.yaml")` before pipeline building
4. **Use in pipelines** by referencing plugin node class paths in your pipeline YAML configurations

**Example plugin manifest:**
```yaml
plugins:
  my_custom_nodes:
    repo: "git@github.com:myorg/cuvis-custom-nodes.git"
    ref: "v1.0.0"
    provides:
      - my_custom_nodes.MyCustomDetector
      - my_custom_nodes.MyPreprocessor
```

See [cuvis-ai-core plugin documentation](https://github.com/cubert-hyperspectral/cuvis-ai-core) for detailed plugin development guide.

## Documentation

- **[tests/README.md](tests/README.md)** - Test fixtures guide and pytest patterns
- **[examples/grpc/readme.md](examples/grpc/readme.md)** - gRPC client examples and workflow guide
- **[cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core)** - Framework repository with plugin system documentation

## Release Notes

See [CHANGELOG.md](CHANGELOG.md) for the consolidated refactor summary and upgrade guidance.

## How to ...

### Getting started

We provide an additional example repository [here](https://github.com/cubert-hyperspectral/cuvis.ai.examples),
covering some basic applications.

Further, we provide a set of example measurements to explore [here](https://cloud.cubert-gmbh.de/s/SrkSRja5FKGS2Tw).
These measurements are also used by the examples mentioned above.

### Getting involved

cuvis.hub welcomes your enthusiasm and expertise!

With providing our SDK wrappers on GitHub, we aim for a community-driven open 
source application development by a diverse group of contributors.
Cubert GmbH aims for creating an open, inclusive, and positive community.
Feel free to branch/fork this repository for later merge requests, open 
issues or point us to your application specific projects.
Contact us, if you want your open source project to be included and shared 
on this hub; either if you search for direct support, collaborators or any 
other input or simply want your project being used by this community.
We ourselves try to expand the code base with further more specific 
applications using our wrappers to provide starting points for research 
projects, embedders or other users.

### Getting help

Directly code related issues can be posted here on the GitHub page, other, more 
general and application related issues should be directed to the 
aforementioned Cubert GmbH [support page](http://support.cubert-hyperspectral.com/).
