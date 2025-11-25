![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# cuvis.ai

*The project is still maturing. We will make occasional breaking changes and add missing features on our way to v1.*

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


## Installation

### Prerequisites

If you want to directly work with cubert session files (.cu3s), you need to install cuvis C SDK from 
[here](https://cloud.cubert-gmbh.de/s/qpxkyWkycrmBK9m).

Local development now relies on [uv](https://docs.astral.sh/uv/) for Python and dependency management.  
If `uv` is not already available on your system you can install it with:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Local development with uv

Create or refresh a development environment at the repository root with:

```bash
uv sync
```

This installs the runtime dependencies declared in `pyproject.toml`. `uv` automatically provisions the Python version declared in the project metadata, so no manual interpreter management is required.

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

Detect lingering CamelCase module filenames with:

```bash
uv run python scripts/check_module_case.py
```

To build the documentation, add the `docs` extra:

```bash
uv sync --locked --extra docs
uv run sphinx-build -M html docs docs/_build
```

Combine extras as needed (e.g. `uv sync --locked --extra dev --extra docs`). Whenever the `pyproject.toml` or `uv.lock` changes, rerun `uv sync --locked` with the extras you need to stay up to date.

## gRPC API

cuvis.ai provides a gRPC API for remote inference and training, enabling integration with C++ clients (cuvis-next) and other language implementations.

### Using the gRPC API

The gRPC API is included in the standard installation. For client usage examples, see the [gRPC examples](examples/grpc/).

Key features:
- Remote inference with hyperspectral data
- Training pipeline management (statistical & gradient-based)
- Pipeline introspection and visualization
- Checkpoint management

### Developing the gRPC API

If you're modifying `.proto` files or working on the gRPC implementation:

1. **Proto Code Generation**: Use grpcio-tools (already installed) to generate Python code:
   ```bash
   python -m grpc_tools.protoc -I=proto \
     --python_out=cuvis_ai/grpc \
     --pyi_out=cuvis_ai/grpc \
     --grpc_python_out=cuvis_ai/grpc \
     proto/cuvis_ai.proto
   ```

2. **Buf CLI (Optional)**: For advanced features like linting and BSR publishing:
   - Install from [GitHub releases](https://github.com/bufbuild/buf/releases)
   - **Do not** use `pip install buf` - that's a different package

3. **Full Development Guide**: See [docs_dev/grpc_development_guide.md](docs_dev/grpc_development_guide.md) for:
   - Proto modification workflow
   - Publishing to Buf Schema Registry
   - Troubleshooting common issues
   - Best practices

**Note:** Generated proto files are committed to the repository, so end users don't need any proto tooling.

### Via pip

If you wish to use cuvis.ai within another project, from within your 
project environment, run 

```
pip install cuvis-ai
```

or add `cuvis-ai` to your project `requirements.txt` or `setup.py`.

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
