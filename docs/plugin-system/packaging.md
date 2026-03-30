!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Packaging & Distribution

## Packaging and Distribution

### Create README.md

```markdown
# CUVIS-AI Custom Anomaly Detection Plugin

Custom anomaly detection algorithms for cuvis-ai framework.

## Installation

### From Git

```bash
# Create plugins.yaml
cat > plugins.yaml << EOF
plugins:
  my_plugin:
    repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
    tag: "v0.1.0"
    provides:

      - cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector
EOF

# Load in Python
from cuvis_ai_core.utils.node_registry import NodeRegistry
registry = NodeRegistry()
registry.load_plugins("plugins.yaml")
```

### From Local Path

```bash
# For development (recommended)
uv sync

# Alternative: editable install
uv pip install -e .
```

## Usage

```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

registry = NodeRegistry()
registry.load_plugin(
    name="my_plugin",
    config={
        "path": "path/to/plugin",
        "provides": ["cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector"]
    }
)

CustomDetector = registry.get("CustomAnomalyDetector", instance=registry)
detector = CustomDetector(threshold=0.95)
```

## Nodes

### CustomAnomalyDetector

Statistical anomaly detection using Mahalanobis distance or contextual methods.

**Parameters:**
- `threshold` (float): Detection threshold (0-1)
- `method` (str): 'simple' or 'advanced'
- `window_size` (int): Contextual window size

**Inputs:**
- `data` (np.ndarray): Hyperspectral cube (H, W, C)

**Outputs:**
- `scores` (np.ndarray): Anomaly scores (H, W)
- `detections` (np.ndarray): Binary detections (H, W)

## Development

```bash
git clone https://github.com/your-org/cuvis-ai-my-plugin.git
cd cuvis-ai-my-plugin
uv sync --extra dev
uv run pytest tests/
```

## License

MIT License
```

### Create .gitignore

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# cuvis cache
.cuvis/
.cuvis_plugins/
```

### Versioning Strategy

Follow semantic versioning (semver):

**Version Format:** `MAJOR.MINOR.PATCH`

- **MAJOR:** Breaking API changes
- **MINOR:** New features, backwards compatible
- **PATCH:** Bug fixes

**Git Tags:**
```bash
# Create annotated tag
git tag -a v0.1.0 -m "Initial release"

# Push tags
git push origin v0.1.0
```

**Update Version:**

1. Update `pyproject.toml`:
   ```toml
   version = "0.2.0"
   ```

2. Update `__init__.py`:
   ```python
   __version__ = "0.2.0"
   ```

3. Create CHANGELOG.md entry:
   ```markdown
   ## [0.2.0] - 2026-02-15
   ### Added
   - New advanced detection method
   - GPU acceleration support

   ### Changed
   - Improved threshold handling

   ### Fixed
   - Bug in score normalization
   ```

4. Commit and tag:
   ```bash
   git add pyproject.toml cuvis_ai_plugin/__init__.py CHANGELOG.md
   git commit -m "Bump version to 0.2.0"
   git tag -a v0.2.0 -m "Release v0.2.0"
   git push origin main v0.2.0
   ```

## Publishing Options

### Option 1: Git Repository (Recommended)

Host on GitHub/GitLab:

```bash
# Create repository on GitHub
gh repo create your-org/cuvis-ai-my-plugin --public

# Push code
git remote add origin https://github.com/your-org/cuvis-ai-my-plugin.git
git push -u origin main
git push origin v0.1.0
```

**Users install via:**
```yaml
plugins:
  my_plugin:
    repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
    tag: "v0.1.0"
    provides:
      - cuvis_ai_plugin.nodes.custom_node.CustomAnomalyDetector
```

### Option 2: PyPI (Public/Private)

Build and upload to PyPI:

```bash
# Install build tools with uv
uv pip install build twine

# Build package
uv run python -m build

# Upload to PyPI
uv run python -m twine upload dist/*

# Or upload to TestPyPI first
uv run python -m twine upload --repository testpypi dist/*
```

**Users install via:**
```bash
# Using uv (recommended)
uv add cuvis-ai-my-plugin

# Or using uv pip
uv pip install cuvis-ai-my-plugin
```

Then use directly (automatic discovery):
```python
from cuvis_ai_core.utils.node_registry import NodeRegistry
CustomDetector = NodeRegistry.get("CustomAnomalyDetector")
```

### Option 3: Private Package Repository

For enterprise/private plugins:

```bash
# Upload to private PyPI server
twine upload --repository-url https://your-pypi.company.com/ dist/*

# Or use artifact repository (e.g., Artifactory, Nexus)
```

## Best Practices

### 1. Node Design

- **Single Responsibility:** Each node should do one thing well
- **Input Validation:** Validate all inputs in `forward()` method
- **Error Handling:** Raise clear, actionable exceptions
- **Port Documentation:** Document all ports in `PortSpec` descriptions
- **Docstrings:** Follow NumPy-style docstrings (see [Docstring Guide](../development/docstrings.md))

### 2. Testing

- **Coverage:** Aim for >90% test coverage
- **Edge Cases:** Test boundary conditions and error cases
- **Integration:** Test nodes in pipeline context
- **Performance:** Benchmark critical operations
- **Fixtures:** Use pytest fixtures for reusable test data

### 3. Documentation

- **README:** Clear installation and usage instructions
- **Examples:** Provide working code examples
- **Changelog:** Maintain version history
- **API Docs:** Document all public APIs
- **Tutorials:** Create guides for complex workflows

### 4. Performance

- **Profiling:** Profile code to identify bottlenecks
- **Vectorization:** Use NumPy vectorized operations
- **GPU Support:** Leverage GPUs for compute-heavy operations
- **Memory:** Minimize memory allocations
- **Caching:** Cache expensive computations when appropriate

### 5. Dependencies

- **Minimal:** Only include necessary dependencies
- **Pinning:** Pin versions for reproducibility
- **Optional:** Use optional dependencies for extras
- **Security:** Audit dependencies for vulnerabilities

## Troubleshooting

### Plugin Not Loading

**Issue:** Plugin fails to load with import errors.

**Solution:**

1. Verify `pyproject.toml` exists
2. Check all `__init__.py` files present
3. Ensure dependencies installed: `uv sync` (or `uv pip install -e .`)
4. Test import manually: `uv run python -c "from cuvis_ai_plugin.nodes import CustomAnomalyDetector"`

### Node Not Found

**Issue:** Node not found after loading plugin.

**Solution:**

1. Check `provides` list in manifest includes full class path
2. Verify class name spelling matches exactly
3. Ensure class inherits from `Node`
4. Check `__all__` exports in `__init__.py`

### Test Failures

**Issue:** Tests fail unexpectedly.

**Solution:**

1. Run with verbose output: `uv run pytest tests/ -v`
2. Check test data shapes match node expectations
3. Verify NumPy random seeds for reproducibility
4. Isolate failing test: `uv run pytest tests/test_custom_node.py::test_name -v`

### Performance Issues

**Issue:** Node is too slow.

**Solution:**

1. Profile code: `uv run python -m cProfile -o profile.stats your_script.py`
2. Analyze: `uv run python -m pstats profile.stats`
3. Vectorize loops using NumPy
4. Use `numba` JIT compilation for hot loops
5. Consider GPU acceleration with PyTorch/CuPy

## Example: Complete Plugin

See the [cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip) plugin for a complete real-world example.

## See Also

- **[Plugin Development Guide](development.md)** - Developing custom nodes
- **[Plugin System Overview](overview.md)** - Plugin architecture and concepts
- **[Plugin Usage Guide](usage.md)** - Using plugins in workflows
- **[Node System Deep Dive](../concepts/node-system-deep-dive.md)** - Node architecture details
- **[Port System Deep Dive](../concepts/port-system-deep-dive.md)** - Port specifications and connections
- **[Two-Phase Training](../concepts/two-phase-training.md)** - Statistical initialization patterns
- **[Docstring Standards](../development/docstrings.md)** - Documentation guidelines
