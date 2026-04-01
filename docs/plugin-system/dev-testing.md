!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Testing & Publishing

Testing best practices, publishing options, and troubleshooting for cuvis-ai plugins.

## Testing Best Practices

### Unit Tests

Test individual node functionality:

```python
def test_node_ports():
    """Test node has correct input/output ports."""
    node = CustomAnomalyDetector()

    assert len(node.INPUT_SPECS) == 1
    assert node.INPUT_SPECS[0].name == "data"

    assert len(node.OUTPUT_SPECS) == 2
    assert node.OUTPUT_SPECS[0].name == "scores"
    assert node.OUTPUT_SPECS[1].name == "detections"


def test_node_deterministic():
    """Test node produces deterministic results."""
    node = CustomAnomalyDetector(threshold=0.9, method="simple")

    np.random.seed(42)
    data = np.random.randn(10, 10, 50).astype(np.float32)

    output1 = node(data=data)
    output2 = node(data=data)

    np.testing.assert_array_equal(output1["scores"], output2["scores"])
    np.testing.assert_array_equal(output1["detections"], output2["detections"])
```

### Integration Tests

Test node in pipeline context:

```python
def test_node_in_pipeline():
    """Test node works in pipeline."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry
    from cuvis_ai_core.pipeline.pipeline import Pipeline

    registry = NodeRegistry()
    registry.load_plugins("examples/plugins.yaml")

    pipeline_dict = {
        "nodes": [
            {
                "class_name": "CustomAnomalyDetector",
                "name": "detector",
                "hparams": {"threshold": 0.95}
            }
        ],
        "connections": []
    }

    pipeline = Pipeline.from_dict(pipeline_dict, node_registry=registry)

    data = np.random.randn(20, 20, 50).astype(np.float32)
    outputs = pipeline(data=data)

    assert "detector" in outputs
    assert "scores" in outputs["detector"]
```

### Performance Tests

```python
import time

def test_node_performance():
    """Test node processes data within time budget."""
    node = CustomAnomalyDetector()
    data = np.random.randn(100, 100, 224).astype(np.float32)

    start = time.time()
    outputs = node(data=data)
    elapsed = time.time() - start

    assert elapsed < 1.0  # Should process 100x100x224 in < 1 second
```

---

## Publishing Options

### Option 1: Git Repository (Recommended)

```bash
gh repo create your-org/cuvis-ai-my-plugin --public
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

### Option 2: PyPI

```bash
uv pip install build twine
uv run python -m build
uv run python -m twine upload dist/*
```

**Users install via:**
```bash
uv add cuvis-ai-my-plugin
```

### Option 3: Private Package Repository

```bash
twine upload --repository-url https://your-pypi.company.com/ dist/*
```

---

## Best Practices

| Area | Guideline |
|------|-----------|
| **Node Design** | Single responsibility, input validation, clear exceptions |
| **Testing** | >90% coverage, edge cases, integration tests, performance benchmarks |
| **Documentation** | Clear README, working examples, changelog |
| **Performance** | NumPy vectorization, GPU support for heavy ops, minimal allocations |
| **Dependencies** | Minimal, pinned versions, use optional deps for extras |

---

## Troubleshooting

### Plugin Not Loading

1. Verify `pyproject.toml` exists
2. Check all `__init__.py` files present
3. Ensure dependencies installed: `uv sync`
4. Test import: `uv run python -c "from cuvis_ai_plugin.nodes import CustomAnomalyDetector"`

### Node Not Found

1. Check `provides` list in manifest includes full class path
2. Verify class name spelling matches exactly
3. Ensure class inherits from `Node`
4. Check `__all__` exports in `__init__.py`

### Test Failures

1. Run with verbose output: `uv run pytest tests/ -v`
2. Check test data shapes match node expectations
3. Verify NumPy random seeds for reproducibility
4. Isolate failing test: `uv run pytest tests/test_custom_node.py::test_name -v`

### Performance Issues

1. Profile: `uv run python -m cProfile -o profile.stats your_script.py`
2. Vectorize loops using NumPy
3. Use `numba` JIT compilation for hot loops
4. Consider GPU acceleration with PyTorch/CuPy

---

## Complete Plugin Example

See the [cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip) plugin for a real-world example.

## See Also

- [Plugin Quick Start](dev-quickstart.md) — create your first plugin
- [Advanced Node Development](dev-advanced.md) — advanced patterns
- [Plugin Packaging](packaging.md) — README templates, versioning, distribution
- [Docstring Standards](../development/docstrings.md) — documentation guidelines
