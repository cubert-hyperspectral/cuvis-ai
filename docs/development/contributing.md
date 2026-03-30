!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Contributing Guide

Welcome to the cuvis-ai community! We're excited to have you contribute to our open-source hyperspectral image analysis framework.

## Introduction & Philosophy

cuvis-ai is built on the principle of community-driven open source development. We believe the best software is created through collaboration and diverse perspectives. Cubert GmbH is committed to fostering an open, inclusive, and positive community where everyone can contribute meaningfully.

### Primary Contribution Path: Plugins

The **recommended way** to extend cuvis-ai is through the **plugin system**. By creating plugins in your own repository, you can:

- Maintain independent version control and release cycles
- Manage your own dependencies without affecting the core framework
- Share your work with the community through the central plugin registry
- Keep your proprietary algorithms separate while still integrating with cuvis-ai pipelines

### Secondary Path: Built-in Nodes

Contributing directly to the core framework is also valuable for:

- Core functionality improvements
- General-purpose nodes that benefit all users
- Performance optimizations to the framework itself

Built-in contributions require more rigorous review and integration testing, but they become part of the official cuvis-ai distribution.

---

## Ways to Contribute

### 🔌 Plugin Nodes (Recommended)

**What:** Create custom nodes in your own repository that extend cuvis-ai functionality.

**When to use:** When you want to add domain-specific algorithms, experimental methods, or proprietary techniques.

**How:**

1. Follow the [Plugin Development Guide](../plugin-system/dev-quickstart.md) to create your plugin
2. Implement nodes inheriting from `cuvis_ai_core.node.Node`
3. Test with cuvis-ai pipelines using local plugin loading
4. Publish to GitHub and submit to the central registry (see [Plugin Contribution Workflow](#plugin-contribution-workflow) below)

**Benefits:** Independent versioning, faster development cycles, easier maintenance, community sharing.

### 🏗️ Built-in Nodes

**What:** Contribute nodes directly to the cuvis-ai core codebase.

**When to use:** When your node provides general-purpose functionality that benefits all users.

**How:**

1. Follow the [Add Built-in Node Guide](../how-to/add-builtin-node.md)
2. Submit a pull request with comprehensive tests
3. Undergo code review and integration testing

**Requirements:** Must follow core coding standards, include thorough documentation, pass all tests.

### 📖 Documentation

Improve guides, tutorials, API documentation, or fix typos. Documentation PRs are always welcome!

### 🐛 Bug Reports

Found a bug? [Open an issue](https://github.com/cubert-hyperspectral/cuvis-ai/issues) with:

- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, CUDA version)
- Minimal code example demonstrating the bug

### 💡 Feature Requests

Have an idea for improvement? [Open a discussion](https://github.com/cubert-hyperspectral/cuvis-ai/discussions) to:

- Describe the feature and its use case
- Explain why it would benefit the community
- Discuss potential implementation approaches

---

## Plugin Contribution Workflow

This section describes the **complete process** for contributing a plugin to the central registry, making your custom nodes discoverable and usable by the entire cuvis-ai community.

### Step 1: Develop Your Plugin

Follow the comprehensive [Plugin Development Guide](../plugin-system/dev-quickstart.md) to create your plugin from scratch.

**Key requirements:**

- **Node Implementation:** Inherit from `cuvis_ai_core.node.Node`
- **Project Structure:** Create proper `pyproject.toml` with dependencies and metadata
- **Documentation:** Write comprehensive README with usage examples
- **Testing:** Add tests using pytest

**Example plugin structure:**
```
my-plugin/
├── my_plugin/
│   ├── __init__.py
│   └── nodes/
│       ├── __init__.py
│       └── custom_detector.py  # Your node classes
├── tests/
│   └── test_custom_detector.py
├── pyproject.toml
├── README.md
└── LICENSE
```

**Example node implementation:**
```python
from cuvis_ai_core.node import Node

class CustomDetector(Node):
    """Custom anomaly detector using proprietary algorithm."""

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def process(self, data):
        # Your implementation here
        return processed_data
```

### Step 2: Test Locally

Before publishing, thoroughly test your plugin with cuvis-ai pipelines.

**Create a local manifest** (`plugins.yaml`):
```yaml
plugins:
  my_plugin:
    path: "../my-plugin"  # Local directory path for development
    provides:
      - my_plugin.nodes.CustomDetector
```

**Test with example pipelines:**
```bash
# Test plugin loading
uv run python -c "
from cuvis_ai_core.utils.node_registry import NodeRegistry
registry = NodeRegistry()
registry.load_plugins('plugins.yaml')
CustomDetector = NodeRegistry.get('CustomDetector', instance=registry)
print('Plugin loaded successfully!')
"

# Test with full pipeline
uv run restore-pipeline \
    --pipeline-path configs/pipeline/my_test_pipeline.yaml \
    --plugins-path plugins.yaml
```

**Verify:**

- ✅ All dependencies install correctly
- ✅ Nodes load without errors
- ✅ Nodes work in actual pipelines
- ✅ All tests pass: `pytest tests/`

### Step 3: Publish to GitHub

Prepare your plugin for public distribution.

**Requirements:**

- **Public GitHub repository** (required for central registry)
- **Semantic version tag:** `v1.0.0`, `v0.1.0-beta`, `v2.0.0-rc.1`
- **LICENSE file:** Permissive open source license recommended (MIT, Apache-2.0, BSD)

**Create a comprehensive README** covering: installation (plugins YAML snippet), nodes provided, usage example, dependencies, and license.

**Create Git tag:**
```bash
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

### Step 4: Submit to Central Registry

Add your plugin to the cuvis-ai central registry so users can discover and use it.

**4.1 Fork the cuvis-ai repository**

Navigate to [github.com/cubert-hyperspectral/cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) and click "Fork".

**4.2 Edit `configs/plugins/registry.yaml`**

Add your plugin entry:
```yaml
plugins:
  # ... existing plugins ...

  my_plugin:
    repo: "https://github.com/yourorg/my-plugin.git"
    tag: "v1.0.0"  # Your release tag
    provides:
      - my_plugin.nodes.CustomDetector
      - my_plugin.nodes.HelperNode  # List all public nodes
```

**4.3 Add documentation entry**

Edit `docs/plugin-system/index.md` under the "Community Plugins" section:
```markdown
### Community Plugins

- **[cuvis-ai-adaclip](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip)** - AdaCLIP vision-language anomaly detection
- **[my-plugin](https://github.com/yourorg/my-plugin)** - Brief description of your plugin's functionality and use cases
```

**4.4 (Optional) Add showcase example**

Create `examples/my-plugin/` directory with:

- Sample pipeline configuration using your plugin
- Example data or instructions
- README explaining the example

### Step 5: Create Pull Request

Submit your plugin registration for review.

**PR title format:** `plugin: add [my-plugin] to registry`

**PR description should include:** plugin name, repository URL, version, purpose, nodes provided, example usage YAML, testing performed, and dependencies.

**Checklist:**

- [ ] Added entry to `configs/plugins/registry.yaml`
- [ ] Added documentation to `docs/plugin-system/index.md`
- [ ] Included LICENSE file
- [ ] README has installation and usage examples
- [ ] (Optional) Added showcase example in `examples/`

**Add label:** `plugin-contribution`

### Step 6: Review Process

The core team will review your submission (typical turnaround: 3-5 business days). We check YAML syntax validity, successful NodeRegistry loading, licensing, documentation, and tests. All feedback is provided through PR comments.

### Step 7: Post-Acceptance Maintenance

Keep your plugin compatible with new cuvis-ai releases. When releasing new versions, submit PRs to update the `tag:` field in `configs/plugins/registry.yaml`. Follow [semantic versioning](https://semver.org/).

---

## Important Requirements for Plugins

- Only Git tags supported (no branches/commits) for reproducibility
- Valid tag formats: `v1.2.3`, `v0.1.0-alpha`, `2.0.0-rc.1` (semantic versioning)
- Proper `pyproject.toml` with all dependencies declared
- Comprehensive README with usage examples
- Permissive open source license (MIT, Apache-2.0, BSD preferred)
- Loading test passes (NodeRegistry can import all provided classes)

During development, you can use local directory paths (`path: "../my-plugin"`) instead of Git repositories in your `plugins.yaml`.

---

## Development Environment Setup

### Prerequisites

- **Python 3.10+** (3.11 recommended)
- **uv** package manager: [Install instructions](https://github.com/astral-sh/uv)
- **Git** for version control
- **CUDA** (optional, for GPU acceleration)

### Clone and Install

```bash
# Clone the repository
git clone https://github.com/cubert-hyperspectral/cuvis-ai.git
cd cuvis-ai

# Install with all dependencies including dev tools
uv sync --all-extras --dev
```

### Set Up Git Hooks

```bash
pre-commit install
```

### Running Tests

```bash
pytest -v                                      # All tests
pytest --cov=cuvis_ai_core --cov-report=html   # With coverage
```

---

## Code Standards

### Style Guide

We follow **PEP 8** with enforcement through **Ruff** (line length: 100, double quotes, 4-space indent). Pre-commit hooks auto-format your code.

### Type Hints

**Type hints are required** for all public functions:

```python
def process_data(
    data: np.ndarray,
    threshold: float = 0.5
) -> tuple[np.ndarray, dict[str, Any]]:
    """Process hyperspectral data."""
    ...
```

### Docstrings

We use **Google-style docstrings**. See the [Docstrings Guide](docstrings.md) for detailed formatting requirements.

**Example:**
```python
class CustomNode(Node):
    """Brief one-line description.

    More detailed explanation of what the node does,
    its purpose, and how it fits into pipelines.

    Args:
        param1: Description of param1
        param2: Description of param2

    Attributes:
        attribute1: Description of attribute1

    Example:
        ```python
        node = CustomNode(param1=value1)
        result = node.process(data)
        ```
    """
    ...
```

### Testing

All new code requires tests:

- **Unit tests** for individual functions/methods
- **Integration tests** for node interactions
- **Use pytest fixtures** for common setup
- **Aim for 80%+ code coverage**

---

## Pull Request Guidelines

### Branch Naming

Use descriptive branch names with prefixes:

- `feature/add-custom-detector` - New features
- `fix/memory-leak-in-pipeline` - Bug fixes
- `docs/update-plugin-guide` - Documentation updates
- `refactor/simplify-node-registry` - Code refactoring

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/) style:

```
type(scope): brief description

Longer explanation if needed (wrap at 72 chars).

- Bullet points for multiple changes
- Use present tense ("add" not "added")
```

**Types:** `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

**Examples:**
```
feat(nodes): add spectral angle mapper node

docs(plugin-system): update contribution workflow

fix(pipeline): resolve memory leak in batch processing
```

### Before Submitting

- [ ] All tests pass: `pytest -v`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Documentation updated (if applicable)
- [ ] Docstrings and type hints included
- [ ] CHANGELOG updated (for significant changes)

---

## Community Values

Cubert GmbH is committed to creating an **open, inclusive, and positive community**. We expect all contributors to:

- Be respectful and welcoming to newcomers
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards others

We have zero tolerance for harassment, discrimination, or abusive behavior.
