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
1. Follow the [Plugin Development Guide](../plugin-system/development.md) to create your plugin
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

Follow the comprehensive [Plugin Development Guide](../plugin-system/development.md) to create your plugin from scratch.

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

**Create comprehensive README:**
```markdown
# My Custom Plugin

Brief description of your plugin and its purpose.

## Installation

\`\`\`yaml
plugins:
  my_plugin:
    repo: "https://github.com/yourorg/my-plugin.git"
    tag: "v1.0.0"
    provides:
      - my_plugin.nodes.CustomDetector
\`\`\`

## Nodes Provided

### CustomDetector

Description of the node, its inputs, outputs, and parameters.

## Usage Example

\`\`\`python
# Example pipeline configuration
...
\`\`\`

## Dependencies

- numpy>=1.20.0
- scikit-learn>=1.0.0

## License

MIT License
```

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

**PR title format:**
```
plugin: add [my-plugin] to registry
```

**PR description template:**
```markdown
## Plugin Information

**Name:** my-plugin
**Repository:** https://github.com/yourorg/my-plugin
**Version:** v1.0.0

## Purpose

Brief description of what the plugin does and why it's useful.

## Nodes Provided

- `CustomDetector`: Description of this node
- `HelperNode`: Description of this node

## Example Usage

\`\`\`yaml
plugins:
  my_plugin:
    repo: "https://github.com/yourorg/my-plugin.git"
    tag: "v1.0.0"
    provides:
      - my_plugin.nodes.CustomDetector

# Use in pipeline
nodes:
  detector:
    _target_: CustomDetector
    threshold: 0.75
\`\`\`

## Testing Performed

- [x] Plugin loads successfully with NodeRegistry
- [x] All tests pass (`pytest tests/`)
- [x] Tested with example pipeline (attached in examples/)
- [x] Documentation is comprehensive

## Dependencies

No significant dependencies beyond cuvis-ai-core requirements.

## Checklist

- [x] Added entry to `configs/plugins/registry.yaml`
- [x] Added documentation to `docs/plugin-system/index.md`
- [x] Included LICENSE file (MIT)
- [x] README has installation and usage examples
- [ ] Added showcase example in `examples/` (optional)
```

**Add label:** `plugin-contribution`

### Step 6: Review Process

The core team will review your submission.

**What we check:**
- ✅ YAML manifest syntax is valid
- ✅ Plugin loads successfully with NodeRegistry
- ✅ Plugin follows development best practices
- ✅ Appropriate open source licensing
- ✅ Documentation is comprehensive and clear
- ✅ Tests exist and pass (if example provided)

**Timeline:**
- Typical review turnaround: 3-5 business days
- We may request changes to manifest formatting or documentation
- Once approved, your plugin will be merged into the registry

**Communication:**
- All feedback will be provided through PR comments
- Please respond to review comments promptly
- Feel free to ask questions or request clarification

### Step 7: Post-Acceptance Maintenance

After your plugin is accepted, keep it maintained for the community.

**Ongoing responsibilities:**
- **Respond to issues:** Help users who encounter problems with your plugin
- **Keep it working:** Ensure compatibility with new cuvis-ai releases
- **Update registry:** When releasing new versions, submit PRs to update the `tag:` field

**Releasing updates:**
```bash
# Create new version tag
git tag -a v1.1.0 -m "Add new feature X"
git push origin v1.1.0

# Update registry (submit PR)
# Edit configs/plugins/registry.yaml:
my_plugin:
  repo: "https://github.com/yourorg/my-plugin.git"
  tag: "v1.1.0"  # Update to new version
  provides:
    - my_plugin.nodes.CustomDetector
```

**Best practices:**
- Use [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github) for better visibility
- Follow [semantic versioning](https://semver.org/) (MAJOR.MINOR.PATCH)
- Write clear release notes describing changes
- Update your plugin's README when adding features

---

## Important Requirements for Plugins

When developing and submitting plugins, ensure they meet these requirements:

- ✅ **Only Git tags supported** (no branches/commits) for reproducibility
- ✅ **Valid tag formats:** `v1.2.3`, `v0.1.0-alpha`, `2.0.0-rc.1` (semantic versioning)
- ✅ **Proper `pyproject.toml`** with all dependencies declared
- ✅ **Comprehensive README** with usage examples and API documentation
- ✅ **Permissive open source license** (MIT, Apache-2.0, BSD preferred)
- ✅ **Loading test passes** (NodeRegistry can import all provided classes)

**Local development support:**

During development, you can use local directory paths instead of Git repositories:

```yaml
plugins:
  # Production plugin from Git
  production_plugin:
    repo: "https://github.com/org/plugin.git"
    tag: "v1.0.0"
    provides:
      - production_plugin.nodes.Node1

  # Local plugin for development
  dev_plugin:
    path: "../my-plugin"  # Relative or absolute directory path
    provides:
      - dev_plugin.nodes.TestNode
```

This is particularly useful for:
- Testing changes before publishing
- Private plugins that won't be on GitHub
- Quick iterations during development

---

## Development Environment Setup

To contribute to cuvis-ai core or develop plugins, set up your development environment.

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

We use pre-commit hooks to ensure code quality:

```bash
# Install pre-commit hooks
pre-commit install

# Run manually on all files (optional)
pre-commit run --all-files
```

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_node_registry.py -v

# Run with coverage
pytest --cov=cuvis_ai_core --cov-report=html
```

### IDE Setup

**VS Code:** Install recommended extensions (Python, Pylance, Ruff)

**PyCharm:** Configure interpreter to use the uv-managed virtual environment

---

## Code Standards

### Style Guide

We follow **PEP 8** with enforcement through **Ruff**:

- Line length: 100 characters (not the PEP 8 default of 79)
- Use double quotes for strings
- 4 spaces for indentation (no tabs)

The pre-commit hooks will automatically format your code.

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

### PR Description

Your PR description should include:

1. **What:** Brief summary of changes
2. **Why:** Motivation and context
3. **How:** Implementation approach (if non-obvious)
4. **Testing:** What tests were added/modified
5. **Breaking changes:** Any backwards-incompatible changes
6. **Screenshots:** If UI changes (for docs/examples)

### Before Submitting

Checklist:

- [ ] All tests pass: `pytest -v`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Documentation updated (if applicable)
- [ ] Docstrings added for new functions/classes
- [ ] Type hints included
- [ ] CHANGELOG updated (for significant changes)

---

## Getting Help

### Community Channels

- **Discord/Slack:** (coming soon!) Real-time chat with other contributors
- **GitHub Discussions:** Ask questions, share ideas, get feedback
- **Issue Tracker:** Report bugs or request features

### Asking Questions

When asking for help:

1. **Search first:** Check if your question has been answered
2. **Be specific:** Provide context, code examples, error messages
3. **Be respectful:** Everyone is volunteering their time

### Contact

For private inquiries or collaboration opportunities:
- Email: [contact info from README or website]
- Open a [GitHub Discussion](https://github.com/cubert-hyperspectral/cuvis-ai/discussions)

---

## Community Values

Cubert GmbH is committed to creating an **open, inclusive, and positive community**. We expect all contributors to:

- Be respectful and welcoming to newcomers
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards others

We have zero tolerance for harassment, discrimination, or abusive behavior.

---

## Thank You!

Thank you for contributing to cuvis-ai! Your work helps build a better tool for the hyperspectral imaging community. We're excited to see what you create!

**Quick Links:**
- [Plugin Development Guide](../plugin-system/development.md) - Complete guide to creating plugins
- [Add Built-in Node Guide](../how-to/add-builtin-node.md) - Contributing to core
- [Node System Deep Dive](../concepts/node-system-deep-dive.md) - Understanding node architecture
- [Central Plugin Registry](../../configs/plugins/registry.yaml) - Browse registered plugins
