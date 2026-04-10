!!! warning "Status: Needs Review"
    This page has not been reviewed for accuracy and completeness. Content may be outdated or contain errors.

---

# Plugin Development Guide

This guide covers the minimum structure needed to build a cuvis-ai plugin that can be loaded through a manifest.

## Required Structure

```text
my-plugin/
├── pyproject.toml
├── my_plugin/
│   ├── __init__.py
│   └── node/
│       ├── __init__.py
│       └── custom_node.py
└── tests/
    └── test_custom_node.py
```

- `pyproject.toml` is required because plugin dependency installation reads project metadata from it.
- Export node classes from import paths that can be listed in a manifest `provides:` section.

## Minimal `pyproject.toml`

```toml
[project]
name = "cuvis-ai-my-plugin"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "cuvis-ai-core>=0.1.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

## Node Requirements

- Inherit from `cuvis_ai_core.node.node.Node`.
- Define `INPUT_SPECS` and `OUTPUT_SPECS`.
- Implement `forward()`.
- Pass serializable constructor arguments through `super().__init__(...)`.

## Manifest for Local Development

```yaml
plugins:
  my_plugin:
    path: "../my-plugin"
    provides:
      - my_plugin.node.custom_node.CustomNode
```

Relative paths resolve from the manifest file location, not from the current shell directory.

## Manifest for a Tagged Release

```yaml
plugins:
  my_plugin:
    repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
    tag: "v0.1.0"
    provides:
      - my_plugin.node.custom_node.CustomNode
```

## Verification

Use `uv` for local validation:

```bash
uv run pytest tests/ -q
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.load_plugins('plugins.yaml'); print(r.list_plugins())"
```

## Release Notes

- Tag releases with semver-style Git tags such as `v0.1.0`.
- Keep `provides` stable across patch releases unless you are intentionally making a breaking change.
- Test the tagged manifest before referencing it from this repo.

See [Plugin System Overview](overview.md) for loader behavior and [Plugin Nodes](../node-catalog/node-catalog-plugins.md) for end-user loading examples.
