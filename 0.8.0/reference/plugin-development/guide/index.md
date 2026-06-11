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
      - class_name: my_plugin.node.custom_node.CustomNode
```

Relative paths resolve from the manifest file location, not from the current shell directory.

## Manifest for a Tagged Release

```yaml
plugins:
  my_plugin:
    repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
    tag: "v0.1.0"
    provides:
      - class_name: my_plugin.node.custom_node.CustomNode
```

Each `provides` entry needs at least `class_name` (a fully-qualified path); it may also carry palette metadata (`category`, `tags`, `icon_svg`, `input_specs`, `output_specs`, `doc_summary`). See [Plugin System Overview](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/reference/plugin-development/overview/index.md).

## Verification

Use `uv` for local validation:

```bash
uv run pytest tests/ -q

# Dev-mode check: load the manifest directly and list the registered plugins
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.load_plugins('plugins.yaml'); print(r.list_plugins())"

# End-to-end: run a pipeline that declares `plugins: [my_plugin]`
uv run restore-pipeline --pipeline-path <pipeline>.yaml --plugins-dir <dir-with-manifest>
```

## Release Notes

- Tag releases with semver-style Git tags such as `v0.1.0`.
- Keep `provides` stable across patch releases unless you are intentionally making a breaking change.
- Test the tagged manifest before referencing it from this repo.

See [Plugin System Overview](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/reference/plugin-development/overview/index.md) for loader behavior and [Plugin Nodes](https://cubert-hyperspectral.github.io/cuvis-ai/0.8.0/catalogs/nodes/index.md) for end-user loading examples.
