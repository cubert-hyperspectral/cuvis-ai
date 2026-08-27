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
- Export node classes from import paths that can be listed in a manifest `capabilities:` section.

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
# my_plugin.yaml (one file per plugin)
name: my_plugin
path: "../my-plugin"
capabilities:
  - class_name: my_plugin.node.custom_node.CustomNode
```

Relative paths resolve from the manifest file location, not from the current shell directory.

## Manifest for a Tagged Release

```yaml
# my_plugin.yaml
name: my_plugin
repo: "https://github.com/your-org/cuvis-ai-my-plugin.git"
tag: "v0.1.0"
capabilities:
  - class_name: my_plugin.node.custom_node.CustomNode
```

Each `capabilities` entry needs at least `class_name` (a fully-qualified path); it may also carry palette metadata (`category`, `tags`, `icon_svg`, `input_specs`, `output_specs`, `doc_summary`). See [Plugin System Overview](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/reference/plugin-development/overview/index.md).

## Dependency resolution in composed child environments

When the orchestrated gRPC server runs a pipeline, it composes an isolated child environment from the declared plugin manifests (see [Cache and Isolation](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/reference/plugin-development/overview/#cache-and-isolation)). Dependency resolution in that environment follows a few rules worth knowing before you publish a plugin:

- **Plugins cannot influence resolver configuration.** The composer owns the child environment's `pyproject.toml`; a plugin contributes only its package as a requirement. Its declared dependencies and version floors still constrain what resolves, but it cannot add indexes or sources. The only manifest-level knob is `extras` on `kind: data_module` capabilities, which selects the pip extras installed for a run that uses that data module.
- **Torch mirrors the host.** As of cuvis-ai-core 0.12.1 the composed child environment mirrors the composing host's installed torch build: the exact `torch` / `torchvision` versions are pinned, and the matching PyTorch wheel index (`cpu`, `cuNNN`, `rocm`, or `xpu`) is emitted with `explicit = true`, so the child resolves the same accelerator build the host runs.
- **Host edge cases.** A host with no torch installed leaves children resolving transitive torch from PyPI (CPU wheels on Windows). A host torch whose local version tag is unrecognized, or mixed across `torch` and `torchvision`, gets its versions pinned without an index, so the child's resolution fails with a no-candidates error; fix the host environment in that case.
- **Floors above the host fail fast.** A plugin whose torch floor is above the host's installed torch fails composition outright. Keep torch floors as low as the plugin genuinely needs.
- **`[tool.uv.sources]` and `[[tool.uv.index]]` do not travel.** Those tables in a plugin repo's `pyproject.toml` apply only to that repo's own development venv, where the plugin is the resolution root. Installs as a git or registry dependency, including composed child environments, never read them.

## Verification

Use `uv` for local validation:

```bash
uv run pytest tests/ -q

# Dev-mode check: load the manifest directly and list the registered plugins
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r=NodeRegistry(); r.register_plugin('plugins.yaml'); print(r.list_plugins())"

# End-to-end: run a pipeline that declares `plugins: [my_plugin]`
uv run restore-pipeline --pipeline-path <pipeline>.yaml --plugins-dir <dir-with-manifest>
```

## Release Notes

- Tag releases with semver-style Git tags such as `v0.1.0`.
- Keep `capabilities` stable across patch releases unless you are intentionally making a breaking change.
- Test the tagged manifest before referencing it from this repo.

See [Plugin System Overview](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/reference/plugin-development/overview/index.md) for loader behavior and [Plugin Nodes](https://cubert-hyperspectral.github.io/cuvis-ai/0.13.3/catalogs/nodes/index.md) for end-user loading examples.
