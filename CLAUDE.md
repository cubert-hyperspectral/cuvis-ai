# cuvis-ai

The **node/operator library** of the Cuvis.AI ecosystem, built on `cuvis-ai-core`. Ships the
concrete processing nodes (channel selectors/mixers, false-RGB, normalization, anomaly
detectors, spectral angle mapper, conversion/decision nodes, mask/compositing ops, video and
JSON writers, visualization helpers) plus ready-to-run pipeline and plugin configs. End-to-end
examples — Jupyter notebooks pairing data, explanation, code, and results — live in `cuvis-ai-cookbook`.

## Part of the Cuvis.AI ecosystem

`cuvis-ai-schemas` (contracts) → `cuvis-ai-core` (framework) → **`cuvis-ai`** (this repo:
nodes + CLIs) → plugins. `cuvis-ai-cookbook` = example notebooks (datasets on Hugging Face:
<https://huggingface.co/cubert-gmbh>); `cuvis-ai-agentic-skills` = private/internal Claude Code
plugin (local checkout only, not published or referenced from public docs);
`dev-docs` = internal ticket docs.

## Layout

- `cuvis_ai/node/` — the node library (one module per family; `anomaly/`, `deciders/` subdirs).
- `cuvis_ai/configs/` — `pipeline/`, `plugins/`, `training/`, `trainrun/`, `data/` YAML manifests
  (packaged with the library so `CONFIG_ROOT` resolves on a pip install, not just a source checkout).
- `tests/` — pytest suite; `tools/` — helper scripts.
- `scripts/` — top-level CLIs (stub generator, pipeline renderers). PEP 420 **namespace package**
  (no `__init__.py`) so it merges with `cuvis-ai-core`'s `scripts/`; register CLIs as `scripts.<mod>:main`.

## Build & test

- Install: `uv sync` (use `uv`, never bare `pip`).
- Tests: `uv run pytest`. Nodes are tested with **pure-tensor mocking** — no heavy model downloads.
- Docs: `uv run mkdocs build --strict` (Material + gen-files node catalog + llmstxt; needs the `docs` extra).
- CLI entry points: `restore-pipeline`, `restore-trainrun`, `dataset`, `create-stubs`.
- Enable hooks once: `git config core.hooksPath .githooks`.
  - **pre-commit**: strips notebook video outputs, `ruff format`, `ruff check --fix`, re-stages.
  - **pre-push**: `uv sync --all-extras` → notebook-video check → `ruff format` → `ruff check --fix`
    → `interrogate cuvis_ai/ --fail-under 95` (docstring coverage) → `pytest -m "not slow and not gpu"`.
- **Pre-push gotcha:** the `uv sync --all-extras` step uninstalls editable deps not listed in
  `pyproject.toml` (e.g. `cuvis-ai-ui`, local experiment packages). Push from a clean venv, or
  re-install editables afterward.

## Plugins

- Each external plugin has a bare manifest at `cuvis_ai/configs/plugins/<name>.yaml` (one file = one plugin):
  an explicit `name:` (never derived from the filename), a source (`path:` or `repo:` + `tag:`), and a
  `capabilities:` list of `class_name` entries (optional `category` / `tags` / `icon_svg` / port specs;
  a `kind: data_module` entry instead carries `data_module_name` + `extras`).
  `cuvis_ai/configs/plugins/cuvis_ai_builtin.yaml` exposes this repo's own nodes the same way.
- Pipelines reference plugins by **bare name only** — a top-level `plugins:` list (e.g.
  `- cuvis_ai_builtin`, `- sam3`). Each name resolves to a manifest in the plugins directory; there
  are no inline or catalog refs.
- Load a pipeline against a plugins directory with `--plugins-dir` (`--plugins-path` was removed).
- Dependency floors are pinned in `pyproject.toml`; the `dep_compat` / `registry_compat` CI
  workflows keep those floors compatible with the published plugin release tags — bump with care.

## Data layer

- cuvis-ai pins **no** `cuvis` SDK. cu3s/cu3 I/O (and `tiff_paired`) live in the
  `cuvis-ai-dataloader` plugin (`[cu3s]` extra); its manifest ships at
  `cuvis_ai/configs/plugins/cuvis_ai_dataloader.yaml`.
- `restore-pipeline` picks data with `--data-module <name>` + repeatable `--data-arg key=value`
  (no `--cu3s-file-path` / `--processing-mode`). It runs **in-process**, so the env must have the
  data plugin installed for a cu3s run; the gRPC orchestrator composes a child env per run instead.
- Trainrun `data:` is `{data_module, splits, params}` with a **selector** split model
  (`{kind: file_indices, source, ids}`), not flat `train_ids`. `SingleCu3sDataset` /
  `SingleCu3sDataModule` are gone (now `Cu3sDataModule` in `cuvis_ai_dataloader`).

## Key patterns for nodes

- Inherit `cuvis_ai_core.node.Node`; define `INPUT_SPECS`/`OUTPUT_SPECS` as class dicts of
  `PortSpec` (`shape=(-1, ...)` for dynamic dims; mark fan-in ports `variadic`). Call
  `super().__init__(**params, **kwargs)` last.
- Register a plugin/package via `NodeRegistry().auto_register_package("pkg.node")` in the
  package `__init__.py`.

## Conventions

- ruff line length **100**.
- New node classes omit the `Node` suffix (e.g. `SpectralAngleMapper`).
- Anomaly nodes expose `scores: [B,H,W,1]` + `anomaly_score: [B]` — never `anomaly_map` / `image_score`.
- No Jira IDs / "Phase N" / migration tags in shipped code, comments, or docstrings.
- No Claude/AI mentions or `Co-Authored-By` trailers in commit messages.
- See `CONTRIBUTING.md` for the full contributor workflow.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **cuvis-ai** (8351 symbols, 13296 relationships, 164 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/cuvis-ai/context` | Codebase overview, check index freshness |
| `gitnexus://repo/cuvis-ai/clusters` | All functional areas |
| `gitnexus://repo/cuvis-ai/processes` | All execution flows |
| `gitnexus://repo/cuvis-ai/process/{name}` | Step-by-step execution trace |

## Cross-Repo Groups

This repository is listed under GitNexus **group(s): cuvis-ai-group** (see `~/.gitnexus/groups/`). For cross-repo analysis, use MCP tools `impact`, `query`, and `context` with `repo` set to `@<groupName>` or `@<groupName>/<memberPath>` (paths match keys in that group’s `group.yaml`). Use `group_list` / `group_sync` for membership and sync. From the terminal: `npx gitnexus group list`, `npx gitnexus group sync <name>`, `npx gitnexus group impact <name> --target <symbol> --repo <group-path>`.

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
