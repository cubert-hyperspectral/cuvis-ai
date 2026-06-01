# cuvis-ai

The **node/operator library** of the Cuvis.AI ecosystem, built on `cuvis-ai-core`. Ships the
concrete processing nodes (channel selectors/mixers, false-RGB, normalization, anomaly
detectors, spectral angle mapper, conversion/decision nodes, mask/compositing ops, video and
JSON writers, visualization helpers) plus ready-to-run pipeline and plugin configs and examples.

## Part of the Cuvis.AI ecosystem

`cuvis-ai-schemas` (contracts) → `cuvis-ai-core` (framework) → **`cuvis-ai`** (this repo:
nodes + CLIs) → plugins. `cuvis-ai-cookbook` = examples; `cuvis-ai-agentic-skills` = Claude
Code plugin; `dev-docs` = internal ticket docs.

## Layout

- `cuvis_ai/node/` — the node library (one module per family; `anomaly/`, `deciders/` subdirs).
- `configs/` — `pipeline/`, `plugins/`, `training/`, `trainrun/`, `data/` YAML manifests.
- `examples/` — runnable end-to-end pipeline scripts.
- `tests/` — pytest suite; `tools/` — helper scripts; `scripts/` — e.g. node-port stub generator.

## Build & test

- Install: `uv sync` (use `uv`, never bare `pip`).
- Tests: `uv run pytest`. Nodes are tested with **pure-tensor mocking** — no heavy model downloads.
- CLI entry points: `restore-pipeline`, `restore-trainrun`, `dataset`, `create-stubs`.

## Key patterns for nodes

- Inherit `cuvis_ai_core.node.Node`; define `INPUT_SPECS`/`OUTPUT_SPECS` as class dicts of
  `PortSpec` (`shape=(-1, ...)` for dynamic dims). Call `super().__init__(**params, **kwargs)` last.
- Register a plugin/package via `NodeRegistry().auto_register_package("pkg.node")` in the
  package `__init__.py`; plugin manifests live at `configs/plugins/<name>.yaml`.

## Conventions

- ruff line length **100**. `.githooks` pre-commit runs `ruff format` + `ruff check --fix`
  (verify it is enabled: `git config core.hooksPath .githooks`).
- New node classes omit the `Node` suffix (e.g. `SpectralAngleMapper`).
- Anomaly nodes expose `scores: [B,H,W,1]` + `anomaly_score: [B]` — never `anomaly_map` / `image_score`.
- No Jira IDs / "Phase N" / migration tags in shipped code, comments, or docstrings.
- No Claude/AI mentions or `Co-Authored-By` trailers in commit messages.
- See `CONTRIBUTING.md` for the full contributor workflow.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **cuvis-ai** (6470 symbols, 10182 relationships, 137 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

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
