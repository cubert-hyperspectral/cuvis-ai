# About Skills

A skill is a piece of structured guidance — name, description,
trigger keywords, and a body of instructions — that a coding assistant
reads when it detects a matching intention in the user's prompt.
`cuvis-ai-agentic-skills` ships eight such skills for cuvis.AI.

Each skill follows the same shape:

- **A description** — when the skill triggers, what it produces, what it doesn't do.
- **A deliverable contract** — a YAML, a node + test file, a deployment-ready transformation. Skills produce artifacts, not free-text answers.
- **Cross-references to other skills** — pipeline authoring hands off to inference; node design hands off to plugin packaging; everything defers runtime setup to `cuvis-ai-env`.

## How skills compose

The eight skills form a directed graph of hand-offs:

```
cuvis-ai-env  (runtime; underpins everything)
   │
   ├─ cuvis-ai-node     ──► cuvis-ai-plugin
   │
   ├─ cuvis-ai-pipeline ──► cuvis-ai-pipeline-visualize
   │                    └─► cuvis-ai-inference ──► cuvis-ai-pipeline-to-cuvisnext
   │
   └─ cuvis-ai-training ──► cuvis-ai-inference
```

- `cuvis-ai-pipeline` emits a pipeline YAML, then auto-invokes
  `cuvis-ai-pipeline-visualize` so you lay eyes on the diagram before
  reviewing the YAML.
- `cuvis-ai-inference` runs the pipeline on cu3s data; Path 2 of
  inference hands off to `cuvis-ai-pipeline-to-cuvisnext` when the
  target is cuvis.next.
- `cuvis-ai-training` builds on a wired pipeline and produces a
  `trainrun.yaml` plus orchestration script.
- Every skill that runs Python defers runtime setup to `cuvis-ai-env`
  — it reads `./.cuvis-ai-env.json` before any `uv run` command and
  invokes `cuvis-ai-env` if the file is absent or invalid.

## Trigger model

Skills auto-trigger when their keywords appear in the prompt — no
explicit invocation required. For example, "build me an RX pipeline
that runs on a single cu3s and outputs an anomaly mp4" triggers
`cuvis-ai-pipeline`, which then chain-triggers
`cuvis-ai-pipeline-visualize` and (if you ask to execute)
`cuvis-ai-inference`.

If you want to invoke a skill explicitly, see [Invoking Skills](invoking.md).

## Distribution

The eight skills ship as a single Claude Code plugin bundle at
[`cubert-hyperspectral/cuvis-ai-agentic-skills`](https://github.com/cubert-hyperspectral/cuvis-ai-agentic-skills).
Install once with `/plugin marketplace add` + `/plugin install`; the
skills become available to every Claude Code surface (CLI, IDE
extensions, Desktop).
