# Invoking Skills

The skills run inside [Claude Code](https://code.claude.com/). Install the plugin bundle once, then trigger skills with plain-English prompts.

## 1. Install Claude Code

If `claude --version` fails in your terminal, install Claude Code first via the [Claude Code quickstart](https://code.claude.com/docs/en/quickstart). The plugin-management slash commands below are only exposed in the standalone CLI session — IDE-extension chat panels (VSCode, JetBrains) do not accept them. Once installed via the CLI, the bundle is loaded by every surface, IDE extensions included.

## 2. Install the cuvis-ai skill bundle

Open a `claude` session and run these slash commands:

```text
/plugin marketplace add cubert-hyperspectral/cuvis-ai-agentic-skills
/plugin install cuvis-ai-agentic-skills@cuvis-ai-agentic-skills
/reload-plugins
```

The eight skills register and become available to subsequent prompts.

## 3. Trigger a skill

Skills auto-trigger on keywords. A few examples:

| Prompt                                                                      | Skill that fires                                                                                                    | Result                                                  |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------- |
| "Set up cuvis-ai on this machine."                                          | `cuvis-ai-env`                                                                                                      | A verified env plus `.cuvis-ai-env.json` recording it.  |
| "Build me a pipeline that runs RX on cu3s data and exports an anomaly MP4." | `cuvis-ai-pipeline` → auto-chains to `cuvis-ai-pipeline-visualize`, then `cuvis-ai-inference` if you ask to run it. | A pipeline YAML, a diagram next to it, and the MP4.     |
| "Visualize this pipeline yaml."                                             | `cuvis-ai-pipeline-visualize`                                                                                       | A `<pipeline>.png` (or Mermaid `.md`) next to the YAML. |
| "Run `rx_statistical.yaml` on `Demo_000.cu3s`."                             | `cuvis-ai-inference`                                                                                                | The artifact set on disk plus the exact command used.   |
| "Design a two-phase training run for this pipeline."                        | `cuvis-ai-training`                                                                                                 | A `trainrun.yaml` plus orchestration script.            |
| "Package this fork of YOLO as a cuvis-ai plugin."                           | `cuvis-ai-plugin` (delegates to `cuvis-ai-node`)                                                                    | A pip-installable plugin package + manifest entry.      |
| "Make this pipeline run in cuvis.next, switch to video mode."               | `cuvis-ai-pipeline-to-cuvisnext`                                                                                    | A `<pipeline>_cuvisnext_video.yaml`.                    |

You can also invoke a skill explicitly with a `/<skill-name>` slash command in the Claude Code prompt — for example, `/cuvis-ai-pipeline`.

## Skill state and configuration

- **Runtime config** — `.cuvis-ai-env.json` lives in the working directory and pins the env path + plugin manifest. `cuvis-ai-env` writes it once; the other skills read it before any `uv run` invocation. If you change Python environments, re-run `cuvis-ai-env`.
- **Plugin manifest** — manifests follow the [Plugin Development](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/reference/plugin-development/overview/index.md) format. `cuvis-ai-pipeline` and `cuvis-ai-inference` honour whatever manifest `cuvis-ai-env` recorded.
- **Working directory** — every skill operates relative to the directory you invoked Claude Code from. Run from your project root, not from inside `docs/` or `examples/`.

## Troubleshooting

| Symptom                                       | Likely cause                       | Fix                                                                                                                                                                                         |
| --------------------------------------------- | ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Slash command "not found" in IDE              | Plugin slash commands are CLI-only | Run the install from a standalone `claude` session, then return to the IDE.                                                                                                                 |
| Skill doesn't trigger on a prompt             | Keywords didn't match              | Invoke explicitly with `/<skill-name>` or include the trigger keyword from the [skills catalog](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/agentic-integration/skills/index.md). |
| `command not found: restore-pipeline` mid-run | Env missing or stale               | The running skill auto-invokes `cuvis-ai-env`. Wait for the env setup to complete and re-prompt.                                                                                            |
