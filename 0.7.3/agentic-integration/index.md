# Agentic Integration

`cuvis-ai-agentic-skills` lets assistants and automation tools interact with cuvis.AI through structured skills.

Use it to:

- design hyperspectral pipelines from a plain-English request,
- inspect, visualise, and execute pipelines,
- run statistical or gradient training runs,
- author and package custom nodes and plugins,
- convert pipelines for cuvis.next deployment.

The skills run inside [Claude Code](https://code.claude.com/) and are distributed as a single plugin bundle from [`cubert-hyperspectral/cuvis-ai-agentic-skills`](https://github.com/cubert-hyperspectral/cuvis-ai-agentic-skills).

## Get started

- **[About Skills](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/agentic-integration/overview/index.md)**

  ______________________________________________________________________

  What skills are, how they hand off to one another, and how `cuvis-ai-env` underpins the rest.

- **[Available Skills](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/agentic-integration/skills/index.md)**

  ______________________________________________________________________

  Catalog of the eight skills: triggers, deliverables, and cross-references.

- **[Invoking Skills](https://cubert-hyperspectral.github.io/cuvis-ai/0.7.3/agentic-integration/invoking/index.md)**

  ______________________________________________________________________

  Install the plugin bundle in Claude Code and trigger skills from a prompt.

## Why a separate integration layer

Pipeline authoring, training, and deployment have a lot of moving parts — node specs, port wiring, execution stages, plugin manifests, trainrun YAMLs, CLI invocations. The agentic-skills bundle turns those moving parts into intentions ("build me an X pipeline that does Y") and emits the artifacts directly: a pipeline YAML, a training config, or a deployment-ready transformation. Skills compose: pipeline authoring hands off to visualisation; visualisation hands off to execution; execution hands off to cuvis.next conversion. Every skill defers runtime setup to `cuvis-ai-env`.
