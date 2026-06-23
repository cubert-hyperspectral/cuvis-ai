# Cuvis.AI Documentation

Cuvis.AI is an open, extensible framework for computer vision, designed for AI workloads at video rate with full spatial and spectral fidelity.
It is developed at [Cubert GmbH](https://cubert-hyperspectral.com/de/), the leader in video spectroscopy and snapshot hyperspectral imaging.

Cuvis.AI works by reduction: every pipeline is assembled from reusable, atomic nodes. Process data, train models, interpret results, and deploy to production, all from the same modular toolkit. Extend it with custom plugins or integrate it into existing systems.

One framework, from sensor to shipped application.


## Features

- **Typed I/O system** — port-based connections with type safety and validation.
- **Statistical initialization** — bootstrap models with non-parametric methods (RX detector, PCA).
- **Gradient-based training** — fine-tune with PyTorch Lightning.
- **Composable node architecture** — preprocessing, feature extraction, and decision modules.
- **Monitoring** — TensorBoard out of the box, extensible to other frameworks.
- **Configuration management** — Hydra-based with CLI overrides.

## Start here

<div class="grid cards" markdown>

-   :material-rocket-launch: **[Get Started](get-started/installation.md)**

    ---

    Install cuvis-ai and run your [first pipeline](get-started/first-pipeline.md) in 5 minutes.

-   :material-school: **[Tutorials](tutorials/index.md)**

    ---

    Notebook-shaped walk-throughs for RX, Channel Selection, Deep SVDD, AdaCLIP, and Blood Perfusion.

-   :material-book-open-page-variant: **[Concepts](concepts/index.md)**

    ---

    The mental model behind cuvis-ai: nodes, ports, pipelines, execution stages, and two-phase training.

</div>

## Browse and do

- **[Catalogs](catalogs/nodes/index.md)** — every built-in and external node, plus the public HuggingFace demo datasets.
- **[Workflows](workflows/index.md)** — task-recipe "I want to…" guides (build, train, run, monitor, profile).
- **[Deployment](deployment/index.md)** — remote and service integration; gRPC client patterns, deployment guide, sequence diagrams.
- **[Reference](reference/configuration/index.md)** — Hydra configuration, Python API, plugin development, and contributing guides.

## Document types

Every page in the docs belongs to one of four types:

| Type | Answers | Voice |
|---|---|---|
| **Tutorial** | "Teach me how to do this." | Step-by-step, narrative. |
| **Workflow** | "I want to do X." | Task-recipe, terse. |
| **Catalog** | "Show me what already exists." | Browsable inventory. |
| **Reference** | "Give me the exact spec." | Exhaustive, dry. |

!!! tip "New to cuvis-ai?"
    Follow the path: [Installation](get-started/installation.md) → [Your First Pipeline](get-started/first-pipeline.md) → [Concepts](concepts/index.md) → [Tutorials](tutorials/index.md).

---

Apache License 2.0 — see [LICENSE](https://github.com/cubert-hyperspectral/cuvis-ai/blob/main/LICENSE).
Report issues at [github.com/cubert-hyperspectral/cuvis-ai/issues](https://github.com/cubert-hyperspectral/cuvis-ai/issues).
