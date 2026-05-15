# Configuration

Cuvis.AI uses Hydra for configuration management — composition,
overrides, and config groups. This section is the reference for the
Hydra layout and the schemas the trainers consume.

If you are looking for the directory layout of a typical
`configs/` tree, see [Config Groups](config-groups.md).

## Configuration Documentation

<div class="grid cards" markdown>

-   :material-file-tree: **[Config Groups](config-groups.md)**

    ---

    Organize configurations using Hydra config groups

-   :material-train: **[TrainRun Schema](trainrun-schema.md)**

    ---

    Complete schema for training run configurations

-   :material-layers: **[Hydra Composition](hydra-basics.md)**

    ---

    Advanced composition and override patterns

</div>

## See also

- [Build Pipeline (YAML)](../../workflows/build-pipeline-yaml.md) — author a pipeline declaratively.
- [Statistical Training](../../workflows/statistical-training.md) and [Gradient Training](../../workflows/gradient-training.md) — the trainers that consume these configs.
