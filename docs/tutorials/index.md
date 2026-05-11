# Tutorials

Notebook-shaped walk-throughs that teach a concept by working through
a runnable example. Each tutorial below links to a script in the
[cuvis-ai-cookbook](https://github.com/cubert-hyperspectral/cuvis-ai-cookbook)
repository — clone it alongside cuvis-ai and run the examples directly.

Tutorials are grouped by training style:

- **Statistical** — pipelines that learn from background statistics alone, no gradient steps. Fast to train, interpretable, strong baselines.
- **Gradient** — pipelines that include trainable parameters fit by backpropagation. More expressive, more compute.

## Statistical

<div class="grid cards" markdown>

-   :material-chart-line: **[RX Anomaly Detection](statistical/rx-anomaly.md)**

    ---

    Classical Mahalanobis-distance anomaly detector. Start here for hyperspectral anomaly fundamentals.

-   :material-select-multiple: **[Channel Selection](statistical/channel-selector.md)**

    ---

    Two-phase training that learns which wavelengths to keep from a hyperspectral cube.

-   :material-water: **[Blood Perfusion](statistical/blood-perfusion.md)**

    ---

    NDVI-style two-band differential rendering blood perfusion as a false-RGB overlay.

</div>

## Gradient

<div class="grid cards" markdown>

-   :material-brain: **[Deep SVDD](gradient/deep-svdd.md)**

    ---

    One-class anomaly detection that learns a compact representation of "normal" data.

-   :material-link-variant: **[AdaCLIP](gradient/adaclip.md)**

    ---

    Vision-language anomaly detection coupling a frozen CLIP backbone with a trainable adapter.

</div>

## Related

- [Concepts → Training](../concepts/training.md) — the two-phase training model behind every cuvis-ai pipeline.
- [Workflows](../workflows/index.md) — task-recipe "I want to…" guides once you know what you're doing.
- [Datasets catalog](../catalogs/datasets/index.md) — the demo datasets each tutorial runs against.
