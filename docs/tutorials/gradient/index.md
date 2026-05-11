# Gradient Tutorials

Pipelines that include trainable parameters fit by backpropagation.
They consume a statistically-initialised pipeline (Phase 1) and refine
it through gradient descent (Phase 2). More expressive than purely
statistical pipelines, at the cost of training compute.

## In this section

- **[Deep SVDD](deep-svdd.md)** — one-class anomaly detection that learns a compact representation of "normal" data. The gradient-trained sibling of [RX](../statistical/rx-anomaly.md).
- **[AdaCLIP](adaclip.md)** — vision-language anomaly detection coupling a frozen CLIP backbone with a small trainable adapter. Three recipes: PCA-reduced baseline, Concrete channel selector, and DRCNN reducer.

## See also

- [Concepts → Training](../../concepts/training.md) — the two-phase training model.
- [Statistical tutorials](../statistical/index.md) — strong baselines, no gradient steps.
