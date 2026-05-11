# Nodes Catalog

A browsable inventory of every node available in cuvis-ai pipelines.
Built-in nodes ship with `cuvis-ai-core`; external nodes are loaded
from plugin manifests at runtime — see
[Plugin Development](../../reference/plugin-development/overview.md)
for how plugins are packaged.

## Categories

<div class="grid cards" markdown>

-   **[Data Nodes](data-nodes.md)**

    ---

    Data entrypoints, JSON readers, NumPy readers, and video frame sources.

-   **[Preprocessing](preprocessing.md)**

    ---

    Normalization, ROI crop, rotation, and occlusion transforms.

-   **[Selectors](selectors.md)**

    ---

    Channel selectors and band reducers (statistical and learnable).

-   **[Statistical](statistical.md)**

    ---

    Statistical anomaly detectors (RX and friends), background-stats accumulators.

-   **[Loss & Metrics](loss-metrics.md)**

    ---

    Loss functions and evaluation metrics. Many run in the metric execution stage.

-   **[Visualization](visualization.md)**

    ---

    Render anomaly heatmaps, false-RGB compositions, and overlay tiles.

-   **[Output](output.md)**

    ---

    Video encoders, COCO writers, NumPy dumpers — the sinks at the end of a pipeline.

-   **[Utility](utility.md)**

    ---

    Mask ops, port adapters, prompt schedulers, and other glue.

-   **[External Nodes](external.md)**

    ---

    Nodes published by plugins (AdaCLIP, Ultralytics, DeepEIoU, TrackEval, SAM3). Installable via plugin manifests.

</div>
