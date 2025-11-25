"""Factory helpers for building CuvisCanvas pipelines used by the gRPC layer.

Pipelines are intentionally lightweight and avoid components that require
statistical pre-fitting (e.g., RXGlobal) so they can run immediately in tests
and basic inference flows. The builder creates three pipeline variants:

- channel_selector: minimal channel selection graph
- statistical: RX-style anomaly detection with per-batch statistics
- gradient: identical topology to statistical (gradient-capable later phases)
"""

from __future__ import annotations

from cuvis_ai.anomaly.rx_detector import RXPerBatch
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.selector import SoftChannelSelector, TopKIndices
from cuvis_ai.pipeline.canvas import CuvisCanvas


class CanvasBuilder:
    """Factory for constructing CuvisCanvas graphs for different pipeline types."""

    DEFAULT_CHANNELS = 61

    @staticmethod
    def create_pipeline(pipeline_type: str, config: dict | None = None) -> CuvisCanvas:
        """Create a pipeline by name.

        Args:
            pipeline_type: One of {"channel_selector", "statistical", "gradient"}.
            config: Optional configuration dictionary.

        Returns:
            Configured CuvisCanvas instance.

        Raises:
            ValueError: If the pipeline type is unknown.
        """
        config = config or {}
        pipeline_key = (pipeline_type or "").lower()

        if pipeline_key == "channel_selector":
            return CanvasBuilder._build_channel_selector(config)
        if pipeline_key == "statistical":
            return CanvasBuilder._build_statistical(config)
        if pipeline_key == "gradient":
            return CanvasBuilder._build_gradient(config)

        raise ValueError(f"Unknown pipeline type: {pipeline_type}")

    @staticmethod
    def _build_channel_selector(config: dict) -> CuvisCanvas:
        """Minimal channel selector graph."""
        channels = CanvasBuilder._resolve_input_channels(config)
        n_select = int(config.get("n_select", 3))

        canvas, data_node, normalizer, selector = CanvasBuilder._base_ingest_and_selector(
            config, channels, n_select
        )
        topk = TopKIndices(k=n_select, name="topk_indices")

        canvas.connect(
            (data_node.outputs.cube, normalizer.inputs.data),
            (normalizer.outputs.normalized, selector.inputs.data),
            (selector.outputs.weights, topk.inputs.weights),
        )

        return canvas

    @staticmethod
    def _build_statistical(config: dict) -> CuvisCanvas:
        """RX-style pipeline using per-batch statistics (no pre-fit required)."""
        channels = CanvasBuilder._resolve_input_channels(config)
        n_select = int(config.get("n_select", 3))

        canvas, data_node, normalizer, selector = CanvasBuilder._base_ingest_and_selector(
            config, channels, n_select
        )
        rx = RXPerBatch(eps=float(config.get("rx_eps", 1e-6)))
        logit_head = RXLogitHead(
            init_scale=float(config.get("rx_logit_scale", 1.0)),
            init_bias=float(config.get("rx_logit_bias", 0.0)),
        )
        decider = BinaryDecider(threshold=float(config.get("threshold", 0.5)))
        topk = TopKIndices(k=n_select, name="topk_indices")

        canvas.connect(
            (data_node.outputs.cube, normalizer.inputs.data),
            (normalizer.outputs.normalized, selector.inputs.data),
            (selector.outputs.selected, rx.inputs.data),
            (rx.outputs.scores, logit_head.inputs.scores),
            (logit_head.outputs.logits, decider.inputs.logits),
            (selector.outputs.weights, topk.inputs.weights),
        )

        return canvas

    @staticmethod
    def _build_gradient(config: dict) -> CuvisCanvas:
        """Gradient-capable pipeline skeleton (same topology as statistical for now)."""
        # Gradient pipeline mirrors statistical topology; later phases can swap in
        # trainable components or callback wiring without changing the factory shape.
        return CanvasBuilder._build_statistical(config)

    @staticmethod
    def _base_ingest_and_selector(
        config: dict, channels: int, n_select: int
    ) -> tuple[CuvisCanvas, LentilsAnomalyDataNode, MinMaxNormalizer, SoftChannelSelector]:
        """Create shared ingest + selector stack."""
        normal_class_ids = config.get("normal_class_ids", [0])

        canvas = CuvisCanvas("channel_selector_pipeline")
        data_node = LentilsAnomalyDataNode(
            normal_class_ids=list(normal_class_ids),
            wavelengths=None,
            name="ingest",
        )
        normalizer = MinMaxNormalizer(
            eps=float(config.get("normalizer_eps", 1e-6)),
            use_running_stats=bool(config.get("use_running_stats", False)),
            name="normalizer",
        )
        selector = SoftChannelSelector(
            n_select=n_select,
            input_channels=channels,
            init_method=str(config.get("init_method", "uniform")),
            temperature_init=float(config.get("temperature_init", 5.0)),
            temperature_min=float(config.get("temperature_min", 0.1)),
            temperature_decay=float(config.get("temperature_decay", 0.9)),
            hard=bool(config.get("hard", False)),
            name="selector",
        )

        return canvas, data_node, normalizer, selector

    @staticmethod
    def _resolve_input_channels(config: dict) -> int:
        """Resolve input channel count from config with sensible defaults."""
        for key in ("input_channels", "num_channels", "channels"):
            if key in config and config[key] is not None:
                try:
                    return int(config[key])
                except (TypeError, ValueError):
                    pass
        return CanvasBuilder.DEFAULT_CHANNELS


__all__ = ["CanvasBuilder"]
