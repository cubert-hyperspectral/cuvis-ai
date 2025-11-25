import pytest
import torch

from cuvis_ai.anomaly.rx_detector import RXPerBatch
from cuvis_ai.anomaly.rx_logit_head import RXLogitHead
from cuvis_ai.deciders.binary_decider import BinaryDecider
from cuvis_ai.grpc.canvas_builder import CanvasBuilder
from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.selector import SoftChannelSelector, TopKIndices
from cuvis_ai.pipeline.canvas import CuvisCanvas


def _node_map(canvas: CuvisCanvas) -> dict[str, object]:
    return {node.name: node for node in canvas.nodes}


class TestCanvasBuilder:
    def test_create_channel_selector_pipeline(self):
        canvas = CanvasBuilder.create_pipeline(
            "channel_selector", {"n_select": 4, "input_channels": 6}
        )

        assert isinstance(canvas, CuvisCanvas)
        nodes = _node_map(canvas)
        assert any(isinstance(n, SoftChannelSelector) for n in nodes.values())
        assert any(isinstance(n, MinMaxNormalizer) for n in nodes.values())
        assert any(isinstance(n, TopKIndices) for n in nodes.values())

        selector = next(n for n in nodes.values() if isinstance(n, SoftChannelSelector))
        cube = torch.randn(2, 3, 3, selector.input_channels)
        outputs = canvas.forward(batch={"cube": cube})

        assert (selector.name, "selected") in outputs

    def test_create_statistical_pipeline(self):
        canvas = CanvasBuilder.create_pipeline("statistical", {"input_channels": 8})
        nodes = _node_map(canvas)

        assert any(isinstance(n, RXPerBatch) for n in nodes.values())
        assert any(isinstance(n, RXLogitHead) for n in nodes.values())
        assert any(isinstance(n, BinaryDecider) for n in nodes.values())

        selector = next(n for n in nodes.values() if isinstance(n, SoftChannelSelector))
        cube = torch.randn(1, 2, 2, selector.input_channels)
        outputs = canvas.forward(batch={"cube": cube})

        decider = next(n for n in nodes.values() if isinstance(n, BinaryDecider))
        assert outputs[(decider.name, "decisions")].dtype == torch.bool

    def test_create_gradient_pipeline_aliases_statistical(self):
        canvas = CanvasBuilder.create_pipeline("gradient", {"input_channels": 4})
        assert isinstance(canvas, CuvisCanvas)
        nodes = _node_map(canvas)
        assert any(isinstance(n, RXPerBatch) for n in nodes.values())

    def test_invalid_pipeline_type(self):
        with pytest.raises(ValueError):
            CanvasBuilder.create_pipeline("unknown", {})

    def test_channel_selector_respects_config(self):
        canvas = CanvasBuilder.create_pipeline("channel_selector", {"n_select": 2, "channels": 5})
        selector = next(n for n in _node_map(canvas).values() if isinstance(n, SoftChannelSelector))
        assert selector.n_select == 2
        assert selector.input_channels == 5
