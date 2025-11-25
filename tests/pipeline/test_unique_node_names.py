"""Tests for counter-based node naming with insertion-order suffixes."""

import pytest

from cuvis_ai.node.normalization import MinMaxNormalizer
from cuvis_ai.node.selector import SoftChannelSelector
from cuvis_ai.pipeline.canvas import CuvisCanvas


class TestCounterBasedNames:
    """Test that Canvas assigns counter-based names for duplicates."""

    def test_counter_based_names(self) -> None:
        canvas = CuvisCanvas("test")

        node1 = MinMaxNormalizer(name="normalizer")
        node2 = MinMaxNormalizer(name="normalizer")
        node3 = MinMaxNormalizer(name="normalizer")

        selector = SoftChannelSelector(n_select=3, input_channels=10, name="selector")

        # Before canvas addition
        assert node1._canvas_counter is None
        assert node1.name == "normalizer"

        # Add first node
        canvas.connect(node1.outputs.normalized, selector.inputs.data)
        assert node1._canvas_counter == 0
        assert node1.name == "normalizer"

        # Add second node
        canvas.connect(node2.outputs.normalized, selector.inputs.data)
        assert node2._canvas_counter == 1
        assert node2.name == "normalizer-1"

        # Add third node
        canvas.connect(node3.outputs.normalized, selector.inputs.data)
        assert node3._canvas_counter == 2
        assert node3.name == "normalizer-2"

    def test_unique_names_get_counter_zero(self) -> None:
        canvas = CuvisCanvas("test")

        node1 = MinMaxNormalizer(name="normalizer_1")
        node2 = MinMaxNormalizer(name="normalizer_2")
        selector = SoftChannelSelector(n_select=3, input_channels=10)

        canvas.connect(node1.outputs.normalized, selector.inputs.data)
        canvas.connect(node2.outputs.normalized, selector.inputs.data)

        # All unique names get counter=0 (no suffix)
        assert node1._canvas_counter == 0
        assert node2._canvas_counter == 0
        assert selector._canvas_counter == 0
        assert node1.name == "normalizer_1"
        assert node2.name == "normalizer_2"

    def test_name_immutable_after_canvas_addition(self) -> None:
        canvas = CuvisCanvas("test")
        node = MinMaxNormalizer(name="normalizer")
        selector = SoftChannelSelector(n_select=3, input_channels=10)

        # Before canvas: cannot set name
        with pytest.raises(AttributeError):
            node.name = "other"

        canvas.connect(node.outputs.normalized, selector.inputs.data)
        with pytest.raises(AttributeError):
            node.name = "other"

    def test_no_id_property(self) -> None:
        node = MinMaxNormalizer(name="normalizer")

        with pytest.raises(AttributeError):
            _ = node.id

    def test_reproducible_names(self) -> None:
        def create_graph() -> list[str]:
            canvas = CuvisCanvas("test")
            nodes = [MinMaxNormalizer(name="norm") for _ in range(3)]
            selector = SoftChannelSelector(n_select=3, input_channels=10)
            for node in nodes:
                canvas.connect(node.outputs.normalized, selector.inputs.data)
            return [n.name for n in nodes]

        names1 = create_graph()
        names2 = create_graph()

        assert names1 == names2
        assert names1 == ["norm", "norm-1", "norm-2"]
