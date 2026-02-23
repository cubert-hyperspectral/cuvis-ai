"""Tests for TRAINABLE_BUFFERS-based freeze/unfreeze round-trips."""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn
from cuvis_ai_core.node import Node

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# 1. __init_subclass__ validation tests
# ---------------------------------------------------------------------------


def test_trainable_buffers_rejects_list() -> None:
    """TRAINABLE_BUFFERS must be a tuple, not a list."""
    with pytest.raises(TypeError, match="must be a tuple of strings"):

        class _Bad(Node):
            TRAINABLE_BUFFERS = ["a", "b"]  # type: ignore[assignment]

            def forward(self, **kw): ...


def test_trainable_buffers_rejects_bare_string() -> None:
    """A bare string is iterable but not a tuple of strings."""
    with pytest.raises(TypeError, match="must be a tuple of strings"):

        class _Bad(Node):
            TRAINABLE_BUFFERS = "oops"  # type: ignore[assignment]

            def forward(self, **kw): ...


def test_trainable_buffers_rejects_non_string_entries() -> None:
    """Entries inside the tuple must all be strings."""
    with pytest.raises(TypeError, match="must be a tuple of strings"):

        class _Bad(Node):
            TRAINABLE_BUFFERS = ("a", 42)  # type: ignore[assignment]

            def forward(self, **kw): ...


def test_trainable_buffers_accepts_empty_tuple() -> None:
    """Empty tuple is the default and should be accepted."""

    class _OK(Node):
        TRAINABLE_BUFFERS = ()

        def forward(self, **kw):
            return {}

    assert _OK.TRAINABLE_BUFFERS == ()


def test_trainable_buffers_accepts_valid_tuple() -> None:
    """Valid tuple of strings should pass validation."""

    class _OK(Node):
        TRAINABLE_BUFFERS = ("a", "b")

        def forward(self, **kw):
            return {}

    assert _OK.TRAINABLE_BUFFERS == ("a", "b")


# ---------------------------------------------------------------------------
# 2. Runtime validation — unfreeze raises if buffer not registered
# ---------------------------------------------------------------------------


def test_unfreeze_raises_for_missing_buffer() -> None:
    """unfreeze() should raise AttributeError if declared name is not registered."""

    class _Bad(Node):
        TRAINABLE_BUFFERS = ("nonexistent",)

        def __init__(self):
            super().__init__()

        def forward(self, **kw):
            return {}

    node = _Bad()
    with pytest.raises(AttributeError, match="nonexistent"):
        node.unfreeze()


def test_freeze_raises_for_missing_buffer() -> None:
    """freeze() should raise AttributeError if declared name is not registered."""

    class _Bad(Node):
        TRAINABLE_BUFFERS = ("nonexistent",)

        def __init__(self):
            super().__init__()

        def forward(self, **kw):
            return {}

    node = _Bad()
    with pytest.raises(AttributeError, match="nonexistent"):
        node.freeze()


# ---------------------------------------------------------------------------
# 3. Round-trip tests for the 6 affected nodes
# ---------------------------------------------------------------------------

_NODE_CONFIGS: list[tuple[str, dict]] = [
    ("cuvis_ai.node.channel_selector:SoftChannelSelector", {"n_select": 3, "input_channels": 10}),
    (
        "cuvis_ai.node.dimensionality_reduction:TrainablePCA",
        {"num_channels": 10, "n_components": 3},
    ),
    ("cuvis_ai.node.conversion:ScoreToLogit", {"init_scale": 1.0, "init_bias": 0.0}),
    ("cuvis_ai.anomaly.lad_detector:LADGlobal", {"num_channels": 10}),
    ("cuvis_ai.anomaly.rx_detector:RXGlobal", {"num_channels": 10}),
    (
        "cuvis_ai.node.channel_mixer:LearnableChannelMixer",
        {"input_channels": 10, "output_channels": 3},
    ),
]


def _import_node(spec: str):
    """Import a node class from 'module:ClassName' spec."""
    module_path, class_name = spec.rsplit(":", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _make_node(spec: str, kwargs: dict):
    """Construct a node from spec and kwargs."""
    cls = _import_node(spec)
    return cls(**kwargs)


@pytest.fixture(params=_NODE_CONFIGS, ids=[s.split(":")[1] for s in [c[0] for c in _NODE_CONFIGS]])
def node_and_spec(request):
    """Parametrized fixture yielding (node_instance, class_spec)."""
    spec, kwargs = request.param
    node = _make_node(spec, kwargs)
    return node, spec


def _get_trainable_names(node: nn.Module) -> tuple[str, ...]:
    """Get TRAINABLE_BUFFERS from a node, or infer from custom freeze/unfreeze."""
    return getattr(node, "TRAINABLE_BUFFERS", ())


def test_initial_state_is_frozen(node_and_spec) -> None:
    """After construction, nodes should have buffers (not parameters) for trainable names."""
    node, spec = node_and_spec
    tb_names = _get_trainable_names(node)

    if tb_names:
        # TRAINABLE_BUFFERS nodes: declared names should be buffers
        buffer_names = {n for n, _ in node.named_buffers()}
        for name in tb_names:
            assert name in buffer_names, f"{name} should be a buffer initially"
    else:
        # LearnableChannelMixer: conv parameters exist but node is constructed unfrozen
        # (nn.Conv2d parameters have requires_grad=True by default)
        pass


def test_unfreeze_converts_to_parameters(node_and_spec) -> None:
    """After unfreeze(), TRAINABLE_BUFFERS names should be nn.Parameters."""
    node, spec = node_and_spec
    tb_names = _get_trainable_names(node)

    node.unfreeze()

    if tb_names:
        param_names = {n for n, _ in node.named_parameters()}
        for name in tb_names:
            assert name in param_names, f"{name} should be a parameter after unfreeze()"
            attr = getattr(node, name)
            assert isinstance(attr, nn.Parameter)
            assert attr.requires_grad
    else:
        # LearnableChannelMixer
        for conv in node.convs:
            for p in conv.parameters():
                assert p.requires_grad


def test_freeze_converts_back_to_buffers(node_and_spec) -> None:
    """After unfreeze() then freeze(), TRAINABLE_BUFFERS names should be buffers again."""
    node, spec = node_and_spec
    tb_names = _get_trainable_names(node)

    node.unfreeze()
    node.freeze()

    if tb_names:
        buffer_names = {n for n, _ in node.named_buffers()}
        for name in tb_names:
            assert name in buffer_names, f"{name} should be a buffer after freeze()"
            attr = getattr(node, name)
            assert not isinstance(attr, nn.Parameter)
    else:
        # LearnableChannelMixer
        for conv in node.convs:
            for p in conv.parameters():
                assert not p.requires_grad


def test_state_dict_keys_preserved(node_and_spec) -> None:
    """state_dict() keys should be the same before and after unfreeze/freeze round-trip."""
    node, spec = node_and_spec
    keys_before = set(node.state_dict().keys())

    node.unfreeze()
    keys_unfrozen = set(node.state_dict().keys())

    node.freeze()
    keys_after = set(node.state_dict().keys())

    assert keys_before == keys_unfrozen, "Keys changed after unfreeze"
    assert keys_before == keys_after, "Keys changed after freeze round-trip"


def test_tensor_values_preserved(node_and_spec) -> None:
    """Tensor values should survive unfreeze/freeze round-trip."""
    node, spec = node_and_spec
    tb_names = _get_trainable_names(node)

    if not tb_names:
        pytest.skip("LearnableChannelMixer uses nn.Conv2d, tested separately")

    # Capture initial values
    initial_values = {name: getattr(node, name).clone() for name in tb_names}

    node.unfreeze()
    node.freeze()

    for name in tb_names:
        current = getattr(node, name)
        assert torch.equal(current, initial_values[name]), f"{name} values changed after round-trip"


def test_double_round_trip(node_and_spec) -> None:
    """freeze -> unfreeze -> freeze should not corrupt state."""
    node, _spec = node_and_spec

    state_before = {k: v.clone() for k, v in node.state_dict().items()}

    node.unfreeze()
    node.freeze()
    node.unfreeze()
    node.freeze()

    state_after = node.state_dict()

    assert set(state_before.keys()) == set(state_after.keys())
    for key in state_before:
        assert torch.equal(state_before[key], state_after[key]), (
            f"Value for '{key}' changed after double round-trip"
        )
