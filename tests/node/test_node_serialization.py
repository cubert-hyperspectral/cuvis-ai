"""Tests for individual node serialization patterns."""

import torch

from cuvis_ai_core.utils.node_registry import NodeRegistry


def test_all_nodes_use_state_dict_only():
    """Verify that all nodes can be serialized using only state_dict."""
    for node_name in NodeRegistry.list_builtin_nodes():
        node_class = NodeRegistry.get(node_name)

        try:
            # Create node with minimal params
            if "ChannelSelector" in node_name:
                node = node_class(n_select=2, input_channels=10)
            elif "TopKIndices" in node_name:
                node = node_class(k=2)
            else:
                node = node_class()

            # Test state_dict works
            state = node.state_dict()
            assert isinstance(state, dict), f"{node_name}.state_dict() must return dict"

            # Test load_state_dict works
            new_node = node_class(**node.hparams) if hasattr(node, "hparams") else node_class()
            new_node.load_state_dict(state, strict=False)

            # Verify states match
            assert set(new_node.state_dict().keys()) == set(state.keys()), (
                f"{node_name} state keys don't match after load"
            )

        except (TypeError, AttributeError):
            # Skip nodes that require special initialization
            continue


def test_no_custom_serialize_methods_needed():
    """Verify that nodes don't need custom serialize/load methods."""
    from cuvis_ai.node.normalization import MinMaxNormalizer
    from cuvis_ai.node.selector import SoftChannelSelector

    # Test stateless node with empty state
    normalizer = MinMaxNormalizer()
    state = normalizer.state_dict()
    new_normalizer = MinMaxNormalizer()
    new_normalizer.load_state_dict(state, strict=False)

    # Test trainable node
    selector = SoftChannelSelector(n_select=3, input_channels=10)
    selector.unfreeze()

    # Modify parameters
    with torch.no_grad():
        selector.channel_logits.fill_(42.0)

    # Serialize and load
    state = selector.state_dict()
    new_selector = SoftChannelSelector(n_select=3, input_channels=10)
    new_selector.load_state_dict(state, strict=False)

    # Verify values preserved
    assert torch.all(new_selector.channel_logits == 42.0)
    # After loading, need to unfreeze again to convert to parameter
    assert not isinstance(new_selector.channel_logits, torch.nn.Parameter)
    new_selector.unfreeze()
    assert isinstance(new_selector.channel_logits, torch.nn.Parameter)


def test_node_inherits_pytorch_serialization():
    """Test that all nodes inherit state_dict/load_state_dict from nn.Module."""
    from torch import nn

    from cuvis_ai.node.normalization import MinMaxNormalizer

    node = MinMaxNormalizer()

    # All nodes inherit from nn.Module through Node base class
    assert isinstance(node, nn.Module)

    # Verify PyTorch serialization methods are available
    assert hasattr(node, "state_dict")
    assert hasattr(node, "load_state_dict")
    assert callable(node.state_dict)
    assert callable(node.load_state_dict)

    # Verify state_dict works
    state = node.state_dict()
    assert isinstance(state, dict)


def test_validate_serialization_support_method():
    """Test node validation method works correctly."""
    from cuvis_ai.node.normalization import MinMaxNormalizer

    node = MinMaxNormalizer()
    is_valid, message = node.validate_serialization_support()

    assert is_valid, f"MinMaxNormalizer should be valid: {message}"
    assert message == "OK"


def test_buffer_registration_for_statistical_nodes():
    """Test that statistical nodes properly register buffers."""
    from cuvis_ai.node.normalization import MinMaxNormalizer

    node = MinMaxNormalizer()

    # Check buffers are registered
    assert hasattr(node, "running_min")
    assert hasattr(node, "running_max")

    # Initially NaN
    assert torch.isnan(node.running_min).item()
    assert torch.isnan(node.running_max).item()

    # Simulate fitting
    test_data = torch.randn(10, 5, 5, 3)
    node.running_min = test_data.min()
    node.running_max = test_data.max()

    # Verify buffers in state_dict
    state = node.state_dict()
    assert "running_min" in state
    assert "running_max" in state

    # Verify serialization preserves values
    new_node = MinMaxNormalizer()
    # First set buffers to non-None so they can accept loaded values
    new_node.running_min = torch.tensor(0.0)
    new_node.running_max = torch.tensor(0.0)
    new_node.load_state_dict(state, strict=False)
    assert torch.equal(new_node.running_min, node.running_min)
    assert torch.equal(new_node.running_max, node.running_max)


def test_trainable_node_parameter_conversion():
    """Test that trainable nodes convert buffers to parameters correctly."""
    from torch import nn

    from cuvis_ai.node.selector import SoftChannelSelector

    node = SoftChannelSelector(n_select=3, input_channels=10)

    # Initially should be a buffer
    assert not isinstance(node.channel_logits, nn.Parameter)

    # Unfreeze should convert to parameter
    node.unfreeze()
    assert isinstance(node.channel_logits, nn.Parameter)
    assert node.channel_logits.requires_grad

    # Test serialization preserves parameter status
    state = node.state_dict()
    new_node = SoftChannelSelector(n_select=3, input_channels=10)
    new_node.load_state_dict(state, strict=False)

    # After loading, need to unfreeze again to convert to parameter
    assert not isinstance(new_node.channel_logits, nn.Parameter)
    new_node.unfreeze()
    assert isinstance(new_node.channel_logits, nn.Parameter)
    assert new_node.channel_logits.requires_grad


def test_non_persistent_buffers_not_serialized():
    """Test that non-persistent buffers are excluded from state_dict."""
    from tests.fixtures import MockStatisticalTrainableNode

    node = MockStatisticalTrainableNode(input_dim=5, hidden_dim=3)

    # Verify temp_cache exists
    assert hasattr(node, "temp_cache")
    assert node.temp_cache is not None

    # Verify it's not in state_dict
    state = node.state_dict()
    assert "temp_cache" not in state


def test_state_dict_handles_nan_buffers():
    """Test that state_dict correctly handles NaN-initialized buffers."""
    from cuvis_ai.node.normalization import MinMaxNormalizer

    node = MinMaxNormalizer()

    # Buffers are initially NaN
    assert torch.isnan(node.running_min).item()
    assert torch.isnan(node.running_max).item()

    # NaN buffers ARE included in state_dict (unlike None buffers)
    state = node.state_dict()
    assert "running_min" in state
    assert "running_max" in state
    assert torch.isnan(state["running_min"]).item()
    assert torch.isnan(state["running_max"]).item()

    # After setting values, they appear in state_dict with actual values
    node.running_min = torch.tensor(-1.0)
    node.running_max = torch.tensor(1.0)
    state = node.state_dict()
    assert "running_min" in state
    assert "running_max" in state
    assert state["running_min"].item() == -1.0
    assert state["running_max"].item() == 1.0

    # Can load state
    new_node = MinMaxNormalizer()
    new_node.load_state_dict(state, strict=False)
    assert torch.equal(new_node.running_min, node.running_min)
    assert torch.equal(new_node.running_max, node.running_max)


def test_hparams_separate_from_state():
    """Test that hyperparameters are separate from state_dict."""
    from cuvis_ai.node.normalization import MinMaxNormalizer

    node = MinMaxNormalizer(eps=1e-5, use_running_stats=True)

    # Hyperparameters should be in hparams
    assert hasattr(node, "hparams")
    assert node.hparams["eps"] == 1e-5
    assert node.hparams["use_running_stats"] is True

    # Hyperparameters should NOT be in state_dict
    state = node.state_dict()
    assert "eps" not in state
    assert "use_running_stats" not in state

    # NaN buffers ARE included in state_dict (with NaN values)
    assert "running_min" in state
    assert "running_max" in state
    assert torch.isnan(state["running_min"]).item()
    assert torch.isnan(state["running_max"]).item()

    # After setting actual values, buffers still in state_dict with those values
    node.running_min = torch.tensor(-1.0)
    node.running_max = torch.tensor(1.0)
    state = node.state_dict()
    assert "running_min" in state
    assert "running_max" in state
    assert state["running_min"].item() == -1.0
    assert state["running_max"].item() == 1.0
