from __future__ import annotations

import pytest

from cuvis_ai.anomaly.rx_detector import RXPerBatch

pytestmark = pytest.mark.unit


def test_node_hparams_auto_population():
    """Test that Node subclasses automatically populate hparams from init parameters."""
    # Test with a single parameter
    eps_value = 1e-99
    model = RXPerBatch(eps=eps_value)

    assert hasattr(model, "hparams"), "Node should have hparams attribute"
    assert "eps" in model.hparams, "hparams should contain eps parameter"
    assert model.hparams["eps"] == eps_value, (
        f"Expected eps={eps_value}, got {model.hparams['eps']}"
    )

    # Test with default parameter
    model_default = RXPerBatch()
    assert hasattr(model_default, "hparams"), "Node with defaults should have hparams attribute"
    assert "eps" in model_default.hparams, "hparams should contain eps parameter with default"
    assert model_default.hparams["eps"] == 1e-6, "Default eps value should be 1e-6"


def test_node_hparams_persistence():
    """Test that hparams are preserved and can be used for node recreation."""
    # Create a node with specific parameters
    eps_value = 1e-8
    original_node = RXPerBatch(eps=eps_value)

    # Verify hparams
    assert original_node.hparams == {"eps": eps_value}

    # Create a new node with the same hparams
    recreated_node = RXPerBatch(**original_node.hparams)

    # Verify the recreated node has the same hyperparameters
    assert recreated_node.hparams == original_node.hparams
    assert recreated_node.eps == original_node.eps


def test_node_initialization_no_error():
    """Test that Node subclasses can be initialized without errors."""
    # This test verifies that the fix for the TypeError is working
    # Previously, this would raise:
    # TypeError: Node.__init__() got an unexpected keyword argument 'eps'
    try:
        model = RXPerBatch(eps=1e-99)
        assert model.eps == 1e-99
    except TypeError as e:
        pytest.fail(f"Node initialization failed with TypeError: {e}")
