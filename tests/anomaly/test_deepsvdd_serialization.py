import pytest
import torch

from cuvis_ai.anomaly.deep_svdd import (
    DeepSVDDCenterTracker,
    DeepSVDDProjection,
    ZScoreNormalizerGlobal,
)
from cuvis_ai.node.losses import DeepSVDDSoftBoundaryLoss

pytestmark = pytest.mark.unit


def _make_stream(tensor: torch.Tensor, key: str) -> list[dict[str, torch.Tensor]]:
    return [{key: tensor}]


def test_deepsvdd_encoder_round_trip() -> None:
    """Test DeepSVDDEncoder serialization with pre-allocated buffers."""
    num_channels = 3
    encoder = ZScoreNormalizerGlobal(num_channels=num_channels, eps=1e-6)

    # Fit the encoder
    data = torch.randn(1, 2, 2, num_channels)
    encoder.statistical_initialization(iter(_make_stream(data, "data")))

    # Get state and create new encoder
    state = encoder.state_dict()
    reloaded = ZScoreNormalizerGlobal(num_channels=num_channels, eps=1e-6)
    reloaded.load_state_dict(state)

    # Test forward pass
    output = reloaded.forward(data)
    assert output["normalized"].shape == data.shape
    assert torch.allclose(encoder.forward(data)["normalized"], output["normalized"])


def test_deepsvdd_projection_round_trip() -> None:
    """Test DeepSVDDProjection serialization with known in_channels."""
    in_channels = 3
    rep_dim = 4
    hidden = 8

    projection = DeepSVDDProjection(in_channels=in_channels, rep_dim=rep_dim, hidden=hidden)
    dummy = torch.randn(1, 2, 2, in_channels)

    # Get original output
    original_output = projection.forward(dummy)["embeddings"]

    # Save and load state
    state = projection.state_dict()
    restored = DeepSVDDProjection(in_channels=in_channels, rep_dim=rep_dim, hidden=hidden)
    restored.load_state_dict(state)

    # Verify outputs match
    restored_output = restored.forward(dummy)["embeddings"]
    assert torch.allclose(original_output, restored_output, atol=1e-6)


def test_deepsvdd_projection_rbf_kernel() -> None:
    """Test DeepSVDDProjection with RBF kernel serialization."""
    in_channels = 7
    rep_dim = 4
    hidden = 8

    projection = DeepSVDDProjection(
        in_channels=in_channels, rep_dim=rep_dim, hidden=hidden, kernel="rbf"
    )
    dummy = torch.randn(1, 2, 2, in_channels)

    # Get original output
    original_output = projection.forward(dummy)["embeddings"]

    # Save and load state
    state = projection.state_dict()
    restored = DeepSVDDProjection(
        in_channels=in_channels, rep_dim=rep_dim, hidden=hidden, kernel="rbf"
    )
    restored.load_state_dict(state)

    # Verify outputs match
    restored_output = restored.forward(dummy)["embeddings"]
    assert torch.allclose(original_output, restored_output, atol=1e-6)


def test_deepsvdd_center_tracker_round_trip() -> None:
    """Test DeepSVDDCenterTracker serialization with known rep_dim."""
    rep_dim = 4
    tracker = DeepSVDDCenterTracker(rep_dim=rep_dim, alpha=0.5)

    # Fit the tracker
    embeddings = torch.randn(1, 2, 2, rep_dim)
    tracker.statistical_initialization(iter(_make_stream(embeddings, "embeddings")))

    # Save and load state
    state = tracker.state_dict()
    restored = DeepSVDDCenterTracker(rep_dim=rep_dim, alpha=0.5)
    restored.load_state_dict(state)

    # Verify center matches
    assert torch.allclose(tracker._tracked_center, restored._tracked_center)


def test_deepsvdd_soft_boundary_loss_round_trip() -> None:
    """Test DeepSVDDSoftBoundaryLoss serialization."""
    loss = DeepSVDDSoftBoundaryLoss(nu=0.05)
    state = loss.state_dict()

    restored = DeepSVDDSoftBoundaryLoss(nu=0.05)
    restored.load_state_dict(state)

    embeddings = torch.randn(1, 2, 2, 4)
    center = torch.zeros(4)
    original = loss.forward(embeddings=embeddings, center=center)["loss"]
    loaded = restored.forward(embeddings=embeddings, center=center)["loss"]
    assert torch.allclose(original, loaded)


def test_deepsvdd_end_to_end_round_trip(tmp_path) -> None:
    """Test end-to-end DeepSVDD pipeline serialization."""
    torch.manual_seed(0)
    num_channels = 30
    rep_dim = 4
    data = torch.randn(2, 200, 200, num_channels)

    # Create and fit encoder
    encoder = ZScoreNormalizerGlobal(num_channels=num_channels, eps=1e-6)
    encoder.statistical_initialization(iter(_make_stream(data, "data")))
    normalized = encoder.forward(data)["normalized"]

    # Create and test projection
    projection = DeepSVDDProjection(in_channels=num_channels, rep_dim=rep_dim, hidden=8)
    embeddings = projection.forward(normalized)["embeddings"]

    # Create and fit tracker
    tracker = DeepSVDDCenterTracker(rep_dim=rep_dim, alpha=0.1)
    tracker.statistical_initialization(iter(_make_stream(embeddings, "embeddings")))
    center = tracker.forward(embeddings)["center"]

    # Create loss node
    loss_node = DeepSVDDSoftBoundaryLoss(nu=0.1, weight=2.0)
    loss_value = loss_node.forward(embeddings=embeddings, center=center)["loss"]

    # Save all states
    checkpoint_path = tmp_path / "deep_svdd_roundtrip.pt"
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "projection": projection.state_dict(),
            "tracker": tracker.state_dict(),
            "loss": loss_node.state_dict(),
        },
        checkpoint_path,
    )

    # Load and verify encoder
    state = torch.load(checkpoint_path)
    encoder_loaded = ZScoreNormalizerGlobal(num_channels=num_channels, eps=1e-6)
    encoder_loaded.load_state_dict(state["encoder"])
    normalized_loaded = encoder_loaded.forward(data)["normalized"]
    assert torch.allclose(normalized, normalized_loaded)

    # Load and verify projection
    projection_loaded = DeepSVDDProjection(in_channels=num_channels, rep_dim=rep_dim, hidden=8)
    projection_loaded.load_state_dict(state["projection"])
    embeddings_loaded = projection_loaded.forward(normalized_loaded)["embeddings"]
    assert torch.allclose(embeddings, embeddings_loaded)

    # Load and verify tracker
    tracker_loaded = DeepSVDDCenterTracker(rep_dim=rep_dim, alpha=0.1)
    tracker_loaded.load_state_dict(state["tracker"])
    center_loaded = tracker_loaded.forward(embeddings_loaded)["center"]
    assert torch.allclose(center, center_loaded)

    # Load and verify loss
    loss_loaded = DeepSVDDSoftBoundaryLoss(nu=0.1, weight=2.0)
    loss_loaded.load_state_dict(state["loss"])
    loss_value_loaded = loss_loaded.forward(embeddings=embeddings_loaded, center=center_loaded)[
        "loss"
    ]
    assert torch.allclose(loss_value, loss_value_loaded)
