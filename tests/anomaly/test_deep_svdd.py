import pytest
import torch

from cuvis_ai.anomaly.deep_svdd import (
    DeepSVDDCenterTracker,
    DeepSVDDEncoder,
    DeepSVDDScores,
)
from cuvis_ai.node.losses import DeepSVDDSoftBoundaryLoss
from cuvis_ai.utils.types import Context, ExecutionStage


def _make_stream(tensor: torch.Tensor):
    def generator():
        yield {"data": tensor}

    return generator()


def test_deep_svdd_fit_and_forward_shapes():
    x = torch.randn(2, 8, 9, 6)
    encoder = DeepSVDDEncoder(rep_dim=4, hidden=16, sample_n=1000, seed=0)
    encoder.fit(_make_stream(x))
    out = encoder.forward(x)["embeddings"]
    assert out.shape == (2, 8, 9, 4)
    assert out.dtype == x.dtype


def test_deep_svdd_forward_requires_fit():
    encoder = DeepSVDDEncoder(rep_dim=2)
    x = torch.randn(1, 4, 4, 3)
    try:
        encoder.forward(x)
    except RuntimeError as exc:
        assert "fit()" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when calling forward before fit()")


def test_deep_svdd_loss_radius_updates():
    embeddings = torch.randn(1, 4, 4, 3)
    center = torch.randn(embeddings.shape[-1])
    loss_node = DeepSVDDSoftBoundaryLoss(nu=0.1)
    optimizer = torch.optim.Adam(loss_node.parameters(), lr=1e-2)

    radius_before = torch.nn.functional.softplus(loss_node.r_unconstrained.clone(), beta=10.0)

    for _ in range(3):
        loss = loss_node.forward(embeddings, center=center)["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    radius_after = torch.nn.functional.softplus(loss_node.r_unconstrained.clone(), beta=10.0)
    assert not torch.allclose(radius_before, radius_after)


def test_deep_svdd_end_to_end_training_loop():
    torch.manual_seed(0)
    x = torch.randn(1, 6, 6, 5)
    encoder = DeepSVDDEncoder(rep_dim=3, hidden=8, sample_n=100, seed=0)
    encoder.fit(_make_stream(x))
    tracker = DeepSVDDCenterTracker(alpha=0.5)
    tracker.fit(_make_stream(encoder.forward(x)["embeddings"]))
    loss_node = DeepSVDDSoftBoundaryLoss(nu=0.05)

    # Unfreeze encoder to enable gradient updates
    encoder.unfreeze()
    params = list(encoder.parameters()) + list(loss_node.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)

    losses = []
    for _ in range(5):
        embeddings = encoder.forward(x)["embeddings"]
        center_out = tracker.forward(embeddings, context=Context(stage=ExecutionStage.TRAIN))
        loss = loss_node.forward(embeddings, center=center_out["center"])["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert torch.isfinite(torch.tensor(losses)).all()
    assert losses[-1] <= losses[0] * 2.0


def test_deep_svdd_scores_node_matches_manual_distance():
    embeddings = torch.randn(2, 3, 3, 4)
    center = torch.randn(4)

    score_node = DeepSVDDScores()
    scores = score_node.forward(embeddings=embeddings, center=center)["scores"]
    assert scores.shape == (2, 3, 3, 1)

    manual = ((embeddings - center.view(1, 1, 1, -1)) ** 2).sum(dim=-1, keepdim=True)
    assert torch.allclose(scores, manual)


def test_deep_svdd_scores_requires_matching_dims():
    score_node = DeepSVDDScores()
    embeddings = torch.randn(1, 2, 2, 3)
    center = torch.randn(4)

    with pytest.raises(RuntimeError, match="must match the size"):
        score_node.forward(embeddings=embeddings, center=center)


def test_deep_svdd_loss_rejects_bad_center_shape():
    embeddings = torch.randn(1, 3, 3, 4)
    bad_center = torch.randn(5)
    good_center = torch.randn(4)
    loss_node = DeepSVDDSoftBoundaryLoss()

    with pytest.raises(RuntimeError, match="must match the size"):
        loss_node.forward(embeddings, center=bad_center)

    result = loss_node.forward(embeddings, center=good_center)
    assert "loss" in result and result["loss"].requires_grad


def test_center_tracker_fit_and_forward_updates_center():
    embeddings = torch.randn(2, 4, 4, 3)
    tracker = DeepSVDDCenterTracker(alpha=0.5)
    tracker.fit(_make_stream(embeddings))

    context = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)
    out = tracker.forward(embeddings, context=context)
    assert out["center"].shape[0] == embeddings.shape[-1]
    assert isinstance(out["metrics"], list)

    new_embeddings = embeddings + 1.0
    updated = tracker.forward(new_embeddings, context=context)
    assert not torch.allclose(out["center"], updated["center"])


def test_center_tracker_skips_eval_updates_by_default():
    embeddings = torch.randn(1, 3, 3, 2)
    tracker = DeepSVDDCenterTracker(alpha=1.0)
    tracker.fit(_make_stream(embeddings))

    train_ctx = Context(stage=ExecutionStage.TRAIN, epoch=0, batch_idx=0)
    val_ctx = Context(stage=ExecutionStage.VAL, epoch=0, batch_idx=0)

    base_center = tracker.forward(embeddings, context=train_ctx)["center"]
    eval_embeddings = embeddings + 10.0
    eval_center = tracker.forward(eval_embeddings, context=val_ctx)["center"]
    assert torch.allclose(base_center, eval_center)
