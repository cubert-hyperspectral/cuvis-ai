import pytest
import torch

from cuvis_ai.node.normalization import (
    IdentityNormalizer,
    MinMaxNormalizer,
    SigmoidNormalizer,
    SigmoidTransform,
    ZScoreNormalizer,
    _ScoreNormalizerBase,
)


@torch.no_grad()
def test_score_normalizer_base_raises_not_implemented() -> None:
    base = _ScoreNormalizerBase()
    x = torch.zeros(2, 3, 4, 5, dtype=torch.float32)

    with pytest.raises(NotImplementedError):
        base._normalize(x)


@torch.no_grad()
def test_identity_normalizer_passes_through() -> None:
    node = IdentityNormalizer()
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32)

    out = node.forward(data=x)["normalized"]
    assert torch.equal(out, x)


def test_minmax_normalizer_init_rejects_bad_max_initialization_frames() -> None:
    with pytest.raises(ValueError):
        MinMaxNormalizer(max_initialization_frames=0, use_running_stats=True)


@torch.no_grad()
def test_minmax_normalizer_running_stats_forward_uses_strict_path() -> None:
    node = MinMaxNormalizer(use_running_stats=True, eps=1e-6, max_initialization_frames=4)

    # Provide two "batches" to exercise statistical initialization logic.
    x1 = torch.tensor(
        [
            [[[1.0, 2.0], [3.0, 4.0]]],
            [[[0.0, 1.0], [2.0, 3.0]]],
        ],
        dtype=torch.float32,
    )  # shape (B=2, H=1, W=2, C=2)
    x2 = torch.tensor(
        [
            [[[2.0, 0.0], [1.0, 3.0]]],
            [[[4.0, 5.0], [6.0, 7.0]]],
        ],
        dtype=torch.float32,
    )

    node.statistical_initialization(iter([{"data": x1}, {"data": x2}]))

    # Forward after init should hit strict running-stats mode.
    y = node.forward(data=x2)["normalized"]
    assert y.shape == x2.shape
    assert torch.isfinite(y).all()
    assert float(y.min()) >= -1e-5
    assert float(y.max()) <= 1.0 + 1e-5


@torch.no_grad()
def test_minmax_normalizer_statistical_initialization_respects_max_initialization_frames() -> None:
    node = MinMaxNormalizer(use_running_stats=True, eps=1e-6, max_initialization_frames=1)

    # First batch has 2 frames but should be sliced to 1.
    x = torch.randn(2, 2, 2, 3, dtype=torch.float32)
    node.statistical_initialization(iter([{"data": x}]))

    out = node.forward(data=x[:1])["normalized"]
    assert out.shape == (1, 2, 2, 3)
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_sigmoid_normalizer_computes_sigmoid_and_handles_zero_std() -> None:
    node = SigmoidNormalizer(std_floor=1e-6)

    # Make each sample constant across spatial dims so std == 0 -> clamp to std_floor.
    x = torch.ones(2, 4, 5, 3, dtype=torch.float32)
    out = node.forward(data=x)["normalized"]
    assert out.shape == x.shape
    assert float(out.min()) >= 0.0
    assert float(out.max()) <= 1.0


@torch.no_grad()
def test_zscore_normalizer_computes_zscore() -> None:
    node = ZScoreNormalizer(dims=[1, 2], eps=1e-6)
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32)

    out = node.forward(data=x)["normalized"]
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


@torch.no_grad()
def test_sigmoid_transform_matches_torch_sigmoid() -> None:
    node = SigmoidTransform()
    x = torch.randn(2, 3, 4, 5, dtype=torch.float32)

    out = node.forward(data=x)["transformed"]
    assert torch.allclose(out, torch.sigmoid(x))
