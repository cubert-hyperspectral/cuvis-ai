"""Unit tests for ScoreThreshold: a raw-score threshold that keeps values below or above it."""

import math

import pytest
import torch

from cuvis_ai.node.deciders.binary_decider import ScoreThreshold

pytestmark = pytest.mark.unit

SCORES = torch.tensor([[[[0.05], [0.1], [0.2], [float("nan")]]]])  # [1, 1, 4, 1]


def test_below_keeps_scores_at_or_under_the_threshold() -> None:
    out = ScoreThreshold(threshold=0.1, keep="below").forward(scores=SCORES)["decisions"]
    assert out[0, 0, :, 0].tolist() == [True, True, False, False]


def test_above_keeps_scores_over_the_threshold() -> None:
    out = ScoreThreshold(threshold=0.1, keep="above").forward(scores=SCORES)["decisions"]
    assert out[0, 0, :, 0].tolist() == [False, False, True, False]


def test_nan_is_never_marked() -> None:
    nan = torch.full((2, 3, 3, 1), float("nan"))
    for keep in ("below", "above"):
        assert not ScoreThreshold(keep=keep).forward(scores=nan)["decisions"].any()


def test_threshold_input_overrides_the_hparam() -> None:
    node = ScoreThreshold(threshold=0.1, keep="below")
    out = node.forward(scores=SCORES, threshold=torch.tensor([0.15]))["decisions"]
    assert out[0, 0, :, 0].tolist() == [True, True, False, False]
    out = node.forward(scores=SCORES, threshold=torch.tensor([0.3]))["decisions"]
    assert out[0, 0, :, 0].tolist() == [True, True, True, False]
    assert node.threshold == 0.1  # the override lasts one call


@pytest.mark.parametrize("value", [torch.tensor([0.1, 0.2]), torch.tensor(0.1), torch.zeros(1, 1)])
def test_threshold_input_of_another_shape_raises(value: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="shape"):
        ScoreThreshold().forward(scores=SCORES, threshold=value)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1])
def test_threshold_input_must_be_finite_and_non_negative(value: float) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ScoreThreshold().forward(scores=SCORES, threshold=torch.tensor([value]))


@pytest.mark.parametrize("value", [float("nan"), math.inf, -1.0, True, "0.1"])
def test_bad_threshold_hparam_raises(value: object) -> None:
    with pytest.raises(ValueError, match="threshold"):
        ScoreThreshold(threshold=value)  # type: ignore[arg-type]


def test_bad_keep_raises() -> None:
    with pytest.raises(ValueError, match="keep"):
        ScoreThreshold(keep="between")


def test_port_contract_and_rebuild() -> None:
    node = ScoreThreshold(threshold=0.25, keep="above")
    out = node.forward(scores=torch.rand(2, 5, 7, 1))
    assert set(out) == set(ScoreThreshold.OUTPUT_SPECS)
    assert out["decisions"].shape == (2, 5, 7, 1) and out["decisions"].dtype == torch.bool
    assert ScoreThreshold.INPUT_SPECS["threshold"].optional
    assert ScoreThreshold(**node.hparams).hparams == node.hparams
