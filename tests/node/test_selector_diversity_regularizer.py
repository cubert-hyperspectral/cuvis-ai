"""SelectorDiversityRegularizer: the loss is the negative, weighted population variance."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.losses import SelectorDiversityRegularizer

pytestmark = pytest.mark.unit


def test_loss_is_the_negative_weighted_population_variance() -> None:
    weights = torch.tensor([0.1, 0.4, 0.2, 0.3])
    loss = SelectorDiversityRegularizer(weight=0.5).forward(weights=weights)["loss"]
    expected = -0.5 * ((weights - weights.mean()) ** 2).mean()
    torch.testing.assert_close(loss, expected, atol=1e-7, rtol=1e-6)


def test_constant_weights_give_a_zero_loss() -> None:
    loss = SelectorDiversityRegularizer(weight=2.0).forward(weights=torch.full((6,), 0.25))["loss"]
    assert loss.shape == ()
    assert float(loss) == pytest.approx(0.0, abs=1e-7)


def test_loss_gradient_pushes_weights_apart() -> None:
    weights = torch.tensor([0.1, 0.4, 0.2, 0.3], requires_grad=True)
    loss = SelectorDiversityRegularizer(weight=1.0).forward(weights=weights)["loss"]
    loss.backward()
    assert weights.grad is not None
    # d/dw of -mean((w - mean(w))^2) is -2 (w - mean(w)) / n: entries above the mean get a
    # negative gradient (descent raises them), entries below a positive one.
    expected = -2.0 * (weights.detach() - weights.detach().mean()) / weights.numel()
    torch.testing.assert_close(weights.grad, expected, atol=1e-7, rtol=1e-6)
