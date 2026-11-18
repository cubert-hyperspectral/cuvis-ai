"""SIMCADecision: moments, limit and decisions against the numpy and scipy reference, the
degrees-of-freedom rules, the false-alarm rate, the fit errors, the alpha-on-restore rule,
the port contract and the export smoke."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import NodeTag

from cuvis_ai.node.deciders.simca_decision import SIMCADecision
from tests.fixtures import simca_reference as ref

pytestmark = pytest.mark.unit


def _scaled_chi_square(
    seed: int, n_pixels: int, mean: float, dof: int, shape: tuple[int, ...] | None = None
) -> torch.Tensor:
    """Draws of ``mean / dof * chi2(dof)`` as a float32 map [1, n_pixels, 1, 1] (or ``shape``)."""
    rng = np.random.default_rng(seed)
    values = mean / dof * rng.chisquare(dof, size=n_pixels)
    return torch.from_numpy(values.astype(np.float32)).reshape(shape or (1, n_pixels, 1, 1))


def _batch(h: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
    return {"score_distance": h, "orthogonal_distance": q}


def _fitted(h: torch.Tensor, q: torch.Tensor, alpha: float = 0.01) -> SIMCADecision:
    node = SIMCADecision(alpha=alpha)
    node.statistical_initialization(iter([_batch(h, q)]))
    return node


def _flat(t: torch.Tensor) -> np.ndarray:
    return t.numpy().reshape(-1).astype(np.float64)


def test_moments_limit_and_decisions_match_the_reference() -> None:
    h = _scaled_chi_square(0, 4000, mean=3.0, dof=4)
    q = _scaled_chi_square(1, 4000, mean=0.02, dof=9)
    node = _fitted(h, q, alpha=0.05)

    h0, n_h = ref.moments(_flat(h))
    q0, n_q = ref.moments(_flat(q))
    critical = ref.limit(0.05, n_h, n_q)
    assert float(node.h0) == pytest.approx(h0, rel=1e-5)
    assert float(node.q0) == pytest.approx(q0, rel=1e-5)
    assert float(node.n_h) == n_h
    assert float(node.n_q) == n_q
    assert float(node.threshold) == pytest.approx(critical, rel=1e-6)

    out = node.forward(**_batch(h, q))
    expected = ref.combined(_flat(h), _flat(q), h0, n_h, q0, n_q)
    assert np.allclose(_flat(out["combined_distance"]), expected, rtol=1e-4)
    assert torch.equal(out["decisions"], out["combined_distance"] > node.threshold)
    clear = np.abs(expected - critical) > 1e-3 * critical  # away from the float32 borderline
    assert np.array_equal(out["decisions"].numpy().reshape(-1)[clear], (expected > critical)[clear])


@pytest.mark.parametrize(
    ("values", "expected_dof"),
    [
        (np.full(500, 2.5), 1.0),  # no spread: the estimate is infinite and is set to 1
        (1.0 + 1e-3 * np.random.default_rng(2).standard_normal(500), 250.0),  # capped
        (np.random.default_rng(3).lognormal(0.0, 2.0, 5000), 1.0),  # spread wider than chi2(1)
    ],
)
def test_degrees_of_freedom_are_rounded_and_clamped(
    values: np.ndarray, expected_dof: float
) -> None:
    h = torch.from_numpy(values.astype(np.float32)).reshape(1, -1, 1, 1)
    q = _scaled_chi_square(4, values.size, mean=1.0, dof=6)
    node = _fitted(h, q)

    assert float(node.n_h) == expected_dof == ref.moments(_flat(h))[1]
    assert float(node.n_q) == ref.moments(_flat(q))[1]


def test_false_alarm_rate_on_fresh_draws_is_near_alpha() -> None:
    """With independent scaled chi-square distances, the model's own assumption, the share of
    normal pixels flagged is the significance level."""
    n = 200_000
    node = _fitted(
        _scaled_chi_square(5, n, mean=3.0, dof=4), _scaled_chi_square(6, n, mean=0.5, dof=7)
    )
    assert (float(node.n_h), float(node.n_q)) == (4.0, 7.0)

    fresh = node.forward(
        **_batch(
            _scaled_chi_square(7, n, mean=3.0, dof=4), _scaled_chi_square(8, n, mean=0.5, dof=7)
        )
    )
    rate = fresh["decisions"].float().mean().item()
    assert rate == pytest.approx(0.01, abs=0.002)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5, float("nan")])
def test_alpha_must_lie_strictly_between_zero_and_one(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        SIMCADecision(alpha=alpha)


def test_alpha_is_a_serialized_hparam() -> None:
    assert SIMCADecision(alpha=0.02).hparams["alpha"] == 0.02
    assert SIMCADecision().alpha == 0.01


def test_requires_a_fit_and_rejects_an_empty_stream() -> None:
    node = SIMCADecision()
    assert node.requires_initial_fit is True
    with pytest.raises(ValueError, match="no pixels"):
        node.statistical_initialization(iter(()))
    assert node._statistically_initialized is False

    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node.forward(**_batch(torch.ones(1, 2, 2, 1), torch.ones(1, 2, 2, 1)))


def test_fit_rejects_a_distance_that_is_zero_everywhere() -> None:
    h = _scaled_chi_square(9, 200, mean=3.0, dof=4)
    node = SIMCADecision()
    with pytest.raises(ValueError, match="orthogonal_distance"):
        node.statistical_initialization(iter([_batch(h, torch.zeros_like(h))]))
    assert node._statistically_initialized is False


def test_fit_rejects_non_finite_distances() -> None:
    h = _scaled_chi_square(10, 200, mean=3.0, dof=4)
    q = _scaled_chi_square(11, 200, mean=1.0, dof=5)
    q[0, 0, 0, 0] = float("inf")
    node = SIMCADecision()
    with pytest.raises(ValueError, match="non-finite"):
        node.statistical_initialization(iter([_batch(h, q)]))
    assert node._statistically_initialized is False


def test_alpha_of_the_restoring_node_sets_the_threshold() -> None:
    """The limit follows ``alpha`` from the pipeline yaml, not the value the weights were
    fitted with: the degrees of freedom come from the weights, the significance from the node."""
    h = _scaled_chi_square(12, 3000, mean=3.0, dof=4)
    q = _scaled_chi_square(13, 3000, mean=0.5, dof=7)
    fitted = _fitted(h, q, alpha=0.01)

    restored = SIMCADecision(alpha=0.001)
    restored.load_state_dict(fitted.state_dict())

    assert restored._statistically_initialized is True
    for name in ("h0", "n_h", "q0", "n_q"):
        assert torch.equal(getattr(restored, name), getattr(fitted, name))
    expected = ref.limit(0.001, float(fitted.n_h), float(fitted.n_q))
    assert float(restored.threshold) == pytest.approx(expected, rel=1e-6)
    assert float(restored.threshold) > float(fitted.threshold)

    same = SIMCADecision(alpha=0.01)
    same.load_state_dict(fitted.state_dict())
    out_a, out_b = fitted.forward(**_batch(h, q)), same.forward(**_batch(h, q))
    assert torch.equal(out_a["decisions"], out_b["decisions"])
    assert torch.equal(out_a["combined_distance"], out_b["combined_distance"])


def test_port_contract() -> None:
    h = _scaled_chi_square(14, 2 * 4 * 5, mean=3.0, dof=4, shape=(2, 4, 5, 1))
    q = _scaled_chi_square(15, 2 * 4 * 5, mean=0.5, dof=7, shape=(2, 4, 5, 1))
    out = _fitted(h, q).forward(**_batch(h, q))

    assert set(SIMCADecision.INPUT_SPECS) == {"score_distance", "orthogonal_distance"}
    assert set(out) == set(SIMCADecision.OUTPUT_SPECS) == {"combined_distance", "decisions"}
    assert out["combined_distance"].shape == out["decisions"].shape == (2, 4, 5, 1)
    assert out["combined_distance"].dtype == torch.float32
    assert out["decisions"].dtype == torch.bool
    for name, spec in SIMCADecision.OUTPUT_SPECS.items():
        assert out[name].dtype == spec.dtype
    assert NodeTag.CLASSIFICATION in SIMCADecision.get_tags()


class _ExportProbe(torch.nn.Module):
    """The node with tensors in and tensors out, the form ``torch.export`` traces."""

    def __init__(self, node: SIMCADecision) -> None:
        super().__init__()
        self.node = node

    def forward(self, h: torch.Tensor, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.node(score_distance=h, orthogonal_distance=q)
        return out["combined_distance"], out["decisions"]


def test_export_smoke() -> None:
    shape = (1, 6, 6, 1)
    h = _scaled_chi_square(16, 36, mean=3.0, dof=4, shape=shape)
    q = _scaled_chi_square(17, 36, mean=0.5, dof=7, shape=shape)
    node = _fitted(h, q).eval()
    program = torch.export.export(_ExportProbe(node), (h, q))

    other_h = _scaled_chi_square(18, 36, mean=6.0, dof=4, shape=shape)
    other_q = _scaled_chi_square(19, 36, mean=1.0, dof=7, shape=shape)
    traced = program.module()(other_h, other_q)
    eager = node.forward(**_batch(other_h, other_q))
    assert torch.allclose(traced[0], eager["combined_distance"], atol=1e-6)
    assert torch.equal(traced[1], eager["decisions"])
    assert eager["decisions"].any() and not eager["decisions"].all()


@pytest.mark.gpu
def test_fits_over_two_cuda_batches() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    batches = [
        _batch(
            _scaled_chi_square(20 + i, 500, mean=3.0, dof=4).to("cuda"),
            _scaled_chi_square(30 + i, 500, mean=0.5, dof=7).to("cuda"),
        )
        for i in range(2)
    ]
    node = SIMCADecision().to("cuda")
    node.statistical_initialization(iter(batches))

    out = node.forward(**batches[0])
    assert out["decisions"].device.type == "cuda"
    assert node.threshold.device.type == "cuda"


def test_a_very_small_alpha_keeps_a_finite_limit() -> None:
    """``1 - alpha`` rounds to 1 for an alpha below 1e-16, so the limit has to come from the
    upper tail directly."""
    h = _scaled_chi_square(21, 3000, mean=3.0, dof=4)
    q = _scaled_chi_square(22, 3000, mean=0.5, dof=7)
    node = _fitted(h, q, alpha=1e-20)

    assert torch.isfinite(node.threshold)
    assert float(node.threshold) > ref.limit(1e-6, float(node.n_h), float(node.n_q))
    far = torch.full((1, 1, 1, 1), 1e6)
    assert node.forward(**_batch(far, far))["decisions"].all()
