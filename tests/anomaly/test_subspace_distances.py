"""SubspaceDistances: T2 and Q against a numpy reference, the rank policy, the fit errors,
the port contract, the state round trip and the export smoke."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import NodeTag

from cuvis_ai.node.anomaly.subspace_distances import SubspaceDistances
from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from tests.fixtures import simca_reference as ref

pytestmark = pytest.mark.unit

N_CHANNELS = 6
N_COMPONENTS = 3


def _pca_stream(pca: TrainablePCA, cubes: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    """What the trainer hands SubspaceDistances: the cube plus the PCA's two outputs."""
    stream = []
    for cube in cubes:
        out = pca.forward(cube)
        stream.append(
            {"data": cube, "projected": out["projected"], "reconstructed": out["reconstructed"]}
        )
    return stream


def _fitted_chain(
    cubes: list[torch.Tensor], whiten: bool = False
) -> tuple[TrainablePCA, SubspaceDistances]:
    pca = TrainablePCA(num_channels=N_CHANNELS, n_components=N_COMPONENTS, whiten=whiten)
    pca.statistical_initialization({"data": cube} for cube in cubes)
    node = SubspaceDistances(n_components=N_COMPONENTS)
    node.statistical_initialization(iter(_pca_stream(pca, cubes)))
    return pca, node


def _score_stream(scores: torch.Tensor) -> list[dict[str, torch.Tensor]]:
    """A stream that carries scores only; the cube and its reconstruction do not enter the fit."""
    cube = torch.zeros(*scores.shape[:-1], N_CHANNELS)
    return [{"data": cube, "projected": scores, "reconstructed": cube}]


def test_distances_match_the_numpy_reference() -> None:
    cubes = ref.low_rank_cubes(seed=0)
    pca, node = _fitted_chain(cubes)
    mean, components, eigenvalues = ref.pca_model(ref.pixels_of(cubes), N_COMPONENTS)

    evaluation = ref.low_rank_cubes(seed=0, n_batches=4)[3]
    out = node.forward(**_pca_stream(pca, [evaluation])[0])
    t2, q = ref.distances(ref.pixels_of([evaluation]), mean, components, eigenvalues)

    assert np.allclose(out["score_distance"].numpy().reshape(-1), t2, rtol=1e-3, atol=1e-5)
    assert np.allclose(out["orthogonal_distance"].numpy().reshape(-1), q, rtol=1e-3, atol=1e-5)


def test_score_distance_is_the_squared_mahalanobis_distance_of_the_scores() -> None:
    rng = np.random.default_rng(1)
    mixing = rng.normal(size=(N_COMPONENTS, N_COMPONENTS))
    scores = rng.normal(size=(2, 8, 8, N_COMPONENTS)) @ mixing.T + np.array([3.0, -1.0, 0.5])
    node = SubspaceDistances(n_components=N_COMPONENTS)
    node.statistical_initialization(iter(_score_stream(torch.from_numpy(scores).float())))

    flat = scores.reshape(-1, N_COMPONENTS)
    centred = flat - flat.mean(axis=0)
    precision = np.linalg.inv(np.cov(flat, rowvar=False))
    expected = np.einsum("nk,kl,nl->n", centred, precision, centred)

    out = node.forward(**_score_stream(torch.from_numpy(scores).float())[0])
    assert np.allclose(out["score_distance"].numpy().reshape(-1), expected, rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("stds", [(1.0, 1.0, 1.0), (1e3, 1.0, 1e-3)])
def test_score_distance_does_not_change_when_the_scores_are_whitened(stds) -> None:
    """Also for a spectrum with a variance ratio of 1e12: no component is dropped by scale."""
    rng = np.random.default_rng(2)
    unit = rng.normal(size=(2, 8, 8, N_COMPONENTS))
    raw = torch.from_numpy(unit * np.asarray(stds)).float()
    whitened = torch.from_numpy(unit).float()

    plain = SubspaceDistances(n_components=N_COMPONENTS)
    plain.statistical_initialization(iter(_score_stream(raw)))
    white = SubspaceDistances(n_components=N_COMPONENTS)
    white.statistical_initialization(iter(_score_stream(whitened)))

    a = plain.forward(**_score_stream(raw)[0])["score_distance"]
    b = white.forward(**_score_stream(whitened)[0])["score_distance"]
    assert torch.allclose(a, b, rtol=1e-3, atol=1e-4)


def test_whitening_the_pca_leaves_both_distances_unchanged() -> None:
    cubes = ref.low_rank_cubes(seed=3)
    pca, node = _fitted_chain(cubes)
    pca_w, node_w = _fitted_chain(cubes, whiten=True)

    plain = node.forward(**_pca_stream(pca, cubes[:1])[0])
    white = node_w.forward(**_pca_stream(pca_w, cubes[:1])[0])
    assert torch.allclose(plain["score_distance"], white["score_distance"], rtol=1e-3, atol=1e-4)
    assert torch.allclose(plain["orthogonal_distance"], white["orthogonal_distance"], atol=1e-5)


def test_port_contract() -> None:
    cubes = ref.low_rank_cubes(seed=4)
    pca, node = _fitted_chain(cubes)
    out = node.forward(**_pca_stream(pca, cubes[:1])[0])

    assert (
        set(out)
        == set(SubspaceDistances.OUTPUT_SPECS)
        == {
            "score_distance",
            "orthogonal_distance",
        }
    )
    for name, spec in SubspaceDistances.OUTPUT_SPECS.items():
        assert out[name].shape == (*cubes[0].shape[:3], 1)
        assert out[name].dtype == spec.dtype == torch.float32
        assert (out[name] >= 0).all()
    assert set(SubspaceDistances.INPUT_SPECS) == {"data", "projected", "reconstructed"}
    assert NodeTag.ANOMALY in SubspaceDistances.get_tags()


def test_the_component_count_is_part_of_the_projected_port() -> None:
    """The symbolic dimension lets port validation catch a count that differs from the PCA's."""
    spec = SubspaceDistances.INPUT_SPECS["projected"]
    assert spec.shape[-1] == "n_components"
    assert spec.resolve_shape(SubspaceDistances(n_components=4)) == (-1, -1, -1, 4)


def test_requires_a_fit_and_rejects_an_empty_stream() -> None:
    node = SubspaceDistances(n_components=N_COMPONENTS)
    assert node.requires_initial_fit is True
    with pytest.raises(ValueError, match="no pixels"):
        node.statistical_initialization(iter(()))
    assert node._statistically_initialized is False

    cube = torch.zeros(1, 2, 2, N_CHANNELS)
    with pytest.raises(RuntimeError, match="statistical_initialization"):
        node.forward(data=cube, projected=torch.zeros(1, 2, 2, N_COMPONENTS), reconstructed=cube)


def test_fit_names_both_counts_when_the_score_width_is_wrong() -> None:
    node = SubspaceDistances(n_components=N_COMPONENTS)
    with pytest.raises(ValueError, match=r"n_components=3.*5 components"):
        node.statistical_initialization(iter(_score_stream(torch.rand(1, 4, 4, 5))))
    assert node._statistically_initialized is False


def test_fit_rejects_a_score_component_without_variance() -> None:
    scores = torch.rand(1, 8, 8, N_COMPONENTS)
    scores[..., 1] = 0.25
    node = SubspaceDistances(n_components=N_COMPONENTS)
    with pytest.raises(ValueError, match="no variance"):
        node.statistical_initialization(iter(_score_stream(scores)))
    assert node._statistically_initialized is False


def test_fit_rejects_non_finite_scores() -> None:
    scores = torch.rand(1, 8, 8, N_COMPONENTS)
    scores[0, 0, 0, 0] = float("nan")
    node = SubspaceDistances(n_components=N_COMPONENTS)
    with pytest.raises(ValueError, match="non-finite"):
        node.statistical_initialization(iter(_score_stream(scores)))
    assert node._statistically_initialized is False


def test_state_dict_round_trip_restores_the_fit() -> None:
    cubes = ref.low_rank_cubes(seed=5)
    pca, node = _fitted_chain(cubes)
    restored = SubspaceDistances(n_components=N_COMPONENTS)
    assert restored._statistically_initialized is False
    restored.load_state_dict(node.state_dict())

    assert restored._statistically_initialized is True
    batch = _pca_stream(pca, cubes[:1])[0]
    for name in SubspaceDistances.OUTPUT_SPECS:
        assert torch.equal(restored.forward(**batch)[name], node.forward(**batch)[name])


class _ExportProbe(torch.nn.Module):
    """The node with tensors in and tensors out, the form ``torch.export`` traces."""

    def __init__(self, node: SubspaceDistances) -> None:
        super().__init__()
        self.node = node

    def forward(
        self, data: torch.Tensor, projected: torch.Tensor, reconstructed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.node(data=data, projected=projected, reconstructed=reconstructed)
        return out["score_distance"], out["orthogonal_distance"]


def test_export_smoke() -> None:
    cubes = ref.low_rank_cubes(seed=6)
    pca, node = _fitted_chain(cubes)
    node.eval()
    example = _pca_stream(pca, cubes[:1])[0]
    program = torch.export.export(_ExportProbe(node), tuple(example.values()))

    other = _pca_stream(pca, cubes[1:2])[0]  # same shapes: the graph is traced at fixed shapes
    traced = program.module()(*other.values())
    eager = node.forward(**other)
    assert torch.allclose(traced[0], eager["score_distance"], atol=1e-6)
    assert torch.allclose(traced[1], eager["orthogonal_distance"], atol=1e-6)


@pytest.mark.gpu
def test_fits_over_two_cuda_batches() -> None:
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    scores = [torch.rand(1, 8, 8, N_COMPONENTS, device="cuda") for _ in range(2)]
    node = SubspaceDistances(n_components=N_COMPONENTS).to("cuda")
    stream = [
        {
            "data": torch.zeros(1, 8, 8, N_CHANNELS, device="cuda"),
            "projected": s,
            "reconstructed": torch.zeros(1, 8, 8, N_CHANNELS, device="cuda"),
        }
        for s in scores
    ]
    node.statistical_initialization(iter(stream))

    out = node.forward(**stream[0])
    assert out["score_distance"].device.type == "cuda"
