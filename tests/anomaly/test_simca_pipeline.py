"""The SIMCA chain in a pipeline: PCA into SubspaceDistances into SIMCADecision, fitted by the
statistical trainer, checked against the numpy and scipy reference, on planted anomalies, and
through a save and reload."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from torch.utils.data import DataLoader

from cuvis_ai.node.anomaly.subspace_distances import SubspaceDistances
from cuvis_ai.node.data import AnomalyDataNode
from cuvis_ai.node.deciders.simca_decision import SIMCADecision
from cuvis_ai.node.dimensionality_reduction import TrainablePCA
from cuvis_ai_core.data.datamodule import BaseCuvisAIDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.trainers import StatisticalTrainer
from tests.fixtures import simca_reference as ref
from tests.fixtures.data_factory import _make_wavelengths_np, _SyntheticDictDataset

pytestmark = pytest.mark.integration

N_CHANNELS = 6
N_COMPONENTS = 3
ALPHA = 0.05
SHAPE = (2, 16, 16)
TRAIN_BATCHES = 6
PLANTED = (slice(None), slice(2, 5), slice(3, 7))  # a 3 x 4 patch in every evaluation frame


def _draws() -> tuple[torch.Tensor, torch.Tensor]:
    """Fit cubes and separate evaluation cubes from one low-rank normal model, float32."""
    cubes = ref.low_rank_cubes(seed=0, n_batches=TRAIN_BATCHES + 4, shape=SHAPE)
    return torch.cat(cubes[:TRAIN_BATCHES]), torch.cat(cubes[TRAIN_BATCHES:])


def _plant(evaluation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Push a patch of every evaluation frame off the model; return the cubes and the mask."""
    offset = torch.from_numpy(np.random.default_rng(9).normal(size=N_CHANNELS)).float()
    cubes = evaluation.clone()
    cubes[PLANTED] += 10.0 * offset / offset.norm()
    mask = torch.zeros(evaluation.shape[:3], dtype=torch.int32)
    mask[PLANTED] = 1
    return cubes, mask


class _NormalFitDataModule(BaseCuvisAIDataModule):
    """Train: normal frames only. Val: fresh normal frames with a planted patch and its mask."""

    def __init__(self) -> None:
        super().__init__()
        train, evaluation = _draws()
        wavelengths = _make_wavelengths_np(N_CHANNELS, (430.0, 910.0))
        normal = torch.zeros(train.shape[:3], dtype=torch.int32)
        self._train = _SyntheticDictDataset(train, normal, wavelengths)
        self._val = _SyntheticDictDataset(*_plant(evaluation), wavelengths)

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=4, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=len(self._val), shuffle=False)


def _simca_pipeline() -> CuvisPipeline:
    pipeline = CuvisPipeline("simca_chain")
    data = AnomalyDataNode(normal_class_ids=[0], name="data")
    pca = TrainablePCA(num_channels=N_CHANNELS, n_components=N_COMPONENTS, name="pca")
    distances = SubspaceDistances(n_components=N_COMPONENTS, name="distances")
    decision = SIMCADecision(alpha=ALPHA, name="decision")
    pipeline.connect(
        (data.outputs.cube, pca.inputs.data),
        (data.outputs.cube, distances.inputs.data),
        (pca.outputs.projected, distances.inputs.projected),
        (pca.outputs.reconstructed, distances.inputs.reconstructed),
        (distances.outputs.score_distance, decision.inputs.score_distance),
        (distances.outputs.orthogonal_distance, decision.inputs.orthogonal_distance),
    )
    return pipeline


@pytest.fixture(scope="module")
def fitted() -> tuple[CuvisPipeline, _NormalFitDataModule]:
    pipeline = _simca_pipeline()
    datamodule = _NormalFitDataModule()
    StatisticalTrainer(pipeline=pipeline, datamodule=datamodule).fit()
    return pipeline, datamodule


def _evaluate(pipeline: CuvisPipeline, datamodule: _NormalFitDataModule) -> dict:
    batch = next(iter(datamodule.val_dataloader()))
    return pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)


def test_the_trainer_fits_the_three_nodes_in_order(fitted) -> None:
    pipeline, _ = fitted
    for node in pipeline.nodes():
        if node.requires_initial_fit:
            assert node._statistically_initialized is True, node.name


def test_the_scores_of_the_fit_frames_are_centred_and_uncorrelated(fitted) -> None:
    """The trainer fits every node on the same frames, so on those frames the PCA scores have
    zero mean and the eigenvalues as their covariance. T2 as a sum over components rests on it.
    """
    pipeline, datamodule = fitted
    pca = next(node for node in pipeline.nodes() if node.name == "pca")
    projected = [
        pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)[("pca", "projected")]
        for batch in datamodule.train_dataloader()
    ]
    scores = torch.cat(projected).reshape(-1, N_COMPONENTS).double()
    sigma = pca._explained_variance.double().sqrt()
    assert torch.allclose(scores.mean(dim=0) / sigma, torch.zeros_like(sigma), atol=1e-4)
    covariance_in_eigenvalues = torch.cov(scores.T) / torch.outer(sigma, sigma)
    assert torch.allclose(covariance_in_eigenvalues, torch.eye(N_COMPONENTS).double(), atol=1e-3)


def test_the_fitted_chain_matches_the_reference(fitted) -> None:
    pipeline, datamodule = fitted
    train, evaluation = _draws()
    train_pixels = ref.pixels_of([train])
    mean, components, eigenvalues = ref.pca_model(train_pixels, N_COMPONENTS)
    h_train, q_train = ref.distances(train_pixels, mean, components, eigenvalues)
    h0, n_h = ref.moments(h_train)
    q0, n_q = ref.moments(q_train)

    decision = next(node for node in pipeline.nodes() if node.name == "decision")
    assert float(decision.h0) == pytest.approx(h0, rel=1e-3)
    assert float(decision.q0) == pytest.approx(q0, rel=1e-3)
    assert (float(decision.n_h), float(decision.n_q)) == (n_h, n_q)
    assert float(decision.threshold) == pytest.approx(ref.limit(ALPHA, n_h, n_q), rel=1e-6)

    out = _evaluate(pipeline, datamodule)
    planted_cubes, _ = _plant(evaluation)
    h, q = ref.distances(ref.pixels_of([planted_cubes]), mean, components, eigenvalues)
    score = out[("distances", "score_distance")].numpy().reshape(-1)
    orthogonal = out[("distances", "orthogonal_distance")].numpy().reshape(-1)
    combined = out[("decision", "combined_distance")].numpy().reshape(-1)
    assert np.allclose(score, h, rtol=1e-3, atol=1e-4)
    assert np.allclose(orthogonal, q, rtol=1e-3, atol=1e-5)
    assert np.allclose(combined, ref.combined(h, q, h0, n_h, q0, n_q), rtol=5e-3, atol=1e-3)


def test_planted_pixels_are_flagged_and_normal_pixels_fire_near_alpha(fitted) -> None:
    pipeline, datamodule = fitted
    out = _evaluate(pipeline, datamodule)
    decisions = out[("decision", "decisions")].squeeze(-1)
    planted = out[("data", "mask")].squeeze(-1).bool()

    assert decisions[planted].all()
    false_alarm_rate = decisions[~planted].float().mean().item()
    assert false_alarm_rate == pytest.approx(ALPHA, abs=0.03)


def test_the_saved_pipeline_reloads_and_scores_the_same(fitted, tmp_path) -> None:
    pipeline, datamodule = fitted
    before = _evaluate(pipeline, datamodule)
    yaml_path = tmp_path / "simca_chain.yaml"
    pipeline.save_to_file(str(yaml_path))

    reloaded = CuvisPipeline.load_pipeline(
        yaml_path, weights_path=yaml_path.with_suffix(".pt"), device="cpu"
    )
    after = _evaluate(reloaded, datamodule)
    for key in (
        ("distances", "score_distance"),
        ("distances", "orthogonal_distance"),
        ("decision", "combined_distance"),
        ("decision", "decisions"),
    ):
        assert torch.equal(after[key], before[key]), key


def test_a_component_count_that_differs_from_the_pca_is_refused() -> None:
    """Typing two different counts into the two nodes fails before any data flows."""
    pipeline = CuvisPipeline("simca_mismatch")
    pca = TrainablePCA(num_channels=N_CHANNELS, n_components=N_COMPONENTS, name="pca")
    distances = SubspaceDistances(n_components=N_COMPONENTS + 1, name="distances")
    with pytest.raises(Exception, match="(?i)shape|dimension|incompatible|n_components"):
        pipeline.connect((pca.outputs.projected, distances.inputs.projected))
