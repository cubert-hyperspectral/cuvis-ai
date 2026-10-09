"""FixedPCAProjection: golden rule against the projection math (projection, fixed scaling, clamp),
the input min-max, port contract, hparams, the opted-out statistical fit and the reload smoke."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.dimensionality_reduction import FixedPCAProjection, TrainablePCA
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit


def _proj(tmp_path, explained: bool = True):
    rng = np.random.default_rng(0)
    C, K = 7, 3
    mean = rng.normal(size=C).astype(np.float32)
    comps = np.linalg.qr(rng.normal(size=(C, C)))[0][:, :K].T.astype(np.float32)
    lo = np.array([-1.0, -2.0, -0.5], np.float32)
    hi = np.array([2.0, 1.5, 0.7], np.float32)
    p = tmp_path / "proj.npz"
    extra = {"explained": np.array([0.7, 0.2, 0.1], np.float32)} if explained else {}
    np.savez(p, mean=mean, comps=comps, lo=lo, hi=hi, **extra)
    return str(p), mean, comps, lo, hi


def test_matches_exporter_math_with_scaling_and_clamp(tmp_path):
    path, mean, comps, lo, hi = _proj(tmp_path)
    # input_global_minmax=False isolates the projection math (the min-max is tested below).
    node = FixedPCAProjection(projection_path=path, input_global_minmax=False)
    rng = np.random.default_rng(1)
    cube = (rng.normal(size=(2, 4, 5, 7)) * 3).astype(np.float32)
    out = node(data=torch.from_numpy(cube))["projected"].numpy()
    ref = (cube.reshape(-1, 7) - mean) @ comps.T
    ref = np.clip((ref - lo) / (hi - lo), 0, 1).reshape(2, 4, 5, 3)
    assert out.shape == (2, 4, 5, 3) and out.dtype == np.float32
    assert np.allclose(out, ref, atol=1e-5)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_raw_projection_when_scaling_disabled(tmp_path):
    path, mean, comps, _, _ = _proj(tmp_path)
    node = FixedPCAProjection(
        projection_path=path, scale_to_unit=False, clamp01=False, input_global_minmax=False
    )
    rng = np.random.default_rng(2)
    cube = (rng.normal(size=(1, 3, 3, 7)) * 2).astype(np.float32)
    out = node(data=torch.from_numpy(cube))["projected"].numpy()
    ref = ((cube.reshape(-1, 7) - mean) @ comps.T).reshape(1, 3, 3, 3)
    assert np.allclose(out, ref, atol=1e-5)


def test_input_global_minmax_is_scale_invariant_and_idempotent(tmp_path):
    # The projection was fitted on cubes min-maxed to [0, 1] as a whole, so it must not depend on
    # the caller's absolute scale (a source may deliver raw-scale reflectance). With the min-max on
    # (the default), a cube and any positive rescaling of it project identically, and a [0, 1]
    # cube passes the min-max unchanged.
    path, *_ = _proj(tmp_path)
    node = FixedPCAProjection(projection_path=path)
    rng = np.random.default_rng(3)
    base = rng.random(size=(1, 4, 4, 7)).astype(np.float32)  # in [0, 1)
    out_unit = node(data=torch.from_numpy(base))["projected"].numpy()
    out_scaled = node(data=torch.from_numpy(base * 10000.0 + 5.0))["projected"].numpy()
    assert np.allclose(out_unit, out_scaled, atol=1e-5)
    b01 = base.copy()
    b01.flat[0], b01.flat[1] = 0.0, 1.0  # min 0, max 1: the min-max is the identity
    gm = FixedPCAProjection._global_minmax(torch.from_numpy(b01)).numpy()
    assert np.allclose(gm, b01, atol=1e-6)


def test_hparams_round_trip_and_initialized(tmp_path):
    path, *_ = _proj(tmp_path)
    node = FixedPCAProjection(projection_path=path, scale_to_unit=True, clamp01=True)
    assert node.hparams["projection_path"] == path
    assert node.hparams["scale_to_unit"] is True and node.hparams["clamp01"] is True
    assert node.hparams["input_global_minmax"] is True
    assert node._statistically_initialized is True


def test_port_contract_exposes_only_the_projection(tmp_path):
    path, *_ = _proj(tmp_path)
    node = FixedPCAProjection(projection_path=path)
    assert set(FixedPCAProjection.OUTPUT_SPECS) == {"projected"}
    out = node(data=torch.rand(2, 5, 6, 7))
    assert set(out) == {"projected"}
    assert out["projected"].shape == (2, 5, 6, 3) and out["projected"].dtype == torch.float32


def test_the_file_is_the_fit(tmp_path):
    # TrainablePCA asks for a statistical pass; the fixed projection must not be refit by one.
    path, *_ = _proj(tmp_path)
    assert TrainablePCA(num_channels=7, n_components=3).requires_initial_fit is True
    assert FixedPCAProjection(projection_path=path).requires_initial_fit is False


def test_rebuilds_from_its_recorded_hparams(tmp_path):
    # The recorded hparams include the parent's num_channels / n_components; a rebuild from them
    # must not clash with the values derived from the file.
    path, *_ = _proj(tmp_path)
    node = FixedPCAProjection(projection_path=path)
    assert node.hparams["num_channels"] == 7 and node.hparams["n_components"] == 3
    rebuilt = FixedPCAProjection(**node.hparams)
    cube = torch.rand(1, 4, 4, 7, generator=torch.Generator().manual_seed(4))
    assert torch.equal(rebuilt(data=cube)["projected"], node(data=cube)["projected"])


def test_mismatched_file_is_rejected(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(
        p,
        mean=np.zeros(5, np.float32),
        comps=np.zeros((3, 7), np.float32),
        lo=np.zeros(3, np.float32),
        hi=np.ones(3, np.float32),
    )
    with pytest.raises(ValueError, match="comps"):
        FixedPCAProjection(projection_path=str(p))


def test_missing_explained_ratios_zero_the_buffer(tmp_path):
    path, *_ = _proj(tmp_path, explained=False)
    node = FixedPCAProjection(projection_path=path)
    assert torch.equal(node._explained_variance, torch.zeros(3))


# ----- reload smoke ---------------------------------------------------------------------------


class _CubeSource(Node):
    """Module-scope test source emitting a fixed raw-scale cube."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def forward(self, **_) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(5)
        return {"cube": torch.rand(1, 6, 5, 7, generator=g) * 4000.0}


def test_pipeline_round_trips_through_yaml(tmp_path):
    path, *_ = _proj(tmp_path)
    pipe = CuvisPipeline("fixed_pca_reload")
    src = _CubeSource(name="src")
    pca = FixedPCAProjection(projection_path=path, name="pca")
    pipe.connect(src.outputs.cube, pca.inputs.data)
    before = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("pca", "projected")]
    yaml_path = tmp_path / "fixed_pca.yaml"
    pipe.save_to_file(str(yaml_path))
    rebuilt = PipelineBuilder().build_from_config(str(yaml_path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("pca", "projected")]
    assert torch.equal(after, before)
