"""Tests for the Standardize node: every axis, stats mode and layout against inline references."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec
from sklearn.preprocessing import StandardScaler

from cuvis_ai.node.pretreatments.standardize import IMAGENET_MEAN, IMAGENET_STD, Standardize
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.factory import PipelineBuilder
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit

ATOL = 1e-6
RTOL = 1e-5
EPS = 1e-8

# Dims each axis reduces per sample, per layout; the fitted statistics also reduce the batch dim 0.
REDUCED = {
    "bhwc": {"channel": [1, 2], "spatial": [1, 2], "spectrum": [3], "global": [1, 2, 3]},
    "nchw": {"channel": [2, 3], "spatial": [2, 3], "spectrum": [1], "global": [1, 2, 3]},
}


# ----- helpers --------------------------------------------------------------------------------


def _cube(layout: str = "bhwc", seed: int = 0, b: int = 2) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    shape = (b, 5, 4, 6) if layout == "bhwc" else (b, 6, 5, 4)
    return torch.rand(*shape, generator=g)


def _frames(layout: str = "bhwc", n: int = 3) -> list[torch.Tensor]:
    return [_cube(layout, seed=10 + i) for i in range(n)]


def _stream(frames, folds=None):
    for i, x in enumerate(frames):
        batch = {"cube": x}
        if folds is not None:
            batch["fold"] = folds[i]
        yield batch


def _reduce_dims(axis, layout: str) -> list[int]:
    return list(axis) if isinstance(axis, list) else REDUCED[layout][axis]


def _ref_sample(x: torch.Tensor, dims: list[int], ddof: int) -> torch.Tensor:
    xd = x.double()
    mean = xd.mean(dim=dims, keepdim=True)
    std = xd.std(dim=dims, correction=ddof, keepdim=True).clamp_min(EPS)
    return ((xd - mean) / std).float()


def _ref_fit(frames, x: torch.Tensor, dims: list[int], ddof: int) -> torch.Tensor:
    data = torch.cat(frames, dim=0).double()
    reduce = [0, *dims]
    mean = data.mean(dim=reduce, keepdim=True)
    std = data.std(dim=reduce, correction=ddof, keepdim=True).clamp_min(EPS)
    return ((x.double() - mean) / std).float()


def _fitted(frames=None, **kw) -> Standardize:
    node = Standardize(**kw)
    node.statistical_initialization(_stream(frames if frames is not None else _frames()))
    return node


def _close(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.allclose(a, b, atol=ATOL, rtol=RTOL)


# ----- defaults and validation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("axis", "stats"),
    [
        ("channel", "fit"),
        ("global", "fit"),
        ("spectrum", "sample"),
        ("spatial", "sample"),
        ([1, 2, 3], "sample"),
    ],
)
def test_default_stats_follow_the_axis(axis, stats):
    node = Standardize(axis=axis)
    assert node.hparams["stats"] == stats
    assert node.requires_initial_fit is (stats == "fit")


def test_given_stats_need_no_fit():
    assert (
        Standardize(stats="given", mean=[0.5], std=[2.0], axis="global").requires_initial_fit
        is False
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"with_mean": False, "with_std": False},
        {"axis": "pixels"},
        {"axis": [0, 1]},
        {"axis": []},
        {"ddof": 2},
        {"layout": "chw"},
        {"stats": "running"},
        {"stats": "given"},
        {"stats": "given", "mean": [0.1, 0.2], "std": [1.0]},
        {"stats": "given", "axis": "spectrum", "mean": [0.1], "std": [1.0]},
        {"axis": "spectrum", "exclude_folds": [1]},
        {"axis": "spatial", "max_initialization_frames": 4},
        {"max_initialization_frames": 0},
        {"mean": [0.1]},
        {"eps_mode": "mul"},
        # A fit has to pool over H and W, or the statistics are tied to one frame size.
        {"axis": "spectrum", "stats": "fit"},
        {"axis": [3], "stats": "fit"},
        {"axis": [1, 3], "stats": "fit"},
        {"axis": [2, 3], "stats": "fit"},
        {"axis": "spectrum", "stats": "fit", "layout": "nchw"},
        {"axis": [1, 2], "stats": "fit", "layout": "nchw"},
    ],
)
def test_invalid_configurations_are_rejected(kwargs):
    with pytest.raises(ValueError):
        Standardize(**kwargs)


# ----- parity: fit, sample, given ----------------------------------------------------------------


@pytest.mark.parametrize("layout", ["bhwc", "nchw"])
@pytest.mark.parametrize("axis", ["channel", "spatial", "global"])
@pytest.mark.parametrize("ddof", [0, 1])
@torch.no_grad()
def test_fit_matches_the_pooled_reference(layout, axis, ddof):
    frames = _frames(layout)
    node = _fitted(frames, axis=axis, stats="fit", ddof=ddof, layout=layout)
    x = _cube(layout, seed=99)
    expected = _ref_fit(frames, x, _reduce_dims(axis, layout), ddof)
    assert _close(node(cube=x)["cube"], expected)


@pytest.mark.parametrize("layout", ["bhwc", "nchw"])
@pytest.mark.parametrize("axis", ["spectrum", "spatial", "global", [1, 2, 3], [3]])
@pytest.mark.parametrize("ddof", [0, 1])
@torch.no_grad()
def test_sample_matches_the_per_sample_reference(layout, axis, ddof):
    x = _cube(layout, seed=3)
    node = Standardize(axis=axis, stats="sample", ddof=ddof, layout=layout)
    expected = _ref_sample(x, _reduce_dims(axis, layout), ddof)
    assert _close(node(cube=x)["cube"], expected)


@torch.no_grad()
def test_given_nchw_reproduces_imagenet_normalization():
    x = torch.rand(2, 3, 8, 8, generator=torch.Generator().manual_seed(4))
    node = Standardize(stats="given", layout="nchw", mean=IMAGENET_MEAN, std=IMAGENET_STD)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    assert _close(node(cube=x)["cube"], (x - mean) / std)


@torch.no_grad()
def test_given_bhwc_per_channel_and_global_scalar():
    x = _cube(seed=5)
    mean, std = [0.1 * i for i in range(6)], [1.0 + i for i in range(6)]
    node = Standardize(stats="given", mean=mean, std=std)
    assert _close(node(cube=x)["cube"], (x - torch.tensor(mean)) / torch.tensor(std))
    scalar = Standardize(stats="given", axis="global", mean=[0.031223], std=[0.049720])
    assert _close(scalar(cube=x)["cube"], (x - 0.031223) / 0.049720)


@torch.no_grad()
def test_channel_fit_on_one_by_one_frames_matches_sklearn():
    # The mean-spectra case: every frame is a single pixel.
    g = torch.Generator().manual_seed(6)
    frames = [torch.rand(4, 1, 1, 61, generator=g) for _ in range(5)]
    node = _fitted(frames, axis="channel", ddof=0)
    rows = torch.cat(frames).reshape(-1, 61).double().numpy()
    expected = StandardScaler().fit(rows).transform(rows)
    got = node(cube=torch.cat(frames))["cube"].reshape(-1, 61).double().numpy()
    assert np.allclose(got, expected, atol=ATOL, rtol=RTOL)


# ----- with_mean / with_std / eps ---------------------------------------------------------------


@torch.no_grad()
def test_with_std_false_only_centres():
    frames = _frames()
    node = _fitted(frames, axis="channel", with_std=False)
    x = _cube(seed=7)
    mean = torch.cat(frames).double().mean(dim=[0, 1, 2]).float()
    assert _close(node(cube=x)["cube"], x - mean)


@torch.no_grad()
def test_with_mean_false_only_scales():
    frames = _frames()
    node = _fitted(frames, axis="channel", with_mean=False)
    x = _cube(seed=8)
    std = torch.cat(frames).double().std(dim=[0, 1, 2], correction=1).float()
    assert _close(node(cube=x)["cube"], x / std)


@pytest.mark.parametrize("axis", [[1, 2], [1, 2, 3]])
@torch.no_grad()
def test_fit_on_a_dim_list_matches_the_pooled_reference(axis):
    frames = _frames()
    node = _fitted(frames, axis=axis, stats="fit")
    x = _cube(seed=98)
    assert _close(node(cube=x)["cube"], _ref_fit(frames, x, axis, 1))


@torch.no_grad()
def test_fitted_statistics_apply_to_another_frame_size():
    node = _fitted(axis="channel")
    assert node.mean.shape == (1, 1, 1, 6)
    out = node(cube=torch.rand(1, 20, 9, 6))["cube"]
    assert out.shape == (1, 20, 9, 6)


@torch.no_grad()
def test_eps_mode_add_reproduces_the_added_eps_bit_for_bit():
    # Reflectance-like cube; the old per-sample z-score divided by (std + eps), ddof 0.
    x = 0.1 + 0.4 * torch.rand(2, 64, 64, 61, generator=torch.Generator().manual_seed(12))
    mean = x.mean(dim=[1, 2], keepdim=True)
    std = x.std(dim=[1, 2], keepdim=True, unbiased=False)
    expected = (x - mean) / (std + 1e-6)
    node = Standardize(axis="spatial", ddof=0, eps=1e-6, eps_mode="add")
    assert torch.equal(node(cube=x)["cube"], expected)
    floored = Standardize(axis="spatial", ddof=0, eps=1e-6)(cube=x)["cube"]
    assert not torch.equal(floored, expected)


@torch.no_grad()
def test_eps_mode_add_with_fitted_and_given_statistics():
    frames = _frames()
    node = _fitted(frames, axis="channel", eps=1e-3, eps_mode="add")
    x = _cube(seed=13)
    assert torch.equal(node(cube=x)["cube"], (x - node.mean) / (node.std + 1e-3))
    given = Standardize(stats="given", axis="global", mean=[0.5], std=[2.0], eps_mode="add")
    assert _close(given(cube=x)["cube"], (x - 0.5) / (2.0 + EPS))


@pytest.mark.parametrize("eps_mode", ["floor", "add"])
@torch.no_grad()
def test_flat_spectrum_is_clamped_not_nan(eps_mode):
    x = torch.full((1, 2, 2, 6), 0.5)  # exact in float32, so the centred values are exactly 0
    out = Standardize(axis="spectrum", eps_mode=eps_mode)(cube=x)["cube"]
    assert torch.isfinite(out).all() and torch.equal(out, torch.zeros_like(x))


# ----- folds, frame cap, empty stream, unfitted forward -----------------------------------------


def test_fold_port_exists_only_with_exclude_folds():
    assert "fold" not in Standardize().INPUT_SPECS
    assert "fold" in Standardize(exclude_folds=[1]).INPUT_SPECS
    assert Standardize.INPUT_SPECS["fold"].optional is True


@torch.no_grad()
def test_exclude_folds_keeps_held_out_frames_out_of_the_fit():
    frames = _frames(n=3)
    shifted = [f.clone() for f in frames]
    shifted[1][0] += 100.0  # frame 0 of batch 1 is in fold 1
    folds = [torch.tensor([2, 3], dtype=torch.int32), torch.tensor([1, 4], dtype=torch.int32)]
    folds.append(torch.tensor([5, 2], dtype=torch.int32))
    node = Standardize(axis="global", ddof=0, exclude_folds=[1])
    node.statistical_initialization(_stream(shifted, folds))
    kept = torch.cat([frames[0], frames[1][1:], frames[2]])
    x = _cube(seed=9)
    assert _close(node(cube=x)["cube"], _ref_fit([kept], x, [1, 2, 3], 0))


def test_exclude_folds_without_fold_in_the_batch_raises():
    node = Standardize(exclude_folds=[1])
    with pytest.raises(ValueError, match="fold"):
        node.statistical_initialization(_stream(_frames()))


@torch.no_grad()
def test_max_initialization_frames_slices_the_last_batch_exactly():
    frames = _frames(n=2)  # four frames
    tail = [_cube(seed=50) + 100.0]
    node = _fitted([*frames, *tail], axis="channel", max_initialization_frames=3)
    x = _cube(seed=11)
    used = torch.cat([frames[0], frames[1][:1]])
    assert _close(node(cube=x)["cube"], _ref_fit([used], x, [1, 2], 1))


def test_empty_stream_raises_and_stays_unfitted():
    node = Standardize()
    with pytest.raises(RuntimeError):
        node.statistical_initialization(_stream([]))
    assert node._statistically_initialized is False


def test_forward_before_fit_raises():
    with pytest.raises(RuntimeError):
        Standardize()(cube=_cube())


# ----- hparams, YAML and weights round trips ------------------------------------------------------

_MODES = [
    {"axis": "channel"},
    {"axis": "global", "ddof": 0, "exclude_folds": [1], "max_initialization_frames": 8},
    {"axis": "spectrum"},
    {"axis": "spatial", "ddof": 0, "eps": 1e-6, "eps_mode": "add"},
    {"axis": [1, 2, 3]},
    {"stats": "given", "layout": "nchw", "mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
    {"axis": "channel", "with_std": False},
]


@pytest.mark.parametrize("kwargs", _MODES)
def test_hparams_are_json_and_rebuild_the_same_node(kwargs):
    node = Standardize(**kwargs)
    json.dumps(node.hparams)
    rebuilt = Standardize(**node.hparams)
    assert rebuilt.hparams == node.hparams
    assert rebuilt.INPUT_SPECS.keys() == node.INPUT_SPECS.keys()


class _CubeSource(Node):
    """Module-scope test source emitting a fixed BHWC cube."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"cube": PortSpec(dtype=torch.float32, shape=(-1, -1, -1, -1))}

    def forward(self, **_) -> dict[str, torch.Tensor]:
        return {"cube": _cube(seed=12)}


@pytest.mark.parametrize(
    "kwargs", [{"axis": "channel"}, {"axis": "global", "ddof": 0}, {"axis": "spectrum"}]
)
def test_pipeline_round_trips_through_yaml_and_weights(tmp_path, kwargs):
    pipe = CuvisPipeline("standardize_reload")
    src = _CubeSource(name="src")
    std = Standardize(name="standardize", **kwargs)
    if std.requires_initial_fit:
        std.statistical_initialization(_stream(_frames()))
    pipe.connect(src.outputs.cube, std.inputs.cube)
    before = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)[("standardize", "cube")]
    path = tmp_path / "standardize.yaml"
    pipe.save_to_file(str(path))
    if std.requires_initial_fit:
        rebuilt = CuvisPipeline.load_pipeline(str(path), weights_path=str(path.with_suffix(".pt")))
    else:
        rebuilt = PipelineBuilder().build_from_config(str(path))
    after = rebuilt.forward(batch={}, stage=ExecutionStage.INFERENCE)[("standardize", "cube")]
    assert torch.equal(after, before)


@torch.no_grad()
def test_fitted_weights_load_into_a_fresh_node():
    node = _fitted(axis="channel")
    state = node.state_dict()
    fresh = Standardize(axis="channel")
    assert fresh.mean.shape == (0,)
    fresh.load_state_dict(state)
    assert fresh.mean.shape == (1, 1, 1, 6) and fresh._statistically_initialized is True
    x = _cube(seed=13)
    assert torch.equal(fresh(cube=x)["cube"], node(cube=x)["cube"])


# ----- freeze / unfreeze lifecycle ------------------------------------------------------------------


def test_freeze_unfreeze_cycle_keeps_values():
    node = _fitted(axis="channel")
    before = {n: getattr(node, n).clone() for n in Standardize.TRAINABLE_BUFFERS}
    node.unfreeze()
    for n in Standardize.TRAINABLE_BUFFERS:
        p = getattr(node, n)
        assert isinstance(p, torch.nn.Parameter) and p.requires_grad
        assert torch.equal(p.detach(), before[n])
    node.freeze()
    for n in Standardize.TRAINABLE_BUFFERS:
        b = getattr(node, n)
        assert not isinstance(b, torch.nn.Parameter) and torch.equal(b, before[n])


def test_gradient_reaches_the_unfrozen_statistics():
    node = _fitted(axis="channel")
    node.unfreeze()
    node(cube=_cube(seed=14))["cube"].pow(2).sum().backward()
    assert node.mean.grad is not None and node.std.grad is not None


@torch.no_grad()
def test_unfreeze_then_fit_keeps_parameters():
    node = Standardize(axis="channel")
    node.unfreeze()
    node.statistical_initialization(_stream(_frames()))
    assert isinstance(node.mean, torch.nn.Parameter) and node.mean.shape == (1, 1, 1, 6)
    reference = _fitted(axis="channel")
    x = _cube(seed=15)
    assert torch.equal(node(cube=x)["cube"], reference(cube=x)["cube"])


@torch.no_grad()
def test_unfrozen_fitted_state_loads_into_an_unfrozen_fresh_node():
    node = _fitted(axis="channel")
    node.unfreeze()
    state = node.state_dict()
    fresh = Standardize(axis="channel")
    fresh.unfreeze()
    fresh.load_state_dict(state)
    assert isinstance(fresh.std, torch.nn.Parameter) and fresh.std.shape == (1, 1, 1, 6)
    fresh.freeze()
    x = _cube(seed=16)
    assert torch.equal(fresh(cube=x)["cube"], node(cube=x)["cube"])


# ----- port contract, non-contiguous input, export ------------------------------------------------


@torch.no_grad()
def test_port_contract():
    node = _fitted(axis="channel")
    out = node(cube=_cube(seed=17))
    assert set(out) == set(Standardize.OUTPUT_SPECS) == {"cube"}
    assert out["cube"].shape == (2, 5, 4, 6) and out["cube"].dtype == torch.float32


@pytest.mark.parametrize("kwargs", [{"axis": "spectrum"}, {"axis": "spatial", "ddof": 0}])
@torch.no_grad()
def test_non_contiguous_input_gives_the_contiguous_result(kwargs):
    x = _cube(seed=18).transpose(1, 2)
    assert not x.is_contiguous()
    node = Standardize(**kwargs)
    assert _close(node(cube=x)["cube"], node(cube=x.contiguous())["cube"])


class _ExportProbe(torch.nn.Module):
    def __init__(self, node: Standardize):
        super().__init__()
        self.node = node

    def forward(self, cube: torch.Tensor) -> torch.Tensor:
        return self.node(cube=cube)["cube"]


@pytest.mark.parametrize("kwargs", [{"axis": "channel"}, {"axis": "spectrum"}])
def test_export_smoke(kwargs):
    node = Standardize(**kwargs)
    if node.requires_initial_fit:
        node.statistical_initialization(_stream(_frames()))
    node.eval()
    program = torch.export.export(_ExportProbe(node), (_cube(seed=19),))
    other = _cube(seed=20)
    assert _close(program.module()(other), node(cube=other)["cube"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs cuda")
@torch.no_grad()
def test_cuda_fit_matches_cpu_fit():
    frames = _frames()
    cpu = _fitted(frames, axis="channel")
    gpu = Standardize(axis="channel").to("cuda")
    gpu.statistical_initialization(_stream([f.cuda() for f in frames]))
    x = _cube(seed=21)
    assert torch.allclose(gpu(cube=x.cuda())["cube"].cpu(), cpu(cube=x)["cube"], atol=1e-5)


# ----- no packaged config, notebook or docs page names a removed class -------------------------

_REMOVED = (
    "MeanCenter",
    "UnitVarianceScaling",
    "SNVCorrection",
    "ZScoreNormalizer",
    "ChannelNormalizeNode",
    "ZScoreNormalizerGlobal",
)


def test_no_packaged_config_notebook_or_doc_names_a_removed_class():
    root = Path(__file__).resolve().parents[2]
    pattern = re.compile(r"\b(" + "|".join(_REMOVED) + r")\b")
    files = [
        *root.glob("cuvis_ai/configs/**/*.yaml"),
        *root.glob("notebooks/**/*.ipynb"),
        *root.glob("docs/**/*.md"),
    ]
    hits = [
        f"{p.relative_to(root)}: {m.group(1)}"
        for p in files
        for m in pattern.finditer(p.read_text(encoding="utf-8", errors="ignore"))
    ]
    assert not hits, "\n".join(hits)
