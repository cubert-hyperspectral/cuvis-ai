"""MaskComposite and ScoreMapSuppression: golden rules, port contract, fan-in, hparam validation."""

from __future__ import annotations

import json

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai.node.mask_ops import MaskComposite, ScoreMapSuppression
from cuvis_ai_core.node.node import Node
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

pytestmark = pytest.mark.unit

B, H, W = 2, 5, 7


def _masks() -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.zeros(B, H, W, 1, dtype=torch.bool)
    a[0, 0, 0, 0] = True  # frame 0 only
    b = torch.zeros(B, H, W, 1, dtype=torch.bool)
    b[:, 1, 1, 0] = True  # both frames
    return a, b


class _MaskSource(Node):
    """Module-scope test source emitting a constant mask (one pixel set, or none)."""

    _category = NodeCategory.SOURCE
    _tags = frozenset({NodeTag.TORCH})
    INPUT_SPECS: dict[str, PortSpec] = {}
    OUTPUT_SPECS = {"decisions": PortSpec(dtype=torch.bool, shape=(-1, -1, -1, 1))}

    def __init__(self, pixel: int = -1, **kwargs) -> None:
        super().__init__(pixel=pixel, **kwargs)
        self.pixel = int(pixel)

    def forward(self, **_) -> dict[str, torch.Tensor]:
        m = torch.zeros(1, H * W, dtype=torch.bool)
        if self.pixel >= 0:
            m[0, self.pixel] = True
        return {"decisions": m.view(1, H, W, 1)}


# ----- 6. MaskComposite -----------------------------------------------------------------------


def test_composite_labels_and_levels_largest_wins_on_overlap():
    shell = torch.zeros(B, H, W, 1, dtype=torch.bool)
    shell[:, 0:2, 0:3, 0] = True
    fo = torch.zeros(B, H, W, 1, dtype=torch.bool)
    fo[0, 1:3, 2:4, 0] = True  # overlaps the shell at (1, 2) in frame 0 only
    out = MaskComposite(labels=[1, 2], levels=[0.5, 1.0])(decisions=[shell, fo])
    exp_mask = torch.zeros(B, H, W, dtype=torch.int32)
    exp_mask[:, 0:2, 0:3] = 1
    exp_mask[0, 1:3, 2:4] = 2
    exp_level = torch.zeros(B, H, W, 1)
    exp_level[:, 0:2, 0:3, 0] = 0.5
    exp_level[0, 1:3, 2:4, 0] = 1.0
    assert torch.equal(out["mask"], exp_mask)
    assert torch.equal(out["scores"], exp_level)
    # the order of the lists, not of the masks' size, decides: a larger label on the first mask wins
    flipped = MaskComposite(labels=[2, 1], levels=[1.0, 0.5])(decisions=[shell, fo])
    assert int(flipped["mask"][0, 1, 2]) == 2 and float(flipped["scores"][0, 1, 2, 0]) == 1.0


def test_composite_defaults_are_shell_1_fo_2():
    node = MaskComposite()
    assert node.labels == [1, 2] and node.levels == [0.5, 1.0]
    empty = torch.zeros(B, H, W, 1, dtype=torch.bool)
    out = node(decisions=[empty, empty])
    assert not out["mask"].any() and not out["scores"].any()


def test_composite_counts_a_pixel_where_any_channel_is_set():
    m = torch.zeros(B, H, W, 2, dtype=torch.bool)
    m[0, 2, 3, 1] = True
    out = MaskComposite(labels=[4], levels=[0.25])(decisions=m)
    assert out["mask"].flatten().nonzero().flatten().tolist() == [2 * W + 3]
    assert float(out["scores"][0, 2, 3, 0]) == 0.25


def test_composite_port_contract():
    a, b = _masks()
    out = MaskComposite()(decisions=[a, b])
    assert set(out) == set(MaskComposite.OUTPUT_SPECS)
    assert out["mask"].shape == (B, H, W) and out["mask"].dtype == torch.int32
    assert out["scores"].shape == (B, H, W, 1) and out["scores"].dtype == torch.float32


def test_composite_mask_count_and_shape_mismatch_raise():
    a, b = _masks()
    with pytest.raises(ValueError):
        MaskComposite()(decisions=[a])  # two labels configured, one mask connected
    with pytest.raises(ValueError):
        MaskComposite()(decisions=[a, torch.zeros(B, H, W + 1, 1, dtype=torch.bool)])


def test_composite_labels_follow_connection_order():
    pipe = CuvisPipeline("composite_order")
    a, b = _MaskSource(3, name="a"), _MaskSource(9, name="b")
    comp = MaskComposite(labels=[1, 2], levels=[0.5, 1.0], name="comp")
    for src in (a, b):
        pipe.connect(src.outputs.decisions, comp.inputs.decisions)
    out = pipe.forward(batch={}, stage=ExecutionStage.INFERENCE)
    mask = out[("comp", "mask")].flatten()
    assert int(mask[3]) == 1 and int(mask[9]) == 2 and int(mask.count_nonzero()) == 2


@pytest.mark.parametrize(
    "bad",
    [
        {"labels": [], "levels": []},
        {"labels": [1, 2], "levels": [1.0]},
        {"labels": [0, 2]},
        {"labels": [1, True]},
        {"labels": [1.5, 2]},
        {"levels": [-0.1, 1.0]},
        {"levels": [0.5, float("nan")]},
        {"levels": [0.5, float("inf")]},
    ],
)
def test_composite_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        MaskComposite(**bad)


def test_composite_hparams_round_trip_json():
    node = MaskComposite(labels=[1, 2], levels=[0.4, 1.0], name="comp")
    assert node.hparams["labels"] == [1, 2] and node.hparams["levels"] == [0.4, 1.0]
    json.dumps(node.hparams)


# ----- 8. ScoreMapSuppression -----------------------------------------------------------------


def _block_mask() -> torch.Tensor:
    """A 3 x 3 block (rows 1-3, cols 2-4) set in both frames of a [B, 5, 7, 1] mask."""
    m = torch.zeros(B, H, W, 1, dtype=torch.bool)
    m[:, 1:4, 2:5, 0] = True
    return m


def test_suppression_zeroes_the_mask_without_erosion():
    s = torch.rand(B, H, W, 1, generator=torch.Generator().manual_seed(3)) + 0.5
    out = ScoreMapSuppression(weight=1.0, erode_px=0)(scores=s, mask=_block_mask())["scores"]
    exp = s.clone()
    exp[:, 1:4, 2:5, 0] = 0.0
    assert torch.equal(out, exp)


def test_suppression_erodes_the_mask_by_erode_px():
    s = torch.ones(B, H, W, 1)
    out = ScoreMapSuppression(weight=1.0, erode_px=1)(scores=s, mask=_block_mask())["scores"]
    exp = torch.ones(B, H, W, 1)
    exp[:, 2, 3, 0] = 0.0  # only the centre of the 3 x 3 block has no outside pixel within 1 px
    assert torch.equal(out, exp)
    # a 2 px margin leaves nothing of a 3 x 3 block
    out2 = ScoreMapSuppression(weight=1.0, erode_px=2)(scores=s, mask=_block_mask())["scores"]
    assert torch.equal(out2, s)


def test_suppression_scales_by_one_minus_weight():
    s = torch.full((B, H, W, 1), 2.0)
    out = ScoreMapSuppression(weight=0.25, erode_px=0)(scores=s, mask=_block_mask())["scores"]
    assert torch.allclose(out[:, 1:4, 2:5], torch.full_like(out[:, 1:4, 2:5], 1.5))
    assert torch.equal(out[:, 0], s[:, 0])


def test_suppression_weight_zero_is_identity():
    s = torch.rand(B, H, W, 1, generator=torch.Generator().manual_seed(4))
    out = ScoreMapSuppression(weight=0.0)(scores=s, mask=_block_mask())["scores"]
    assert torch.equal(out, s)


def test_suppression_image_border_does_not_erode_the_mask():
    s = torch.ones(B, H, W, 1)
    full = torch.ones(B, H, W, 1, dtype=torch.bool)
    out = ScoreMapSuppression(weight=1.0, erode_px=2)(scores=s, mask=full)["scores"]
    assert torch.equal(out, torch.zeros_like(s))


def test_suppression_counts_a_pixel_where_any_mask_channel_is_set():
    s = torch.ones(B, H, W, 1)
    m = torch.zeros(B, H, W, 2, dtype=torch.bool)
    m[:, 0, 0, 1] = True
    out = ScoreMapSuppression(weight=1.0, erode_px=0)(scores=s, mask=m)["scores"]
    assert out[:, 0, 0, 0].tolist() == [0.0, 0.0] and float(out.sum()) == B * (H * W - 1)


def test_suppression_is_differentiable_in_the_scores():
    s = torch.rand(B, H, W, 1, generator=torch.Generator().manual_seed(5)).requires_grad_(True)
    out = ScoreMapSuppression(weight=0.5, erode_px=0)(scores=s, mask=_block_mask())["scores"]
    out.sum().backward()
    exp = torch.ones(B, H, W, 1)
    exp[:, 1:4, 2:5, 0] = 0.5
    assert torch.equal(s.grad, exp)


def test_suppression_port_contract():
    s = torch.rand(B, H, W, 1)
    out = ScoreMapSuppression()(scores=s, mask=_block_mask())
    spec = ScoreMapSuppression.OUTPUT_SPECS["scores"]
    assert set(out) == {"scores"}
    assert out["scores"].shape == s.shape and out["scores"].dtype == spec.dtype


def test_suppression_shape_mismatch_raises():
    with pytest.raises(ValueError):
        ScoreMapSuppression()(
            scores=torch.rand(B, H, W, 1), mask=torch.zeros(B, H, W + 1, 1, dtype=torch.bool)
        )


@pytest.mark.parametrize(
    "bad",
    [
        {"weight": -0.1},
        {"weight": 1.5},
        {"weight": True},
        {"weight": "1"},
        {"erode_px": -1},
        {"erode_px": 1.5},
        {"erode_px": True},
    ],
)
def test_suppression_invalid_hparams_raise(bad):
    with pytest.raises(ValueError):
        ScoreMapSuppression(**bad)


def test_suppression_hparams_round_trip_json():
    hp = ScoreMapSuppression(weight=1, erode_px=3, name="sup").hparams
    assert hp["weight"] == 1.0 and hp["erode_px"] == 3
    json.dumps(hp)
    default = ScoreMapSuppression().hparams
    assert default["weight"] == 1.0 and default["erode_px"] == 4
