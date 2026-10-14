"""FrameScoreGate: gating, frame score, mask mode, decisions mask, port contract, hparams."""

from __future__ import annotations

import pytest
import torch

from cuvis_ai.node.deciders.frame_score_gate import FrameScoreGate

pytestmark = pytest.mark.unit


def _scores() -> torch.Tensor:
    # [2, 4, 4, 1]: frame 0 all 0.1 (clean); frame 1 has 4 hot pixels = 5.0, rest 0.1.
    x = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    x[1, 0, :, 0] = 5.0  # 4 pixels of 16 -> matches topk_frac=0.25 (k=4)
    return x


def test_gate_heatmap_passthrough_and_blank():
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="heatmap")
    out = node(scores=_scores())
    # frame 0: top-4 mean = 0.1 <= 1.0 -> blanked; frame 1: top-4 mean = 5.0 > 1.0 -> passed through
    assert torch.equal(out["scores"][0], torch.zeros(4, 4, 1))
    assert torch.equal(out["scores"][1], _scores()[1])
    assert torch.allclose(out["frame_score"], torch.tensor([0.1, 5.0]), atol=1e-6)
    assert torch.equal(out["passed"], torch.tensor([0, 1], dtype=torch.int32))


def test_gate_mask_mode_binarises_passing_frame():
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="mask")
    out = node(scores=_scores())
    assert torch.equal(out["scores"][0], torch.zeros(4, 4, 1))  # clean frame blank
    expected = (_scores()[1] > 1.0).float()  # the 4 hot pixels -> 1.0, rest 0.0
    assert torch.equal(out["scores"][1], expected)
    assert float(out["scores"][1].sum()) == 4.0


def test_alarm_scores_drives_gate_of_a_different_display_map():
    # Display map is low-amplitude on both frames (looks clean); the alarm map is hot on frame 1
    # only. The gate must pass the DISPLAY map through on frame 1 (alarm fired) and blank frame 0.
    display = torch.full((2, 4, 4, 1), 0.3, dtype=torch.float32)
    display[1, 2, :, 0] = 0.4  # low-amplitude localisation, never above `threshold` on its own
    alarm = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    alarm[1, 0, :, 0] = 5.0  # frame 1 alarm top-4 mean = 5.0
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="heatmap")
    out = node(scores=display, alarm_scores=alarm)
    # frame 0: alarm 0.1 <= 1.0 -> blanked; frame 1: alarm fired -> display passed through as-is
    assert torch.equal(out["scores"][0], torch.zeros(4, 4, 1))
    assert torch.equal(out["scores"][1], display[1])
    # the frame score comes from the alarm map
    assert torch.allclose(out["frame_score"], torch.tensor([0.1, 5.0]), atol=1e-6)
    assert torch.equal(out["passed"], torch.tensor([0, 1], dtype=torch.int32))


def test_alarm_scores_batch_mismatch_raises():
    node = FrameScoreGate(threshold=1.0)
    with pytest.raises(RuntimeError):
        node(scores=torch.rand(2, 4, 4, 1), alarm_scores=torch.rand(3, 4, 4, 1))


def test_mask_threshold_binarises_display_on_its_own_scale():
    # Alarm on a hot map; the display map is on another scale, binarised at mask_threshold
    # (not at `threshold`).
    display = torch.full((1, 4, 4, 1), 0.2, dtype=torch.float32)
    display[0, 1, :, 0] = 0.8  # 4 pixels above 0.5
    alarm = torch.full((1, 4, 4, 1), 5.0, dtype=torch.float32)  # alarm always fires
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mode="mask", mask_threshold=0.5)
    out = node(scores=display, alarm_scores=alarm)
    assert float(out["scores"].sum()) == 4.0  # only the 0.8 pixels cross 0.5
    assert node.hparams["mask_threshold"] == 0.5


def test_port_contract():
    node = FrameScoreGate(threshold=0.5)
    out = node(scores=torch.rand(3, 8, 8, 1, dtype=torch.float32))
    assert out["scores"].shape == (3, 8, 8, 1) and out["scores"].dtype == torch.float32
    assert out["frame_score"].shape == (3,) and out["frame_score"].dtype == torch.float32
    assert out["passed"].shape == (3,) and out["passed"].dtype == torch.int32
    assert out["decisions"].shape == (3, 8, 8, 1) and out["decisions"].dtype == torch.bool
    assert out["core"].shape == (3, 8, 8, 1) and out["core"].dtype == torch.bool
    assert set(out) == set(FrameScoreGate.OUTPUT_SPECS)


def test_decisions_mask_the_passing_frames_only():
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25)
    out = node(scores=_scores())
    assert not out["decisions"][0].any()  # clean frame: no mask
    assert torch.equal(out["decisions"][1], _scores()[1] > 1.0)  # the 4 hot pixels
    assert int(out["decisions"][1].sum()) == 4


def test_decisions_use_mask_threshold_on_the_display_scale():
    # The alarm map fires; the display map is on its own scale and masked at mask_threshold.
    display = torch.full((1, 4, 4, 1), 0.2, dtype=torch.float32)
    display[0, 1, :2, 0] = 0.8  # 2 pixels above 0.5
    alarm = torch.full((1, 4, 4, 1), 5.0, dtype=torch.float32)
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mask_threshold=0.5)
    out = node(scores=display, alarm_scores=alarm)
    assert int(out["decisions"].sum()) == 2
    assert torch.equal(out["scores"], display)  # heatmap mode still shows the display map


def test_decisions_equal_the_binary_scores_of_mask_mode():
    for mask_threshold in (None, 0.05):
        node = FrameScoreGate(
            threshold=1.0, topk_frac=0.25, mode="mask", mask_threshold=mask_threshold
        )
        out = node(scores=_scores())
        assert torch.equal(out["scores"], out["decisions"].to(torch.float32))


def test_decisions_area_follows_the_object():
    # An absolute mask_threshold marks every hot pixel of a passing frame, however many there are.
    x = torch.full((2, 8, 8, 1), 0.1, dtype=torch.float32)
    x[0, 0, 0, 0] = 5.0  # small object: 1 px
    x[1, :4, :4, 0] = 5.0  # large object: 16 px
    node = FrameScoreGate(threshold=1.0, topk_frac=1 / 64, mask_threshold=1.0)
    out = node(scores=x)
    assert out["decisions"].flatten(1).sum(1).tolist() == [1, 16]


def test_decisions_follow_the_smoothed_gate():
    node = FrameScoreGate(threshold=1.0, topk_frac=1.0, smooth_k=3)
    masked = []
    for v in (0.1, 0.1, 5.0, 5.0):  # the isolated spike stays closed; the sustained one opens
        masked.append(bool(node(scores=torch.full((1, 2, 2, 1), v))["decisions"].any()))
    assert masked == [False, False, False, True]


def test_core_is_the_confident_part_of_the_mask():
    x = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    x[:, 0, :, 0] = torch.tensor([1.2, 1.4, 2.0, 5.0])  # both frames: hot row
    alarm = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    alarm[1] = 5.0  # only frame 1 passes
    node = FrameScoreGate(threshold=1.0, topk_frac=0.25, mask_threshold=1.0, core_ratio=1.3)
    out = node(scores=x, alarm_scores=alarm)
    assert not out["core"][0].any() and not out["decisions"][0].any()  # blank frame: no core
    assert torch.equal(out["core"][1], x[1] > 1.3)  # 3 pixels: 1.4, 2.0, 5.0
    assert int(out["core"][1].sum()) == 3 and int(out["decisions"][1].sum()) == 4
    assert not (out["core"] & ~out["decisions"]).any()  # a subset of the mask
    assert node.hparams["core_ratio"] == 1.3


def test_core_defaults_to_the_decisions():
    out = FrameScoreGate(threshold=1.0, topk_frac=0.25)(scores=_scores())
    assert torch.equal(out["core"], out["decisions"])
    assert FrameScoreGate(threshold=1.0).hparams["core_ratio"] == 1.0


@pytest.mark.parametrize("kw", [{"core_ratio": 0.9}, {"core_ratio": True}, {"core_ratio": "1.3"}])
def test_invalid_core_ratio_raises(kw):
    with pytest.raises(ValueError):
        FrameScoreGate(threshold=1.0, **kw)


def test_core_ratio_above_one_refuses_a_negative_cutoff():
    """On a negative cutoff the ratio would loosen it, and ``core`` would leave ``decisions``."""
    with pytest.raises(ValueError, match="core_ratio"):
        FrameScoreGate(threshold=-5.0, mask_threshold=-2.0, core_ratio=2.0)
    with pytest.raises(ValueError, match="core_ratio"):
        FrameScoreGate(threshold=-1.0, core_ratio=1.3)
    FrameScoreGate(threshold=-1.0, mask_threshold=0.5, core_ratio=1.3)
    FrameScoreGate(threshold=-1.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"threshold": "x"},
        {"threshold": 1.0, "topk_frac": 0.0},
        {"threshold": 1.0, "mode": "binary"},
    ],
)
def test_invalid_hparams_raise(kw):
    with pytest.raises(ValueError):
        FrameScoreGate(**kw)


def test_hparams_round_trip():
    node = FrameScoreGate(threshold=2.5, topk_frac=0.002, mode="mask")
    assert node.hparams["threshold"] == 2.5
    assert node.hparams["topk_frac"] == 0.002
    assert node.hparams["mode"] == "mask"


def test_smooth_k_gates_on_the_rolling_median_one_frame_at_a_time():
    # Streaming: one frame per forward. An isolated spike (frame 2) is suppressed by the median of
    # the last 3 scores; a sustained anomaly (frames 4-5) passes once it holds the majority.
    node = FrameScoreGate(threshold=1.0, topk_frac=1.0, smooth_k=3)
    passed, raw = [], []
    for v in (0.1, 0.1, 5.0, 0.1, 5.0, 5.0):
        out = node(scores=torch.full((1, 2, 2, 1), v))
        passed.append(int(out["passed"]))
        raw.append(float(out["frame_score"]))
    assert passed == [0, 0, 0, 0, 1, 1]
    assert raw == pytest.approx([0.1, 0.1, 5.0, 0.1, 5.0, 5.0])  # frame_score stays unsmoothed


def test_log_scores_logs_every_frame_decision():
    from loguru import logger

    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO", format="{message}")
    try:
        node = FrameScoreGate(threshold=1.0, topk_frac=0.25, log_scores=True, name="gate")
        node(scores=_scores())
    finally:
        logger.remove(sink)
    assert len(lines) == 2
    assert "[gate] frame_score=0.1000 threshold=1.0000 passed=0" in lines[0]
    assert "[gate] frame_score=5.0000 threshold=1.0000 passed=1" in lines[1]


def test_log_scores_logs_the_display_maps_highest_pixel():
    """pmax = the displayed map's peak (what mask_threshold cuts), not the alarm map's."""
    from loguru import logger

    shown = torch.full((2, 4, 4, 1), 0.1, dtype=torch.float32)
    shown[0, 1, 2, 0] = 9.0  # one hot pixel: top-4 mean 2.325, peak 9.0
    alarm = torch.full((2, 4, 4, 1), 0.2, dtype=torch.float32)
    alarm[1, 3, 3, 0] = 7.0
    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO", format="{message}")
    try:
        kw = {"threshold": 1.0, "topk_frac": 0.25, "log_scores": True}
        FrameScoreGate(**kw, name="g")(scores=shown)
        FrameScoreGate(**kw, name="g2")(scores=shown, alarm_scores=alarm)
    finally:
        logger.remove(sink)
    msgs = [m.rstrip() for m in lines]
    assert msgs == [
        "[g] frame_score=2.3250 threshold=1.0000 passed=1 pmax=9.0000",
        "[g] frame_score=0.1000 threshold=1.0000 passed=0 pmax=0.1000",
        "[g2] frame_score=0.2000 threshold=1.0000 passed=0 pmax=9.0000",  # alarms on alarm_scores
        "[g2] frame_score=1.9000 threshold=1.0000 passed=1 pmax=0.1000",
    ]


@pytest.mark.parametrize("kw", [{"smooth_k": 0}, {"smooth_k": 2.5}, {"mask_threshold": "x"}])
def test_invalid_smoothing_and_mask_hparams_raise(kw):
    with pytest.raises(ValueError):
        FrameScoreGate(threshold=1.0, **kw)


def test_smooth_k_refuses_a_batch_of_several_frames():
    node = FrameScoreGate(threshold=1.0, topk_frac=1.0, smooth_k=3)
    with pytest.raises(RuntimeError, match="one frame per forward"):
        node(scores=_scores())  # two frames in one call would share one rolling window


def test_reset_forgets_the_rolling_frame_scores():
    node = FrameScoreGate(threshold=1.0, topk_frac=1.0, smooth_k=3)
    for v in (0.1, 0.1):
        node(scores=torch.full((1, 2, 2, 1), v))
    assert int(node(scores=torch.full((1, 2, 2, 1), 5.0))["passed"]) == 0  # the median holds it
    node.reset()
    assert int(node(scores=torch.full((1, 2, 2, 1), 5.0))["passed"]) == 1  # a fresh window


# ----- the one top-k rule ---------------------------------------------------------------------


@pytest.mark.parametrize("n", [1_080_000, 270_000, 576])
def test_frame_score_uses_the_core_topk_rule_at_the_deployed_sizes(n):
    # The plugin's floor rule and core's ceil rule give the same k at the deployed map sizes, so
    # the frame score is the mean of the same pixels as before.
    from cuvis_ai_core.node.metric_utils import topk_count

    frac = 0.001
    assert topk_count(n, frac) == max(1, int(frac * n))
    scores = torch.rand(1, n, 1, 1, generator=torch.Generator().manual_seed(n))
    out = FrameScoreGate(threshold=0.5, topk_frac=frac)(scores=scores)
    k = topk_count(n, frac)
    expected = torch.topk(scores.reshape(1, -1), k, dim=1).values.mean(dim=1)
    assert torch.equal(out["frame_score"], expected)


def test_gate_reloads_from_its_hparams():
    node = FrameScoreGate(
        threshold=1.25,
        topk_frac=0.25,
        mode="mask",
        mask_threshold=0.5,
        smooth_k=1,
        core_ratio=1.3,
        name="gate",
    )
    rebuilt = FrameScoreGate(**node.hparams)
    out_a, out_b = node(scores=_scores()), rebuilt(scores=_scores())
    for key in out_a:
        assert torch.equal(out_a[key], out_b[key])
