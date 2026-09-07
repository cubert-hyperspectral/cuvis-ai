"""Tests for the streaming AnomalyAUROCMetrics node (torchmetrics BinaryAUROC)."""

from __future__ import annotations

import math

import pytest
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from torchmetrics.classification import BinaryAUROC

from cuvis_ai.node.metrics import AnomalyAUROCMetrics


def _ctx(stage: ExecutionStage = ExecutionStage.VAL, epoch: int = 0, batch_idx: int = 0) -> Context:
    return Context(stage=stage, epoch=epoch, batch_idx=batch_idx, global_step=batch_idx)


def _batch(b: int, h: int, w: int, score_value: float, all_anomaly: bool):
    scores = torch.full((b, h, w, 1), score_value)
    targets = torch.full((b, h, w, 1), all_anomaly, dtype=torch.bool)
    anomaly_score = torch.full((b,), score_value)
    return scores, targets, anomaly_score


def _names(metrics) -> set[str]:
    return {m.name for m in metrics}


def test_forward_emits_running_auroc_metrics() -> None:
    """Each forward emits running auroc_pixel + auroc_image as Metric objects (no callback)."""
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=0.5, all_anomaly=True)
    out = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx())
    assert _names(out["metrics"]) == {"auroc_pixel", "auroc_image"}
    for m in out["metrics"]:
        assert m.stage == ExecutionStage.VAL and isinstance(m.value, float)


def test_running_auroc_perfect_separation() -> None:
    """After a both-classes epoch with separable scores, the running AUROC is ~1.0."""
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)  # positives, high score
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=0))
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)  # negatives, low score
    out = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=1))
    vals = {m.name: m.value for m in out["metrics"]}
    assert vals["auroc_pixel"] == pytest.approx(1.0, abs=1e-3)
    assert vals["auroc_image"] == pytest.approx(1.0, abs=1e-3)


def test_resets_on_stage_epoch_boundary() -> None:
    """A new (stage, epoch) restarts accumulation: a fresh epoch seeing only one class
    yields AUROC 0.0 (undefined), proving prior-epoch state was cleared."""
    node = AnomalyAUROCMetrics()
    # Epoch 0 — both classes (perfect separation).
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=0, batch_idx=0))
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)
    out0 = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=0, batch_idx=1))
    assert {m.name: m.value for m in out0["metrics"]}["auroc_pixel"] == pytest.approx(1.0, abs=1e-3)
    # Epoch 1 — single all-positive batch. If state carried over, AUROC would be ~1.0;
    # after reset it sees one class only -> 0.0.
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    out1 = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(epoch=1, batch_idx=0))
    assert {m.name: m.value for m in out1["metrics"]}["auroc_pixel"] == pytest.approx(0.0, abs=1e-6)


def test_explicit_reset_clears_state() -> None:
    node = AnomalyAUROCMetrics()
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=0))
    s, t, a = _batch(1, 4, 4, score_value=-5.0, all_anomaly=False)
    node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=1))
    node.reset()
    # After reset, a single one-class batch -> AUROC 0.0 (undefined), not the prior 1.0.
    s, t, a = _batch(1, 4, 4, score_value=5.0, all_anomaly=True)
    out = node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=0))
    assert {m.name: m.value for m in out["metrics"]}["auroc_pixel"] == pytest.approx(0.0, abs=1e-6)


def test_streaming_state_is_bounded_no_cpu_concat() -> None:
    """Regression for the memory fix: state is torchmetrics BinaryAUROC (O(thresholds)),
    not the old unbounded per-pixel CPU lists."""
    node = AnomalyAUROCMetrics(thresholds=128)
    assert isinstance(node.pixel_auroc, BinaryAUROC)
    assert isinstance(node.image_auroc, BinaryAUROC)
    assert not hasattr(node, "_pixel_preds")  # the old couple-GB-per-epoch concat is gone
    # Feeding many large batches must not grow any Python-side buffer.
    for i in range(8):
        s, t, a = _batch(1, 64, 64, score_value=float(i), all_anomaly=bool(i % 2))
        node.forward(scores=s, targets=t, anomaly_score=a, context=_ctx(batch_idx=i))
    # No list attribute should be accumulating tensors.
    assert not any(isinstance(v, list) and v for v in vars(node).values())


def test_stage_filter_val_test_only() -> None:
    node = AnomalyAUROCMetrics()
    assert ExecutionStage.VAL in node.execution_stages
    assert ExecutionStage.TEST in node.execution_stages
    assert ExecutionStage.TRAIN not in node.execution_stages


# --- pooled epoch-end reduction ----------------------------------------------------------


def _spatial_batch(scores_hw: torch.Tensor, mask_hw: torch.Tensor):
    """Wrap [H, W] score + bool mask into the node's [B, H, W, 1] ports + an image score."""
    scores = scores_hw[None, :, :, None].float()
    targets = mask_hw[None, :, :, None].bool()
    anomaly_score = scores.flatten(1).max(dim=1).values
    return scores, targets, anomaly_score


def test_pooled_metric_names_declared() -> None:
    """The trainer keys off POOLED_METRIC_NAMES to skip per-batch logging of these names."""
    assert AnomalyAUROCMetrics.POOLED_METRIC_NAMES == frozenset({"auroc_pixel", "auroc_image"})


def test_pooled_metrics_empty_before_any_batch() -> None:
    """Nothing to log for a run that never produced scores."""
    assert AnomalyAUROCMetrics(thresholds=200).pooled_metrics() == {}


def test_pooled_compute_is_exact_not_per_batch_mean() -> None:
    """pooled_metrics() gives the exact pooled AUROC, not the biased per-batch mean.

    Batch A is an all-normal frame -> its running AUROC is undefined and torchmetrics
    returns 0.0, which poisons a per-batch mean. The pooled accumulator across A + B is
    unaffected.
    """
    torch.manual_seed(0)
    node = AnomalyAUROCMetrics(thresholds=200)

    a_scores, a_mask = torch.rand(8, 8), torch.zeros(8, 8, dtype=torch.bool)
    b_scores = torch.cat([torch.rand(4, 8) + 3.0, torch.rand(4, 8)], dim=0)
    b_mask = torch.zeros(8, 8, dtype=torch.bool)
    b_mask[:4, :] = True

    running = []
    for i, (s, m) in enumerate([(a_scores, a_mask), (b_scores, b_mask)]):
        out = node.forward(*_spatial_batch(s, m), context=_ctx(ExecutionStage.TEST, batch_idx=i))
        running.append({x.name: x.value for x in out["metrics"]}["auroc_pixel"])

    ref = BinaryAUROC(thresholds=200)
    ref.update(
        torch.sigmoid(torch.cat([a_scores.flatten(), b_scores.flatten()])),
        torch.cat([a_mask.flatten(), b_mask.flatten()]).long(),
    )

    pooled = node.pooled_metrics()
    assert set(pooled) == {"auroc_pixel", "auroc_image"}
    node_pooled = float(pooled["auroc_pixel"].compute())
    assert node_pooled == pytest.approx(float(ref.compute()), abs=1e-6)
    assert node_pooled > sum(running) / len(running) + 0.2


def test_pooled_metrics_reset_on_new_epoch() -> None:
    """The (stage, epoch) boundary clears the pooled accumulator."""
    node = AnomalyAUROCMetrics(thresholds=200)
    s = torch.cat([torch.rand(4, 8) + 3.0, torch.rand(4, 8)], dim=0)
    m = torch.zeros(8, 8, dtype=torch.bool)
    m[:4, :] = True
    node.forward(*_spatial_batch(s, m), context=_ctx(ExecutionStage.TEST, epoch=0))
    first = float(node.pooled_metrics()["auroc_pixel"].compute())
    node.forward(*_spatial_batch(s, m), context=_ctx(ExecutionStage.TEST, epoch=1))
    assert float(node.pooled_metrics()["auroc_pixel"].compute()) == pytest.approx(first, abs=1e-6)


# --- pixel_stride subsampling ------------------------------------------------------------


def _spy_update(monkeypatch: pytest.MonkeyPatch, metric: BinaryAUROC) -> list[tuple]:
    """Record every (preds, target) pair passed to ``metric.update`` and forward the call."""
    calls: list[tuple] = []
    original = metric.update

    def recording_update(preds: torch.Tensor, target: torch.Tensor):
        calls.append((preds, target))
        return original(preds, target)

    monkeypatch.setattr(metric, "update", recording_update)
    return calls


def _random_batch(b: int, h: int, w: int, seed: int = 0):
    """A [B, H, W, 1] score/mask pair with both classes present plus its image scores."""
    torch.manual_seed(seed)
    scores = torch.randn(b, h, w, 1)
    targets = scores > 0.0
    anomaly_score = scores.flatten(1).max(dim=1).values
    return scores, targets, anomaly_score


def test_pixel_stride_defaults_to_one_and_is_an_hparam() -> None:
    """Default is no subsampling, and the value lands in hparams for save/restore."""
    node = AnomalyAUROCMetrics()
    assert node.pixel_stride == 1
    assert node.hparams["pixel_stride"] == 1


def test_pixel_stride_round_trips_through_hparams() -> None:
    """A non-default stride survives the hparams round trip a pipeline yaml goes through."""
    node = AnomalyAUROCMetrics(thresholds=64, pixel_stride=4)
    assert node.hparams["thresholds"] == 64
    assert node.hparams["pixel_stride"] == 4
    restored = AnomalyAUROCMetrics(**node.hparams)
    assert restored.pixel_stride == 4
    assert restored.hparams == node.hparams


def test_pixel_update_element_count_is_strided(monkeypatch: pytest.MonkeyPatch) -> None:
    """The pixel accumulator sees ceil(H/s) * ceil(W/s) * B elements, not the full frame."""
    node = AnomalyAUROCMetrics(thresholds=32, pixel_stride=3)
    pixel_calls = _spy_update(monkeypatch, node.pixel_auroc)
    image_calls = _spy_update(monkeypatch, node.image_auroc)

    b, h, w = 2, 9, 11
    scores, targets, anomaly_score = _random_batch(b, h, w)
    node.forward(scores=scores, targets=targets, anomaly_score=anomaly_score, context=_ctx())

    expected = math.ceil(h / 3) * math.ceil(w / 3) * b
    preds, target = pixel_calls[0]
    assert preds.numel() == expected
    assert target.numel() == expected
    assert expected < b * h * w  # the stride actually removed work
    # The image pair is never subsampled: one score and one label per frame.
    img_preds, img_target = image_calls[0]
    assert img_preds.numel() == b and img_target.numel() == b


@pytest.mark.parametrize("stride", [1, 2, 3, 5])
def test_pixel_update_bounded_for_every_stride(
    stride: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Odd H/W: the element count matches the ceil bound exactly for each stride."""
    node = AnomalyAUROCMetrics(thresholds=32, pixel_stride=stride)
    pixel_calls = _spy_update(monkeypatch, node.pixel_auroc)

    b, h, w = 3, 7, 13
    scores, targets, anomaly_score = _random_batch(b, h, w, seed=stride)
    node.forward(scores=scores, targets=targets, anomaly_score=anomaly_score, context=_ctx())

    expected = math.ceil(h / stride) * math.ceil(w / stride) * b
    preds, target = pixel_calls[0]
    assert preds.numel() == expected
    assert target.numel() == expected


def test_image_label_comes_from_the_full_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lone anomalous pixel the stride skips still yields image label 1."""
    node = AnomalyAUROCMetrics(thresholds=32, pixel_stride=2)
    pixel_calls = _spy_update(monkeypatch, node.pixel_auroc)
    image_calls = _spy_update(monkeypatch, node.image_auroc)

    scores = torch.zeros(1, 6, 6, 1)
    targets = torch.zeros(1, 6, 6, 1, dtype=torch.bool)
    targets[0, 1, 1, 0] = True  # odd row/column: stride 2 samples 0, 2, 4 only
    scores[0, 1, 1, 0] = 5.0
    anomaly_score = scores.flatten(1).max(dim=1).values

    node.forward(scores=scores, targets=targets, anomaly_score=anomaly_score, context=_ctx())

    _, pixel_target = pixel_calls[0]
    assert not bool(pixel_target.any())  # the stride skipped the only positive pixel
    _, image_target = image_calls[0]
    assert image_target.dtype == torch.bool
    assert bool(image_target[0])  # the image label still says "anomalous"


def test_stride_one_is_bit_exact_against_the_unstrided_reference() -> None:
    """pixel_stride=1 must be the identity: same confusion matrix, same AUROC."""
    node = AnomalyAUROCMetrics(thresholds=200, pixel_stride=1)
    b, h, w = 2, 8, 10
    scores, targets, anomaly_score = _random_batch(b, h, w, seed=7)
    node.forward(scores=scores, targets=targets, anomaly_score=anomaly_score, context=_ctx())

    ref = BinaryAUROC(thresholds=200)
    ref.update(
        torch.sigmoid(scores.flatten().float()),
        targets.squeeze(-1).flatten().long(),
    )

    assert torch.equal(node.pixel_auroc.confmat, ref.confmat)
    assert float(node.pixel_auroc.compute()) == float(ref.compute())


def test_bool_targets_match_long_targets() -> None:
    """Feeding the mask as bool (no int64 promotion) gives the same state as long."""
    b, h, w = 2, 8, 10
    scores, targets, anomaly_score = _random_batch(b, h, w, seed=11)

    bool_node = AnomalyAUROCMetrics(thresholds=200)
    bool_node.forward(scores=scores, targets=targets, anomaly_score=anomaly_score, context=_ctx())
    long_node = AnomalyAUROCMetrics(thresholds=200)
    long_node.forward(
        scores=scores,
        targets=targets.long(),
        anomaly_score=anomaly_score,
        context=_ctx(),
    )

    assert torch.equal(bool_node.pixel_auroc.confmat, long_node.pixel_auroc.confmat)
    assert torch.equal(bool_node.image_auroc.confmat, long_node.image_auroc.confmat)
    assert float(bool_node.pixel_auroc.compute()) == float(long_node.pixel_auroc.compute())


def test_binary_auroc_skips_argument_validation() -> None:
    """validate_args=False drops torchmetrics' per-update unique()/sort over every pixel."""
    node = AnomalyAUROCMetrics()
    assert node.pixel_auroc.validate_args is False
    assert node.image_auroc.validate_args is False


@pytest.mark.parametrize("bad", [0, -1, 1.5, "2"])
def test_invalid_pixel_stride_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="pixel_stride"):
        AnomalyAUROCMetrics(pixel_stride=bad)


@pytest.mark.parametrize("bad", [1, 0, -3, 2.5, "200"])
def test_invalid_thresholds_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="thresholds"):
        AnomalyAUROCMetrics(thresholds=bad)
