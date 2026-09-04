"""Calibrate binary-decider thresholds from a trained pipeline's scores.

A trained anomaly pipeline ships its decider with fixed threshold hparams. Those go
stale per checkpoint: training moves the score distribution, so values calibrated for
one set of weights misfit the next. This tool runs the trained pipeline over a labeled
split, sweeps the decider's actual decision rule against the ground truth, and reports
F1-optimal hparams ready to paste back into the pipeline yaml.

The sweep is dispatched on the pipeline's decider class:

- ``TwoStageBinaryDecider`` - 2-D grid over the stage-1 image gate (the decider's own
  mean-of-top-k statistic, raw score space) and an absolute stage-2 ``pixel_threshold``
  (raw score space). The stages couple (a frame killed by the gate contributes all-zero
  pixels), so the grid is joint rather than two independent maxima.
- ``BinaryDecider`` - single sweep over the elementwise cutoff. The node thresholds
  ``sigmoid(logits)``, so the sweep runs in raw score space and the reported
  ``threshold`` is the sigmoid of the raw optimum.
- ``QuantileBinaryDecider`` - single sweep over ``quantile``. The node recomputes its
  cutoff from each frame's own score distribution, so there is no absolute threshold to
  calibrate; the sweep finds the F1-best flagged-pixel fraction instead (which still
  flags that fraction in every frame, anomalous or clean - the metrics show the cost).

Any other decider class aborts with an explicit unsupported-decider error.

The scores are collected from the port wired into the decider's ``logits`` input (the
space the decider actually thresholds), not from a global port scan; ground truth is
the pipeline's single bool ``mask`` output. Both can be overridden with
``--scores-port`` / ``--mask-port`` (``node.port``), and ``--decider-node`` picks the
decider when a pipeline has several.

Works on both training layouts:

- CuvisNEXT GUI runs: ``<run>/saved/trainrun_pipeline.yaml`` + ``<run>/saved/trainrun.pt``
  (the yaml/pt stems differ, so the implicit ``.pt`` sibling lookup never applies here).
- ``restore-trainrun`` runs: ``<run>/trained_models/<name>_restored.{yaml,pt}``.

Lightning ``.ckpt`` files are not consumed directly; convert one first with
``cuvis-ai-dinomaly/examples/export_dinomaly_multifile_pipeline_from_ckpt.py``.

Examples
--------
Calibrate on the validation split of a CuvisNEXT training run::

    calibrate-thresholds --trainrun-dir "C:/Program Files/Cuvis/user/training_runs/<run>"

Explicit artifacts and the test split::

    calibrate-thresholds --trainrun-dir <run> --split test \\
        --pipeline-yaml <run>/saved/trainrun_pipeline.yaml \\
        --weights-path <run>/saved/trainrun.pt
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from cuvis_ai_schemas.enums import ExecutionStage
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PipelineConfig
from loguru import logger

from cuvis_ai.node.deciders import _calibration as _calib
from cuvis_ai.utils.grpc_workflow import CONFIG_ROOT
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training.config import TrainRunConfig
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_core.utils.plugin_resolver import resolve_pipeline_plugins
from cuvis_ai_core.utils.restore import _create_datamodule_from_config, _discover_plugins_dirs

#: Decider classes whose decision rule the sweeps model, keyed by class name tail.
_SUPPORTED_DECIDERS = {
    "TwoStageBinaryDecider": "two_stage",
    "BinaryDecider": "binary",
    "QuantileBinaryDecider": "quantile",
}

#: Node hparams that make score distributions depend on frame order / warm-up history.
_HISTORY_HPARAMS = (
    "running_warmup_frames",
    "freeze_running_bounds_after_frames",
    "max_initialization_frames",
)


def _resolve_artifacts(
    trainrun_dir: Path,
    trainrun_yaml: Path | None,
    pipeline_yaml: Path | None,
    weights_path: Path | None,
) -> tuple[Path, Path, Path]:
    """Locate the trainrun yaml, pipeline yaml and pipeline weights for a run directory.

    Explicit overrides always win. Otherwise the CuvisNEXT GUI layout (``saved/``) is
    tried first, then the ``restore-trainrun`` layout (``trained_models/``); for the
    trainrun yaml the fallback considers top-level ``*.yaml`` files that carry a
    ``data:`` section and refuses to guess between several.
    """
    if trainrun_yaml is None:
        gui_trainrun = trainrun_dir / "saved" / "trainrun.yaml"
        if gui_trainrun.exists():
            trainrun_yaml = gui_trainrun
        else:
            candidates = [
                candidate
                for candidate in sorted(trainrun_dir.glob("*.yaml"))
                if re.search(r"^data:", candidate.read_text(encoding="utf-8"), re.MULTILINE)
            ]
            if len(candidates) == 1:
                trainrun_yaml = candidates[0]
            elif candidates:
                raise FileNotFoundError(
                    f"Several trainrun-like yamls under {trainrun_dir}: "
                    f"{[c.name for c in candidates]}; pass --trainrun-yaml explicitly."
                )
            else:
                raise FileNotFoundError(
                    f"No trainrun yaml found (looked at {gui_trainrun} and top-level "
                    f"*.yaml files with a data: section under {trainrun_dir}); "
                    "pass --trainrun-yaml explicitly."
                )

    if pipeline_yaml is None:
        for candidate in [
            trainrun_dir / "saved" / "trainrun_pipeline.yaml",
            *sorted((trainrun_dir / "trained_models").glob("*_restored.yaml")),
        ]:
            if candidate.exists():
                pipeline_yaml = candidate
                break
        else:
            raise FileNotFoundError(
                f"No pipeline yaml under {trainrun_dir}; pass --pipeline-yaml explicitly."
            )

    if weights_path is None:
        # CuvisNEXT has written both stems over time: trainrun.pt and (matching the
        # pipeline yaml stem) trainrun_pipeline.pt.
        for candidate in [
            trainrun_dir / "saved" / "trainrun.pt",
            trainrun_dir / "saved" / "trainrun_pipeline.pt",
            *sorted((trainrun_dir / "trained_models").glob("*_restored.pt")),
        ]:
            if candidate.exists():
                weights_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"No pipeline weights (.pt) under {trainrun_dir}; pass --weights-path. "
                "A Lightning .ckpt is not consumed directly - export a pipeline .pt first."
            )

    return Path(trainrun_yaml), Path(pipeline_yaml), Path(weights_path)


def _load_trainrun(trainrun_yaml: Path, trainrun_dir: Path) -> TrainRunConfig:
    """Load the trainrun config, repairing a stale absolute ``splits_path`` if needed.

    CuvisNEXT writes absolute paths into ``saved/trainrun.yaml``; renaming the run
    directory afterwards leaves them dangling. When the recorded ``splits_path`` does
    not exist but ``<trainrun_dir>/splits.json`` does, the local copy is used.
    """
    config = TrainRunConfig.load_from_file(trainrun_yaml)
    splits = config.data.splits
    if splits is not None and splits.splits_path:
        recorded = Path(splits.splits_path)
        fallback = trainrun_dir / "splits.json"
        if not recorded.exists() and fallback.exists():
            logger.warning(
                f"splits_path {recorded} does not exist (renamed run directory?); "
                f"using {fallback} instead."
            )
            repaired = splits.model_copy(update={"splits_path": str(fallback.resolve())})
            config = config.model_copy(
                update={"data": config.data.model_copy(update={"splits": repaired})}
            )
    return config


def _plugins_candidates(anchor: Path, plugins_dirs: list[str] | None) -> list[Path]:
    """Candidate plugin-manifest directories for the resolver.

    Wraps the core discovery (walk up from ``anchor``, then explicit ``--plugins-dir``
    values; later entries win on name collisions) and prepends the manifests packaged
    with cuvis-ai as the weakest fallback - a CuvisNEXT run directory has no
    ``configs/plugins`` ancestor to discover.
    """
    candidates = _discover_plugins_dirs(anchor, plugins_dirs)
    packaged = CONFIG_ROOT / "plugins"
    if packaged.is_dir() and packaged not in candidates:
        candidates.insert(0, packaged)
    return candidates


def _resolve_decider(
    config: PipelineConfig, decider_node: str | None
) -> tuple[str, str, dict[str, Any]]:
    """Pick the decider node to calibrate; returns ``(node_name, class_tail, hparams)``.

    Selection is by class identity against the supported decider set, never by name
    substring. Zero supported deciders or an ambiguous set abort with an explicit error.
    """
    candidates = [
        node for node in config.nodes if node.class_name.rsplit(".", 1)[-1] in _SUPPORTED_DECIDERS
    ]
    if decider_node is not None:
        matches = [node for node in candidates if node.name == decider_node]
        if not matches:
            raise ValueError(
                f"--decider-node {decider_node!r} does not name a supported decider; "
                f"supported decider nodes in this pipeline: "
                f"{sorted(node.name for node in candidates) or 'none'}."
            )
        node = matches[0]
    elif len(candidates) == 1:
        node = candidates[0]
    elif not candidates:
        decider_like = [
            f"{node.name} ({node.class_name.rsplit('.', 1)[-1]})"
            for node in config.nodes
            if "decider" in node.class_name.lower() or "decider" in (node.name or "").lower()
        ]
        raise ValueError(
            f"No supported decider in the pipeline. Calibration models "
            f"{sorted(_SUPPORTED_DECIDERS)}; decider-like nodes found: "
            f"{decider_like or 'none'}."
        )
    else:
        raise ValueError(
            f"Multiple supported deciders {sorted(node.name for node in candidates)}; "
            "pass --decider-node to pick one."
        )
    return node.name, node.class_name.rsplit(".", 1)[-1], dict(node.hparams or {})


def _score_source(config: PipelineConfig, decider_name: str) -> tuple[str, str]:
    """Resolve the ``(node, port)`` wired into the decider's ``logits`` input.

    That port carries the tensor the decider actually thresholds, which is the only
    correct calibration space - a transform between the detector and the decider would
    silently invalidate a sweep on the detector's own output.
    """
    for connection in config.connections:
        if connection.to_node == decider_name and connection.to_port == "logits":
            return connection.from_node, connection.from_port
    raise ValueError(f"No connection targets {decider_name}.inputs.logits; pass --scores-port.")


def _parse_port(spec: str, flag: str) -> tuple[str, str]:
    """Parse a ``node.port`` (or ``node.outputs.port``) CLI override."""
    parts = spec.split(".")
    if len(parts) == 3 and parts[1] in ("outputs", "inputs"):
        return parts[0], parts[2]
    if len(parts) == 2:
        return parts[0], parts[1]
    raise ValueError(f"{flag} must be 'node.port', got {spec!r}.")


def _build_pipeline(
    pipeline_yaml: Path,
    weights_path: Path,
    device: str,
    plugins_dirs: list[str] | None,
) -> CuvisPipeline:
    """Build the pipeline with plugin resolution and load the trained weights.

    Aborts when a weight-bearing node is absent from the checkpoint - calibrating
    freshly initialized weights would produce meaningless thresholds.
    """
    candidate_dirs = _plugins_candidates(pipeline_yaml, plugins_dirs)
    registry: NodeRegistry | None = None
    pipeline_cfg = PipelineConfig.load_from_file(pipeline_yaml)
    if pipeline_cfg.plugins or candidate_dirs:
        resolved = resolve_pipeline_plugins(pipeline_cfg, candidate_dirs)
        if resolved:
            registry = NodeRegistry()
            registry.register_plugins_installed(resolved)
            logger.info(f"Materialised plugins: {sorted(resolved)}")

    load_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    pipeline = CuvisPipeline.load_pipeline(
        str(pipeline_yaml), weights_path=None, device=load_device, node_registry=registry
    )
    missing = pipeline._restore_weights_from_checkpoint(
        str(weights_path), strict_weight_loading=False, device=load_device
    )
    nodes_by_name = {getattr(node, "name", None): node for node in pipeline.nodes()}
    parametrized_missing = [
        name
        for name in missing
        if name in nodes_by_name
        and (
            any(True for _ in nodes_by_name[name].parameters())
            or any(True for _ in nodes_by_name[name].buffers())
        )
    ]
    if parametrized_missing:
        raise RuntimeError(
            f"Checkpoint {weights_path} carries no weights for {parametrized_missing}; "
            "refusing to calibrate freshly initialized nodes. Wrong --weights-path?"
        )
    return pipeline


def _collect_frames(
    pipeline: CuvisPipeline,
    datamodule: Any,
    split: str,
    scores_key: tuple[str, str],
    mask_key: tuple[str, str] | None,
    probe_builder: Callable[[int], np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, dict[float, np.ndarray]]:
    """Run the pipeline over the split and collect pixel scores and ground truth.

    Returns ``(pixel_scores, gt_masks, frame_quantiles)``: float32/bool ``[N, H, W]``
    arrays plus, per requested quantile probe, the per-frame quantile computed over the
    FULL ``[H, W, C]`` score tensor - exactly the statistic ``torch.quantile`` gives the
    deciders at runtime, so the multi-channel case stays bit-honest even though only the
    channel-reduced map is retained. The loop mirrors ``Predictor`` (eval mode,
    ``no_grad``, per-batch CPU detach) but iterates the val/test dataloader directly:
    ``Predictor.predict`` re-runs ``setup("predict")``, and with an empty ``predict``
    selector list that rebuilds the predict dataset as the whole universe rather than
    the requested split.
    """
    # _create_datamodule_from_config already ran setup("fit"); only test needs another.
    if split == "test":
        datamodule.setup(stage="test")
    loader = datamodule.val_dataloader() if split == "val" else datamodule.test_dataloader()

    for module in pipeline.torch_layers:
        module.eval()
    for node in pipeline.nodes():
        reset_fn = getattr(node, "reset", None)
        if callable(reset_fn):
            reset_fn()

    device = torch.device("cpu")
    for layer in pipeline.torch_layers:
        for param in layer.parameters():
            device = param.device
            break
        else:
            continue
        break

    scores_frames: list[np.ndarray] = []
    mask_frames: list[np.ndarray] = []
    probes: np.ndarray | None = None
    probe_rows: list[np.ndarray] = []
    stage = ExecutionStage.VAL if split == "val" else ExecutionStage.TEST
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                moved = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                context = Context(stage=stage, batch_idx=batch_idx, global_step=batch_idx)
                outputs = pipeline.forward(batch=moved, context=context)

                if scores_key not in outputs:
                    raise RuntimeError(
                        f"Pipeline outputs carry no {scores_key[0]}.{scores_key[1]}; "
                        f"available: {sorted(outputs)}. Wrong --scores-port?"
                    )
                scores = outputs[scores_key]
                if not isinstance(scores, torch.Tensor) or not scores.dtype.is_floating_point:
                    raise RuntimeError(
                        f"{scores_key[0]}.{scores_key[1]} is not a float tensor; "
                        "pass --scores-port pointing at the decider's score input."
                    )
                if mask_key is not None:
                    if mask_key not in outputs:
                        raise RuntimeError(
                            f"Pipeline outputs carry no {mask_key[0]}.{mask_key[1]}; "
                            f"available: {sorted(outputs)}. Wrong --mask-port?"
                        )
                    mask = outputs[mask_key]
                    if not isinstance(mask, torch.Tensor):
                        raise RuntimeError(f"{mask_key[0]}.{mask_key[1]} is not a tensor.")
                    mask = mask.to(torch.bool)
                else:
                    mask = _single_port(outputs, "mask", torch.bool)

                if probe_builder is not None and probes is None:
                    probes = np.asarray(probe_builder(int(scores[0].numel())), dtype=np.float64)
                if probes is not None and probes.size:
                    for frame in scores:
                        flat = frame.detach().to("cpu", torch.float32).numpy().ravel()
                        probe_rows.append(np.quantile(flat, probes))

                # [B,H,W,C] -> per-pixel max over channels, then split the batch
                pixel = scores.max(dim=-1).values.detach().cpu().numpy()
                # Collapse a channel dim only when one exists ([B,H,W,C] vs [B,H,W]).
                gt = (mask.any(dim=-1) if mask.dim() == 4 else mask).detach().cpu().numpy()
                scores_frames.extend(np.asarray(frame, dtype=np.float32) for frame in pixel)
                mask_frames.extend(np.asarray(frame, dtype=bool) for frame in gt)
    finally:
        for node in pipeline.nodes():
            close_fn = getattr(node, "close", None)
            if callable(close_fn):
                close_fn()

    if not scores_frames:
        raise RuntimeError(f"The {split} split produced no frames - nothing to calibrate.")
    frame_quantiles: dict[float, np.ndarray] = {}
    if probes is not None and probes.size:
        stacked = np.stack(probe_rows)  # [N, P]
        frame_quantiles = {float(q): stacked[:, i] for i, q in enumerate(probes)}
    return np.stack(scores_frames), np.stack(mask_frames), frame_quantiles


def _single_port(
    outputs: dict[tuple[str, str], Any], port: str, dtype: torch.dtype
) -> torch.Tensor:
    """Pick the single tensor output named ``port`` with the given dtype.

    Raises when zero or multiple nodes emit a matching port, naming the candidates so
    the caller can disambiguate with the port-override flags rather than guessing.
    """
    hits = {
        (node, name): value
        for (node, name), value in outputs.items()
        if name == port and isinstance(value, torch.Tensor) and value.dtype == dtype
    }
    if len(hits) != 1:
        raise RuntimeError(
            f"Expected exactly one {dtype} output port {port!r}, found "
            f"{sorted(node for node, _ in hits) or 'none'}; pass --mask-port."
        )
    return next(iter(hits.values()))


def _warn_history_dependent(config: PipelineConfig) -> list[str]:
    """Warn about nodes whose score distribution depends on frame order / warm-up."""
    flagged = []
    for node in config.nodes:
        hparams = node.hparams or {}
        hits = [key for key in _HISTORY_HPARAMS if hparams.get(key)]
        if hits:
            flagged.append(f"{node.name}: {hits}")
            logger.warning(
                f"Node {node.name} has history-dependent hparams {hits}; the calibrated "
                "thresholds assume the same warm-up behavior at deployment."
            )
    return flagged


def calibrate_thresholds(
    trainrun_dir: str | Path,
    trainrun_yaml: str | Path | None = None,
    pipeline_yaml: str | Path | None = None,
    weights_path: str | Path | None = None,
    split: str = "val",
    device: str = "auto",
    decider_node: str | None = None,
    scores_port: str | None = None,
    mask_port: str | None = None,
    top_k_fraction: float | None = None,
    num_candidates: int = 256,
    output: str | Path | None = None,
    plugins_dirs: list[str] | None = None,
) -> dict[str, Any]:
    """Calibrate decider thresholds for a trained pipeline against a labeled split.

    Parameters
    ----------
    trainrun_dir : str | Path
        Training run directory (CuvisNEXT GUI or ``restore-trainrun`` layout).
    trainrun_yaml, pipeline_yaml, weights_path : str | Path | None
        Explicit artifact overrides; resolved from the run layout when omitted.
    split : str
        ``val`` (default) or ``test``.
    device : str
        ``auto`` (default: cuda when available, else cpu), ``cpu`` or ``cuda``.
    decider_node : str | None
        Decider node name, for pipelines with several supported deciders.
    scores_port, mask_port : str | None
        ``node.port`` overrides for the score source (default: whatever feeds the
        decider's ``logits`` input) and the bool ground-truth mask (default: the
        pipeline's single bool ``mask`` output).
    top_k_fraction : float | None
        Stage-1 top-k fraction (two-stage decider only); read from the decider's
        hparams when omitted.
    num_candidates : int
        Sweep resolution for threshold/quantile candidates.
    output : str | Path | None
        JSON report path (default ``<trainrun_dir>/calibration_<split>.json``).
    plugins_dirs : list[str] | None
        Extra plugin manifest directories for node resolution.

    Returns
    -------
    dict
        The calibration report (also written to ``output``).
    """
    trainrun_dir = Path(trainrun_dir)
    resolved_trainrun, resolved_yaml, resolved_weights = _resolve_artifacts(
        trainrun_dir,
        Path(trainrun_yaml) if trainrun_yaml else None,
        Path(pipeline_yaml) if pipeline_yaml else None,
        Path(weights_path) if weights_path else None,
    )
    logger.info(f"pipeline: {resolved_yaml}\nweights:  {resolved_weights}")

    config = _load_trainrun(resolved_trainrun, trainrun_dir)
    seed = config.training.seed if config.training is not None else 42
    pl.seed_everything(seed, workers=True)

    pipeline_cfg = PipelineConfig.load_from_file(resolved_yaml)
    decider_name, decider_class, decider_defaults = _resolve_decider(pipeline_cfg, decider_node)
    mode = _SUPPORTED_DECIDERS[decider_class]
    logger.info(f"decider: {decider_name} ({decider_class}) -> {mode} sweep")
    history_flags = _warn_history_dependent(pipeline_cfg)

    if scores_port is not None:
        scores_key = _parse_port(scores_port, "--scores-port")
    else:
        scores_key = _score_source(pipeline_cfg, decider_name)
    mask_key = _parse_port(mask_port, "--mask-port") if mask_port is not None else None
    logger.info(f"scores from {scores_key[0]}.{scores_key[1]}")

    if top_k_fraction is not None and mode != "two_stage":
        logger.warning(f"--top-k-fraction is ignored for {decider_class} (no image gate).")
    if top_k_fraction is None:
        top_k_fraction = float(decider_defaults.get("top_k_fraction", 0.001))

    # Quantile probes computed per frame over the FULL [H,W,C] tensor at collection
    # time, matching the deciders' torch.quantile semantics exactly (grid depends on
    # the per-frame pixel count, hence the deferred builder).
    preset_quantile = float(decider_defaults.get("quantile", 0.995))
    probe_builder: Callable[[int], np.ndarray] | None = None
    if mode == "quantile":
        probe_builder = _calib.quantile_grid_builder(preset_quantile, num_candidates)
    elif mode == "two_stage" and decider_defaults.get("pixel_threshold") is None:
        probe_builder = _calib.preset_probe_builder(preset_quantile)

    pipeline = _build_pipeline(resolved_yaml, resolved_weights, device, plugins_dirs)
    candidate_dirs = _plugins_candidates(resolved_trainrun, plugins_dirs)
    datamodule = _create_datamodule_from_config(config, candidate_dirs, resolved_trainrun.parent)

    pixel_scores, gt_masks, frame_quantiles = _collect_frames(
        pipeline, datamodule, split, scores_key, mask_key, probe_builder
    )
    frame_labels = gt_masks.any(axis=(1, 2))
    n_anomalous = int(frame_labels.sum())
    logger.info(
        f"{split} split: {len(frame_labels)} frames, "
        f"{n_anomalous} anomalous / {len(frame_labels) - n_anomalous} normal"
    )
    if n_anomalous == 0 or n_anomalous == len(frame_labels):
        raise RuntimeError(
            f"The {split} split is single-class ({n_anomalous} anomalous of "
            f"{len(frame_labels)}); thresholds cannot be calibrated on it. If the split "
            "should be mixed, the COCO image_id to measurement mapping may be off."
        )

    report: dict[str, Any] = {
        "trainrun_dir": str(trainrun_dir),
        "pipeline_yaml": str(resolved_yaml),
        "weights_path": str(resolved_weights),
        "split": split,
        "stage": "VAL" if split == "val" else "TEST",
        "frames": int(len(frame_labels)),
        "anomalous_frames": n_anomalous,
        "decider": {"node": decider_name, "class_name": decider_class, "mode": mode},
        "score_source": f"{scores_key[0]}.{scores_key[1]}",
        "history_dependent_nodes": history_flags,
    }

    if mode == "two_stage":
        image_scores = _calib.topk_mean_scores(pixel_scores, top_k_fraction)
        best_image, joint_best, conditional_best = _calib.sweep_two_stage(
            pixel_scores, gt_masks, image_scores, frame_labels, num_candidates
        )
        current = _current_preset_two_stage(
            pixel_scores, gt_masks, image_scores, frame_labels, decider_defaults, frame_quantiles
        )
        report.update(
            {
                "top_k_fraction": top_k_fraction,
                "score_space": "raw (no sigmoid; the space TwoStageBinaryDecider thresholds in)",
                "image": {
                    "auroc": _calib.binary_auroc(image_scores, frame_labels),
                    "f1_max": best_image,
                },
                "pixel": {
                    "joint_optimum": joint_best,
                    "conditional_on_image_f1max": conditional_best,
                },
                "current_preset": current,
                "calibrated_decider_hparams": {
                    "image_threshold": float(best_image["threshold"]),
                    "top_k_fraction": top_k_fraction,
                    "pixel_threshold": float(conditional_best["pixel_threshold"]),
                },
            }
        )
    elif mode == "binary":
        frame_max = pixel_scores.max(axis=(1, 2))
        best = _calib.sweep_absolute(pixel_scores, gt_masks, num_candidates)
        current = _current_preset_binary(pixel_scores, gt_masks, decider_defaults)
        report.update(
            {
                "score_space": (
                    "sigmoid probability (BinaryDecider thresholds sigmoid(logits); the "
                    "sweep ran in raw space and the optimum was mapped through sigmoid)"
                ),
                "frame_auroc_max_score": _calib.binary_auroc(frame_max, frame_labels),
                "pixel": {"optimum": best},
                "current_preset": current,
                "calibrated_decider_hparams": {
                    "threshold": float(_calib.sigmoid(best["raw_threshold"]))
                },
            }
        )
    else:  # quantile
        frame_max = pixel_scores.max(axis=(1, 2))
        best = _calib.sweep_quantile(pixel_scores, gt_masks, frame_quantiles)
        current = _current_preset_quantile(pixel_scores, gt_masks, preset_quantile, frame_quantiles)
        report.update(
            {
                "score_space": (
                    "per-frame adaptive (QuantileBinaryDecider recomputes its cutoff from "
                    "each frame's own scores; no absolute threshold exists to calibrate - "
                    "the swept quantile fixes the flagged-pixel fraction per frame)"
                ),
                "frame_auroc_max_score": _calib.binary_auroc(frame_max, frame_labels),
                "pixel": {"optimum": best},
                "current_preset": current,
                "calibrated_decider_hparams": {"quantile": float(best["quantile"])},
            }
        )

    output_path = Path(output) if output else trainrun_dir / f"calibration_{split}.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"calibration written to {output_path}")
    _print_report(report)
    return report


def _current_preset_two_stage(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    image_scores: np.ndarray,
    frame_labels: np.ndarray,
    decider_defaults: dict[str, Any],
    frame_quantiles: dict[float, np.ndarray],
) -> dict[str, Any]:
    """Metrics at the two-stage decider values currently in the pipeline.

    Mirrors the decider's precedence: an already-calibrated absolute ``pixel_threshold``
    wins over the quantile fallback, and the fallback quantile is the per-frame value
    over the full ``[H,W,C]`` tensor collected alongside the scores. An unset
    ``image_threshold`` (``null`` in the yaml, the training presets' default) means the
    gate is off: every frame reaches stage 2 and there are no image-level metrics.
    """
    raw_image_thr = decider_defaults.get("image_threshold")
    image_thr = float(raw_image_thr) if raw_image_thr is not None else None
    quantile = float(decider_defaults.get("quantile", 0.995))
    pixel_thr = decider_defaults.get("pixel_threshold")
    if image_thr is None:
        gate = np.ones(len(frame_labels), dtype=bool)
    else:
        gate = image_scores >= image_thr
    if pixel_thr is not None:
        thresholds = np.full(len(frame_labels), float(pixel_thr))
        stage2 = {"mode": "absolute", "pixel_threshold": float(pixel_thr)}
    else:
        thresholds = frame_quantiles[quantile]
        stage2 = {"mode": "quantile", "quantile": quantile}
    flagged = (pixel_scores >= thresholds[:, None, None]) & gate[:, None, None]
    tp = float((flagged & gt_masks).sum())
    fp = float((flagged & ~gt_masks).sum())
    fn = float(gt_masks.sum()) - tp
    return {
        "image_threshold": image_thr,
        "stage2": stage2,
        "image": (
            _calib.image_metrics_at(image_scores, frame_labels, image_thr)
            if image_thr is not None
            else None
        ),
        "pixel": _calib.prf(tp, fp, fn),
    }


def _current_preset_binary(
    pixel_scores: np.ndarray, gt_masks: np.ndarray, decider_defaults: dict[str, Any]
) -> dict[str, Any]:
    """Metrics at the ``BinaryDecider`` sigmoid-space threshold currently shipped."""
    threshold = float(decider_defaults.get("threshold", 0.5))
    flagged = _calib.sigmoid(pixel_scores) >= threshold
    tp = float((flagged & gt_masks).sum())
    fp = float((flagged & ~gt_masks).sum())
    fn = float(gt_masks.sum()) - tp
    return {"threshold": threshold, "pixel": _calib.prf(tp, fp, fn)}


def _current_preset_quantile(
    pixel_scores: np.ndarray,
    gt_masks: np.ndarray,
    preset_quantile: float,
    frame_quantiles: dict[float, np.ndarray],
) -> dict[str, Any]:
    """Metrics at the ``QuantileBinaryDecider`` quantile currently shipped."""
    thresholds = frame_quantiles[preset_quantile][:, None, None]
    flagged = pixel_scores >= thresholds
    tp = float((flagged & gt_masks).sum())
    fp = float((flagged & ~gt_masks).sum())
    fn = float(gt_masks.sum()) - tp
    return {"quantile": preset_quantile, "pixel": _calib.prf(tp, fp, fn)}


def _print_report(report: dict[str, Any]) -> None:
    """Print the human-readable calibration summary."""
    mode = report["decider"]["mode"]
    print(  # noqa: T201
        f"\n=== calibration ({report['split']}, {report['frames']} frames, "
        f"{report['anomalous_frames']} anomalous) ==="
    )
    print(  # noqa: T201
        f"decider: {report['decider']['node']} ({report['decider']['class_name']}), "
        f"scores from {report['score_source']}"
    )
    if mode == "two_stage":
        image = report["image"]["f1_max"]
        joint = report["pixel"]["joint_optimum"]
        conditional = report["pixel"]["conditional_on_image_f1max"]
        current = report["current_preset"]
        print(  # noqa: T201
            f"image AUROC (top-k mean scores): {report['image']['auroc']:.4f}"
        )
        print(  # noqa: T201
            f"image F1-max: {image['f1']:.4f} at image_threshold={image['threshold']:.6f} "
            f"(P={image['precision']:.3f} R={image['recall']:.3f})"
        )
        print(  # noqa: T201
            f"pixel F1 at that gate: {conditional['f1']:.4f} at "
            f"pixel_threshold={conditional['pixel_threshold']:.6f} "
            f"(P={conditional['precision']:.3f} R={conditional['recall']:.3f} "
            f"IoU={conditional['iou']:.3f})"
        )
        print(  # noqa: T201
            f"pixel F1 joint optimum: {joint['f1']:.4f} at image_threshold="
            f"{joint['image_threshold']:.6f}, pixel_threshold={joint['pixel_threshold']:.6f}"
        )
        stage2 = current["stage2"]
        stage2_desc = (
            f"pixel_threshold={stage2['pixel_threshold']}"
            if stage2["mode"] == "absolute"
            else f"quantile={stage2['quantile']}"
        )
        if current["image_threshold"] is None:
            gate_desc, image_desc = "gate off", ""
        else:
            gate_desc = f"image_threshold={current['image_threshold']}"
            image_desc = f"image F1={current['image']['f1']:.4f}, "
        print(  # noqa: T201
            f"current preset ({gate_desc}, {stage2_desc}): "
            f"{image_desc}pixel F1={current['pixel']['f1']:.4f}"
        )
    else:
        best = report["pixel"]["optimum"]
        current = report["current_preset"]
        print(  # noqa: T201
            f"frame AUROC (max pixel score, informational): {report['frame_auroc_max_score']:.4f}"
        )
        knob = "quantile" if mode == "quantile" else "raw_threshold"
        print(  # noqa: T201
            f"pixel F1 optimum: {best['f1']:.4f} at {knob}={best[knob]:.6f} "
            f"(P={best['precision']:.3f} R={best['recall']:.3f} IoU={best['iou']:.3f})"
        )
        current_knob = "quantile" if mode == "quantile" else "threshold"
        print(  # noqa: T201
            f"current preset ({current_knob}={current[current_knob]}): "
            f"pixel F1={current['pixel']['f1']:.4f}"
        )
    print("\ndecider hparams to paste into the pipeline yaml:")  # noqa: T201
    for key, value in report["calibrated_decider_hparams"].items():
        print(f"    {key}: {value}")  # noqa: T201


def main() -> None:
    """CLI entry point for ``calibrate-thresholds``."""
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate binary-decider thresholds from a trained pipeline "
            "(TwoStageBinaryDecider, BinaryDecider or QuantileBinaryDecider)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  calibrate-thresholds --trainrun-dir <run>\n"
            "  calibrate-thresholds --trainrun-dir <run> --split test --device cpu\n"
            "  calibrate-thresholds --trainrun-dir <run> "
            "--weights-path <run>/saved/trainrun.pt\n"
            "  calibrate-thresholds --trainrun-dir <run> --decider-node decider "
            "--scores-port dinomaly_detector.scores\n"
        ),
    )
    parser.add_argument(
        "--trainrun-dir", required=True, help="Training run directory (GUI or CLI layout)"
    )
    parser.add_argument("--trainrun-yaml", default=None, help="Explicit trainrun yaml override")
    parser.add_argument("--pipeline-yaml", default=None, help="Explicit pipeline yaml override")
    parser.add_argument("--weights-path", default=None, help="Explicit pipeline .pt override")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--decider-node",
        default=None,
        help="Decider node name (for pipelines with several supported deciders)",
    )
    parser.add_argument(
        "--scores-port",
        default=None,
        help="Score source as node.port (default: the port wired into the decider's logits)",
    )
    parser.add_argument(
        "--mask-port",
        default=None,
        help="Ground-truth mask as node.port (default: the single bool 'mask' output)",
    )
    parser.add_argument(
        "--top-k-fraction",
        type=float,
        default=None,
        help="Stage-1 top-k fraction, two-stage decider only (default: from its hparams)",
    )
    parser.add_argument(
        "--num-candidates", type=int, default=256, help="Threshold/quantile sweep resolution"
    )
    parser.add_argument("--output", default=None, help="JSON report path")
    parser.add_argument(
        "--plugins-dir",
        action="append",
        default=None,
        help="Extra plugins directory (repeatable)",
    )

    args = parser.parse_args()
    try:
        calibrate_thresholds(
            trainrun_dir=args.trainrun_dir,
            trainrun_yaml=args.trainrun_yaml,
            pipeline_yaml=args.pipeline_yaml,
            weights_path=args.weights_path,
            split=args.split,
            device=args.device,
            decider_node=args.decider_node,
            scores_port=args.scores_port,
            mask_port=args.mask_port,
            top_k_fraction=args.top_k_fraction,
            num_candidates=args.num_candidates,
            output=args.output,
            plugins_dirs=args.plugins_dir,
        )
    except Exception as error:  # pragma: no cover - CLI surface
        logger.error(f"calibration failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
