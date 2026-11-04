"""Rewrite pipelines from the walnut plugins' node copies to the cuvis-ai built-ins.

cuvis-ai 0.19 took over the generic score, mask, spatial and tiling nodes that
cuvis-ai-patchcore, cuvis-ai-steervit and cuvis-ai-rfdetr carried. This tool rewrites a
pipeline yaml and its fitted ``.pt`` so that every such node runs on the built-in class,
with the same outputs:

- Sixteen classes keep their hparams and ports and only change ``class_name``
  (the fusion, mask, objectness, spatial, gate and tiling nodes, ``FixedPCAProjection``).
- ``ScoreRangeNormalizer`` (patchcore) becomes ``PercentileNormalizer`` in ``statistical``
  mode with the pooled fit: ``low`` / ``high`` (percent) become ``quantile_low`` /
  ``quantile_high`` (fractions), ``floor`` becomes ``clamp: floor`` (or ``none``), the input
  port ``scores`` becomes ``data`` and the buffers ``lo`` / ``hi`` become ``running_min`` /
  ``running_max``.
- ``JointPercentileStretch`` (steervit) becomes ``PercentileNormalizer`` in ``per_frame``
  mode with ``per_frame_quantiles``, ``joint_channels`` (``not per_channel``) and its
  quantisation. It had no state; the normalizer's three buffers are added unset.
- ``ScoreFusion`` and ``ScoreIntersection`` (rfdetr) become ``ScoreMapFusion``: the inputs
  ``a`` and ``b`` become two connections to the variadic ``scores`` port, ``a`` first. When
  both come from the same port (an identity tap that only renames an output), one
  connection is kept: the maximum, minimum or mean of one map is the map itself.

Classes without a built-in (``PatchCoreDetector``, ``SteerViTExtractor``,
``RFDETRSegmenter``, ``MaskPersistence``, ``SamShellGate`` ...) are left as they are and listed
in the report. The ``plugins:`` list gains ``cuvis_ai_builtin`` when needed; a plugin is
dropped only with ``--plugins-dir``, when none of the classes its manifest lists remains.

Examples
--------
Write the migrated pair next to a copy of the catalog::

    migrate-plugin-nodes walnut_best/walnut_best_refit_cuvisnext_cube.yaml --out-dir migrated/

Several pipelines, with the plugin manifests to prune the ``plugins:`` list::

    migrate-plugin-nodes walnut_best/*.yaml --out-dir migrated/ --plugins-dir cuvis_ai/configs/plugins
"""

from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import yaml

BUILTIN_PLUGIN = "cuvis_ai_builtin"

#: Plugin class path -> built-in class path, hparams and ports unchanged.
RENAMED: dict[str, str] = {
    "cuvis_ai_patchcore.node.fusion.ScoreMapFusion": "cuvis_ai.node.fusion.ScoreMapFusion",
    "cuvis_ai_patchcore.node.fusion.DecisionFusion": "cuvis_ai.node.fusion.DecisionFusion",
    "cuvis_ai_patchcore.node.fusion.MaskComposite": "cuvis_ai.node.mask_ops.MaskComposite",
    "cuvis_ai_patchcore.node.fusion.ScoreMapSuppression": (
        "cuvis_ai.node.mask_ops.ScoreMapSuppression"
    ),
    "cuvis_ai_patchcore.node.morphology.MaskMinArea": "cuvis_ai.node.mask_ops.MaskMinArea",
    "cuvis_ai_patchcore.node.morphology.MaskBlobGate": "cuvis_ai.node.mask_ops.MaskBlobGate",
    "cuvis_ai_patchcore.node.morphology.MaskPeakGate": "cuvis_ai.node.mask_ops.MaskPeakGate",
    "cuvis_ai_patchcore.node.objectness.MaskBlobFilter": "cuvis_ai.node.mask_ops.MaskBlobFilter",
    "cuvis_ai_patchcore.node.objectness.SpectralObjectMask": (
        "cuvis_ai.node.spectral_angle_mapper.SpectralObjectMask"
    ),
    "cuvis_ai_patchcore.node.spatial.GridSubsample": "cuvis_ai.node.spatial.GridSubsample",
    "cuvis_ai_patchcore.node.spatial.ScoreUpsample": "cuvis_ai.node.spatial.ScoreUpsample",
    "cuvis_ai_patchcore.node.spatial.ScoreMapSmoothing": "cuvis_ai.node.spatial.ScoreMapSmoothing",
    "cuvis_ai_patchcore.node.gate.FrameScoreGate": (
        "cuvis_ai.node.deciders.frame_score_gate.FrameScoreGate"
    ),
    "cuvis_ai_steervit.node.tiling.ImageTiler": "cuvis_ai.node.tiling.ImageTiler",
    "cuvis_ai_steervit.node.tiling.GridStitcher": "cuvis_ai.node.tiling.GridStitcher",
    "cuvis_ai_rfdetr.node.fixed_pca_projection.FixedPCAProjection": (
        "cuvis_ai.node.dimensionality_reduction.FixedPCAProjection"
    ),
}
SCORE_RANGE = "cuvis_ai_patchcore.node.calibration.ScoreRangeNormalizer"
STRETCH = "cuvis_ai_steervit.node.stretch.JointPercentileStretch"
SCORE_FUSION = "cuvis_ai_rfdetr.node.score_fusion.ScoreFusion"
SCORE_INTERSECTION = "cuvis_ai_rfdetr.node.score_intersection.ScoreIntersection"
PERCENTILE = "cuvis_ai.node.normalization.PercentileNormalizer"
SCORE_MAP_FUSION = "cuvis_ai.node.fusion.ScoreMapFusion"
#: Plugin classes that moved, or were dropped, but have no automatic rewrite here.
NOT_REWRITTEN = {
    "cuvis_ai_patchcore.node.temporal.MaskPersistence": "no built-in (decision on ALL-6670)",
    "cuvis_ai_rfdetr.node.sam_shell_gate.SamShellGate": (
        "rewrite by hand to SpectralAngleMapper + BinaryDecider + ScoreMapSuppression"
    ),
}
#: ScoreFusion modes the built-in reproduces bit for bit.
EXACT_FUSION_MODES = ("mean", "min", "max")


class MigrationError(ValueError):
    """A node or connection the tool cannot rewrite without changing the outputs."""


@dataclass
class Report:
    """What one pipeline's migration did."""

    yaml_path: Path
    renamed: dict[str, str] = field(default_factory=dict)
    rewritten: dict[str, str] = field(default_factory=dict)
    kept: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        """Human-readable summary."""
        out = [f"{self.yaml_path.name}:"]
        for title, rows in (
            ("renamed", self.renamed),
            ("rewritten", self.rewritten),
            ("left on the plugin", self.kept),
        ):
            for node, what in rows.items():
                out.append(f"  {title}: {node} ({what})")
        out += [f"  note: {n}" for n in self.notes]
        return out


# --------------------------------------------------------------------- hparams


def _score_range_hparams(hp: dict[str, Any]) -> dict[str, Any]:
    """ScoreRangeNormalizer hparams as PercentileNormalizer hparams."""
    low = float(hp.get("low", 1.0))
    high = float(hp.get("high", 99.0))
    new = {
        "n_channels": int(hp.get("n_channels", 1)),
        "norm_mode": "statistical",
        "statistical_fit": "pooled",
        "quantile_low": low / 100.0,
        "quantile_high": high / 100.0,
        "clamp": "floor" if hp.get("floor", True) else "none",
        "eps": float(hp.get("eps", 1e-9)),
        "fit_subsample": int(hp.get("fit_subsample", 4)),
        "max_fit_values": int(hp.get("max_fit_values", 4_000_000)),
        "seed": int(hp.get("seed", 0)),
        "invert": bool(hp.get("invert", False)),
    }
    if "name" in hp:
        new["name"] = hp["name"]
    return new


def _stretch_hparams(hp: dict[str, Any], n_channels: int) -> dict[str, Any]:
    """JointPercentileStretch hparams as PercentileNormalizer hparams."""
    new = {
        "n_channels": int(n_channels),
        "norm_mode": "per_frame",
        "per_frame_quantiles": True,
        "joint_channels": not bool(hp.get("per_channel", False)),
        "quantile_low": float(hp.get("low", 2.0)) / 100.0,
        "quantile_high": float(hp.get("high", 98.0)) / 100.0,
        "quantize_levels": hp.get("quantize_levels"),
        "eps": float(hp.get("eps", 1e-6)),
    }
    if "name" in hp:
        new["name"] = hp["name"]
    return new


def _fusion_hparams(hp: dict[str, Any], mode: str, node: str) -> dict[str, Any]:
    """ScoreFusion / ScoreIntersection hparams as ScoreMapFusion hparams."""
    if mode not in EXACT_FUSION_MODES:
        raise MigrationError(
            f"{node}: ScoreFusion mode {mode!r} has no bit-identical built-in "
            f"(only {EXACT_FUSION_MODES}); rewrite it by hand"
        )
    new: dict[str, Any] = {"mode": mode}
    if "name" in hp:
        new["name"] = hp["name"]
    return new


# --------------------------------------------------------------------- yaml


def _split(endpoint: str) -> tuple[str, str, str]:
    """``node.outputs.port`` -> ``(node, "outputs", port)``."""
    node, kind, port = endpoint.split(".", 2)
    return node, kind, port


def _stretch_channels(doc: dict[str, Any], node: str, state: dict[str, Any] | None) -> int:
    """Channels of the tensor feeding a stretch: from its source node's fitted bounds."""
    for conn in doc.get("connections", []):
        target, _, port = _split(conn["target"])
        if target == node and port == "data":
            source = _split(conn["source"])[0]
            bounds = (state or {}).get(source, {}).get("running_min")
            if bounds is not None:
                return int(bounds.shape[0])
            break
    return 3


def migrate_doc(
    doc: dict[str, Any], state: dict[str, Any] | None, report: Report
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the migrated pipeline document and node states (inputs are not modified)."""
    doc = yaml.safe_load(yaml.safe_dump(doc))  # deep copy, plain types
    state = dict(state) if state is not None else None
    fused: dict[str, str] = {}  # ScoreFusion-like nodes
    range_nodes: set[str] = set()

    for node in doc.get("nodes", []):
        name, cls = node["name"], node["class_name"]
        hp = dict(node.get("hparams") or {})
        if cls in RENAMED:
            node["class_name"] = RENAMED[cls]
            report.renamed[name] = f"{cls.rsplit('.', 1)[1]} -> {RENAMED[cls]}"
        elif cls == SCORE_RANGE:
            node["class_name"] = PERCENTILE
            node["hparams"] = _score_range_hparams(hp)
            range_nodes.add(name)
            report.rewritten[name] = "ScoreRangeNormalizer -> PercentileNormalizer (pooled fit)"
            if state is not None and name in state:
                old = state[name]
                missing = {"lo", "hi"} - set(old)
                if missing:
                    raise MigrationError(f"{name}: .pt state lacks {sorted(missing)}")
                state[name] = {
                    "running_min": old["lo"].clone(),
                    "running_max": old["hi"].clone(),
                    "_norm_frame_count": torch.zeros((), dtype=torch.long),
                }
        elif cls == STRETCH:
            channels = _stretch_channels(doc, name, state)
            node["class_name"] = PERCENTILE
            node["hparams"] = _stretch_hparams(hp, channels)
            report.rewritten[name] = (
                f"JointPercentileStretch -> PercentileNormalizer (per_frame, {channels} channels)"
            )
            if state is not None:
                if state.get(name):
                    raise MigrationError(f"{name}: a stretch with state is unexpected")
                state[name] = {
                    "running_min": torch.full((channels,), math.nan),
                    "running_max": torch.full((channels,), math.nan),
                    "_norm_frame_count": torch.zeros((), dtype=torch.long),
                }
        elif cls in (SCORE_FUSION, SCORE_INTERSECTION):
            mode = "min" if cls == SCORE_INTERSECTION else str(hp.get("mode", "gmean"))
            node["class_name"] = SCORE_MAP_FUSION
            node["hparams"] = _fusion_hparams(hp, mode, name)
            fused[name] = mode
            report.rewritten[name] = f"{cls.rsplit('.', 1)[1]}({mode}) -> ScoreMapFusion"
        elif cls in NOT_REWRITTEN:
            report.kept[name] = f"{cls.rsplit('.', 1)[1]}: {NOT_REWRITTEN[cls]}"
        elif not cls.startswith("cuvis_ai."):
            report.kept[name] = cls.rsplit(".", 1)[1]

    # connections: ScoreRangeNormalizer.inputs.scores -> data; ScoreFusion a/b -> scores
    conns = doc.get("connections", [])
    for conn in conns:
        target, kind, port = _split(conn["target"])
        if target in range_nodes and kind == "inputs" and port == "scores":
            conn["target"] = f"{target}.inputs.data"
    sources: dict[str, dict[str, str]] = {name: {} for name in fused}
    for conn in conns:
        target, kind, port = _split(conn["target"])
        if target in fused and kind == "inputs" and port in ("a", "b"):
            sources[target][port] = conn["source"]
    for name, ab in sources.items():
        if set(ab) != {"a", "b"}:
            raise MigrationError(f"{name}: expected inputs a and b, found {sorted(ab)}")
        if ab["a"] == ab["b"]:
            report.notes.append(f"{name}: a and b came from {ab['a']}; one connection kept")
    # the variadic port collects in connection order: emit a then b where the first of them stood
    rewired: list[dict[str, Any]] = []
    done: set[str] = set()
    for conn in conns:
        target, kind, port = _split(conn["target"])
        if target in fused and kind == "inputs" and port in ("a", "b"):
            if target not in done:
                ab = sources[target]
                for source in dict.fromkeys((ab["a"], ab["b"])):
                    rewired.append({"source": source, "target": f"{target}.inputs.scores"})
                done.add(target)
            continue
        rewired.append(conn)
    doc["connections"] = rewired

    plugins = list(doc.get("plugins") or [])
    if (report.renamed or report.rewritten) and BUILTIN_PLUGIN not in plugins:
        plugins.insert(0, BUILTIN_PLUGIN)
    doc["plugins"] = plugins
    return doc, state


def prune_plugins(doc: dict[str, Any], plugins_dir: Path, report: Report) -> None:
    """Drop listed plugins none of whose manifest classes the pipeline still uses."""
    used = {n["class_name"] for n in doc.get("nodes", [])}
    kept = []
    for plugin in doc.get("plugins") or []:
        manifest = plugins_dir / f"{plugin}.yaml"
        if plugin == BUILTIN_PLUGIN or not manifest.is_file():
            kept.append(plugin)
            continue
        caps = yaml.safe_load(manifest.read_text(encoding="utf-8")).get("capabilities") or []
        classes = {c.get("class_name") for c in caps if isinstance(c, dict)}
        if classes & used:
            kept.append(plugin)
        else:
            report.notes.append(f"plugin {plugin} dropped: none of its classes remains")
    doc["plugins"] = kept


# --------------------------------------------------------------------- files


def migrate_file(yaml_path: Path, out_dir: Path, plugins_dir: Path | None = None) -> Report:
    """Migrate one pipeline yaml (and its ``.pt`` sibling, when present) into ``out_dir``."""
    report = Report(yaml_path)
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    pt_path = yaml_path.with_suffix(".pt")
    checkpoint = (
        torch.load(pt_path, map_location="cpu", weights_only=True) if pt_path.is_file() else None
    )
    state = checkpoint.get("state_dict") if checkpoint is not None else None
    new_doc, new_state = migrate_doc(doc, state, report)
    if plugins_dir is not None:
        prune_plugins(new_doc, plugins_dir, report)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = out_dir / yaml_path.name
    out_yaml.write_text(
        yaml.safe_dump(new_doc, sort_keys=False, allow_unicode=True, width=100), encoding="utf-8"
    )
    if checkpoint is not None:
        torch.save({**checkpoint, "state_dict": new_state}, out_dir / pt_path.name)
    else:
        report.notes.append("no .pt sibling; yaml only")
    return report


def main(argv: Iterable[str] | None = None) -> int:
    """Command-line entry point (``migrate-plugin-nodes``)."""
    ap = argparse.ArgumentParser(prog="migrate-plugin-nodes", description=__doc__.split("\n\n")[0])
    ap.add_argument("pipelines", nargs="+", type=Path, help="pipeline yaml files")
    ap.add_argument("--out-dir", type=Path, required=True, help="where the migrated pairs go")
    ap.add_argument(
        "--plugins-dir",
        type=Path,
        default=None,
        help="plugin manifests; with it, plugins none of whose classes remain are dropped",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    failed = 0
    for path in args.pipelines:
        try:
            report = migrate_file(path, args.out_dir, args.plugins_dir)
        except MigrationError as exc:
            print(f"{path.name}: NOT migrated: {exc}", file=sys.stderr)
            failed += 1
            continue
        print("\n".join(report.lines()))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
