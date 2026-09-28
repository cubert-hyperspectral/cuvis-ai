"""Build the combined walnut pipelines for cuvis.next (cube mode): FO anomaly + SEG shells on one cube.

Two FO families, each joined with the SEG ensemble `walnut_seg_ens_rgb_cir_mean_v2`:
- `walnut_combo_or`:    FO `walnut_fo_multiscale_effad_or_gated` (multi-scale gate OR EfficientAD gate);
- `walnut_combo_gated`: FO `walnut_fo_multiscale_gated` (the multi-scale gate alone).

The FO branch is frame-gated as deployed: its map shows only on anomalous frames (zeros otherwise). The SEG branch is
the ensemble as deployed: a mask threshold on the fused score plus the heatmap, no frame gate. Outputs to pick in
cuvis.next's Displayed Output: the FO map (`display.scores` / `gate.scores`), `ShellMask.decisions` and
`ShellHeatmap.scores`.

Tiers (FO precision x SEG tier): `tf32_exact`, `tf32_fast`, `tf32_trt32`, `tf32_trt16`, `fp16_trt16`.
- There is no FO float32 tier: `import rfdetr` sets PyTorch's float32 matmul precision to "high" (TF32) for the
  whole process, so next to the SEG models the FO branch runs TF32 anyway. The FO tiers are the validated `_tf32` /
  `_fp16` variants with their own calibrated thresholds.
- The SEG tiers are the deployed `_exact` (fp32, bit-identical to the default), `_fast`, `_trt_fp32`, `_trt_fp16`.

Build: both standalone pipelines are restored with their weights (CPU, no forward). The SEG nodes are connected into
the FO pipeline following the SEG yaml's own connection list:
- one CU3SDataNode, FO's `cu3s_data`; SEG's `DataSource` is dropped;
- SEG `Fuse` becomes `ShellFuse` (a fresh ScoreFusion with the same hparams; it holds no state), because FO already
  has a `fuse` node and the names differ only by case.
The pipeline is saved (yaml + .pt) and post-processed like the FO deploy yamls: bare plugin names, empty
CU3SDataNode hparams, canonical key order. No forward runs before the save, so the SEG selectors keep their fresh
state and calibrate on the first 20 frames of a session. Every tier of a family has the same state, so one .pt per
family is kept and hardlinked under the other tiers' names (each tier's own save is compared tensor by tensor first).

Run in an env with all five plugins, e.g. a throwaway uv overlay on the SEG child env:
    python build_combo.py [--families or,gated] [--tiers tf32_exact,...] --scratch <dir on a drive with space>
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
PIPE = HERE.parent
FO_DIR = PIPE / "walnut_fo"
SEG_DIR = PIPE / "walnut_seg"
SEG_BASE = "walnut_seg_ens_rgb_cir_mean_v2"
AUTHOR = "raj@cubert-gmbh.de"

FAMILIES = {
    "or": {"fo": "walnut_fo_multiscale_effad_or_gated", "fo_out": "display.scores",
           "fo_text": "multi-scale t1+t2 gate OR EfficientAD gate; display = the multi-scale gated map when its gate "
                      "opens, else the EfficientAD gated map, zeros when neither passes"},
    "gated": {"fo": "walnut_fo_multiscale_gated", "fo_out": "gate.scores",
              "fo_text": "multi-scale t1+t2 feature banks through FrameScoreGate; the map shows only when the gate "
                         "opens, zeros otherwise"},
}
# tier: (FO variant suffix, SEG tier suffix, FO text, SEG text)
TIERS = {
    "tf32_exact": ("_tf32", "_exact", "TF32", "fp32 PyTorch with GPU input (bit-identical to the SEG default)"),
    "tf32_fast": ("_tf32", "_fast", "TF32", "fp16 + JIT trace + GPU input"),
    "tf32_trt32": ("_tf32", "_trt_fp32", "TF32", "TensorRT fp32 engine (TF32 allowed)"),
    "tf32_trt16": ("_tf32", "_trt_fp16", "TF32", "TensorRT fp16 engine"),
    "fp16_trt16": ("_fp16", "_trt_fp16", "float16 autocast", "TensorRT fp16 engine"),
}
RENAME = {"DataSource": "cu3s_data", "Fuse": "ShellFuse"}
PLUGIN_OF = (("cuvis_ai.", "cuvis_ai_builtin"), ("cuvis_ai_patchcore.", "patchcore"), ("cuvis_ai_steervit.", "steervit"),
             ("cuvis_ai_efficientad.", "efficientad"), ("cuvis_ai_rfdetr.", "rfdetr_seg"))


def stem(family: str, tier: str) -> str:
    return f"walnut_combo_{family}_{tier}_cuvisnext_cube"


def thresholds(cfg: dict) -> dict[str, float]:
    return {n["name"]: n["hparams"]["threshold"] for n in cfg["nodes"] if n["name"] in ("gate", "gate_effad")}


def class_defaults(class_name: str) -> dict:
    """Constructor defaults across the class hierarchy (a node may pass **kwargs to its base), enums by value."""
    import enum
    import importlib
    import inspect

    module, _, cls = class_name.rpartition(".")
    defaults: dict = {}
    for klass in getattr(importlib.import_module(module), cls).__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for k, p in inspect.signature(init).parameters.items():
            if p.default is inspect.Parameter.empty or k in defaults:
                continue
            defaults[k] = p.default.value if isinstance(p.default, enum.Enum) else p.default
    return defaults


def source_hparams(node: dict, source: dict) -> dict:
    """The source yaml's hparams for `node`, after checking that the saved extras are only class defaults.

    save_to_file writes every hparam of a node, defaults included; the deployed yamls list only the ones that
    differ. Keeping the source's list keeps the combined yaml node-for-node equal to its two sources.
    """
    saved = {k: v for k, v in (node.get("hparams") or {}).items() if k != "name"}
    src = {k: v for k, v in (source.get("hparams") or {}).items() if k != "name"}
    if node["class_name"] != source["class_name"]:
        raise SystemExit(f"{node['name']}: class {node['class_name']} != source {source['class_name']}")
    wrong = {k: (saved.get(k), v) for k, v in src.items() if saved.get(k) != v}
    if wrong:
        raise SystemExit(f"{node['name']}: saved hparams differ from the source: {wrong}")
    defaults = class_defaults(node["class_name"])
    extra = {k: v for k, v in saved.items() if k not in src}
    not_default = {k: v for k, v in extra.items() if k not in defaults or defaults[k] != v}
    if not_default:
        raise SystemExit(f"{node['name']}: saved hparams beyond the source that are not defaults: {not_default}")
    return src


def build(family: str, tier: str, out_yaml: Path) -> None:
    """Restore the two standalone pipelines, join them and save `out_yaml` (+ its sibling .pt)."""
    from cuvis_ai_core.training.config import PipelineMetadata
    from cuvis_ai_core.utils.restore import restore_pipeline
    from cuvis_ai_rfdetr.node.score_fusion import ScoreFusion

    fo_sfx, seg_sfx, fo_txt, seg_txt = TIERS[tier]
    fo_stem = f"{FAMILIES[family]['fo']}{fo_sfx}_cuvisnext_cube"
    seg_stem = f"{SEG_BASE}{seg_sfx}_cuvisnext_cube"
    fo_cfg = yaml.safe_load(open(FO_DIR / f"{fo_stem}.yaml", encoding="utf-8"))
    seg_cfg = yaml.safe_load(open(SEG_DIR / f"{seg_stem}.yaml", encoding="utf-8"))
    fo = restore_pipeline(str(FO_DIR / f"{fo_stem}.yaml"), weights_path=str(FO_DIR / f"{fo_stem}.pt"), device="cpu")
    seg = restore_pipeline(str(SEG_DIR / f"{seg_stem}.yaml"), weights_path=str(SEG_DIR / f"{seg_stem}.pt"),
                           device="cpu")
    fo_nodes = {n.name: n for n in fo.nodes if not isinstance(n, str)}
    seg_nodes = {n.name: n for n in seg.nodes if not isinstance(n, str)}
    fuse_hp = {k: v for k, v in next(n for n in seg_cfg["nodes"] if n["name"] == "Fuse")["hparams"].items()
               if k != "name"}
    nodes = {k: v for k, v in seg_nodes.items() if k not in ("DataSource", "Fuse")}
    clash = set(nodes) & set(fo_nodes)
    if clash:
        raise SystemExit(f"node names in both pipelines: {sorted(clash)}")
    nodes["cu3s_data"] = fo_nodes["cu3s_data"]
    nodes["ShellFuse"] = ScoreFusion(name="ShellFuse", **fuse_hp)
    for c in seg_cfg["connections"]:
        src_node, _, src_port = c["source"].split(".")
        dst_node, _, dst_port = c["target"].split(".")
        src, dst = nodes[RENAME.get(src_node, src_node)], nodes[RENAME.get(dst_node, dst_node)]
        fo.connect(getattr(src.outputs, src_port), getattr(dst.inputs, dst_port))

    thr = thresholds(fo_cfg)
    thr_txt = ", ".join(f"{k} {v}" for k, v in thr.items())
    fam = FAMILIES[family]
    desc = (
        f"Combined walnut pipeline, FO anomaly + SEG shells on one cube. FO ({fo_stem.replace('_cuvisnext_cube', '')}, "
        f"{fo_txt}; thresholds {thr_txt}): {fam['fo_text']}. SEG ({seg_stem.replace('_cuvisnext_cube', '')}, "
        f"{seg_txt}): RGB 640/550/470 + CIR 850/660/550 RF-DETR-Seg-L at 504 px, mean fusion; mask = fused score >= "
        f"0.5, no frame gate; its selectors calibrate on the first 20 frames of a session. Outputs to pick in "
        f"cuvis.next: {fam['fo_out']} (FO heatmap, gated), ShellMask.decisions (shell mask), ShellHeatmap.scores "
        f"(shell heatmap). Weights: the family .pt (every tier of walnut_combo_{family} shares it); the SEG models load "
        f"from their checkpoint_path."
    )
    tags = sorted(set(fo_cfg["metadata"].get("tags") or []) | set(seg_cfg["metadata"].get("tags") or [])
                  | {"combined", "segmentation", "walnut_combo"})
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    fo.save_to_file(str(out_yaml), metadata=PipelineMetadata(name=out_yaml.stem, description=desc, author=AUTHOR,
                                                              tags=tags))
    cfg = yaml.safe_load(open(out_yaml, encoding="utf-8"))
    cfg["plugins"] = [p for prefix, p in PLUGIN_OF if any(n["class_name"].startswith(prefix) for n in cfg["nodes"])]
    # The source node of every combined node: FO nodes by name, SEG nodes by name, ShellFuse = SEG's Fuse.
    sources = {n["name"]: n for n in fo_cfg["nodes"]}
    sources.update({RENAME.get(n["name"], n["name"]): n for n in seg_cfg["nodes"] if n["name"] != "DataSource"})
    for n in cfg["nodes"]:
        n["hparams"] = ({} if n["class_name"] == "cuvis_ai.node.data.CU3SDataNode"
                        else source_hparams(n, sources[n["name"]]))
    ordered = {k: cfg[k] for k in ("metadata", "plugins", "nodes", "connections") if k in cfg}
    ordered.update({k: v for k, v in cfg.items() if k not in ordered})
    out_yaml.write_text(yaml.safe_dump(ordered, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"built {out_yaml.name}: {len(cfg['nodes'])} nodes, {len(cfg['connections'])} connections, plugins "
          f"{cfg['plugins']}, FO {fo_stem}, SEG {seg_stem}, thresholds {thr}", flush=True)


def same_state(a: Path, b: Path) -> bool:
    sa = torch.load(a, map_location="cpu", weights_only=False)["state_dict"]
    sb = torch.load(b, map_location="cpu", weights_only=False)["state_dict"]
    if sa.keys() != sb.keys():
        return False
    for node in sa:
        if sa[node].keys() != sb[node].keys():
            return False
        for k, v in sa[node].items():
            w = sb[node][k]
            if isinstance(v, torch.Tensor):
                if not (isinstance(w, torch.Tensor) and v.dtype == w.dtype and v.shape == w.shape
                        and torch.equal(v.nan_to_num(123.0), w.nan_to_num(123.0))
                        and torch.equal(v.isnan(), w.isnan())):
                    return False
            elif v != w:
                return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="or,gated")
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--scratch", required=True, help="where the tiers' own .pt land before the tensor comparison")
    a = ap.parse_args()
    scratch = Path(a.scratch)
    for family in a.families.split(","):
        tiers = a.tiers.split(",")
        family_pt = HERE / f"{stem(family, tiers[0])}.pt"
        for i, tier in enumerate(tiers):
            name = stem(family, tier)
            if i == 0:
                build(family, tier, HERE / f"{name}.yaml")
                continue
            build(family, tier, scratch / f"{name}.yaml")
            if not same_state(scratch / f"{name}.pt", family_pt):
                raise SystemExit(f"{name}: state differs from {family_pt.name}; not hardlinking")
            shutil.move(str(scratch / f"{name}.yaml"), HERE / f"{name}.yaml")
            os.remove(scratch / f"{name}.pt")
            target = HERE / f"{name}.pt"
            if target.exists():
                target.unlink()
            os.link(family_pt, target)
            print(f"  {name}.pt = hardlink of {family_pt.name} (state identical)", flush=True)
    print("BUILD DONE")


if __name__ == "__main__":
    main()
