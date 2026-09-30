"""Build the walnut FO `gmm_or` family for cuvis.next (cube mode): the multi-scale gate OR a gate on the fused map.

`walnut_fo_multiscale_gmm_or_gated<tier>_cuvisnext_cube` replaces EfficientAD in the deployed OR
(`walnut_fo_multiscale_effad_or_gated`) with a spectral branch; walnut FO report Steps 37-38. The graph is the gated
multi-scale pipeline of the same tier (`walnut_fo_multiscale_gated<tier>`: SteerViT t1 + t2 feature banks -> `fuse`
-> `gate`, node for node and threshold for threshold) plus, on the same cube:

    cu3s_data.cube -> mix_grid (GridSubsample, every 4th pixel) -> mix_snv (SNVCorrection)
      -> mix_gmm (GaussianMixtureClusterer, 32 full-covariance components, per-pixel log-likelihood)
      -> mix_up (ScoreUpsample to the cube, bilinear) -> mix_norm (ScoreRangeNormalizer, invert: the negative
         log-likelihood onto the 1-99 % range of the 34 clean VAL frames)
    fuse -> ms_norm (ScoreRangeNormalizer onto the same frames' 1-99 % range of the multi-scale map)
    [ms_norm, mix_norm] -> fused (ScoreMapFusion softmin, beta 10) -> gate_fused (FrameScoreGate)
    [gate_fused, gate] -> display (ScoreMapFusion first);  [gate_fused, gate].decisions -> display_mask (DecisionFusion)
    fused -> fused_map, mix_norm -> mixture_map (one-input ScoreMapFusion taps: the ungated maps in Displayed Output)

`gate_fused` comes before `gate` in the nodes list and in the connections: ScoreMapFusion(first) and
DecisionFusion(first) take their inputs in the yaml's connection order (and `save_to_file` writes a fan-in in the
order its source nodes entered the graph), so the display is the fused gated map when its gate opens, else the
multi-scale gated map, zeros when neither passes.

Tiers (the gated family's): float32 "", _tf32, _fp16, _fp16trt. The multi-scale part of a tier is its gated yaml
verbatim; the spectral branch is float32 PyTorch in every tier. Its matmuls need exact float32 (the Gaussian
quadratic term cancels): TF32 must stay off for it, which holds in FO-only pipelines (the SteerViT / PatchCore `tf32`
options set TF32 around their own matmuls and restore it), not next to `import rfdetr` (walnut_combo).

State: the gated tier's .pt (identical for every tier) plus the mixture (`--gmm`, the experiment's
runs/gmm_v1/gmm_snv_k32/fitted.pt: SNV spectra of the 200 v3 TRAIN frames, 400k pixels, seed 0) and the two
normalisers (lo / hi = `--lohi`, the experiment's g4_checks.json `lohi_VAL`). One .pt for the family, hardlinked
under the other tiers' names. The gates are stateless: `gate` keeps the tier's calibrated values; `gate_fused` gets
`--fused-threshold` / `--fused-mask-threshold` until calibrate_live.py --pipeline gmm_or sets them per session.

Needs torch + PyYAML; with the plugins importable (cuvis-ai + cuvis-ai-patchcore >= the softmin / invert / spatial
nodes) it also builds the new nodes from their hparams, loads their state and runs them on a random cube.
    python build_gmm_or.py --gmm <fitted.pt> --lohi <g4_checks.json> --out <dir> [--tiers ,_tf32,_fp16,_fp16trt]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path

import torch
import yaml

HERE = Path(__file__).resolve().parent
BASE = "walnut_fo_multiscale_gated"
FAMILY = "walnut_fo_multiscale_gmm_or_gated"
TIERS = ("", "_tf32", "_fp16", "_fp16trt")
BETA = 10.0
PATCHCORE = "cuvis_ai_patchcore.node."
NORM = {"n_channels": 1, "low": 1.0, "high": 99.0, "floor": True, "fit_subsample": 4, "max_fit_values": 4000000,
        "seed": 0, "eps": 1e-09}  # norm_t1's hparams; lo / hi are set from the VAL maps, not fitted here
# how the experiment fitted the mixture (inference needs only its buffers)
MIXTURE = {"n_components": 32, "covariance_type": "full", "reg_covar": 0.001, "max_iter": 300, "n_init": 1,
           "random_state": 0, "max_fit_pixels": 400000, "fit_seed": 0}
GMM_KEYS = ("_initialized", "means", "precisions_chol", "weights")


def stem(prefix: str, tier: str) -> str:
    return f"{prefix}{tier}_cuvisnext_cube"


def node(name: str, class_name: str, hparams: dict) -> dict:
    return {"name": name, "class_name": class_name, "hparams": hparams}


def conn(src: str, dst: str) -> dict:
    return {"source": src, "target": dst}


def spectral_nodes(gate_hp: dict, thr: float, mask_thr: float, k: int) -> tuple[list[dict], list[dict]]:
    """The new nodes before `gate` and after it, the fused gate a copy of `gate`'s hparams with its own values."""
    fused_hp = {**gate_hp, "threshold": thr, "mask_threshold": mask_thr}
    before = [
        node("ms_norm", PATCHCORE + "calibration.ScoreRangeNormalizer", dict(NORM)),
        node("mix_grid", PATCHCORE + "spatial.GridSubsample", {"stride": 4}),
        node("mix_snv", "cuvis_ai.node.pretreatments.snv.SNVCorrection", {"eps": 1e-08}),
        node("mix_gmm", "cuvis_ai.node.clustering.gmm.GaussianMixtureClusterer", {**MIXTURE, "n_components": k}),
        node("mix_up", PATCHCORE + "spatial.ScoreUpsample", {"mode": "bilinear"}),
        node("mix_norm", PATCHCORE + "calibration.ScoreRangeNormalizer", {**NORM, "invert": True}),
        node("fused", PATCHCORE + "fusion.ScoreMapFusion", {"mode": "softmin", "weights": None, "beta": BETA}),
        node("gate_fused", PATCHCORE + "gate.FrameScoreGate", fused_hp),
    ]
    after = [
        node("display", PATCHCORE + "fusion.ScoreMapFusion", {"mode": "first", "weights": None}),
        node("display_mask", PATCHCORE + "fusion.DecisionFusion", {"mode": "first"}),
        node("fused_map", PATCHCORE + "fusion.ScoreMapFusion", {"mode": "mean", "weights": None}),
        node("mixture_map", PATCHCORE + "fusion.ScoreMapFusion", {"mode": "mean", "weights": None}),
    ]
    return before, after


SPECTRAL_CONNECTIONS = [  # before fuse -> gate: gate_fused enters the graph first
    conn("fuse.outputs.scores", "ms_norm.inputs.scores"),
    conn("cu3s_data.outputs.cube", "mix_grid.inputs.cube"),
    conn("mix_grid.outputs.cube", "mix_snv.inputs.cube"),
    conn("mix_snv.outputs.cube", "mix_gmm.inputs.cube"),
    conn("mix_gmm.outputs.scores", "mix_up.inputs.scores"),
    conn("cu3s_data.outputs.cube", "mix_up.inputs.reference"),
    conn("mix_up.outputs.scores", "mix_norm.inputs.scores"),
    conn("ms_norm.outputs.normalized", "fused.inputs.scores"),
    conn("mix_norm.outputs.normalized", "fused.inputs.scores"),
    conn("fused.outputs.scores", "gate_fused.inputs.scores"),
]
DISPLAY_CONNECTIONS = [  # after fuse -> gate, fused gate first
    conn("gate_fused.outputs.scores", "display.inputs.scores"),
    conn("gate.outputs.scores", "display.inputs.scores"),
    conn("gate_fused.outputs.decisions", "display_mask.inputs.decisions"),
    conn("gate.outputs.decisions", "display_mask.inputs.decisions"),
    conn("fused.outputs.scores", "fused_map.inputs.scores"),
    conn("mix_norm.outputs.normalized", "mixture_map.inputs.scores"),
]


def tier_text(base_desc: str) -> str:
    """The base's speed-variant sentence(s), if any (float32 has none)."""
    i = base_desc.find("Speed variant:")
    return "" if i < 0 else " " + base_desc[i:].strip()


def build_cfg(base: dict, tier: str, thr: float, mask_thr: float, k: int) -> dict:
    names = [n["name"] for n in base["nodes"]]
    if names[-1] != "gate" or names.count("gate") != 1:
        raise SystemExit(f"{stem(BASE, tier)}: expected `gate` as the last node, got {names}")
    gate_cfg = base["nodes"][-1]
    before, after = spectral_nodes(dict(gate_cfg["hparams"]), thr, mask_thr, k)
    clash = {n["name"] for n in before + after} & set(names)
    if clash:
        raise SystemExit(f"node names already in {stem(BASE, tier)}: {sorted(clash)}")
    nodes = base["nodes"][:-1] + before + [gate_cfg] + after
    gate_in = [c for c in base["connections"] if c["target"].startswith("gate.")]
    if gate_in != [conn("fuse.outputs.scores", "gate.inputs.scores")]:
        raise SystemExit(f"{stem(BASE, tier)}: unexpected gate inputs {gate_in}")
    others = [c for c in base["connections"] if c not in gate_in]
    connections = others + SPECTRAL_CONNECTIONS + gate_in + DISPLAY_CONNECTIONS
    desc = (
        "OR-gated walnut FO detector with a spectral branch (replaces EfficientAD). Multi-scale t1+t2 feature banks "
        "-> fuse -> gate, OR fused -> gate_fused: fused = soft minimum (beta 10) of the multi-scale map (ms_norm) and "
        f"a Gaussian-mixture map (mix_norm), each on the 1-99 % range of the 34 clean VAL frames. Mixture: the "
        f"61-band spectra of every 4th pixel (mix_grid), SNV (mix_snv), a {k}-component full-covariance Gaussian "
        "mixture fitted on the 200 v3 TRAIN frames (mix_gmm; negative log-likelihood), bilinear back to the cube "
        "(mix_up). Display = the fused gated map when gate_fused opens, else the multi-scale gated map "
        "(ScoreMapFusion first; zeros when neither passes); FO mask: display_mask.decisions (DecisionFusion first, "
        "same order). Ungated inspection maps: fused_map.scores, mixture_map.scores. Thresholds: max clean x 1.10 "
        "per gate, set per session with calibrate_live.py --pipeline gmm_or (a pooled threshold misses fakes on "
        "another day). The mixture runs in float32 with TF32 off." + tier_text(base["metadata"]["description"])
    )
    meta = dict(base["metadata"])
    meta.update(name=stem(FAMILY, tier), description=desc,
                tags=sorted(set(meta.get("tags") or []) | {"gaussian_mixture", "or_gate", "snv"}))
    cfg = {"metadata": meta, "plugins": list(base["plugins"]), "nodes": nodes, "connections": connections}
    if set(cfg["plugins"]) != {"cuvis_ai_builtin", "patchcore", "steervit"}:
        raise SystemExit(f"unexpected plugins {cfg['plugins']}")
    return cfg


def write_yaml(path: Path, cfg: dict) -> None:
    text = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True)
    path.write_bytes(text.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))  # CRLF like the stack yamls


def family_state(base_pt: Path, gmm: Path, lohi: dict, meta: dict, k: int, names: list[str]) -> dict:
    """The base .pt's node states plus the new nodes', in the yaml's node order (`names`); the stateless nodes the
    base .pt lacks (its `gate`: the gated family loads the ungated .pt) get an empty entry, as save_to_file writes."""
    ck = torch.load(base_pt, map_location="cpu", weights_only=False)
    g = torch.load(gmm, map_location="cpu", weights_only=False)
    if int(g["hparams"]["k"]) != k or g["hparams"]["prep"] != "snv":
        raise SystemExit(f"{gmm}: a {g['hparams']['prep']} K {g['hparams']['k']} mixture, expected snv K {k}")
    if tuple(sorted(g["state_dict"])) != GMM_KEYS or not bool(g["state_dict"]["_initialized"].all()):
        raise SystemExit(f"{gmm}: unexpected mixture state {sorted(g['state_dict'])}")
    sd = ck["state_dict"]
    lohi_key = f"k{k}"

    def lh(key: str) -> OrderedDict:
        lo, hi = (float(v) for v in lohi[key])
        if not hi > lo:
            raise SystemExit(f"lohi {key}: hi {hi} <= lo {lo}")
        return OrderedDict(lo=torch.tensor([lo], dtype=torch.float32), hi=torch.tensor([hi], dtype=torch.float32))

    new = OrderedDict(
        ms_norm=lh("ms"), mix_grid=OrderedDict(), mix_snv=OrderedDict(),
        mix_gmm=OrderedDict((key, g["state_dict"][key].clone()) for key in GMM_KEYS),
        mix_up=OrderedDict(), mix_norm=lh(lohi_key), fused=OrderedDict(), gate_fused=OrderedDict(),
        display=OrderedDict(), display_mask=OrderedDict(), fused_map=OrderedDict(), mixture_map=OrderedDict())
    clash = set(new) & set(sd)
    if clash:
        raise SystemExit(f"{base_pt.name} already has state for {sorted(clash)}")
    unknown = (set(sd) | set(new)) - set(names)
    if unknown:  # cuvis.next loads weights strictly: no state for a node the yaml does not have
        raise SystemExit(f"state for nodes not in the yaml: {sorted(unknown)}")
    stateless = [n for n in names if n not in sd and n not in new]
    if stateless != ["gate"]:
        raise SystemExit(f"nodes without state besides gate: {stateless}")
    out = OrderedDict((n, sd[n] if n in sd else new.get(n, OrderedDict())) for n in names)
    return {"state_dict": out, "metadata": {**ck.get("metadata", {}), **meta}}


def check_nodes(cfg: dict, state: dict) -> None:
    """Build the spectral nodes from the yaml's hparams, load their state, run them on a random cube; the fused gate
    and the display order on a crafted frame."""
    import importlib

    make = {}
    for n in cfg["nodes"]:
        if n["name"] in ("ms_norm", "mix_grid", "mix_snv", "mix_gmm", "mix_up", "mix_norm", "fused", "gate_fused",
                         "gate", "display", "display_mask"):
            mod, _, cls = n["class_name"].rpartition(".")
            obj = getattr(importlib.import_module(mod), cls)(**n["hparams"])
            obj.load_state_dict(state["state_dict"][n["name"]])
            obj._statistically_initialized = True
            make[n["name"]] = obj.eval()
    gen = torch.Generator().manual_seed(0)
    cube = torch.rand(1, 40, 44, 61, generator=gen) * 3000 + 500
    grid = make["mix_grid"](cube=cube)["cube"]
    ll = make["mix_gmm"](cube=make["mix_snv"](cube=grid)["cube"])["scores"]
    up = make["mix_up"](scores=ll, reference=cube)["scores"]
    mix = make["mix_norm"](scores=up)["normalized"]
    fuse = torch.rand(1, 40, 44, 1, generator=gen) * 2
    ms = make["ms_norm"](scores=fuse)["normalized"]
    fused = make["fused"](scores=[ms, mix])["scores"]
    lo = torch.minimum(ms, mix)
    if not (torch.isfinite(fused).all() and (fused >= lo - 1e-6).all() and (fused <= lo + 0.0694).all()):
        raise SystemExit("fused map outside [min, min + log(2) / beta]")
    # display order: both gates open -> the fused gated map; only gate -> the multi-scale gated map
    make["gate_fused"].threshold, make["gate"].threshold = -1.0, -1.0
    make["gate_fused"].mask_threshold, make["gate"].mask_threshold = 0.5, 0.5
    gf, gm = make["gate_fused"](scores=fused), make["gate"](scores=fuse)
    both = make["display"](scores=[gf["scores"], gm["scores"]])["scores"]
    make["gate_fused"].threshold = 1e9
    gf2 = make["gate_fused"](scores=fused)
    only = make["display"](scores=[gf2["scores"], gm["scores"]])["scores"]
    mask = make["display_mask"](decisions=[gf["decisions"], gm["decisions"]])["decisions"]
    if not (torch.equal(both, fused) and torch.equal(only, fuse) and torch.equal(mask, fused > 0.5)):
        raise SystemExit("display / display_mask priority wrong")
    print(f"  node check OK: grid {tuple(grid.shape)}, log-lik {float(ll.min()):.1f}..{float(ll.max()):.1f}, "
          f"mixture map {float(mix.min()):.2f}..{float(mix.max()):.2f}, fused {float(fused.min()):.3f}.."
          f"{float(fused.max()):.3f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--gmm", required=True, help="the experiment's fitted.pt of the mixture (gmm_snv_k32)")
    ap.add_argument("--lohi", required=True, help="g4_checks.json with lohi_VAL (ms, k32)")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default=str(HERE), help="output folder (default: this walnut_fo folder)")
    ap.add_argument("--tiers", default=",".join(TIERS), help="comma list of tier suffixes; '' = float32")
    ap.add_argument("--fused-threshold", type=float, required=True, help="gate_fused threshold until calibrated")
    ap.add_argument("--fused-mask-threshold", type=float, required=True, help="gate_fused mask_threshold")
    ap.add_argument("--no-check", action="store_true", help="skip the node check (plugins not importable)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tiers = a.tiers.split(",")
    lohi = json.loads(Path(a.lohi).read_text(encoding="utf-8"))["lohi_VAL"]
    base_pts = [HERE / f"{stem(BASE, t)}.pt" for t in tiers]
    ref = base_pts[0]
    for p in base_pts[1:]:  # every tier's gated .pt is the same file (hardlinks) -> one family .pt
        if not os.path.samefile(p, ref):
            raise SystemExit(f"{p.name} is not a hardlink of {ref.name}; compare the states first")
    family_pt = out / f"{stem(FAMILY, tiers[0])}.pt"
    for i, tier in enumerate(tiers):
        base = yaml.safe_load((HERE / f"{stem(BASE, tier)}.yaml").read_text(encoding="utf-8"))
        cfg = build_cfg(base, tier, a.fused_threshold, a.fused_mask_threshold, a.k)
        write_yaml(out / f"{stem(FAMILY, tier)}.yaml", cfg)
        if i == 0:
            meta = {k: cfg["metadata"][k] for k in ("name", "description", "tags")}
            state = family_state(ref, Path(a.gmm), lohi, meta, a.k, [n["name"] for n in cfg["nodes"]])
            torch.save(state, family_pt)
            if not a.no_check:
                check_nodes(cfg, state)
        else:
            target = out / f"{stem(FAMILY, tier)}.pt"
            if target.exists():
                target.unlink()
            os.link(family_pt, target)
        gate = next(n for n in cfg["nodes"] if n["name"] == "gate")["hparams"]
        print(f"built {stem(FAMILY, tier)}: {len(cfg['nodes'])} nodes, {len(cfg['connections'])} connections; gate "
              f"{gate['threshold']} / {gate.get('mask_threshold')}, gate_fused {a.fused_threshold} / "
              f"{a.fused_mask_threshold}" + ("" if i == 0 else f"; .pt = hardlink of {family_pt.name}"), flush=True)
    print("BUILD DONE")


if __name__ == "__main__":
    main()
