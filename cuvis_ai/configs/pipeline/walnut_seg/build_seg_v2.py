"""Build the SEG v2 walnut_seg pipelines (models retrained 23 Sep on old + 22-Sep data with aug v2) for cuvis.next.

  walnut_seg_cir_v2_cuvisnext_cube              C: CIR 850/660/550 single            (weights/cir_v2_ema.pth)
  walnut_seg_rgb_v2_cuvisnext_cube              B: RGB 640/550/470 single            (weights/rgb_v2_ema.pth)
  walnut_seg_ens_rgb_cir_mean_v2_cuvisnext_cube B+C: members at 0.05 + ScoreFusion(mean)

Same node recipe as the deployed *_full pipelines (repro/build_seg_stack.py) + the ShellMask output from
add_shell_mask.py (builtin BinaryDecider, threshold sigmoid(0.5) = scores >= 0.5). Each pipeline is built, saved
(yaml + .pt), reloaded WITH its .pt and forwarded on the float32 cu3s frame real_world_live_000 f0 exactly as the
cu3s reader delivers it (exported on asai1 by export_ref_frame.py). Acceptance = ShellMask == (scores >= 0.5) exactly,
and mask-agreement IoU >= 0.999 with asai1's eval_v2_height.py mask for that frame run with the SAME rfdetr as this
venv (1.10.1; argv[2] = that run's dump). Exact pixel equality across machines is not expected: torch 2.11/cu128
(laptop) vs 2.13/cu130 (asai1) differ in the last float bits, which flips a few mask cells (measured 1-26 px of ~71k). The eval/training env on asai1 uses rfdetr 1.8.3 - its small effect is measured separately over all 163 test
frames (SEG_V2_HEIGHT_ENSEMBLE_2026-09-24.md). Also prints the px on the float16 .npy reference cube as the drift
reference for future re-checks (like 72424 for rgb_full). argv[1] = dir with real_world_live_000_f0000_{f32,wl}.npy.
Run in the stack cuvis-ai/.venv.
"""

import math
import os
import shutil
import sys

import numpy as np
import cuvis_ai_core
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
W = f"{SEG}/weights"
THR = 1.0 / (1.0 + math.exp(-0.5))  # sigmoid(0.5): BinaryDecider sigmoids its input, scores are probabilities
PLUG = ["cuvis_ai_builtin", "rfdetr_seg"]
RGB_BANDS, CIR_BANDS = [640.0, 550.0, 470.0], [850.0, 660.0, 550.0]
DS = {"name": "DataSource", "class_name": "cuvis_ai.node.data.CU3SDataNode", "hparams": {}}
MASK = {"name": "ShellMask", "class_name": "cuvis_ai.node.deciders.binary_decider.BinaryDecider",
        "hparams": {"threshold": THR}}


def sel(n, b):
    return {"name": n, "class_name": "cuvis_ai.node.channel_selector.FixedWavelengthSelector",
            "hparams": {"target_wavelengths": b, "normalize_output": True, "norm_mode": "per_frame", "apply_gamma": False}}


def seg(n, ck, thr=0.5):
    return {"name": n, "class_name": "cuvis_ai_rfdetr.node.rfdetr_segmenter.RFDETRSegmenter",
            "hparams": {"checkpoint_path": ck, "variant": "large", "resolution": 504, "threshold": thr,
                        "class_filter": 0, "tiling": "whole"}}


def wire(s, g):
    return [{"source": "DataSource.outputs.cube", "target": f"{s}.inputs.cube"},
            {"source": "DataSource.outputs.wavelengths", "target": f"{s}.inputs.wavelengths"},
            {"source": f"{s}.outputs.rgb_image", "target": f"{g}.inputs.rgb_image"}]


def to_mask(term):
    return [{"source": f"{term}.outputs.scores", "target": "ShellMask.inputs.logits"}]


PIPES = {
    "cir_v2": dict(ref="C", term="Seg", what="C = CIR 850/660/550 single, v2 data + aug v2",
                   nodes=[DS, sel("Selector", CIR_BANDS), seg("Seg", f"{W}/cir_v2_ema.pth"), MASK],
                   connections=wire("Selector", "Seg") + to_mask("Seg")),
    "rgb_v2": dict(ref="B", term="Seg", what="B = RGB 640/550/470 single, v2 data + aug v2",
                   nodes=[DS, sel("Selector", RGB_BANDS), seg("Seg", f"{W}/rgb_v2_ema.pth"), MASK],
                   connections=wire("Selector", "Seg") + to_mask("Seg")),
    "ens_rgb_cir_mean_v2": dict(
        ref="ens_BC", term="Fuse", what="B+C = RGB v2 + CIR v2 mean ensemble",
        nodes=[DS, sel("SelRGB", RGB_BANDS), seg("SegRGB", f"{W}/rgb_v2_ema.pth", 0.05), sel("SelCIR", CIR_BANDS),
               seg("SegCIR", f"{W}/cir_v2_ema.pth", 0.05),
               {"name": "Fuse", "class_name": "cuvis_ai_rfdetr.node.score_fusion.ScoreFusion",
                "hparams": {"mode": "mean", "weight": 0.5}}, MASK],
        connections=wire("SelRGB", "SegRGB") + wire("SelCIR", "SegCIR") + [
            {"source": "SegRGB.outputs.scores", "target": "Fuse.inputs.a"},
            {"source": "SegCIR.outputs.scores", "target": "Fuse.inputs.b"}] + to_mask("Fuse")),
}

d = np.load(sys.argv[2])
H, Wd = (int(x) for x in d["shape"])
MREF = {k[2:]: np.unpackbits(d[k])[: H * Wd].reshape(H, Wd).astype(bool) for k in d.files if k.startswith("m_")}
REF = {k: int(v.sum()) for k, v in MREF.items()}
print("asai1 reference px (rfdetr 1.10.1, real_world_live_000_f0000):", REF, flush=True)
cube = torch.from_numpy(np.load(f"{sys.argv[1]}/real_world_live_000_f0000_f32.npy"))[None]
wl = torch.from_numpy(np.load(f"{sys.argv[1]}/real_world_live_000_f0000_wl.npy").astype(np.int32))[None]
npy16 = torch.from_numpy(np.load(r"D:/walnuts/deploy_wise08/real_world_live_000_f0000.npy").astype(np.float32))[None]
wl_grid = torch.from_numpy((430 + 8 * np.arange(61)).astype(np.int32))[None]


def run(p, term, x=None, w=None):
    with torch.no_grad():
        out = p.forward(batch={"cube": cube if x is None else x, "wavelengths": wl if w is None else w})
    return out[(term, "scores")].detach().cpu(), out.get(("ShellMask", "decisions"))


# sanity: the deployed rgb_full on the same float32 frame == asai1 with the same rfdetr version
p0 = CuvisPipeline.load_pipeline(f"{SEG}/walnut_seg_rgb_full_cuvisnext_cube.yaml",
                                 weights_path=f"{SEG}/walnut_seg_rgb_full_cuvisnext_cube.pt")
s0, _ = run(p0, "Seg")
px0 = int((s0[0, :, :, 0] >= 0.5).sum())
m0 = (s0[0, :, :, 0] >= 0.5).numpy()
agree0 = float((m0 & MREF["rgb_full"]).sum() / max(1, (m0 | MREF["rgb_full"]).sum()))
print(f"{'OK  ' if agree0 >= 0.999 else 'FAIL'} input check: stack rgb_full px={px0} asai1={REF['rgb_full']} "
      f"mask agreement IoU={agree0:.5f}", flush=True)
del p0

allok = agree0 >= 0.999
for short, b in PIPES.items():
    name = f"walnut_seg_{short}_cuvisnext_cube"
    yml, pt = f"{SEG}/{name}.yaml", f"{SEG}/{name}.pt"
    doc = {"metadata": {"name": name, "author": "raj@cubert-gmbh.de"}, "plugins": PLUG,
           "nodes": b["nodes"], "connections": b["connections"]}
    yaml.safe_dump(doc, open(yml, "w"), sort_keys=False)
    p = CuvisPipeline.load_pipeline(yml)
    p.save_to_file(yml)  # canonical yaml + sibling .pt (cuvis.next picker needs both)
    doc = yaml.safe_load(open(yml))
    doc.setdefault("metadata", {})["cuvis_ai_version"] = cuvis_ai_core.__version__  # load_pipeline path stamps schemas'
    doc["metadata"]["description"] = (
        f"Walnut shell segmentation ({short}: {b['what']}). Outputs: {b['term']}.scores = shell probability heatmap; "
        f"ShellMask.decisions = boolean shell mask (scores >= 0.5; BinaryDecider threshold = sigmoid(0.5)).")
    yaml.safe_dump(doc, open(yml, "w"), sort_keys=False)
    p2 = CuvisPipeline.load_pipeline(yml, weights_path=pt)
    s, dec = run(p2, b["term"])
    px = int((s[0, :, :, 0] >= 0.5).sum())
    pxm = int(dec.sum()) if dec is not None else -1
    m = dec[0].squeeze(-1).numpy() if dec.ndim == 4 else dec[0].numpy()
    agree = float((m & MREF[b["ref"]]).sum() / max(1, (m | MREF[b["ref"]]).sum()))
    ok = pxm == px and dec.dtype == torch.bool and agree >= 0.999
    allok &= ok
    shutil.copy2(pt, f"{SEG}/cuvis_picker_{name}.pt")
    s16, _ = run(p2, b["term"], npy16, wl_grid)
    print(f"{'OK  ' if ok else 'FAIL'} {name}: px(scores>=0.5)={px} ShellMask px={pxm} asai1 ref={REF[b['ref']]} mask agreement IoU={agree:.5f} "
          f"dtype={getattr(dec, 'dtype', None)} | .npy drift reference px={int((s16[0, :, :, 0] >= 0.5).sum())}", flush=True)
    del p, p2
print("BUILD SEG V2 DONE", "ALL OK" if allok else "WITH FAILURES")
