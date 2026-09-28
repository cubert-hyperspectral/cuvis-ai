"""Expose BOTH the shell heatmap and the shell mask as selectable outputs in every walnut_seg cuvis.next pipeline.

cuvis.next's Displayed Output dropdown lists the pipeline's terminal (unconnected) output ports and classifies them
by PORT NAME: `decisions` / `*.mask` -> mask, `*scores` -> heatmap, `*_thickness` / `*value_map` -> value map,
`*rgb_image` / `false_color` -> image; ports with other names are not offered. Since ShellMask (BinaryDecider) was
added on 23 Sep it consumes the model's `scores`, so the heatmap (`Seg.scores` / `Fuse.scores` / `Inter.scores` /
`Gate.scores`) is no longer terminal and only `ShellMask.decisions` was selectable. A first fix (IdentityNormalizer
tap, port `normalized`) stayed invisible because of the port name. Fix: fan the scores out to a `ScoreFusion` node
named `ShellHeatmap` with mode `max` and BOTH inputs on the same scores (max(x, x) == x, bit-identical); its
`scores` port stays unconnected, so cuvis.next offers `ShellHeatmap.scores` (heatmap) next to `ShellMask.decisions`
(mask).

Each pipeline is restored WITH its .pt from `_backup_pre_heatmap_2026-09-24/` (the state before any heatmap tap),
the node is connected, the pipeline is re-saved (yaml + .pt), reloaded and forwarded on the float32 reference frame
(`ref/real_world_live_000_f0000_reflectance.npz`, the cu3s frame exactly as the reader delivers it). Acceptance:
ShellHeatmap.scores is bit-identical to the model scores before the change, ShellMask.decisions is bit-identical to
before, and both ports are in get_output_specs(). The cuvis_picker_<name>.pt copies are refreshed. Run in the stack
cuvis-ai/.venv.
"""

import glob
import os
import shutil

import numpy as np
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_rfdetr.node.score_fusion import ScoreFusion

SEG = os.path.dirname(os.path.abspath(__file__))
BK = os.path.join(SEG, "_backup_pre_heatmap_2026-09-24")
REF = np.load(os.path.join(SEG, "ref", "real_world_live_000_f0000_reflectance.npz"))
cube = torch.from_numpy(REF["cube"].astype(np.float32))[None]
wl = torch.from_numpy(REF["wavelengths"].astype(np.int32))[None]
os.makedirs(BK, exist_ok=True)


def run(p):
    with torch.no_grad():
        return p.forward(batch={"cube": cube, "wavelengths": wl})


allok = True
for yml in sorted(glob.glob(os.path.join(SEG, "walnut_seg_*_cuvisnext_cube.yaml"))):
    name = os.path.basename(yml)[:-5]
    pt = os.path.join(SEG, name + ".pt")
    for f in (yml, pt, os.path.join(SEG, f"cuvis_picker_{name}.pt")):
        if os.path.exists(f) and not os.path.exists(os.path.join(BK, os.path.basename(f))):
            shutil.copy2(f, BK)
    src = yaml.safe_load(open(os.path.join(BK, name + ".yaml")))
    if any(n.get("name") == "ShellHeatmap" for n in src["nodes"]):
        raise SystemExit(f"{name}: backup already has ShellHeatmap - refusing to stack a second one")
    term = next(c["source"].split(".")[0] for c in src["connections"] if c["target"] == "ShellMask.inputs.logits")
    p = CuvisPipeline.load_pipeline(os.path.join(BK, name + ".yaml"), weights_path=os.path.join(BK, name + ".pt"))
    o0 = run(p)
    s0, d0 = o0[(term, "scores")].detach().cpu(), o0[("ShellMask", "decisions")].detach().cpu()
    node = next(n for n in p.nodes if getattr(n, "name", None) == term)
    heat = ScoreFusion(mode="max", name="ShellHeatmap")
    p.connect(node.outputs.scores, heat.inputs.a)
    p.connect(node.outputs.scores, heat.inputs.b)
    p.save_to_file(yml)
    doc = yaml.safe_load(open(yml))
    doc["metadata"]["description"] = (
        f"{src['metadata'].get('description', name).split(' Outputs:')[0]} Outputs (both selectable in cuvis.next): "
        f"ShellHeatmap.scores = shell probability heatmap (exact copy of {term}.scores: ScoreFusion max of the map "
        f"with itself, so the port is terminal and named 'scores'); ShellMask.decisions = boolean shell mask "
        f"(scores >= 0.5; BinaryDecider threshold = sigmoid(0.5)).")
    yaml.safe_dump(doc, open(yml, "w"), sort_keys=False)
    p2 = CuvisPipeline.load_pipeline(yml, weights_path=pt)
    o1 = run(p2)
    h1, d1 = o1[("ShellHeatmap", "scores")].detach().cpu(), o1[("ShellMask", "decisions")].detach().cpu()
    outs = sorted(p2.get_output_specs())
    ok = (torch.equal(h1, s0) and torch.equal(d1, d0) and "ShellHeatmap.scores" in outs
          and "ShellMask.decisions" in outs)
    allok &= ok
    shutil.copy2(pt, os.path.join(SEG, f"cuvis_picker_{name}.pt"))
    print(f"{'OK  ' if ok else 'FAIL'} {name}: term={term} heatmap==scores {torch.equal(h1, s0)} "
          f"mask unchanged {torch.equal(d1, d0)} mask px={int(d1.sum())} outputs={outs}", flush=True)
    del p, p2
print("ADD HEATMAP OUTPUT DONE", "ALL OK" if allok else "WITH FAILURES")
