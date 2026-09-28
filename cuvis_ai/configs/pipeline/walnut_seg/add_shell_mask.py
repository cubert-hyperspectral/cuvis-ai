"""Add a boolean shell-mask output (ShellMask) to the 7 walnut_seg cuvis.next pipelines.

cuvis.next offers a "Mask" display only for a boolean decision output; these pipelines ended in a
continuous score map (heatmap only). ShellMask = builtin BinaryDecider on the terminal `scores`.
BinaryDecider applies a sigmoid before thresholding, and the scores are already probabilities in
[0, 1], so threshold = sigmoid(0.5) = 0.62246 gives exactly `scores >= 0.5` -- the operating point
all walnut_seg parity numbers use (the same trick the 7-Sep walnut deploy pipelines used).

Each pipeline is restored WITH its .pt (PCA buffers, SAM-gate reference), the node is connected,
and the pipeline is re-saved (yaml + .pt). Acceptance on the reference cube: the terminal scores
are bit-identical before/after, and ShellMask pixels == (scores >= 0.5) == the recorded parity
pixel count. Originals are kept in _backup_pre_mask_2026-09-23/. The cuvis_picker_<name>.pt
copies are refreshed from the new .pt. Run in the stack cuvis-ai/.venv.
"""

import math
import os
import shutil

import numpy as np
import torch
import yaml
from cuvis_ai.node.deciders.binary_decider import BinaryDecider
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__))
BK = os.path.join(SEG, "_backup_pre_mask_2026-09-23")
TERM = {"rgb_full": "Seg", "cir_full": "Seg", "pca_full": "Seg", "ens_rgb_cir_mean": "Fuse",
        "int_rgb_cir": "Inter", "ens_sam_t13": "Gate", "ens_sam_t11": "Gate"}
PARITY = {"rgb_full": 72424, "cir_full": 63257, "pca_full": 62712, "ens_rgb_cir_mean": 71899,
          "int_rgb_cir": 62394, "ens_sam_t13": 69100, "ens_sam_t11": 66232}
THR = 1.0 / (1.0 + math.exp(-0.5))  # sigmoid(0.5)
cube = torch.from_numpy(np.load(r"D:/walnuts/deploy_wise08/real_world_live_000_f0000.npy").astype(np.float32))[None]
wl = torch.from_numpy((430 + 8 * np.arange(61)).astype(np.int32))[None]
os.makedirs(BK, exist_ok=True)


def run(p, term):
    with torch.no_grad():
        out = p.forward(batch={"cube": cube, "wavelengths": wl})
    return out[(term, "scores")].detach().cpu(), out.get(("ShellMask", "decisions"))


for short, term in TERM.items():
    name = f"walnut_seg_{short}_cuvisnext_cube"
    yml, pt = os.path.join(SEG, name + ".yaml"), os.path.join(SEG, name + ".pt")
    for f in (yml, pt, os.path.join(SEG, f"cuvis_picker_{name}.pt")):
        if os.path.exists(f) and not os.path.exists(os.path.join(BK, os.path.basename(f))):
            shutil.copy2(f, BK)
    src_yml = os.path.join(BK, name + ".yaml")
    if any(n.get("name") == "ShellMask" for n in yaml.safe_load(open(src_yml))["nodes"]):
        raise SystemExit(f"{name}: backup already has ShellMask - refusing to stack a second one")
    p = CuvisPipeline.load_pipeline(src_yml, weights_path=os.path.join(BK, name + ".pt"))
    s0, _ = run(p, term)
    px0 = int((s0[0, :, :, 0] >= 0.5).sum())
    node = next(n for n in p.nodes if getattr(n, "name", None) == term)
    mask = BinaryDecider(threshold=THR, name="ShellMask")
    p.connect(node.outputs.scores, mask.inputs.logits)
    p.save_to_file(yml)
    doc = yaml.safe_load(open(yml))
    doc.setdefault("metadata", {})["description"] = (
        f"Walnut shell segmentation ({short}). Outputs: {term}.scores = shell probability heatmap; "
        f"ShellMask.decisions = boolean shell mask (scores >= 0.5; BinaryDecider threshold = sigmoid(0.5)).")
    yaml.safe_dump(doc, open(yml, "w"), sort_keys=False)
    p2 = CuvisPipeline.load_pipeline(yml, weights_path=pt)
    s1, dec = run(p2, term)
    same = torch.equal(s0, s1)
    pxm = int(dec.sum()) if dec is not None else -1
    ok = same and pxm == px0 == PARITY[short] and dec.dtype == torch.bool
    shutil.copy2(pt, os.path.join(SEG, f"cuvis_picker_{name}.pt"))
    print(f"{'OK  ' if ok else 'FAIL'} {name}: scores bit-identical={same} px(scores>=0.5)={px0} "
          f"ShellMask px={pxm} parity={PARITY[short]} dtype={getattr(dec, 'dtype', None)}", flush=True)
print("ADD SHELL MASK DONE")
