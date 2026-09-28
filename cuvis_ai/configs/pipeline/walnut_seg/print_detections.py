"""Print the RF-DETR detections (confidence, box, area in the mask) of one walnut_seg pipeline on the reference frame.

Diagnoses cross-machine mask differences: an instance whose confidence sits at the segmenter's `threshold`
(0.5 for the singles) appears on one machine and not on the other.
  python print_detections.py walnut_seg_pca_full_cuvisnext_cube [--device cuda]
"""

import os
import sys

import numpy as np
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__))
name = sys.argv[1]
device = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else None
d = np.load(os.path.join(SEG, "ref", "real_world_live_000_f0000_reflectance.npz"))
cube = torch.from_numpy(d["cube"].astype(np.float32))[None]
wl = torch.from_numpy(d["wavelengths"].astype(np.int32))[None]
p = CuvisPipeline.load_pipeline(os.path.join(SEG, name + ".yaml"), weights_path=os.path.join(SEG, name + ".pt"), device=device)
with torch.no_grad():
    o = p.forward(batch={"cube": cube.to(device) if device else cube, "wavelengths": wl})
for (node, port), v in o.items():
    if port == "detections":
        dets = sorted(v[0], key=lambda r: -r["confidence"])
        print(f"{node}: {len(dets)} detections")
        for r in dets:
            x1, y1, x2, y2 = (int(round(c)) for c in r["xyxy"])
            print(f"  conf={r['confidence']:.5f} class={r['class_id']} box=({x1},{y1},{x2},{y2})")
