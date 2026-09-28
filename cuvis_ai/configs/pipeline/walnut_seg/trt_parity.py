"""Mask-level parity of a TensorRT engine vs the fp32 PyTorch network (walnut rgb_v2, reference frame).

Reads ``ref.pt`` (fp32 PyTorch outputs + metadata, from trt_onnx_export.py) and trtexec's ``--exportOutput`` JSON of
the engine run on the same ``input.bin``; both output sets go through rfdetr's own post-processing (node threshold,
class filter) and are rasterised like the node (per pixel max of mask x confidence). Reports detections, shell-mask
IoU vs fp32 and the max abs difference of the raw outputs.

Usage (child-env python; builds the rfdetr model once for its postprocess):
  python trt_parity.py <walnut_seg dir> <dir> [trtexec outputs json, default trt_outputs.json]
"""

import json
import os
import sys

import numpy as np
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

seg_dir, d = sys.argv[1], sys.argv[2]
ref = torch.load(os.path.join(d, "ref.pt"), weights_only=False)
y = os.path.join(seg_dir, "walnut_seg_rgb_v2_cuvisnext_cube.yaml")
seg = next(n for n in CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda").nodes
           if n.name == "Seg")
m = seg._build_model()
hw, thr, cls = tuple(ref["hw"]), ref["threshold"], ref["class_filter"]


def score_map(outs):
    pred = {"pred_boxes": outs[0].float().cuda(), "pred_logits": outs[1].float().cuda(),
            "pred_masks": outs[2].float().cuda()}
    r = m.model.postprocess(pred, target_sizes=torch.tensor([hw], device="cuda"), score_threshold=thr)[0]
    keep = (r["scores"] > thr) & (r["labels"] == cls)
    canvas = torch.zeros(hw, device="cuda")
    if int(keep.sum()):
        masks = r["masks"][keep].reshape(int(keep.sum()), *hw).bool()
        canvas = torch.where(masks, r["scores"][keep].float().view(-1, 1, 1), torch.zeros((), device="cuda")).amax(0)
    return canvas, int(keep.sum())


trt_json = json.load(open(os.path.join(d, sys.argv[3] if len(sys.argv) > 3 else "trt_outputs.json")))
by_name = {o["name"]: o for o in trt_json}
trt = []
for name, ref_t in zip(("dets", "labels", "masks"), ref["outputs"]):
    vals = np.asarray(by_name[name]["values"], dtype=np.float32).reshape(tuple(ref_t.shape))
    trt.append(torch.from_numpy(vals))
a, na = score_map(ref["outputs"])
b, nb = score_map(trt)
ma, mb = a >= 0.5, b >= 0.5
iou = float((ma & mb).sum()) / max(1, int((ma | mb).sum()))
diffs = {n: float((t - r).abs().max()) for n, t, r in zip(("dets", "labels", "masks"), trt, ref["outputs"])}
print(json.dumps({"detections_fp32": na, "detections_trt": nb, "shell_px_fp32": int(ma.sum()),
                  "shell_px_trt": int(mb.sum()), "shell_mask_iou_vs_fp32": round(iou, 5),
                  "max_abs_diff_raw": {k: round(v, 4) for k, v in diffs.items()}}))
