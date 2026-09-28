"""PyTorch fp32 with TF32 math on vs off, for the walnut RF-DETR-Seg-L network (rgb_v2) on the real reference frame.

Separates what a TensorRT fp32 engine gains from TensorRT itself vs from TF32 tensor-core math (TensorRT allows TF32
in "fp32" builds unless --noTF32; PyTorch keeps matmuls in full fp32 unless told otherwise). Same preparation and
parity as trt_probe.py (rfdetr post-processing + the node's rasterisation, shell-mask IoU vs strict fp32).
Usage:  python tf32_probe.py <walnut_seg dir>
"""
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import trt_probe as tp  # noqa: E402  (reuses bench / score_map)

import numpy as np  # noqa: E402
import torchvision.transforms.functional as F  # noqa: E402
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline  # noqa: E402
from cuvis_ai_rfdetr.functional import to_unit_frames  # noqa: E402

seg_dir = sys.argv[1]
y = os.path.join(seg_dir, "walnut_seg_rgb_v2_cuvisnext_cube.yaml")
nodes = {n.name: n for n in CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda").nodes}
seg, sel = nodes["Seg"], nodes["Selector"]
d = np.load(os.path.join(seg_dir, "ref", "real_world_live_000_f0000_reflectance.npz"))
with torch.no_grad():
    rgb = sel.forward(cube=torch.from_numpy(d["cube"].astype(np.float32))[None].cuda(),
                      wavelengths=d["wavelengths"].astype(np.int32))["rgb_image"]
hw = (int(rgb.shape[1]), int(rgb.shape[2]))
m = seg._build_model()
res = int(m.model.resolution)
x = to_unit_frames(rgb)[0].permute(2, 0, 1).contiguous()
x = F.normalize(F.resize(x, [res, res], antialias=False), m.means, m.stds)[None].cuda()
net = copy.deepcopy(m.model.model).to(m.model.device).eval()
net.export()
rows = {}
ref = None
for tf32 in (False, True):
    torch.backends.cuda.matmul.allow_tf32 = tf32
    torch.backends.cudnn.allow_tf32 = tf32
    ms, out = tp.bench(net, x)
    sm, n = tp.score_map(m, out, hw, seg.threshold, seg.class_filter or 0)
    if ref is None:
        ref = sm
    a, b = sm >= 0.5, ref >= 0.5
    rows["fp32 PyTorch, TF32 " + ("on" if tf32 else "off")] = dict(
        ms=round(ms, 2), detections=n, iou_vs_strict_fp32=round(float((a & b).sum()) / max(1, int((a | b).sum())), 5))
print(json.dumps(rows, indent=1))
