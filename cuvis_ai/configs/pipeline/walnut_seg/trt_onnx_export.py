"""Export the walnut RF-DETR-Seg-L network (rgb_v2) to ONNX the way rfdetr's own export does, for a TensorRT build.

rfdetr ``export(format="onnx")`` = ``model.export()`` + ``torch.onnx.export(dynamo=False, opset 17)`` with input
``input`` and outputs ``dets`` / ``labels`` / ``masks``; this script makes the same call on the fine-tuned
checkpoint (no graph-surgeon pass, so only the ``onnx`` package is needed). It also writes, for a parity check of
the TensorRT engine:
  input.bin  the REAL preprocessed reference frame (float32, 1x3xRxR, as rfdetr's predict prepares it) for
             ``trtexec --loadInputs=input:input.bin``
  ref.pt     the fp32 PyTorch outputs on that input, the frame size and the node's threshold / class filter.

Usage (overlay env with ``onnx``):  python trt_onnx_export.py <walnut_seg dir> <out dir>
"""

import copy
import os
import sys

import numpy as np
import torch
import torchvision.transforms.functional as F
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_rfdetr.functional import to_unit_frames

seg_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
y = os.path.join(seg_dir, "walnut_seg_rgb_v2_cuvisnext_cube.yaml")
pipe = CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda")
nodes = {n.name: n for n in pipe.nodes}
seg, sel = nodes["Seg"], nodes["Selector"]
d = np.load(os.path.join(seg_dir, "ref", "real_world_live_000_f0000_reflectance.npz"))
cube = torch.from_numpy(d["cube"].astype(np.float32))[None].cuda()
with torch.no_grad():
    rgb = sel.forward(cube=cube, wavelengths=d["wavelengths"].astype(np.int32))["rgb_image"]
m = seg._build_model()
res = int(m.model.resolution)
x = to_unit_frames(rgb)[0].permute(2, 0, 1).contiguous()
x = F.normalize(F.resize(x, [res, res], antialias=False), m.means, m.stds)[None].float()

net = copy.deepcopy(m.model.model).to(m.model.device).eval()
net.export()
with torch.no_grad():
    out = net(x.cuda())
outs = list(out) if isinstance(out, (tuple, list)) else [out[k] for k in ("pred_boxes", "pred_logits", "pred_masks")]
print("network outputs:", [tuple(o.shape) for o in outs], flush=True)
torch.save({"outputs": [o.detach().cpu() for o in outs], "hw": (int(rgb.shape[1]), int(rgb.shape[2])),
            "threshold": float(seg.threshold), "class_filter": int(seg.class_filter or 0), "resolution": res},
           os.path.join(out_dir, "ref.pt"))
x.cpu().numpy().astype(np.float32).tofile(os.path.join(out_dir, "input.bin"))

net_cpu = net.cpu()
onnx_path = os.path.join(out_dir, f"rgb_v2_{res}.onnx")
kwargs = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
torch.onnx.export(net_cpu, (x.cpu(),), onnx_path, input_names=["input"], output_names=["dets", "labels", "masks"],
                  dynamic_axes=None, opset_version=17, do_constant_folding=True, **kwargs)
print("ONNX written:", onnx_path, round(os.path.getsize(onnx_path) / 2**20, 1), "MiB", flush=True)
