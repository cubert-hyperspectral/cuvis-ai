"""Where does a segmentation frame's time go? Network (the part TensorRT would replace) vs everything else.

Loads walnut_seg_rgb_v2{suffix} pipelines, runs pipeline.forward on the reference frame (the real cu3s frame), and
times with CUDA sync: the whole pipeline, the RF-DETR node, and inside it the network call alone (the exported /
traced module rfdetr calls in predict, wrapped with a timer). Median of N runs after warm-up.
Usage: python net_share_probe.py <seg dir> [suffix,...]   e.g. "_fast,_fp16,_exact"
"""
import os
import statistics
import sys
import time

import numpy as np
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = sys.argv[1]
SUFFIXES = sys.argv[2].split(",") if len(sys.argv) > 2 else ["_fast", "_fp16", "_exact"]
N, WARM = 40, 5
d = np.load(os.path.join(SEG, "ref", "real_world_live_000_f0000_reflectance.npz"))
batch = {"cube": torch.from_numpy(d["cube"].astype(np.float32))[None].cuda(),
         "wavelengths": torch.from_numpy(d["wavelengths"].astype(np.int32))[None]}


class Timed(torch.nn.Module):
    """Wraps the network module rfdetr calls; records its CUDA-synchronised wall time."""

    def __init__(self, inner, log):
        super().__init__()
        self.inner, self.log = inner, log

    def forward(self, *a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = self.inner(*a, **k)
        torch.cuda.synchronize(); self.log.append(1000 * (time.perf_counter() - t))
        return out


def med(v):
    return round(statistics.median(v), 1)


for suf in SUFFIXES:
    y = os.path.join(SEG, f"walnut_seg_rgb_v2{suf}_cuvisnext_cube.yaml")
    p = CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda")
    seg = next(n for n in p.nodes if type(n).__name__ == "RFDETRSegmenter")
    m = seg._ensure_model()
    net_log, node_log, pipe_log = [], [], []
    holder = m.model
    attr = "inference_model" if getattr(holder, "inference_model", None) is not None else "model"
    setattr(holder, attr, Timed(getattr(holder, attr), net_log))
    orig_fwd = seg.forward

    def timed_forward(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = orig_fwd(*a, **k)
        torch.cuda.synchronize(); node_log.append(1000 * (time.perf_counter() - t))
        return out

    seg.forward = timed_forward
    for i in range(WARM + N):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            p.forward(batch=batch)
        torch.cuda.synchronize(); pipe_log.append(1000 * (time.perf_counter() - t))
    net, node, pipe = net_log[WARM:], node_log[WARM:], pipe_log[WARM:]
    print(f"rgb_v2{suf or ' (default)'}: pipeline {med(pipe)} ms | RF-DETR node {med(node)} ms | network alone "
          f"{med(net)} ms ({100 * statistics.median(net) / statistics.median(pipe):.0f} % of the pipeline) | "
          f"rest of the node (pre/post-processing, mask transfer + paste) {med(node) - med(net):.1f} ms | "
          f"other nodes {med(pipe) - med(node):.1f} ms | network module {attr}", flush=True)
    del p, seg, m
    torch.cuda.empty_cache()
