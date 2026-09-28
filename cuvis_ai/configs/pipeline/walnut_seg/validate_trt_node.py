"""163-frame accuracy + timing of the `_trt_fp32` / `_trt_fp16` walnut pipelines (RFDETRSegmenter backend tensorrt).

Unlike validate_trt.py (a hand-built engine next to the node), this runs the deployable code path: the Seg nodes of
walnut_seg_{rgb_v2,cir_v2,ens_rgb_cir_mean_v2}_{trt_fp32,trt_fp16}_cuvisnext_cube (engines built beforehand by
build_fast_pipelines.py / cuvis_ai_rfdetr.trt_engine build-pipeline) against the Seg nodes of the deployed default
pipelines (PyTorch fp32), on the 163 cached test frames (frame_cache.pt of validate_speed_options.py). Masks: singles
>= 0.5, ensemble = mean of the member maps >= 0.5. Rows as validate_trt.py -> summarize_validate.py; also the median
CUDA-synchronised node forward time per model and backend.

Usage (overlay env with tensorrt; never install into the stack or cuvis.next child envs):
  python validate_trt_node.py --cache <frame_cache.pt> --out <dir> [--limit N]
"""

import argparse
import json
import os
import platform
import statistics
import time

import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__))
BACKENDS = {"pytorch_fp32": "", "trt_fp32": "_trt_fp32", "trt_fp16": "_trt_fp16"}


def load(short):
    y = os.path.join(SEG, f"walnut_seg_{short}_cuvisnext_cube.yaml")
    return {n.name: n for n in CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda").nodes}


def agreement(a, b):
    union = int((a | b).sum())
    return 1.0 if union == 0 else float((a & b).sum()) / union


def timed(node, img, times):
    torch.cuda.synchronize()
    t = time.perf_counter()
    s = node.forward(rgb_image=img)["scores"][0, :, :, 0]
    torch.cuda.synchronize()
    times.append(1000.0 * (time.perf_counter() - t))
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    import cuvis_ai_rfdetr
    import tensorrt

    env = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "tensorrt": tensorrt.__version__, "cuvis_ai_rfdetr": cuvis_ai_rfdetr.__file__}
    print("ENV", json.dumps(env), flush=True)
    frames = torch.load(args.cache, weights_only=False)
    frames = frames[: args.limit] if args.limit else frames
    nodes = {}
    for b, suffix in BACKENDS.items():
        rgb, cir, ens = load("rgb_v2" + suffix), load("cir_v2" + suffix), load("ens_rgb_cir_mean_v2" + suffix)
        nodes[b] = {"rgb1": rgb["Seg"], "cir1": cir["Seg"], "ensr": ens["SegRGB"], "ensc": ens["SegCIR"]}
        seg = nodes[b]["rgb1"]
        print(f"loaded {b}: backend={seg.backend} precision={seg.precision} gpu_input={seg.gpu_input}", flush=True)
    times = {b: {k: [] for k in ("rgb1", "cir1", "ensr", "ensc")} for b in BACKENDS}
    first_ms = {}
    rows = []
    for i, fr in enumerate(frames):
        img = {k: fr[k].cuda().float() / 255.0 for k in ("rgb", "cir")}
        masks = {}
        with torch.no_grad():
            for b, nd in nodes.items():
                if i == 0:  # first call builds the model (+ loads the engine): timed separately
                    t = time.perf_counter()
                    for k, n in nd.items():
                        n.forward(rgb_image=img["rgb" if k in ("rgb1", "ensr") else "cir"])
                    torch.cuda.synchronize()
                    first_ms[b] = round(1000.0 * (time.perf_counter() - t))
                s = {k: timed(n, img["rgb" if k in ("rgb1", "ensr") else "cir"], times[b][k]) for k, n in nd.items()}
                masks[b] = {"rgb_v2": s["rgb1"] >= 0.5, "cir_v2": s["cir1"] >= 0.5,
                            "ens_rgb_cir_mean_v2": 0.5 * (s["ensr"] + s["ensc"]) >= 0.5}
        gs, gf, unl = fr["gs"], fr["gf"], fr["unl"]
        for b, per in masks.items():
            for pipe_name, mt in per.items():
                mk = mt.cpu().numpy()
                ref = masks["pytorch_fp32"][pipe_name].cpu().numpy()
                rows.append(dict(option=b, pipeline=pipe_name, set=fr["set"], scene=fr["scene"], name=fr["name"],
                                 tp=int((mk & gs).sum()), fp=int((mk & ~gs).sum()), fn=int((gs & ~mk).sum()),
                                 gs=int(gs.sum()), objfp=int((mk & unl).sum()), f2n=int((mk & gf).sum()),
                                 gf=int(gf.sum()), agree=agreement(mk, ref)))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frames)} frames", flush=True)
    timing = {b: {k: round(statistics.median(v), 2) for k, v in t.items()} for b, t in times.items()}
    host = platform.node()
    json.dump(rows, open(os.path.join(args.out, f"validate_trt_node_{host}.json"), "w"))
    json.dump({"env": env, "node_forward_median_ms": timing, "first_forward_ms_all_four_models": first_ms},
              open(os.path.join(args.out, f"validate_trt_node_meta_{host}.json"), "w"), indent=1)
    print("NODE FORWARD median ms", json.dumps(timing), "| first forward ms", json.dumps(first_ms), flush=True)
    print("VALIDATE TRT NODE DONE", flush=True)


if __name__ == "__main__":
    main()
