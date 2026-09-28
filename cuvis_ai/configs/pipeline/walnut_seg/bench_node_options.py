"""Interleaved A/B timing of the RFDETRSegmenter speed options on real cu3s frames (the decision numbers).

profile_seg_cu3s.py times each variant in its own pass, with a 120-380 ms cu3s decode between frames, so GPU
clocks and background load drift between variants and differences of a few ms drown in the spread (laptop std
7-24 ms). Here all frames are decoded once and kept in host memory (37 x 61-band cubes = 9.7 GB would not fit an 8 GB
laptop GPU); every variant of a pipeline is loaded side by side and, frame by frame, the cube is moved to the GPU once
(untimed) and all variants run back to back on it in a rotating order. Timed: CUDA-synchronised wall clock of
pipeline.forward (cube on the GPU -> ShellMask + ShellHeatmap), i.e. what a host that already holds the cube pays.

Variants (per pipeline):
  legacy_paste  fast_paste=0: per-instance CPU paste (the node before the speed options)
  default       the yaml as saved (fast paste on)
  exact         the _exact yaml: fp32 + gpu_input + fast paste (bit-identical to default)
  fp16          the _fp16 yaml: fp16 (no JIT trace) + gpu_input + fast paste
  fast          the _fast yaml: fp16 + jit_trace + gpu_input + fast paste
  trt_fp32      the _trt_fp32 yaml: TensorRT fp32 engine (TF32 allowed) + gpu_input + fast paste
  trt_fp16      the _trt_fp16 yaml: TensorRT fp16 engine + gpu_input + fast paste
                (the TensorRT variants need tensorrt in the env and this machine's engines, see build_fast_pipelines.py)
Reported: median / mean / p90 ms, speed-up vs legacy_paste (median), mask IoU vs legacy_paste (pooled over frames),
and first_frame_ms = the cold first forward (lazy RF-DETR build + checkpoint load, + the JIT trace for `fast`) -
the delay before the first mask when a pipeline starts in cuvis.next.

Usage (stack env / Thor child env, + cu3s reader overlay):
  python bench_node_options.py --cu3s <a.cu3s> [--cu3s <b.cu3s>] [--reps 3] [--out bench_<host>.json]
                                [--variants legacy_paste,default,...]  (legacy_paste is always included)
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time

import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule

SEG = os.path.dirname(os.path.abspath(__file__))
WARMUP = 3
VARIANTS = [("legacy_paste", "", {"fast_paste": False}), ("default", "", {}),
            ("exact", "_exact", {}), ("fp16", "_fp16", {}), ("fast", "_fast", {}),
            ("trt_fp32", "_trt_fp32", {}), ("trt_fp16", "_trt_fp16", {})]


def load(short, suffix, flips):
    y = os.path.join(SEG, f"walnut_seg_{short}{suffix}_cuvisnext_cube.yaml")
    p = CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda")
    for n in p.nodes:
        if type(n).__name__ == "RFDETRSegmenter":
            for k, v in flips.items():
                setattr(n, k, v)
    return p


def read_frames(paths):
    frames = []
    for c in paths:
        dm = Cu3sDataModule(cu3s_file_path=c, processing_mode="Reflectance", batch_size=1, num_workers=0)
        dm.setup(stage="predict")
        for batch in Predictor._iter_batches(dm.predict_dataloader()):
            frames.append({"cube": batch["cube"].float(), "wavelengths": batch["wavelengths"]})
    return frames


def on_gpu(fr):
    return {"cube": fr["cube"].cuda(non_blocking=False), "wavelengths": fr["wavelengths"]}


def run(p, fr):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        o = p.forward(batch=fr)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0, o[("ShellMask", "decisions")].bool()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cu3s", action="append", required=True)
    ap.add_argument("--pipelines", default="rgb_v2,cir_v2,ens_rgb_cir_mean_v2")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variants", default=",".join(v[0] for v in VARIANTS))
    args = ap.parse_args()
    wanted = {"legacy_paste", *args.variants.split(",")}
    variants = [v for v in VARIANTS if v[0] in wanted]
    env = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
    print("ENV", json.dumps(env), flush=True)
    frames = read_frames(args.cu3s)
    print(f"{len(frames)} frames in memory, {args.reps} reps", flush=True)
    rows = []
    for short in args.pipelines.split(","):
        pipes = {name: load(short, suffix, flips) for name, suffix, flips in variants}
        first = {}
        for name, p in pipes.items():  # warm-up (the fast variant traces its network here)
            first[name] = run(p, on_gpu(frames[0]))[0]
            for fr in frames[1:WARMUP]:
                run(p, on_gpu(fr))
        ms = {name: [] for name in pipes}
        inter = {name: 0 for name in pipes}
        union = {name: 0 for name in pipes}
        order = list(pipes)
        for rep in range(args.reps):
            for i, fr_host in enumerate(frames):
                fr = on_gpu(fr_host)  # one untimed transfer per frame, shared by all variants
                k = (i + rep) % len(order)
                masks = {}
                for name in order[k:] + order[:k]:  # rotate who runs first
                    t, masks[name] = run(pipes[name], fr)
                    ms[name].append(t)
                if rep == 0:
                    ref = masks["legacy_paste"]
                    for name, m in masks.items():
                        inter[name] += int((m & ref).sum())
                        union[name] += int((m | ref).sum())
        base = statistics.median(ms["legacy_paste"])
        for name, v in ms.items():
            v = sorted(v)
            med = statistics.median(v)
            row = dict(pipeline=short, variant=name, n=len(v), median_ms=round(med, 1),
                       mean_ms=round(statistics.fmean(v), 1), p90_ms=round(v[int(0.9 * (len(v) - 1))], 1),
                       fps=round(1000.0 / med, 1), speedup_vs_legacy=round(base / med, 2),
                       mask_iou_vs_legacy=round(inter[name] / max(1, union[name]), 5),
                       first_frame_ms=round(first[name]), **env)
            rows.append(row)
            print("BENCH", json.dumps({k: row[k] for k in list(row)[:10]}), flush=True)
        del pipes
        gc.collect()  # pipelines hold reference cycles; without this the models stay on the GPU
        torch.cuda.empty_cache()
        print(f"  after {short}: {torch.cuda.memory_allocated() / 2**20:.0f} MiB still allocated", flush=True)
    print("\n| pipeline | variant | median ms | fps | x vs legacy paste | p90 ms | mask IoU vs legacy | first frame ms |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['pipeline']} | {r['variant']} | {r['median_ms']} | {r['fps']} | {r['speedup_vs_legacy']} | "
              f"{r['p90_ms']} | {r['mask_iou_vs_legacy']} | {r['first_frame_ms']} |")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
