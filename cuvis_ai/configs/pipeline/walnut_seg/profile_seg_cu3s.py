"""Profile walnut_seg pipelines on real cu3s recordings with cuvis-ai's built-in node profiler.

Path 3 of the cuvis-ai-inference skill (CuvisPipeline + Cu3sDataModule + Predictor). The restore-pipeline CLI
(Path 1, run separately as the reference) switches profiling on with no CUDA synchronisation and no warm-up skip,
so its per-node split mixes asynchronous GPU work between nodes and its mean includes the first (warm-up) frame.
Here the profiler runs with `synchronize_cuda=True` (per-node GPU wall-clock) and `skip_first_n=WARMUP`.

Per pipeline and cu3s file it reports:
  * the per-node table from `format_profiling_summary` (count / mean / std / min / max / median per node),
  * pipeline ms/frame = sum of the per-node means (steady state),
  * data ms/frame = cu3s decode + reflectance processing (Cu3sDataModule iterated alone),
  * end-to-end ms/frame = Predictor.predict wall time / frames (data + pipeline, sequential),
  * peak CUDA memory, and the ShellMask px of the first frame (sanity vs the eval numbers).

`--variant NAME[:key=val,...]` (repeatable) runs every pipeline once per variant, with the listed RFDETRSegmenter
attributes flipped after loading (the attributes the hparams set, e.g. `legacy_paste:fast_paste=0`,
`gpu_input:gpu_input=1`); plain `default` runs the yaml as saved. Rows carry the variant name.

Usage (stack env + the cu3s dataloader as a uv overlay, never installed into the stack venv):
  uv run --no-sync --with "cuvis-ai-dataloader[cu3s,coco] @ file:///<stack>/cuvis-ai-dataloader" \
      python profile_seg_cu3s.py --out <dir> --cu3s <a.cu3s> [--cu3s <b.cu3s>] [--pipelines rgb_v2,cir_v2,...] \
      [--variant default --variant legacy_paste:fast_paste=0] [--tag <json suffix>]
"""

import argparse
import gc
import json
import os
import platform
import time

import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule
from cuvis_ai_schemas.enums import ExecutionStage

SEG = os.path.dirname(os.path.abspath(__file__))
WARMUP = 2


def data_only_ms(cu3s: str) -> tuple[float, int]:
    dm = Cu3sDataModule(cu3s_file_path=cu3s, processing_mode="Reflectance", batch_size=1, num_workers=0)
    dm.setup(stage="predict")
    times, t = [], time.perf_counter()
    for _ in dm.predict_dataloader():
        now = time.perf_counter()
        times.append((now - t) * 1000.0)
        t = now
    steady = times[WARMUP:] or times
    return sum(steady) / len(steady), len(times)


def apply_variant(pipe, spec: str) -> dict:
    """Flip RFDETRSegmenter attributes per `key=val,...`; a precision / jit_trace change rebuilds the model."""
    kv = dict(item.split("=", 1) for item in spec.split(",")) if spec else {}
    for node in pipe.nodes:
        if type(node).__name__ != "RFDETRSegmenter":
            continue
        for key, val in kv.items():
            cur = getattr(node, key)
            setattr(node, key, bool(int(val)) if isinstance(cur, bool) else type(cur)(val))
        if {"precision", "jit_trace"} & set(kv):
            node._model = None
    return kv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cu3s", action="append", required=True)
    ap.add_argument("--pipelines", default="rgb_v2,cir_v2,ens_rgb_cir_mean_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variant", action="append", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    variants = [(v.split(":", 1) + [""])[:2] for v in (args.variant or ["default"])]
    os.makedirs(args.out, exist_ok=True)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    env = {"host": platform.node(), "gpu": gpu, "torch": torch.__version__, "python": platform.python_version()}
    print("ENV", json.dumps(env), flush=True)
    data = {c: data_only_ms(c) for c in args.cu3s}
    for c, (ms, n) in data.items():
        print(f"DATA {os.path.basename(c)}: {n} frames, cu3s decode + reflectance {ms:.1f} ms/frame", flush=True)
    results = []
    for short, (vname, vspec) in [(s, v) for s in args.pipelines.split(",") for v in variants]:
        name = f"walnut_seg_{short}_cuvisnext_cube"
        yml = os.path.join(SEG, name + ".yaml")
        pipe = CuvisPipeline.load_pipeline(yml, weights_path=yml[:-5] + ".pt", device=args.device)
        flips = apply_variant(pipe, vspec)
        print(f"\n##### {name} variant {vname} {flips or '(yaml as saved)'}", flush=True)
        for c in args.cu3s:
            torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
            pipe.set_profiling(enabled=True, synchronize_cuda=True, reset=True, skip_first_n=WARMUP)
            dm = Cu3sDataModule(cu3s_file_path=c, processing_mode="Reflectance", batch_size=1, num_workers=0)
            t0 = time.perf_counter()
            outs = Predictor(pipeline=pipe, datamodule=dm).predict(collect_outputs=True, collect_ports={"decisions"})
            wall = time.perf_counter() - t0
            nfr = len(outs)
            stats = pipe.get_profiling_summary(stage=ExecutionStage.INFERENCE)
            pipe_ms = sum(s.mean_ms for s in stats)
            mask0 = int(next(v for (n, p), v in outs[0].items() if p == "decisions").sum())
            peak = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
            table = pipe.format_profiling_summary(stage=ExecutionStage.INFERENCE, total_frames=nfr)
            print(f"\n=== {name} [{vname}] on {os.path.basename(c)} ({nfr} frames) ===\n{table}", flush=True)
            row = dict(pipeline=short, variant=vname, flips=flips, cu3s=os.path.basename(c), frames=nfr,
                       pipeline_ms=round(pipe_ms, 1),
                       pipeline_fps=round(1000.0 / pipe_ms, 2), data_ms=round(data[c][0], 1),
                       e2e_ms=round(1000.0 * wall / nfr, 1), e2e_fps=round(nfr / wall, 2),
                       peak_cuda_mib=round(peak), mask_px_frame0=mask0,
                       nodes={s.node_name: dict(mean=round(s.mean_ms, 2), median=round(s.median_ms, 2),
                                                std=round(s.std_ms, 2), count=s.count) for s in stats}, **env)
            results.append(row)
            print("RESULT", json.dumps({k: v for k, v in row.items() if k != "nodes"}), flush=True)
        pipe.set_profiling(enabled=False)
        del pipe, outs
        gc.collect()  # pipelines hold reference cycles; without this earlier variants' models stay on the GPU
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    with open(os.path.join(args.out, f"profile_{platform.node()}{args.tag}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("PROFILE DONE", flush=True)


if __name__ == "__main__":
    main()
