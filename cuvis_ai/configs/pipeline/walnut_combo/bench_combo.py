"""Interleaved A/B timing: a combined walnut pipeline against its two standalone pipelines, in one process.

    python bench_combo.py --combo C.yaml --fo F.yaml --seg S.yaml --ref-cubes R.npz [--reps 10] --out bench.json

All three pipelines (yaml + sibling .pt, CUDA) are loaded side by side and warmed up (3 passes each: JIT traces,
TensorRT engines, cuDNN). Then every rep runs every frame through all three in a rotating order, each call
CUDA-synchronised. Frames: the SEG reference frame and the FO reference cubes (see check_combo.py). Reported per
pipeline: median / p90 ms per frame, and the combined median against FO + SEG. The standalone FO pipeline runs in
the combined process's global state (the SEG plugin imports rfdetr, which sets TF32 for float32 matmuls); its TF32 /
float16 nodes set their precision themselves, so that matches its own env.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from check_combo import frames


def main() -> None:
    import torch
    from cuvis_ai_core.utils.restore import restore_pipeline
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True)
    ap.add_argument("--fo", required=True)
    ap.add_argument("--seg", required=True)
    ap.add_argument("--ref-cubes", required=True)
    ap.add_argument("--seg-ref", default=str(Path(__file__).resolve().parent.parent / "walnut_seg" / "ref"
                                             / "real_world_live_000_f0000_reflectance.npz"))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    pipes = {}
    for key in ("combo", "fo", "seg"):
        y = Path(getattr(a, key)).resolve()
        pipes[key] = restore_pipeline(str(y), weights_path=str(y.with_suffix(".pt")), device="cuda")
    ctx = Context(stage=ExecutionStage.INFERENCE)
    batches = [(tag, {"cube": torch.from_numpy(c)[None].cuda(), "wavelengths": torch.from_numpy(w)[None].cuda()})
               for tag, c, w in frames(a.ref_cubes, a.seg_ref)]
    first = {}
    with torch.no_grad():
        for key, p in pipes.items():
            for i in range(3):
                t0 = time.perf_counter()
                p.forward(batch=batches[i % len(batches)][1], context=ctx)
                torch.cuda.synchronize()
                if i == 0:
                    first[key] = (time.perf_counter() - t0) * 1e3
        times = {k: [] for k in pipes}
        order = list(pipes)
        for rep in range(a.reps):
            for _, batch in batches:
                for key in order[rep % 3:] + order[:rep % 3]:
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    pipes[key].forward(batch=batch, context=ctx)
                    torch.cuda.synchronize()
                    times[key].append((time.perf_counter() - t0) * 1e3)
    res = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "reps": a.reps,
           "frames": [t for t, _ in batches], "peak_gpu_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
           "pipelines": {k: str(Path(getattr(a, k)).name) for k in pipes}}
    for key, t in times.items():
        res[key] = {"ms_median": round(float(np.median(t)), 1), "ms_p90": round(float(np.percentile(t, 90)), 1),
                    "first_frame_ms": round(first[key], 1), "n": len(t)}
    res["fo_plus_seg_ms"] = round(res["fo"]["ms_median"] + res["seg"]["ms_median"], 1)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{Path(a.combo).stem}: combo {res['combo']['ms_median']} ms (p90 {res['combo']['ms_p90']}) | FO "
          f"{res['fo']['ms_median']} + SEG {res['seg']['ms_median']} = {res['fo_plus_seg_ms']} ms | first frame combo "
          f"{res['combo']['first_frame_ms']} ms | peak {res['peak_gpu_GB']} GB ({res['device']})", flush=True)


if __name__ == "__main__":
    main()
