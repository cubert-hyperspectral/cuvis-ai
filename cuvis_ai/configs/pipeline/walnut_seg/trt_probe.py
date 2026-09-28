"""Torch-TensorRT feasibility probe for the walnut RF-DETR-Seg-L network (rgb_v2).

The network is prepared exactly as rfdetr's ``inference()`` does (deep copy, ``eval()``, ``export()`` mode, dtype cast)
and fed one REAL input: the reference cu3s frame -> the pipeline's own Selector node -> rfdetr's predict
preprocessing (quantized [0, 1] frame, bilinear resize to the model resolution without antialias, mean/std
normalisation). Timed (median of N CUDA-synchronised calls after warm-up), network only:
  fp32 eager | fp16 eager | fp16 torch.jit.trace (= the `_fast` pipelines) | Torch-TensorRT fp16 (dynamo, if importable)
Each variant's outputs go through rfdetr's own post-processing (threshold 0.5, class 0) and are rasterised like the
node (per pixel max of mask x confidence); reported: shell-mask IoU vs fp32, detections, compile/build time.

Usage (an overlay env with torch-tensorrt; never install into the stack or cuvis.next child envs):
  python trt_probe.py <walnut_seg dir> [--out result.json] [--skip-trt]
"""

import argparse
import copy
import json
import os
import platform
import statistics
import time
import traceback

import numpy as np
import torch
import torchvision.transforms.functional as F
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_rfdetr.functional import to_unit_frames

N, WARM = 50, 10


def bench(fn, x):
    with torch.no_grad():
        for _ in range(WARM):
            fn(x)
        ts = []
        for _ in range(N):
            torch.cuda.synchronize()
            t = time.perf_counter()
            out = fn(x)
            torch.cuda.synchronize()
            ts.append(1000.0 * (time.perf_counter() - t))
    return statistics.median(ts), out


def as_dict(out):
    """Network tuple -> the dict rfdetr's postprocess expects (same mapping as RFDETR.predict)."""
    if isinstance(out, dict):
        return {k: v.float() for k, v in out.items()}
    d = {"pred_logits": out[1].float(), "pred_boxes": out[0].float()}
    if len(out) == 3:
        d["pred_masks"] = out[2].float()
    return d


def score_map(m, out, hw, threshold, class_id):
    res = m.model.postprocess(as_dict(out), target_sizes=torch.tensor([hw], device="cuda"), score_threshold=threshold)[0]
    keep = (res["scores"] > threshold) & (res["labels"] == class_id)
    canvas = torch.zeros(hw, device="cuda")
    if int(keep.sum()) and "masks" in res:
        masks = res["masks"][keep].reshape(int(keep.sum()), *hw).bool()
        conf = res["scores"][keep].float().view(-1, 1, 1)
        canvas = torch.where(masks, conf, torch.zeros((), device="cuda")).amax(0)
    return canvas, int(keep.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("seg_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-trt", action="store_true")
    args = ap.parse_args()
    env = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "python": platform.python_version()}
    try:
        import torch_tensorrt
        import tensorrt
        env.update(torch_tensorrt=torch_tensorrt.__version__, tensorrt=tensorrt.__version__)
    except Exception as exc:  # noqa: BLE001 - report, do not hide
        env["torch_tensorrt_import_error"] = f"{type(exc).__name__}: {exc}"[:400]
    print("ENV", json.dumps(env), flush=True)

    y = os.path.join(args.seg_dir, "walnut_seg_rgb_v2_cuvisnext_cube.yaml")
    pipe = CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda")
    nodes = {n.name: n for n in pipe.nodes}
    seg, sel = nodes["Seg"], nodes["Selector"]
    d = np.load(os.path.join(args.seg_dir, "ref", "real_world_live_000_f0000_reflectance.npz"))
    cube = torch.from_numpy(d["cube"].astype(np.float32))[None].cuda()
    with torch.no_grad():
        rgb = sel.forward(cube=cube, wavelengths=d["wavelengths"].astype(np.int32))["rgb_image"]
    hw = (int(rgb.shape[1]), int(rgb.shape[2]))
    m = seg._build_model()
    res = int(m.model.resolution)
    x = to_unit_frames(rgb)[0].permute(2, 0, 1).contiguous()
    x = F.normalize(F.resize(x, [res, res], antialias=False), m.means, m.stds)[None].cuda()

    def exported(dtype):
        net = copy.deepcopy(m.model.model).to(m.model.device).eval()
        net.export()
        return net.to(dtype=dtype)

    rows, ref = {}, None
    net32 = exported(torch.float32)
    ms, out = bench(net32, x)
    ref, n_ref = score_map(m, out, hw, seg.threshold, seg.class_filter or 0)
    rows["fp32 eager"] = dict(ms=round(ms, 2), detections=n_ref, iou=1.0)
    del net32
    net16 = exported(torch.float16)
    x16 = x.half()

    def record(name, fn, t_build=None):
        ms, out = bench(fn, x16)
        sm, n = score_map(m, out, hw, seg.threshold, seg.class_filter or 0)
        a, b = sm >= 0.5, ref >= 0.5
        rows[name] = dict(ms=round(ms, 2), detections=n, iou=round(float((a & b).sum()) / max(1, int((a | b).sum())), 5),
                          build_s=None if t_build is None else round(t_build, 1))
        print(name, json.dumps(rows[name]), flush=True)

    record("fp16 eager", net16)
    t = time.perf_counter()
    with torch.no_grad():
        traced = torch.jit.trace(net16, x16)
    record("fp16 jit.trace (= _fast)", traced, time.perf_counter() - t)
    del traced
    if not args.skip_trt and "torch_tensorrt" in env:
        import torch_tensorrt

        variants = (
            ("torch-tensorrt fp16 (dynamo)", {}),
            # torch-tensorrt 2.14's SDPA converter passes a constant `value` straight to add_attention_v2
            # (TypeError); either decompose attention into matmul + softmax before conversion ...
            ("torch-tensorrt fp16, attention decomposed", {"enable_experimental_decompositions": True}),
            # ... or keep only the attention op in PyTorch (graph breaks around every attention block).
            # (torch 2.14 exports aten.scaled_dot_product_attention, torch 2.11 its _efficient overload; the
            # failing `value` operand is a frozen parameter of the network)
            ("torch-tensorrt fp16, attention in PyTorch",
             {"torch_executed_ops": {"torch.ops.aten.scaled_dot_product_attention.default",
                                     "torch.ops.aten._scaled_dot_product_efficient_attention.default",
                                     "torch.ops.aten._scaled_dot_product_flash_attention.default",
                                     "torch.ops.aten._scaled_dot_product_cudnn_attention.default"}}),
            # explicit typing (default in some releases) forces every layer to the model dtype (fp16); weak typing
            # lets TensorRT keep precision-sensitive layers in fp32 - same as the torch-tensorrt 2.14 default
            ("torch-tensorrt fp16 weak typing, attention in PyTorch",
             {"use_explicit_typing": False,
              "torch_executed_ops": {"torch.ops.aten.scaled_dot_product_attention.default",
                                     "torch.ops.aten._scaled_dot_product_efficient_attention.default",
                                     "torch.ops.aten._scaled_dot_product_flash_attention.default",
                                     "torch.ops.aten._scaled_dot_product_cudnn_attention.default"}}),
        )
        for label, kwargs in variants:
            t = time.perf_counter()
            try:
                with torch.no_grad():
                    try:
                        trt_mod = torch_tensorrt.compile(net16, ir="dynamo", inputs=[x16],
                                                         enabled_precisions={torch.half}, **kwargs)
                    except AssertionError as exc:
                        # releases with explicit typing on by default reject enabled_precisions; the fp16
                        # model's own dtypes then decide the precision
                        if "use_explicit_typing" not in str(exc):
                            raise
                        trt_mod = torch_tensorrt.compile(net16, ir="dynamo", inputs=[x16],
                                                         use_explicit_typing=True, **kwargs)
                record(label, trt_mod, time.perf_counter() - t)
                del trt_mod
            except Exception as exc:  # noqa: BLE001
                rows[label] = dict(error=f"{type(exc).__name__}: {exc}"[:600], build_s=round(time.perf_counter() - t, 1))
                print(label, "FAILED", rows[label]["error"], flush=True)
                traceback.print_exc()
    print("\n| variant | network ms | detections | shell-mask IoU vs fp32 | build s |")
    print("|---|---|---|---|---|")
    for k, r in rows.items():
        print(f"| {k} | {r.get('ms', '-')} | {r.get('detections', '-')} | {r.get('iou', '-')} | {r.get('build_s') or '-'} |"
              + (f" {r['error'][:120]}" if "error" in r else ""))
    if args.out:
        json.dump({"env": env, "rows": rows}, open(args.out, "w"), indent=1)
    print("TRT PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
