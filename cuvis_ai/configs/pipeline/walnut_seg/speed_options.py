"""Benchmark inference-speed options for the walnut_seg RF-DETR pipelines on real cu3s frames.

Each option is applied to the loaded pipelines' ``RFDETRSegmenter`` nodes WITHOUT changing the plugin code
(rfdetr's own ``model.inference(...)`` API, a GPU input path, a smaller model resolution, parallel ensemble
members), timed per frame with CUDA synchronisation (first WARMUP frames skipped), and checked against the fp32
baseline masks (agreement IoU) and the ground truth (shell IoU, false-positive px).

Also reports a breakdown of the baseline Seg node: pure network forward vs rfdetr ``predict`` (pre/post-processing)
vs the node's CPU mask pasting.

Usage (seg child env + cu3s reader overlay, see SEG_V2_PROFILING_2026-09-24.md):
  python speed_options.py --out <dir> --gt-dir <dir with GT npz> --cu3s a.cu3s [--cu3s b.cu3s] [--options o0,o2,...]
"""

import argparse
import copy
import glob
import json
import os
import platform
import threading
import time

import numpy as np
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule

import cuvis_ai_rfdetr.node.rfdetr_segmenter as segmod

SEG = os.path.dirname(os.path.abspath(__file__))
WARMUP = 2
PIPES = ["rgb_v2", "cir_v2", "ens_rgb_cir_mean_v2"]

# name -> (dtype | None, jit_trace, gpu_input, resolution | None, parallel_members, fast_paste[, compile_mode[, autocast]])
OPTIONS = {
    "o0_baseline_fp32": (None, False, False, None, False, False),
    "o1_export_fp32": (torch.float32, False, False, None, False, False),
    "o2_fp16": (torch.float16, False, False, None, False, False),
    "o3_bf16": (torch.bfloat16, False, False, None, False, False),
    "o4_fp16_jit": (torch.float16, True, False, None, False, False),
    "o5_fp16_gpuin": (torch.float16, False, True, None, False, False),
    "o9_fp32_gpuin_fastpaste": (None, False, True, None, False, True),
    "o10_fp16_gpuin_fastpaste": (torch.float16, False, True, None, False, True),
    "o11_fp16_jit_gpuin_fastpaste": (torch.float16, True, True, None, False, True),
    "o6_fp16_gpuin_fastpaste_res432": (torch.float16, False, True, 432, False, True),
    "o7_fp16_gpuin_fastpaste_res384": (torch.float16, False, True, 384, False, True),
    "o8_fp16_gpuin_fastpaste_parallel": (torch.float16, False, True, None, True, True),
    "o12_fp16_gpuin_fastpaste_compile": (torch.float16, False, True, None, False, True, "default"),
    "o13_fp16_gpuin_fastpaste_compile_cudagraphs": (torch.float16, False, True, None, False, True, "reduce-overhead"),
    # mixed precision: fp32 weights, torch.autocast keeps LayerNorm/softmax/reductions in fp32, matmuls in half
    "o14_autocast_fp16_gpuin_fastpaste": (None, False, True, None, False, True, None, torch.float16),
    "o15_autocast_bf16_gpuin_fastpaste": (None, False, True, None, False, True, None, torch.bfloat16),
}
AUTOCAST = {"dtype": None}

_orig_to_uint8 = segmod.to_uint8_frames
_orig_predict_frame = segmod.RFDETRSegmenter._predict_frame


def gpu_frames(rgb_image):
    """Same values as to_uint8_frames (x*255 rounded, as uint8), but kept on the GPU as CHW float in [0, 1]."""
    x = rgb_image.detach().float()
    if x.numel() > 0 and float(x.max()) <= 1.5:
        x = x * 255.0
    x = x.clamp(0.0, 255.0).round() / 255.0
    return [x[i].permute(2, 0, 1).contiguous() for i in range(x.shape[0])]


def predict_frame_no_source(self, model, frame):
    result = model.predict(frame, threshold=self.threshold, include_source_image=False)
    if isinstance(result, (list, tuple)):
        result = result[0] if result else None
    if result is None or getattr(result, "xyxy", None) is None:
        return []
    rows = []
    for j in range(len(result.xyxy)):
        x1, y1, x2, y2 = (float(v) for v in result.xyxy[j])
        conf = float(result.confidence[j]) if result.confidence is not None else 0.0
        cid = int(result.class_id[j]) if result.class_id is not None else -1
        m = result.mask[j] if result.mask is not None else None
        rows.append((x1, y1, x2, y2, conf, cid, m))
    return rows


_orig_forward = segmod.RFDETRSegmenter.forward


def fast_forward(self, rgb_image, context=None, **_):
    """Same outputs as RFDETRSegmenter.forward (tiling="whole"): the per-instance CPU boolean paste is replaced by
    ONE max over all instance masks on the image's device, max(where(mask_i, conf_i, 0)) — identical values."""
    model = self._ensure_model()
    batch, height, width = rgb_image.shape[0], rgb_image.shape[1], rgb_image.shape[2]
    device = rgb_image.device
    frames = segmod.to_uint8_frames(rgb_image)
    scores = torch.zeros((batch, height, width, 1), dtype=torch.float32, device=device)
    anomaly_score = torch.zeros((batch,), dtype=torch.float32)
    detections = []
    for idx in range(batch):
        rows, masks, confs = [], [], []
        if AUTOCAST["dtype"] is not None:
            with torch.autocast("cuda", dtype=AUTOCAST["dtype"]):
                preds = self._predict_frame(model, frames[idx])
        else:
            preds = self._predict_frame(model, frames[idx])
        for x1, y1, x2, y2, conf, cid, m in preds:
            if self.class_filter is not None and cid != self.class_filter:
                continue
            rows.append((x1, y1, x2, y2, conf, cid))
            masks.append(m)
            confs.append(conf)
        if masks:
            mm = torch.from_numpy(np.stack(masks)).to(device, non_blocking=True)
            cc = torch.tensor(confs, dtype=torch.float32, device=device).view(-1, 1, 1)
            scores[idx, :, :, 0] = torch.where(mm, cc, torch.zeros((), device=device)).amax(0)
        detections.append([{"xyxy": [r[0], r[1], r[2], r[3]], "confidence": r[4], "class_id": r[5]} for r in rows])
        anomaly_score[idx] = max((r[4] for r in rows), default=0.0)
    return {"scores": scores, "detections": detections, "anomaly_score": anomaly_score.to(device)}


def set_patches(gpu_input: bool, fast_paste: bool = False) -> None:
    segmod.to_uint8_frames = gpu_frames if gpu_input else _orig_to_uint8
    segmod.RFDETRSegmenter._predict_frame = predict_frame_no_source if gpu_input else _orig_predict_frame
    segmod.RFDETRSegmenter.forward = fast_forward if fast_paste else _orig_forward


def load_frames(cu3s_list, gt_dir):
    frames = []
    for c in cu3s_list:
        stem = os.path.basename(c)[:-5]
        dm = Cu3sDataModule(cu3s_file_path=c, processing_mode="Reflectance", batch_size=1, num_workers=0)
        dm.setup(stage="predict")
        for batch in Predictor._iter_batches(dm.predict_dataloader()):
            idx = int(batch["mesu_index"][0])
            name = f"{stem}_f{idx:04d}"
            gt = os.path.join(gt_dir, name + ".npz")
            if not os.path.exists(gt):
                continue
            g = np.load(gt)
            frames.append(dict(name=name, cube=batch["cube"].float(), wl=batch["wavelengths"],
                               mi=batch["mesu_index"], gs=g["gs"], gf=g["gf"]))
    return frames


def build(short, resolution):
    yml = os.path.join(SEG, f"walnut_seg_{short}_cuvisnext_cube.yaml")
    if resolution is not None:
        doc = yaml.safe_load(open(yml))
        for n in doc["nodes"]:
            if n["class_name"].endswith("RFDETRSegmenter"):
                n["hparams"]["resolution"] = int(resolution)
        yml2 = os.path.join(SEG, f"_tmp_speed_{short}_{resolution}.yaml")
        yaml.safe_dump(doc, open(yml2, "w"), sort_keys=False)
        p = CuvisPipeline.load_pipeline(yml2, weights_path=yml[:-5] + ".pt", device="cuda")
        os.remove(yml2)
        return p
    return CuvisPipeline.load_pipeline(yml, weights_path=yml[:-5] + ".pt", device="cuda")


def seg_nodes(pipe):
    return [n for n in pipe.nodes if isinstance(n, segmod.RFDETRSegmenter)]


def apply_dtype(pipe, dtype, jit, compile_mode=None):
    for n in seg_nodes(pipe):
        m = n._ensure_model()
        if dtype is not None:
            m.inference(compile=jit, batch_size=1, dtype=dtype)
        if compile_mode is not None:  # torch.compile (inductor) the exported inference model; static 1x3xRxR input
            m.model.inference_model = torch.compile(m.model.inference_model, mode=compile_mode, dynamic=False)


def instrument(pipe):
    """Wrap each segmenter's forward and its rfdetr model's predict with CUDA-synchronised timers."""
    stats = {}
    for n in seg_nodes(pipe):
        m = n._ensure_model()
        st = stats[n.name] = dict(node=[], pred=[], ninst=[])

        def fwd(*a, _o=n.forward, _st=st, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = _o(*a, **k)
            torch.cuda.synchronize()
            _st["node"].append(1000.0 * (time.perf_counter() - t))
            _st["ninst"].append(len(r["detections"][0]))
            return r

        def pred(*a, _o=m.predict, _st=st, **k):
            torch.cuda.synchronize()
            t = time.perf_counter()
            r = _o(*a, **k)
            torch.cuda.synchronize()
            _st["pred"].append(1000.0 * (time.perf_counter() - t))
            return r

        n.forward = fwd
        m.predict = pred
    return stats


def run_pipe(pipe, fr):
    with torch.no_grad():
        o = pipe.forward(batch={"cube": fr["cube"].cuda(), "wavelengths": fr["wl"], "mesu_index": fr["mi"]})
    return o[("ShellMask", "decisions")][0, :, :, 0].detach().cpu().numpy()


def run_ens_parallel(pipe, fr):
    """Ensemble with the two members' selector+segmenter branches in two threads, then fuse + threshold."""
    nodes = {n.name: n for n in pipe.nodes}
    ds = nodes["DataSource"].forward(cube=fr["cube"].cuda(), wavelengths=fr["wl"], mesu_index=fr["mi"])
    res = {}

    def branch(sel, seg):
        stream = torch.cuda.Stream()  # own stream per branch, else the kernels serialize on the default stream
        with torch.no_grad(), torch.cuda.stream(stream):
            img = nodes[sel].forward(cube=ds["cube"], wavelengths=ds["wavelengths"])["rgb_image"]
            res[seg] = nodes[seg].forward(rgb_image=img)["scores"]
        stream.synchronize()

    ts = [threading.Thread(target=branch, args=a) for a in (("SelRGB", "SegRGB"), ("SelCIR", "SegCIR"))]
    [t.start() for t in ts]
    [t.join() for t in ts]
    fused = nodes["Fuse"].forward(a=res["SegRGB"], b=res["SegCIR"])["scores"]
    return (fused[0, :, :, 0] >= 0.5).cpu().numpy()


def network_ms(pipe, n=20):
    """Pure LW-DETR forward (dummy input at the model resolution), per segmenter node."""
    out = {}
    for node in seg_nodes(pipe):
        m = node._ensure_model()
        net = m.model.inference_model if getattr(m, "_is_optimized_for_inference", False) else m.model.model
        dt = m._optimized_dtype if getattr(m, "_is_optimized_for_inference", False) else torch.float32
        x = torch.randn(1, 3, m.model.resolution, m.model.resolution, device="cuda", dtype=dt)
        if hasattr(net, "eval"):
            net.eval()
        with torch.no_grad():
            for _ in range(3):
                net(x)
            torch.cuda.synchronize()
            t = time.perf_counter()
            for _ in range(n):
                net(x)
            torch.cuda.synchronize()
        out[node.name] = 1000.0 * (time.perf_counter() - t) / n
    return out


def breakdown(pipe, frames, n=10):
    """Baseline Seg node: node forward vs rfdetr predict vs pure network (single-model pipelines)."""
    node = seg_nodes(pipe)[0]
    m = node._ensure_model()
    sel = next(n for n in pipe.nodes if n.name.startswith("Sel"))
    t_node, t_pred = [], []
    orig = type(m).predict

    def timed_predict(self, *a, **k):
        torch.cuda.synchronize()
        t = time.perf_counter()
        r = orig(self, *a, **k)
        torch.cuda.synchronize()
        t_pred.append(1000.0 * (time.perf_counter() - t))
        return r

    type(m).predict = timed_predict
    try:
        for fr in frames[:n]:
            with torch.no_grad():
                img = sel.forward(cube=fr["cube"].cuda(), wavelengths=fr["wl"])["rgb_image"]
                torch.cuda.synchronize()
                t = time.perf_counter()
                node.forward(rgb_image=img)
                torch.cuda.synchronize()
                t_node.append(1000.0 * (time.perf_counter() - t))
    finally:
        type(m).predict = orig
    return float(np.median(t_node[WARMUP:])), float(np.median(t_pred[WARMUP:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--cu3s", action="append", required=True)
    ap.add_argument("--options", default=",".join(OPTIONS))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    env = dict(host=platform.node(), gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    print("ENV", json.dumps(env), flush=True)
    frames = load_frames(args.cu3s, args.gt_dir)
    print(f"frames with GT: {len(frames)}", flush=True)
    base_masks, results = {}, []
    for opt in args.options.split(","):
        dtype, jit, gpu_in, res, parallel, fast, *rest = OPTIONS[opt]
        compile_mode = rest[0] if rest else None
        AUTOCAST["dtype"] = rest[1] if len(rest) > 1 else None
        for short in PIPES:
            if parallel and not short.startswith("ens"):
                continue
            set_patches(gpu_in, fast)
            try:
                pipe = build(short, res)
                apply_dtype(pipe, dtype, jit, compile_mode)
                if compile_mode is not None:  # compile + CUDA-graph capture happen on the first calls: warm up untimed
                    t = time.perf_counter()
                    for fr in frames[:4]:
                        run_pipe(pipe, fr)
                    print(f"  {opt} {short}: compile warm-up {time.perf_counter() - t:.0f} s", flush=True)
                stats = {} if parallel else instrument(pipe)  # device-wide syncs would serialize the branches
                times, masks = [], []
                for fr in frames:
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    m = run_ens_parallel(pipe, fr) if parallel else run_pipe(pipe, fr)
                    torch.cuda.synchronize()
                    times.append(1000.0 * (time.perf_counter() - t))
                    masks.append(m)
                net = network_ms(pipe)
                bd = None
            except Exception as e:  # keep going; record why an option failed
                print(f"FAIL {opt} {short}: {type(e).__name__}: {str(e)[:300]}", flush=True)
                results.append(dict(option=opt, pipeline=short, error=f"{type(e).__name__}: {str(e)[:300]}"))
                continue
            finally:
                set_patches(False)
            if opt == "o0_baseline_fp32":
                base_masks[short] = masks
            ious, fps_, agree = [], [], []
            for fr, m, i in zip(frames, masks, range(len(frames))):
                gs = fr["gs"]
                inter, union = (m & gs).sum(), (m | gs).sum()
                ious.append(inter / union if union else np.nan)
                fps_.append(int((m & ~gs).sum()))
                b = base_masks.get(short.replace("_parallel", ""), [None] * len(frames))[i]
                if b is not None:
                    u = (m | b).sum()
                    agree.append((m & b).sum() / u if u else 1.0)
            st = np.array(times[WARMUP:])
            row = dict(option=opt, pipeline=short, ms_median=round(float(np.median(st)), 1),
                       ms_mean=round(float(st.mean()), 1), fps_median=round(1000.0 / float(np.median(st)), 1),
                       network_ms={k: round(v, 1) for k, v in net.items()},
                       shell_iou=round(float(np.nanmean(ious)), 4), fp_px=round(float(np.mean(fps_))),
                       agreement_iou_vs_fp32=round(float(np.mean(agree)), 5) if agree else None, **env)
            row["per_seg"] = {
                k: dict(node_ms=round(float(np.median(v["node"][WARMUP:])), 1),
                        predict_ms=round(float(np.median(v["pred"][WARMUP:])), 1) if v["pred"] else None,
                        instances=int(np.median(v["ninst"][WARMUP:])))
                for k, v in stats.items() if v["node"]}
            results.append(row)
            print("RESULT", json.dumps(row), flush=True)
            del pipe
            torch.cuda.empty_cache()
    with open(os.path.join(args.out, f"speed_options_{platform.node()}.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("SPEED OPTIONS DONE", flush=True)


if __name__ == "__main__":
    main()
