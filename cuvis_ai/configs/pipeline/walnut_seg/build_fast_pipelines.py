"""Build the speed variants of the v2 walnut_seg pipelines (rgb_v2, cir_v2, ens_rgb_cir_mean_v2).

Same graph, weights and outputs as the default pipeline; only the RFDETRSegmenter speed hparams differ.
Every pipeline already combines the instance masks on the GPU (`fast_paste`, the node default, bit-identical).

  `_exact`  precision fp32, jit_trace false, gpu_input TRUE
            the frame is handed to rfdetr as a CUDA tensor instead of a CPU uint8 array -> BIT-IDENTICAL to the
            default pipeline (acceptance: heatmap and mask torch.equal on the reference frame)
  `_fp16`   precision fp16, jit_trace false, gpu_input TRUE
            rfdetr model.inference(dtype=float16, compile=False): network exported + cast to fp16, NOT traced ->
            no start-up trace (first frame ~ as fp32), a bit slower per frame than `_fast` (acceptance as `_fast`)
  `_fast`   precision fp16, jit_trace TRUE, gpu_input TRUE
            rfdetr model.inference(dtype=float16, compile=True): network exported, cast to fp16 and traced
            (fixed 1x3x504x504 input; first frame +few s). fp16 rounds scores, so pixels at the 0.5 cut can flip:
            the 163-frame GT check (validate_speed_options.py, option o11) gives ensemble shell IoU 0.963 = fp32,
            mask agreement 0.995 (acceptance here: mask IoU vs default >= 0.99 on the reference frame)
  `_trt_fp32` backend tensorrt, precision fp32, gpu_input TRUE
            RF-DETR's network as a TensorRT fp32 engine (TF32 allowed = like-for-like with the deployed fp32),
            rfdetr's own pre-/post-processing around it (acceptance as `_fast`)
  `_trt_fp16` backend tensorrt, precision fp16, gpu_input TRUE (TensorRT FP16 builder flag; acceptance as `_fast`)
            The TensorRT variants need the `tensorrt` package (10.x) + `onnx` in the env and build this machine's
            engines first (cuvis_ai_rfdetr.trt_engine build-pipeline, next to the checkpoints: weights/*.pth.trt/).

The yaml is copied with the hparams set (the .pt / cuvis_picker .pt hold the non-RF-DETR state and are copied
unchanged), then loaded and run on the float32 reference frame next to the default pipeline; both outputs must be
present. Needs CUDA and the cuvis-ai-rfdetr version with the speed hparams. Run in the stack cuvis-ai/.venv:
  python build_fast_pipelines.py [exact,fp16,fast]
TensorRT variants, in a throwaway overlay (never install into the stack venv / cuvis.next child envs), laptop:
  uv run --no-project --python <stack .venv python> --with "tensorrt-cu12==10.15.1.29" --with onnx
      python build_fast_pipelines.py trt_fp32,trt_fp16
"""

import os
import shutil
import sys

import numpy as np
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__))
VARIANTS = {
    "exact": ({"precision": "fp32", "jit_trace": False, "gpu_input": True, "fast_paste": True},
              "EXACT variant: RF-DETR gets the frame on the GPU (gpu_input) - bit-identical to"),
    "fp16": ({"precision": "fp16", "jit_trace": False, "gpu_input": True, "fast_paste": True},
             "FP16 variant: RF-DETR runs fp16 (not JIT-traced, so no start-up trace) with the frame handed over on the"
             " GPU (precision fp16, gpu_input). Same weights and outputs as"),
    "fast": ({"precision": "fp16", "jit_trace": True, "gpu_input": True, "fast_paste": True},
             "FAST variant: RF-DETR runs fp16 + JIT-traced with the frame handed over on the GPU (precision fp16,"
             " jit_trace, gpu_input); first frame takes a few seconds longer (trace). Same weights and outputs as"),
    "trt_fp32": ({"backend": "tensorrt", "precision": "fp32", "jit_trace": False, "gpu_input": True,
                  "fast_paste": True},
                 "TRT_FP32 variant: RF-DETR's network runs as a TensorRT fp32 engine (TF32 math allowed, like the"
                 " PyTorch default) with rfdetr's own pre- and post-processing (backend tensorrt, precision fp32,"
                 " gpu_input). Needs the tensorrt package (10.x) in the env and this machine's engine: python -m"
                 " cuvis_ai_rfdetr.trt_engine build-pipeline <this yaml>. Same weights and outputs as"),
    "trt_fp16": ({"backend": "tensorrt", "precision": "fp16", "jit_trace": False, "gpu_input": True,
                  "fast_paste": True},
                 "TRT_FP16 variant: RF-DETR's network runs as a TensorRT fp16 engine with rfdetr's own pre- and"
                 " post-processing (backend tensorrt, precision fp16, gpu_input). Needs the tensorrt package (10.x) in"
                 " the env and this machine's engine: python -m cuvis_ai_rfdetr.trt_engine build-pipeline <this yaml>."
                 " Same weights and outputs as"),
}
REF = np.load(os.path.join(SEG, "ref", "real_world_live_000_f0000_reflectance.npz"))
cube = torch.from_numpy(REF["cube"].astype(np.float32))[None].cuda()
wl = torch.from_numpy(REF["wavelengths"].astype(np.int32))[None]


def outputs(p):
    with torch.no_grad():
        o = p.forward(batch={"cube": cube, "wavelengths": wl})
    return o[("ShellHeatmap", "scores")].cpu(), o[("ShellMask", "decisions")].cpu().bool()


allok = True
for vname in (sys.argv[1].split(",") if len(sys.argv) > 1 else list(VARIANTS)):
    hp, blurb = VARIANTS[vname]
    for short in ("rgb_v2", "cir_v2", "ens_rgb_cir_mean_v2"):
        src = os.path.join(SEG, f"walnut_seg_{short}_cuvisnext_cube")
        dst = os.path.join(SEG, f"walnut_seg_{short}_{vname}_cuvisnext_cube")
        doc = yaml.safe_load(open(src + ".yaml"))
        n_seg = 0
        for n in doc["nodes"]:
            if n["class_name"].endswith("RFDETRSegmenter"):
                n["hparams"].update(hp)
                n_seg += 1
        doc["metadata"]["name"] = os.path.basename(dst)
        doc["metadata"]["description"] = (
            doc["metadata"]["description"].replace(f"({short}:", f"({short} {vname.upper()}:", 1)
            + f" {blurb} walnut_seg_{short}_cuvisnext_cube.")
        yaml.safe_dump(doc, open(dst + ".yaml", "w"), sort_keys=False)
        shutil.copy2(src + ".pt", dst + ".pt")
        shutil.copy2(os.path.join(SEG, f"cuvis_picker_walnut_seg_{short}_cuvisnext_cube.pt"),
                     os.path.join(SEG, f"cuvis_picker_walnut_seg_{short}_{vname}_cuvisnext_cube.pt"))
        if hp.get("backend") == "tensorrt":
            from cuvis_ai_rfdetr import trt_engine
            trt_engine.main(["build-pipeline", dst + ".yaml"])
        base = CuvisPipeline.load_pipeline(src + ".yaml", weights_path=src + ".pt", device="cuda")
        var = CuvisPipeline.load_pipeline(dst + ".yaml", weights_path=dst + ".pt", device="cuda")
        segs = [n for n in var.nodes if type(n).__name__ == "RFDETRSegmenter"]
        hp_ok = len(segs) == n_seg and all(
            (s.precision, s.jit_trace, s.gpu_input, s.fast_paste, getattr(s, "backend", "torch"))
            == (hp["precision"], hp["jit_trace"], hp["gpu_input"], hp["fast_paste"], hp.get("backend", "torch"))
            for s in segs)
        h0, m0 = outputs(base)
        h1, m1 = outputs(var)
        iou = float((m0 & m1).sum()) / max(1, int((m0 | m1).sum()))
        outs = sorted(var.get_output_specs())
        same = torch.equal(h0, h1) and torch.equal(m0, m1)
        ok = hp_ok and "ShellHeatmap.scores" in outs and "ShellMask.decisions" in outs and (
            same if vname == "exact" else iou >= 0.99)
        allok &= ok
        print(f"{'OK  ' if ok else 'FAIL'} {os.path.basename(dst)}: {n_seg} segmenter(s) {hp}, mask px "
              f"{int(m1.sum())} vs default {int(m0.sum())} (IoU {iou:.4f}, bit-identical {same}), heatmap "
              f"max|diff| {float((h1 - h0).abs().max()):.4f}, outputs {outs}", flush=True)
        del base, var
        torch.cuda.empty_cache()
print("BUILD SPEED VARIANTS DONE", "ALL OK" if allok else "WITH FAILURES")
