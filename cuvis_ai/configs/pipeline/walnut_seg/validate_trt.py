"""163-frame accuracy check of TensorRT engines for the walnut v2 RF-DETR models vs the deployed PyTorch fp32 path.

Per model (rgb_v2, cir_v2): an rfdetr-style ONNX export (``model.export()`` + ``torch.onnx.export``, opset 17) and
TensorRT engines built on THIS machine with the installed TensorRT Python API, cached in ``--work``:
  fp32  TF32 allowed (the TensorRT default) = like-for-like with the deployed "fp32", which runs TF32 because
        ``import rfdetr`` sets torch's float32 matmul precision to "high"
  fp16  ``BuilderFlag.FP16`` (mixed precision: TensorRT keeps a layer in fp32 where that is faster or needed)
For each of the 163 cached test frames (``frame_cache.pt`` of validate_speed_options.py: uint8 RGB / CIR model inputs
from the pipelines' own selector nodes + the GT masks):
  reference  the deployed RFDETRSegmenter node forward (fp32 PyTorch, fast paste, no GPU input)
  TensorRT   the node's preprocessing (uint8 -> [0, 1], bilinear resize without antialias, mean/std) -> engine ->
             rfdetr's own post-processing (node threshold, class filter) -> the node's rasterisation (max mask x conf)
Masks: rgb_v2 and cir_v2 singles (their thresholds) and the B+C mean ensemble (member thresholds from the ensemble
pipeline, mean >= 0.5). Rows like validate_speed_options.py (tp / fp / fn / gs / objfp / f2n / gf / agreement vs the
reference) -> summarize_validate.py. Also reports the engine build times and the median engine call time.

Usage (overlay env with tensorrt + onnx; never install into the stack or cuvis.next child envs):
  python validate_trt.py --cache <frame_cache.pt> --work <dir> [--limit N]
"""

import argparse
import copy
import json
import os
import platform
import statistics
import time

import numpy as np
import tensorrt as trt
import torch
import torchvision.transforms.functional as F
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_rfdetr.functional import to_unit_frames

SEG = os.path.dirname(os.path.abspath(__file__))
LOGGER = trt.Logger(trt.Logger.WARNING)
TORCH_DTYPE = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32,
               trt.int64: torch.int64, trt.bool: torch.bool}


def load(short):
    y = os.path.join(SEG, f"walnut_seg_{short}_cuvisnext_cube.yaml")
    return {n.name: n for n in CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda").nodes}


def preprocess(m, img):
    """Exactly the node's input path: re-quantize to uint8 levels / 255, resize, normalise (rfdetr predict)."""
    res = int(m.model.resolution)
    x = to_unit_frames(img)[0].permute(2, 0, 1).contiguous()
    return F.normalize(F.resize(x, [res, res], antialias=False), m.means, m.stds)[None].contiguous()


def export_onnx(m, x, path):
    net = copy.deepcopy(m.model.model).to(m.model.device).eval()
    net.export()
    kwargs = {"dynamo": False} if "dynamo" in torch.onnx.export.__code__.co_varnames else {}
    torch.onnx.export(net.cpu(), (x.cpu(),), path, input_names=["input"], output_names=["dets", "labels", "masks"],
                      dynamic_axes=None, opset_version=17, do_constant_folding=True, **kwargs)
    del net


def build_engine(onnx_path, engine_path, fp16):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(onnx_path):
        raise RuntimeError([parser.get_error(i).desc() for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    t = time.perf_counter()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    with open(engine_path, "wb") as f:
        f.write(blob)
    return time.perf_counter() - t


class Engine:
    """Minimal TensorRT runner on torch CUDA tensors (static shapes, batch 1): input 'input', outputs by name."""

    def __init__(self, path):
        self.runtime = trt.Runtime(LOGGER)
        self.engine = self.runtime.deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        self.out = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                shape = tuple(self.engine.get_tensor_shape(name))
                self.out[name] = torch.empty(shape, dtype=TORCH_DTYPE[self.engine.get_tensor_dtype(name)], device="cuda")
                self.ctx.set_tensor_address(name, self.out[name].data_ptr())
        self.times = []

    def __call__(self, x):
        x = x.to(dtype=torch.float32).contiguous()
        self.ctx.set_tensor_address("input", x.data_ptr())
        torch.cuda.synchronize()
        t = time.perf_counter()
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.synchronize()
        self.times.append(1000.0 * (time.perf_counter() - t))
        return self.out["dets"], self.out["labels"], self.out["masks"]


def score_map(m, outs, hw, threshold, class_id):
    """rfdetr post-processing + the node's rasterisation (per pixel max of mask x confidence)."""
    pred = {"pred_boxes": outs[0].float(), "pred_logits": outs[1].float(), "pred_masks": outs[2].float()}
    r = m.model.postprocess(pred, target_sizes=torch.tensor([hw], device="cuda"), score_threshold=threshold)[0]
    keep = (r["scores"] > threshold) & (r["labels"] == class_id)
    canvas = torch.zeros(hw, device="cuda")
    if int(keep.sum()):
        masks = r["masks"][keep].reshape(int(keep.sum()), *hw).bool()
        canvas = torch.where(masks, r["scores"][keep].float().view(-1, 1, 1), torch.zeros((), device="cuda")).amax(0)
    return canvas


def agreement(a, b):
    union = int((a | b).sum())
    return 1.0 if union == 0 else float((a & b).sum()) / union


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.work, exist_ok=True)
    env = {"host": platform.node(), "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
           "tensorrt": trt.__version__}
    print("ENV", json.dumps(env), flush=True)
    frames = torch.load(args.cache, weights_only=False)
    frames = frames[: args.limit] if args.limit else frames
    rgb1, cir1, ens = load("rgb_v2")["Seg"], load("cir_v2")["Seg"], load("ens_rgb_cir_mean_v2")
    ens_rgb, ens_cir = ens["SegRGB"], ens["SegCIR"]
    models = {"rgb": rgb1._build_model(), "cir": cir1._build_model()}
    img0 = {k: frames[0][k].cuda().float() / 255.0 for k in ("rgb", "cir")}
    engines, builds = {}, {}
    for k, m in models.items():
        onnx_path = os.path.join(args.work, f"{k}_v2_{int(m.model.resolution)}.onnx")
        if not os.path.exists(onnx_path):
            export_onnx(m, preprocess(m, img0[k]), onnx_path)
            print(f"exported {onnx_path}", flush=True)
        for prec in ("fp32", "fp16"):
            path = os.path.join(args.work, f"{k}_v2_{int(m.model.resolution)}_{prec}_trt{trt.__version__}.engine")
            if not os.path.exists(path):
                builds[f"{k}_{prec}"] = round(build_engine(onnx_path, path, prec == "fp16"), 1)
                print(f"built {path} in {builds[f'{k}_{prec}']} s", flush=True)
            engines[(k, prec)] = Engine(path)
    rows = []
    for i, fr in enumerate(frames):
        img = {k: fr[k].cuda().float() / 255.0 for k in ("rgb", "cir")}
        hw = (int(img["rgb"].shape[1]), int(img["rgb"].shape[2]))
        masks = {}
        with torch.no_grad():
            s = {"rgb1": rgb1.forward(rgb_image=img["rgb"])["scores"][0, :, :, 0],
                 "cir1": cir1.forward(rgb_image=img["cir"])["scores"][0, :, :, 0],
                 "ensr": ens_rgb.forward(rgb_image=img["rgb"])["scores"][0, :, :, 0],
                 "ensc": ens_cir.forward(rgb_image=img["cir"])["scores"][0, :, :, 0]}
            masks["pytorch_fp32"] = {"rgb_v2": s["rgb1"] >= 0.5, "cir_v2": s["cir1"] >= 0.5,
                                     "ens_rgb_cir_mean_v2": 0.5 * (s["ensr"] + s["ensc"]) >= 0.5}
            for prec in ("fp32", "fp16"):
                out = {}
                for k, m in models.items():
                    raw = engines[(k, prec)](preprocess(m, img[k]))
                    node1, noden = (rgb1, ens_rgb) if k == "rgb" else (cir1, ens_cir)
                    out[k + "1"] = score_map(m, raw, hw, node1.threshold, node1.class_filter or 0)
                    out[k + "n"] = score_map(m, raw, hw, noden.threshold, noden.class_filter or 0)
                masks[f"trt_{prec}"] = {"rgb_v2": out["rgb1"] >= 0.5, "cir_v2": out["cir1"] >= 0.5,
                                        "ens_rgb_cir_mean_v2": 0.5 * (out["rgbn"] + out["cirn"]) >= 0.5}
        gs, gf, unl = fr["gs"], fr["gf"], fr["unl"]
        for opt, per in masks.items():
            for pipe_name, mt in per.items():
                mk = mt.cpu().numpy()
                ref = masks["pytorch_fp32"][pipe_name].cpu().numpy()
                rows.append(dict(option=opt, pipeline=pipe_name, set=fr["set"], scene=fr["scene"], name=fr["name"],
                                 tp=int((mk & gs).sum()), fp=int((mk & ~gs).sum()), fn=int((gs & ~mk).sum()),
                                 gs=int(gs.sum()), objfp=int((mk & unl).sum()), f2n=int((mk & gf).sum()),
                                 gf=int(gf.sum()), agree=agreement(mk, ref)))
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(frames)} frames", flush=True)
    timing = {f"{k}_{p}": round(statistics.median(e.times), 2) for (k, p), e in engines.items()}
    json.dump(rows, open(os.path.join(args.work, f"validate_trt_{platform.node()}.json"), "w"))
    json.dump({"env": env, "engine_build_s": builds, "engine_call_median_ms": timing},
              open(os.path.join(args.work, f"validate_trt_meta_{platform.node()}.json"), "w"), indent=1)
    print("ENGINE CALL median ms", json.dumps(timing), "| builds", json.dumps(builds), flush=True)
    print("VALIDATE TRT DONE", flush=True)


if __name__ == "__main__":
    main()
