"""Accuracy check of the fast-inference options on ALL 163 test frames (1-Sep 106, 15-Sep 14, 22-Sep d 43).

The speed benchmark (speed_options.py, 37 frames) is too small to judge the single models: a few knife-edge frames
(whole-hand false positives, instances at the confidence cut) flip with any numeric change. Here every option is
scored on the full test sets with the eval-harness metrics (shell IoU, FP px, FP on unlabeled objects/hands,
fake->shell) per set and scene, plus mask agreement vs fp32.

Frames are read once through the cu3s DataModule; the two 3-band model inputs (RGB 640/550/470 and CIR 850/660/550
from the pipelines' own FixedWavelengthSelector nodes) are cached as uint8 — exact, because RFDETRSegmenter quantizes
its input to uint8 anyway. The options are applied with speed_options.py's helpers (same code paths as on Thor).

Usage (stack env + cu3s reader overlay):
  python validate_speed_options.py --gt-root <dir with sep01_gt/ sep15_gt/ sep22_gt/> --out <dir> --options o0,o10,...
"""

import argparse
import glob
import json
import os
import platform

import numpy as np
import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import Predictor
from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule

import speed_options as so

SEG = os.path.dirname(os.path.abspath(__file__))
SETS = {"sep01_gt": "D:/Cubert/2026_09_01", "sep15_gt": "D:/Cubert/2026_09_15", "sep22_gt": "D:/Cubert/2026_09_22"}
INPUT_OF = {"rgb_v2:Seg": "rgb", "cir_v2:Seg": "cir",
            "ens_rgb_cir_mean_v2:SegRGB": "rgb", "ens_rgb_cir_mean_v2:SegCIR": "cir"}


def scene(n):
    s = n.rsplit("_f", 1)[0]
    if s.startswith("real_world_live"): return "live_hand_objects"
    if s.startswith("fake_upside_down"): return "fake_upsidedown"
    if s.startswith("fake_shells_real"): return "fake_real_kernels"
    if s.startswith("fake_shells"): return "fake_only"
    if s.startswith("real_fake_and_kernel_overlap"): return "fake_on_real_15sep"
    if s.startswith("kernel_shell_hand_other"): return "hand_other_22sep"
    if s.startswith("kernel_shell_fakes"): return "fakes_22sep"
    if s.startswith("kernel_shell_fo"): return "fo_22sep"
    if s.startswith("kernel_shell") or s.startswith("shell_only"): return "clean_22sep"
    return "normal_baseline"


def load(pipe_name):
    y = os.path.join(SEG, f"walnut_seg_{pipe_name}_cuvisnext_cube.yaml")
    return CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda")


def cache_frames(gt_root, limit_files=0):
    sel_rgb = next(n for n in load("rgb_v2").nodes if n.name == "Selector")
    sel_cir = next(n for n in load("cir_v2").nodes if n.name == "Selector")
    frames = []
    for gset, root in SETS.items():
        names = sorted(os.path.basename(p)[:-4] for p in glob.glob(f"{gt_root}/{gset}/*.npz"))
        by = {}
        for n in names:
            hit = glob.glob(f"{root}/*/{n.rsplit('_f', 1)[0]}.cu3s")
            if hit:
                by.setdefault(hit[0], {})[int(n.rsplit("_f", 1)[1])] = n
        files = sorted(by.items())
        for c, idxs in (files[:limit_files] if limit_files else files):
            dm = Cu3sDataModule(cu3s_file_path=c, processing_mode="Reflectance", batch_size=1, num_workers=0)
            dm.setup(stage="predict")
            for batch in Predictor._iter_batches(dm.predict_dataloader()):
                i = int(batch["mesu_index"][0])
                if i not in idxs:
                    continue
                g = np.load(f"{gt_root}/{gset}/{idxs[i]}.npz")
                cube = batch["cube"].float().cuda()
                wl = batch["wavelengths"][0].cpu().numpy()
                with torch.no_grad():
                    imgs = {k: s.forward(cube=cube, wavelengths=wl)["rgb_image"] for k, s in (("rgb", sel_rgb), ("cir", sel_cir))}
                u8 = {k: (v.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu() for k, v in imgs.items()}
                frames.append(dict(set=gset, name=idxs[i], scene=scene(idxs[i]), rgb=u8["rgb"], cir=u8["cir"],
                                   gs=g["gs"], gf=g["gf"], unl=g["obj"] & ~g["gs"] & ~g["gf"]))
            print(f"  cached {gset} {os.path.basename(c)}: total {len(frames)} frames", flush=True)
    return frames


def agreement(m, b):
    """Mask IoU between two predictions; two empty masks agree perfectly."""
    union = int((m | b).sum())
    return 1.0 if union == 0 else float((m & b).sum()) / union


def masks_for(opt, frames):
    """Per frame masks for rgb_v2, cir_v2 and the B+C mean ensemble under one option."""
    dtype, jit, gpu_in, res, parallel, fast, *rest = so.OPTIONS[opt]
    so.AUTOCAST["dtype"] = rest[1] if len(rest) > 1 else None
    so.set_patches(gpu_in, fast)
    try:
        segs = {}
        for pipe_name, names in (("rgb_v2", ["Seg"]), ("cir_v2", ["Seg"]), ("ens_rgb_cir_mean_v2", ["SegRGB", "SegCIR"])):
            p = so.build(pipe_name, res)
            so.apply_dtype(p, dtype, jit, rest[0] if rest else None)
            nodes = {n.name: n for n in p.nodes}
            for nm in names:
                segs[f"{pipe_name}:{nm}"] = nodes[nm]
        out = {"rgb_v2": [], "cir_v2": [], "ens_rgb_cir_mean_v2": []}
        for fr in frames:
            img = {k: fr[k].cuda().float() / 255.0 for k in ("rgb", "cir")}  # cached as [1, H, W, 3] uint8
            with torch.no_grad():
                s = {k: n.forward(rgb_image=img[INPUT_OF[k]])["scores"][0, :, :, 0] for k, n in segs.items()}
            out["rgb_v2"].append((s["rgb_v2:Seg"] >= 0.5).cpu().numpy())
            out["cir_v2"].append((s["cir_v2:Seg"] >= 0.5).cpu().numpy())
            fused = 0.5 * (s["ens_rgb_cir_mean_v2:SegRGB"] + s["ens_rgb_cir_mean_v2:SegCIR"])
            out["ens_rgb_cir_mean_v2"].append((fused >= 0.5).cpu().numpy())
        return out
    finally:
        so.set_patches(False)
        so.AUTOCAST["dtype"] = None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-files", type=int, default=0, help="smoke test: read only the first N cu3s per set")
    ap.add_argument("--options", default="o0_baseline_fp32,o9_fp32_gpuin_fastpaste,o10_fp16_gpuin_fastpaste,o11_fp16_jit_gpuin_fastpaste")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache = os.path.join(args.out, "frame_cache.pt")
    if os.path.exists(cache) and not args.limit_files:  # reading ~30 cu3s takes ~20 min; reuse the cache across runs
        frames = torch.load(cache, weights_only=False)
    else:
        frames = cache_frames(args.gt_root, args.limit_files)
        if not args.limit_files:
            torch.save(frames, cache)
    print(f"cached {len(frames)} frames", flush=True)
    rows, base = [], {}
    for opt in args.options.split(","):
        masks = masks_for(opt, frames)
        if opt == "o0_baseline_fp32":
            base = masks
        for pipe_name, ms in masks.items():
            for fr, m, b in zip(frames, ms, base.get(pipe_name, [None] * len(frames))):
                gs, gf, unl = fr["gs"], fr["gf"], fr["unl"]
                tp, fp, fn = int((m & gs).sum()), int((m & ~gs).sum()), int((gs & ~m).sum())
                rows.append(dict(option=opt, pipeline=pipe_name, set=fr["set"], scene=fr["scene"], name=fr["name"],
                                 tp=tp, fp=fp, fn=fn, gs=int(gs.sum()), objfp=int((m & unl).sum()),
                                 f2n=int((m & gf).sum()), gf=int(gf.sum()),
                                 agree=None if b is None else agreement(m, b)))
        print(f"done {opt}", flush=True)
    with open(os.path.join(args.out, f"validate_speed_{platform.node()}.json"), "w") as f:
        json.dump(rows, f)
    print("VALIDATE DONE", flush=True)


if __name__ == "__main__":
    main()
