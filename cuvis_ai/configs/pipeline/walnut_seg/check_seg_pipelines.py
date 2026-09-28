"""Forward every walnut_seg pipeline on the reference frame and check it against a reference-mask file.

Input: `ref/real_world_live_000_f0000_reflectance.npz` (frame 0 of the 1-Sep live cu3s exactly as the cu3s reader
delivers it: raw-scale reflectance, uint16-lossless). For each `walnut_seg_*_cuvisnext_cube.yaml` (+ its .pt) it
prints the ShellMask px, checks ShellHeatmap.scores == the model scores and ShellMask == (scores >= 0.5), and,
given `--ref <npz>`, the mask-agreement IoU with the reference machine's masks (>= 0.998 across machines; a `_fast` / `_fp16`
pipeline without its own reference is compared with its fp32 counterpart's mask at IoU >= 0.99: fp16 rounding flips
boundary cells). An `_exact` pipeline must also be bit-identical (heatmap and mask) to its default pipeline on THIS
machine. A `_trt_fp32` / `_trt_fp16` pipeline (RFDETRSegmenter backend tensorrt) is compared like `_fast` (IoU >= 0.99
vs fp32) and is SKIPPED in envs without the tensorrt package; it needs this machine's engines (build_fast_pipelines.py
or cuvis_ai_rfdetr.trt_engine build-pipeline). `--write <npz>` stores this
machine's masks (packed) as a new reference. Same-machine, same-env reruns must be bit-identical (IoU 1.0);
across machines / torch builds expect IoU >= 0.998 (last-bit float noise flips a few mask cells; the mean ensembles
land at ~0.9989 laptop vs Thor).

  python check_seg_pipelines.py --write ref/seg_ref_masks_laptop.npz          # on the laptop
  python check_seg_pipelines.py --ref ref/seg_ref_masks_laptop.npz --device cuda   # anywhere else
"""

import argparse
import glob
import importlib.util
import os
import platform

import numpy as np
import torch
import yaml
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=None)
    ap.add_argument("--write", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    d = np.load(os.path.join(SEG, "ref", "real_world_live_000_f0000_reflectance.npz"))
    cube = torch.from_numpy(d["cube"].astype(np.float32))[None]
    wl = torch.from_numpy(d["wavelengths"].astype(np.int32))[None]
    ref = np.load(args.ref) if args.ref else None
    out, allok, seen = {}, True, {}
    have_trt = importlib.util.find_spec("tensorrt") is not None
    print(f"host={platform.node()} torch={torch.__version__} device={args.device or 'default'} tensorrt={have_trt}",
          flush=True)
    for yml in sorted(glob.glob(os.path.join(SEG, "walnut_seg_*_cuvisnext_cube.yaml"))):
        name = os.path.basename(yml)[:-5]
        if "_trt_" in name and not have_trt:
            print(f"SKIP {name:48} (backend tensorrt: no tensorrt package in this env)", flush=True)
            continue
        cfg = yaml.safe_load(open(yml))
        term = next(c["source"].split(".")[0] for c in cfg["connections"] if c["target"] == "ShellMask.inputs.logits")
        p = CuvisPipeline.load_pipeline(yml, weights_path=yml[:-5] + ".pt", device=args.device)
        x = cube.to(args.device) if args.device else cube
        with torch.no_grad():
            o = p.forward(batch={"cube": x, "wavelengths": wl})
        s = o[(term, "scores")].detach().cpu()
        h = o[("ShellHeatmap", "scores")].detach().cpu()
        m = o[("ShellMask", "decisions")].detach().cpu()[0, :, :, 0].numpy()
        ok = torch.equal(h, s) and np.array_equal(m, (s[0, :, :, 0] >= 0.5).numpy())
        msg = f"{name:48} px={int(m.sum()):6d} heatmap==scores/mask==thr {ok}"
        key, need = f"m_{name}", 0.998
        if "_exact_cuvisnext" in name:
            base = name.replace("_exact_cuvisnext", "_cuvisnext")
            if key not in (ref.files if ref is not None else []):
                key = f"m_{base}"
            if base in seen:
                bit = torch.equal(h, seen[base][0]) and np.array_equal(m, seen[base][1])
                msg += f" bit-identical to {base[len('walnut_seg_'):-len('_cuvisnext_cube')]}: {bit}"
                ok &= bit
        for fp16_suffix in ("_trt_fp32_cuvisnext", "_trt_fp16_cuvisnext", "_fast_cuvisnext", "_fp16_cuvisnext"):
            if ref is not None and key not in ref.files and fp16_suffix in name:
                key, need = f"m_{name.replace(fp16_suffix, '_cuvisnext')}", 0.99
                break
        if ref is not None and key in ref.files:
            r = np.unpackbits(ref[key])[: m.size].reshape(m.shape).astype(bool)
            iou = float((m & r).sum() / max(1, (m | r).sum()))
            msg += f" ref px={int(r.sum()):6d} agreement IoU={iou:.5f}" + (" (vs fp32)" if need < 0.998 else "")
            ok &= iou >= need
        allok &= ok
        out[f"m_{name}"] = np.packbits(m)
        seen[name] = (h, m)
        print(("OK   " if ok else "FAIL ") + msg, flush=True)
        del p
    if args.write:
        np.savez_compressed(args.write, **out)
        print("wrote", args.write)
    print("CHECK SEG PIPELINES", "ALL OK" if allok else "WITH FAILURES")


if __name__ == "__main__":
    main()
