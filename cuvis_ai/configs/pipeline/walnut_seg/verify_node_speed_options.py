"""Bit-exactness check of RFDETRSegmenter's speed options on the real v2 models and all 163 test frames.

Compares, per frame and per segmenter node, the score map of
  A  fast_paste=False                  (the per-instance CPU paste, the node's behaviour before the options)
  B  fast_paste=True                   (the new default)
  C  fast_paste=True + gpu_input=True  (frame handed to rfdetr as a CUDA tensor)
with torch.equal, plus the B+C ensemble's fused mask (mean >= 0.5). The inputs are the 163 cached 3-band model
inputs of validate_speed_options.py (`frame_cache.pt`: uint8 RGB/CIR from the pipelines' own selector nodes; the
node re-quantizes its input to uint8, so feeding u8/255 is exact). The switches are flipped on the loaded nodes
(the same attributes the hparams set). precision / jit_trace are NOT bit-exact by design and are covered by
validate_speed_options.py (accuracy on GT). Without the cache (e.g. on Thor), `--cu3s` reads recordings and feeds
the selector outputs of the ensemble pipeline (SelRGB / SelCIR) straight into the segmenter nodes.

Usage (stack env, CUDA):
  python verify_node_speed_options.py --cache <frame_cache.pt> [--out out.json]
  python verify_node_speed_options.py --cu3s <a.cu3s> [--cu3s <b.cu3s>] [--out out.json]   # + cu3s reader overlay
"""

import argparse
import json
import os

import torch
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

SEG = os.path.dirname(os.path.abspath(__file__))
NODES = {"rgb_v2": {"Seg": "rgb"}, "cir_v2": {"Seg": "cir"},
         "ens_rgb_cir_mean_v2": {"SegRGB": "rgb", "SegCIR": "cir"}}
VARIANTS = {"A_per_instance_paste": (False, False), "B_fast_paste": (True, False),
            "C_fast_paste_gpu_input": (True, True)}


def cu3s_frames(paths, pipe):
    """The 3-band model inputs of every frame, from the ensemble pipeline's own selector nodes."""
    from cuvis_ai_core.training import Predictor
    from cuvis_ai_dataloader.data.datamodule_cu3s import Cu3sDataModule

    sel = {n.name: n for n in pipe.nodes}
    frames = []
    for c in paths:
        dm = Cu3sDataModule(cu3s_file_path=c, processing_mode="Reflectance", batch_size=1, num_workers=0)
        dm.setup(stage="predict")
        for batch in Predictor._iter_batches(dm.predict_dataloader()):
            cube, wl = batch["cube"].float().cuda(), batch["wavelengths"][0].cpu().numpy()
            with torch.no_grad():
                frames.append({k: sel[n].forward(cube=cube, wavelengths=wl)["rgb_image"]
                               for k, n in (("rgb", "SelRGB"), ("cir", "SelCIR"))})
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None)
    ap.add_argument("--cu3s", action="append", default=[])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_json = args.out
    segs, ens = {}, None
    for short, names in NODES.items():
        y = os.path.join(SEG, f"walnut_seg_{short}_cuvisnext_cube.yaml")
        p = CuvisPipeline.load_pipeline(y, weights_path=y[:-5] + ".pt", device="cuda")
        nodes = {n.name: n for n in p.nodes}
        for nm, band in names.items():
            segs[f"{short}:{nm}"] = (nodes[nm], band)
        ens = p if short.startswith("ens") else ens
    # cached uint8 inputs are moved to the GPU one frame at a time; the node re-quantizes to uint8, so u8/255 is exact
    frames = torch.load(args.cache, weights_only=False) if args.cache else cu3s_frames(args.cu3s, ens)
    stats = {f"{v}:{k}": {"equal": 0, "max_abs": 0.0} for v in VARIANTS if v[0] != "A" for k in segs}
    stats.update({f"{v}:ens_mask": {"equal": 0, "max_abs": 0.0} for v in VARIANTS if v[0] != "A"})
    for i, fr in enumerate(frames):
        img = {k: fr[k].cuda().float() / 255.0 if fr[k].dtype == torch.uint8 else fr[k] for k in ("rgb", "cir")}
        res = {}
        for v, (fast, gpu_in) in VARIANTS.items():
            res[v] = {}
            for key, (node, band) in segs.items():
                node.fast_paste, node.gpu_input = fast, gpu_in
                with torch.no_grad():
                    res[v][key] = node.forward(rgb_image=img[band])["scores"][0, :, :, 0].cpu()
            res[v]["ens_mask"] = (0.5 * (res[v]["ens_rgb_cir_mean_v2:SegRGB"]
                                         + res[v]["ens_rgb_cir_mean_v2:SegCIR"]) >= 0.5).float()
        for v in VARIANTS:
            if v[0] == "A":
                continue
            for key, t in res[v].items():
                s = stats[f"{v}:{key}"]
                s["equal"] += int(torch.equal(t, res["A_per_instance_paste"][key]))
                s["max_abs"] = max(s["max_abs"], float((t - res["A_per_instance_paste"][key]).abs().max()))
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(frames)} frames", flush=True)
    for node, _ in segs.values():  # restore the yaml defaults
        node.fast_paste, node.gpu_input = True, False
    print(f"{len(frames)} frames: bit-identical frames vs A (per-instance CPU paste)")
    for k, s in stats.items():
        print(f"  {k:48s} {s['equal']:4d}/{len(frames)}  max|diff| {s['max_abs']:.3g}")
    if out_json:
        json.dump({"frames": len(frames), "stats": stats}, open(out_json, "w"), indent=1)
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
