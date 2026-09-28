"""Parity check: each branch of a combined walnut pipeline against its standalone pipeline.

    dump    --yaml Y --ref-cubes R.npz --seg-ref S.npz --out O.npz
            run one pipeline (yaml + its sibling .pt, CUDA) over the frame sequence and save its outputs
    compare --combo C.npz --fo F.npz --seg G.npz [--fo-own F2.npz] --name N --md report.md
            compare the combined pipeline's FO and SEG outputs with the standalone dumps

Frame sequence (the same for every pipeline, from a fresh state): the SEG reference frame (1-Sep live f0,
`walnut_seg/ref/real_world_live_000_f0000_reflectance.npz`), then the FO reference cubes of `R.npz` (`<tag>_cube`,
`<tag>_wl`; e.g. clean, fo). The SEG selectors calibrate over the first frames of a sequence, so a fresh state and one
sequence make the runs comparable.

Saved per frame: every terminal output (get_output_specs) plus the gated maps and the fused SEG score (`gate.scores`,
`gate_effad.scores`, `fuse.scores`, `Seg*.scores`, `ShellFuse.scores` / `Fuse.scores`).

Expected:
- same tier, same process: bit-identical;
- the FO standalone in its own cuvis.next env (no rfdetr imported, `--fo-own`): the TF32 variants bit-identical,
  since their nodes set TF32 themselves; any difference there is the process-wide TF32 of `import rfdetr`.
- gate decisions identical at the validated thresholds; shell mask IoU >= 0.99 otherwise.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXTRA = ("gate.scores", "gate_effad.scores", "fuse.scores", "SegRGB.scores", "SegCIR.scores", "ShellFuse.scores",
         "Fuse.scores")
SEG_RENAME = {"ShellFuse": "Fuse"}  # combined -> standalone SEG node name


def frames(ref_cubes: str, seg_ref: str):
    s = np.load(seg_ref)
    out = [("seg_live_f0", s["cube"].astype(np.float32), s["wavelengths"].astype(np.int32))]
    r = np.load(ref_cubes)
    for tag in sorted({k[:-5] for k in r.files if k.endswith("_cube")}):
        out.append((tag, r[f"{tag}_cube"].astype(np.float32), r[f"{tag}_wl"].astype(np.int32)))
    return out


def cmd_dump(a) -> None:
    if a.import_rfdetr:
        # A standalone FO pipeline in the same global state as the combined process, where the SEG plugin's
        # `import rfdetr` sets float32 matmul precision "high".
        import rfdetr  # noqa: F401
    import torch
    from cuvis_ai_core.utils.restore import restore_pipeline
    from cuvis_ai_schemas.enums import ExecutionStage
    from cuvis_ai_schemas.execution import Context

    y = Path(a.yaml).resolve()
    pipe = restore_pipeline(str(y), weights_path=str(y.with_suffix(".pt")), device="cuda")
    wanted = set(pipe.get_output_specs()) | set(EXTRA)
    ctx = Context(stage=ExecutionStage.INFERENCE)
    saved = {}
    with torch.no_grad():
        for tag, cube, wl in frames(a.ref_cubes, a.seg_ref):
            o = pipe.forward(batch={"cube": torch.from_numpy(cube)[None].cuda(),
                                    "wavelengths": torch.from_numpy(wl)[None].cuda()}, context=ctx)
            for (node, port), v in o.items():
                key = f"{node}.{port}"
                if key in wanted and isinstance(v, torch.Tensor):
                    saved[f"{tag}|{key}"] = v.detach().cpu().numpy()
    np.savez_compressed(a.out, **saved)
    print(f"dumped {y.name}: {len(saved)} arrays, float32 matmul precision "
          f"{torch.get_float32_matmul_precision()!r} -> {a.out}", flush=True)


def _cmp(x: np.ndarray, y: np.ndarray) -> tuple[bool, float, str]:
    if x.shape != y.shape:
        return False, float("nan"), f"shape {x.shape} vs {y.shape}"
    if x.dtype == bool or y.dtype == bool:
        same = np.array_equal(x, y)
        inter, union = np.logical_and(x, y).sum(), np.logical_or(x, y).sum()
        return same, float("nan"), f"IoU {inter / union:.5f}" if union else "IoU 1 (empty)"
    same = np.array_equal(x, y, equal_nan=True)
    d = float(np.nanmax(np.abs(x.astype(np.float64) - y.astype(np.float64)))) if x.size else 0.0
    return same, d, ""


def cmd_compare(a) -> None:
    c, fo, seg = np.load(a.combo), np.load(a.fo), np.load(a.seg)
    own = np.load(a.fo_own) if a.fo_own else None
    rows, count, ok, own_ok, unmatched = [], {"FO": 0, "SEG": 0}, {"FO": True, "SEG": True}, True, []
    for key in sorted(c.files):
        tag, port = key.split("|")
        node, pname = port.split(".", 1)
        if node == "cu3s_data":
            continue
        if key in fo.files:
            branch, ref = "FO", fo[key]
        else:
            skey = f"{tag}|{SEG_RENAME.get(node, node)}.{pname}"
            if skey not in seg.files:
                unmatched.append(key)
                continue
            branch, ref = "SEG", seg[skey]
        same, d, note = _cmp(c[key], ref)
        count[branch] += 1
        ok[branch] &= same
        own_txt = ""
        if branch == "FO" and own is not None and key in own.files:
            s2, d2, n2 = _cmp(c[key], own[key])
            own_ok &= s2
            own_txt = "identical" if s2 else f"max |d| {d2:.3g} {n2}".strip()
        rows.append((branch, tag, port, "identical" if same else f"max |d| {d:.3g} {note}".strip(), own_txt,
                     _val(c[key])))
    # Coverage the other way round: every standalone array (the SEG data node aside) has a combined counterpart.
    back = {v: k for k, v in SEG_RENAME.items()}
    missing = [k for k in fo.files if k not in c.files]
    for k in seg.files:
        tag, port = k.split("|")
        node, pname = port.split(".", 1)
        if node != "DataSource" and f"{tag}|{back.get(node, node)}.{pname}" not in c.files:
            missing.append(k)
    verdict = (f"FO {count['FO']} arrays {'bit-identical' if ok['FO'] else 'DIFFER'}; SEG {count['SEG']} arrays "
               f"{'bit-identical' if ok['SEG'] else 'DIFFER'}"
               + (f"; FO vs its own env {'bit-identical' if own_ok else 'differs (see column)'}" if own is not None
                  else ""))
    lines = [f"# {a.name}: combined vs standalone", "", f"**{verdict}.**", ""]
    if missing or unmatched:
        lines += [f"- standalone arrays missing in the combined dump: {missing}",
                  f"- combined arrays without a standalone counterpart: {unmatched}", ""]
    lines += ["| branch | frame | port | vs standalone (same process) | vs FO own env | value |",
              "|---|---|---|---|---|---|"]
    lines += [f"| {b} | {t} | {p} | {r} | {o} | {v} |" for b, t, p, r, o, v in rows]
    Path(a.md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{a.name}: {verdict}" + (f"; missing {len(missing)}, unmatched {len(unmatched)}"
                                    if missing or unmatched else ""), flush=True)


def _val(x: np.ndarray) -> str:
    if x.dtype == bool:
        return f"{int(x.sum())} px"
    if x.size <= 4:
        return ", ".join(f"{float(v):.4f}" for v in x.ravel())
    return f"max {float(np.nanmax(x)):.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dump")
    d.add_argument("--yaml", required=True)
    d.add_argument("--ref-cubes", required=True)
    d.add_argument("--seg-ref", default=str(HERE.parent / "walnut_seg" / "ref" / "real_world_live_000_f0000_reflectance.npz"))
    d.add_argument("--out", required=True)
    d.add_argument("--import-rfdetr", action="store_true",
                   help="import rfdetr first (a standalone FO run in the combined process's global state)")
    c = sub.add_parser("compare")
    c.add_argument("--combo", required=True)
    c.add_argument("--fo", required=True)
    c.add_argument("--seg", required=True)
    c.add_argument("--fo-own", default=None)
    c.add_argument("--name", required=True)
    c.add_argument("--md", required=True)
    a = ap.parse_args()
    cmd_dump(a) if a.cmd == "dump" else cmd_compare(a)


if __name__ == "__main__":
    main()
