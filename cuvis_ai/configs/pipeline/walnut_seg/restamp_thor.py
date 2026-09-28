"""Re-stamp the stack's absolute paths for a new machine (e.g. Thor).

The seg + FO manifests and pipeline yamls hold absolute `D:/walnuts/walnut_fo_stack/...` paths
(checkpoint_path, projection_path, plugin `path:`). After copying the stack to another root, run this
once to rewrite them. Covers BOTH seg and FO (all plugins/*.yaml + pipeline/**/*.yaml) in one pass.

    python restamp_thor.py --root /home/anish/walnut_fo_stack        # apply
    python restamp_thor.py --root /home/anish/walnut_fo_stack --dry  # preview only

Stdlib only. Rewrites the YAMLs (authoritative for hparams); the sibling .pt files carry buffers only,
not paths, so they need no edit. Idempotent — safe to re-run.
"""
import argparse
from pathlib import Path

OLD = ["D:/walnuts/walnut_fo_stack", "D:\\walnuts\\walnut_fo_stack"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="new stack root on this machine (forward slashes ok)")
    ap.add_argument("--dry", action="store_true", help="preview, do not write")
    args = ap.parse_args()
    new = args.root.replace("\\", "/").rstrip("/")

    configs = Path(__file__).resolve().parents[2]  # .../cuvis_ai/configs
    targets = sorted(configs.glob("plugins/*.yaml")) + sorted(configs.glob("pipeline/**/*.yaml"))
    if not targets:
        print(f"no yamls found under {configs}"); return

    total = 0
    for f in targets:
        text = f.read_text(encoding="utf-8")
        hits = sum(text.count(o) for o in OLD)
        if not hits:
            continue
        new_text = text
        for o in OLD:
            new_text = new_text.replace(o, new)
        total += hits
        rel = f.relative_to(configs)
        print(f"{'[dry] ' if args.dry else ''}{rel}: {hits} path(s) -> {new}")
        if not args.dry:
            f.write_text(new_text, encoding="utf-8")
    print(f"\n{'would rewrite' if args.dry else 'rewrote'} {total} path(s) across {len(targets)} yaml(s). new root = {new}")

if __name__ == "__main__":
    main()
