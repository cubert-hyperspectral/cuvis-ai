"""Is TensorRT in the cuvis.next child env the walnut seg pipelines use?

This covers the time until TensorRT can be requested by the plugin (cuvis-ai-rfdetr#19, cuvis-ai-core#89). cuvis.next
composes one child env per plugin set under ~/.cuvis_runs and reuses it as is, so a package installed there by hand
stays. It is lost when that env is composed anew:
- after a cuvis.next / cuvis-ai-core update;
- after a change to a plugin's pyproject.toml;
- when the env is evicted (more than 10 cached envs, least recently used first, envs used in the last hour are safe).

This script finds the seg envs (runtime pyproject with cuvis-ai-rfdetr and no FO plugin) and, in the most recently used
one, checks `import tensorrt` (pinned version) and this machine's four engines (rgb_v2 / cir_v2 x fp32 / fp16, next to
the checkpoints). `--fix` installs the pinned TensorRT into that env: purely additive, 3 packages (tensorrt-cu13 on
Linux aarch64 = Thor, tensorrt-cu12 on Windows = laptop), ~2.1 GB download the first time, cached after.

  python check_trt_env.py          # report; exit 1 if TensorRT or an engine is missing
  python check_trt_env.py --fix    # install TensorRT into the current seg env if it is missing, then re-check
  python check_trt_env.py --include-fo --fix   # env of a combined FO + SEG pipeline (has FO plugins too)
  python check_trt_env.py --env <hash> --fix   # one explicit env under ~/.cuvis_runs

Run it after cuvis.next has loaded any walnut_seg pipeline once (that composes / touches the seg env), then load the
`_trt` pipelines. Stdlib only; any Python >= 3.11 works.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

TRT_VERSION = "10.15.1.29"
FO_PLUGINS = {"cuvis-ai-patchcore", "cuvis-ai-steervit", "cuvis-ai-ssft", "cuvis-ai-efficientad"}
CHECKPOINTS = ("rgb_v2_ema.pth", "cir_v2_ema.pth")
SEG = Path(__file__).resolve().parent


def trt_package() -> str:
    if sys.platform.startswith("linux") and platform.machine() == "aarch64":
        return "tensorrt-cu13"  # Thor: torch +cu130
    return "tensorrt-cu12"  # laptop: torch +cu128


def env_python(env: Path) -> Path:
    return env / ".venv" / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def seg_envs(root: Path, include_fo: bool = False) -> list[tuple[float, Path]]:
    """Composed envs whose runtime pyproject lists cuvis-ai-rfdetr (and, unless ``include_fo``, no FO plugin).

    Most recently used first.
    """
    found = []
    for env in root.iterdir() if root.is_dir() else []:
        ready, pyproject = env / ".ready", env / "pyproject.toml"
        if not (ready.is_file() and pyproject.is_file()):
            continue
        deps = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]["dependencies"]
        names = {re.split(r"[\s\[=<>!~;@]", d, maxsplit=1)[0].lower() for d in deps}
        if "cuvis-ai-rfdetr" in names and (include_fo or not names & FO_PLUGINS):
            found.append((ready.stat().st_mtime, env))
    return sorted(found, reverse=True)


def run_in(env: Path, code: str) -> tuple[bool, str]:
    r = subprocess.run([str(env_python(env)), "-c", code], capture_output=True, text=True, timeout=300)
    out = (r.stdout.strip().splitlines() or [""])[-1] if r.returncode == 0 else (r.stderr.strip().splitlines() or ["?"])[-1]
    return r.returncode == 0, out


def install(env: Path) -> bool:
    uv = shutil.which("uv") or ("/snap/bin/uv" if Path("/snap/bin/uv").exists() else None)
    if uv is None:
        print("  uv not found on PATH")
        return False
    cmd = [uv, "pip", "install", "--no-config", "--python", str(env_python(env)), f"{trt_package()}=={TRT_VERSION}"]
    print("  $", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=Path.home()).returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fix", action="store_true", help="install the pinned TensorRT into the current seg env")
    ap.add_argument("--root", default=os.environ.get("CUVIS_RUN_CACHE_ROOT") or str(Path.home() / ".cuvis_runs"))
    ap.add_argument("--include-fo", action="store_true",
                    help="also consider envs with FO plugins (the env of a combined FO + SEG pipeline)")
    ap.add_argument("--env", default=None, help="check this env (directory name under --root) instead")
    args = ap.parse_args()
    root = Path(args.root)
    envs = seg_envs(root, args.include_fo)
    if args.env:
        envs = [(mtime, env) for mtime, env in envs if env.name == args.env]
        if not envs:
            print(f"MISSING: {args.env} is not a ready env with cuvis-ai-rfdetr under {root}"
                  + ("" if args.include_fo else " (add --include-fo for an env with FO plugins)"))
            return 1
    kind = "rfdetr (incl. FO)" if args.include_fo else "walnut seg"
    print(f"{len(envs)} {kind} env(s) under {root} (most recently used first):")
    for mtime, env in envs:
        print(f"  {env.name}  last used {time.strftime('%Y-%m-%d %H:%M', time.localtime(mtime))}")
    if not envs:
        print("MISSING: no seg env yet - load any walnut_seg pipeline in cuvis.next once, then run this again.")
        return 1
    env = envs[0][1]
    ok, version = run_in(env, "import tensorrt; print(tensorrt.__version__)")
    trt_ok = ok and version == TRT_VERSION
    print(f"current seg env {env.name}: tensorrt {version if ok else 'NOT INSTALLED'}"
          + ("" if trt_ok else f" (need {trt_package()}=={TRT_VERSION})"))
    if not trt_ok and args.fix:
        if install(env):
            ok, version = run_in(env, "import tensorrt; print(tensorrt.__version__)")
            trt_ok = ok and version == TRT_VERSION
            print(f"  after install: tensorrt {version if ok else 'still NOT importable'}")
    ok, tag = run_in(env, "from cuvis_ai_rfdetr.trt_engine import gpu_tag; print(gpu_tag())")
    missing = []
    if not ok:
        print(f"could not read this machine's GPU tag in {env.name}: {tag}")
        missing.append("gpu tag")
    else:
        for ckpt in CHECKPOINTS:
            for precision in ("fp32", "fp16"):
                engine = SEG / "weights" / f"{ckpt}.trt" / f"{precision}_r504_{tag}_trt{TRT_VERSION}.engine"
                if not engine.is_file():
                    missing.append(str(engine))
        print(f"engines for {tag} / TensorRT {TRT_VERSION}: " + ("all 4 present" if not missing else "MISSING"))
        for m in missing:
            print(f"  missing {m} -> python build_fast_pipelines.py trt_fp32,trt_fp16 (in an env with tensorrt)")
    good = trt_ok and not missing
    print("TRT ENV OK" if good else ("TRT ENV NOT READY" + ("" if args.fix else " - run with --fix")))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
