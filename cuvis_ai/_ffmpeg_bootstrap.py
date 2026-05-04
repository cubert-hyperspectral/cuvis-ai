"""Register FFmpeg shared-library directories on Windows.

Python 3.8+ on Windows no longer searches ``PATH`` to resolve DLL
dependencies of native extensions; modules that need a sibling DLL must
register its directory via ``os.add_dll_directory``. ``torchcodec``'s
``libtorchcodec_core*.dll`` links against FFmpeg's ``avcodec-*.dll``
family, so without this registration ``import torchcodec`` raises
``RuntimeError: Could not load libtorchcodec`` even when the FFmpeg bin
directory is on ``PATH``.

This module walks ``PATH`` and registers any directory containing an
``avcodec-*.dll``. No-op on non-Windows platforms.
"""

from __future__ import annotations

import os
import sys
from glob import glob


def _register_ffmpeg_dll_dirs() -> None:
    if sys.platform != "win32":
        return
    seen: set[str] = set()
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or entry in seen:
            continue
        seen.add(entry)
        try:
            if os.path.isdir(entry) and glob(os.path.join(entry, "avcodec-*.dll")):
                os.add_dll_directory(entry)
        except OSError:
            continue


_register_ffmpeg_dll_dirs()
