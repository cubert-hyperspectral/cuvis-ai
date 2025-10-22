"""Utilities for exposing the installed package version."""

from __future__ import annotations

import os
from importlib import metadata

__all__ = ["__version__", "get_version"]

_PACKAGE_NAME = "cuvis-ai"
_FALLBACK_VERSION = os.environ.get("CUVIS_AI_VERSION", "0.1.3")


def get_version() -> str:
    """Return the installed package version, falling back to the source version."""
    try:
        return metadata.version(_PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return _FALLBACK_VERSION


__version__ = get_version()
