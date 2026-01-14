"""Smoke test ensuring all cuvis_ai modules import with absolute paths."""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Iterable

import pytest

import cuvis_ai


def iter_module_names() -> Iterable[str]:
    """Yield fully-qualified module names under cuvis_ai."""
    for module in pkgutil.walk_packages(cuvis_ai.__path__, prefix="cuvis_ai."):
        yield module.name


@pytest.mark.parametrize("module_name", sorted(set(iter_module_names())))
def test_module_is_importable(module_name: str) -> None:
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing_root = (exc.name or "").split(".")[0]
        if missing_root != "cuvis_ai":
            pytest.skip(
                f"Optional dependency '{missing_root}' required for "
                f"'{module_name}' is not installed."
            )
        raise
