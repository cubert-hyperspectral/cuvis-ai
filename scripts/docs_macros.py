"""mkdocs-macros hook: expose project metadata to docs."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement

_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
_LOWER_BOUND_OPS = (">=", "==", "~=")


def _read_cuvis_floor() -> str:
    """Return the lower-bound version pinned for the `cuvis` dependency."""
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    deps = data.get("project", {}).get("dependencies", [])
    for dep in deps:
        req = Requirement(dep)
        if req.name == "cuvis":
            for spec in req.specifier:
                if spec.operator in _LOWER_BOUND_OPS:
                    return spec.version
            raise RuntimeError(f"`cuvis` dependency in {_PYPROJECT} has no lower-bound specifier")
    raise RuntimeError(f"`cuvis` not found in [project].dependencies of {_PYPROJECT}")


def define_env(env) -> None:
    env.variables["cuvis_sdk_version"] = _read_cuvis_floor()
