"""Validate Hydra trainrun configs for structure and composition."""

from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "cuvis_ai" / "configs"
TRAINRUN_DIR = CONFIG_ROOT / "trainrun"


def _first_non_empty_line(text: str) -> str:
    """Return the first non-empty line from text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _has_self(defaults: list) -> bool:
    """Whether the defaults list carries ``_self_`` (list-item or single-key-dict form)."""
    return any(
        item == "_self_" or (isinstance(item, dict) and "_self_" in item) for item in defaults
    )


def _data_is_resolvable(config_dict: dict, defaults: list | None) -> bool:
    """A trainrun resolves ``data`` via a ``/data@`` default group or an inline ``data:`` key."""
    if "data" in config_dict:
        return True
    if isinstance(defaults, list):
        return any("/data@" in str(item) for item in defaults)
    return False


def validate_trainrun_config(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a single trainrun config file.

    Trainruns come in two shapes: Hydra-composition (a ``defaults:`` block pulling
    in a ``/data@`` group and optionally ``/training@``, with ``_self_`` and a
    ``# @package _global_`` header) and fully-inlined (``data:`` / ``training:``
    written directly, no defaults block). The header and ``_self_`` are required
    only for the composition shape; the one invariant across both is that ``data``
    is resolvable. The Hydra compose below is the authoritative check for either.
    """
    errors: list[str] = []

    if not config_path.exists():
        return False, [f"File not found: {config_path}"]

    raw = config_path.read_text(encoding="utf-8")

    try:
        config_dict = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return False, [f"Invalid YAML: {exc}"]

    defaults = config_dict.get("defaults")
    if defaults is not None and not isinstance(defaults, list):
        errors.append("defaults must be a list")
        defaults = None

    # Composition-style trainruns (with a defaults block) need the _global_ package header
    # and _self_ for override precedence; fully-inlined trainruns legitimately omit both.
    if defaults is not None:
        if _first_non_empty_line(raw) != "# @package _global_":
            errors.append("Missing '# @package _global_' directive at top of file")
        if not _has_self(defaults):
            errors.append("defaults must include '_self_' for override precedence")

    if not _data_is_resolvable(config_dict, defaults):
        errors.append("no data source: expected a '/data@' default or an inline 'data:' key")

    # Hydra composition check — the authoritative validation for either shape.
    try:
        relative_name = config_path.relative_to(CONFIG_ROOT).with_suffix("").as_posix()
        with initialize_config_dir(config_dir=str(CONFIG_ROOT), version_base="1.3"):
            compose(config_name=relative_name)
    except Exception as exc:  # pragma: no cover - runtime validation
        errors.append(f"Hydra compose failed: {exc}")

    return len(errors) == 0, errors


def main() -> None:
    any_errors = False
    for yaml_file in sorted(TRAINRUN_DIR.glob("*.yaml")):
        valid, errors = validate_trainrun_config(yaml_file)
        status = "OK" if valid else "FAIL"
        print(f"[{status}] {yaml_file.relative_to(CONFIG_ROOT)}")
        if errors:
            any_errors = True
            for err in errors:
                print(f"  - {err}")

    if any_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
