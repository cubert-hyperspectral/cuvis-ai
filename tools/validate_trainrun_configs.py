"""Validate Hydra trainrun configs for structure and composition."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir

CONFIG_ROOT = Path(__file__).resolve().parents[1] / "configs"
TRAINRUN_DIR = CONFIG_ROOT / "trainrun"
REQUIRED_DIRECTIVES = ("/pipeline@", "/data@", "/training@")


def _first_non_empty_line(text: str) -> str:
    """Return the first non-empty line from text."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _validate_defaults(defaults: Iterable, errors: list[str]) -> None:
    """Validate defaults section contents."""
    if not isinstance(defaults, list):
        errors.append("defaults must be a list")
        return

    has_self = any(
        item == "_self_" or getattr(item, "get", lambda _=None: None)("_self_") for item in defaults
    )
    if not has_self:
        errors.append("defaults must include '_self_' for override precedence")

    # Check for composition directives
    joined = "\n".join(str(item) for item in defaults)
    for directive in REQUIRED_DIRECTIVES:
        if directive not in joined:
            errors.append(f"defaults missing composition entry containing '{directive}'")


def validate_trainrun_config(config_path: Path) -> tuple[bool, list[str]]:
    """Validate a single trainrun config file."""
    errors: list[str] = []

    if not config_path.exists():
        return False, [f"File not found: {config_path}"]

    raw = config_path.read_text(encoding="utf-8")
    if _first_non_empty_line(raw) != "# @package _global_":
        errors.append("Missing '# @package _global_' directive at top of file")

    try:
        config_dict = yaml.safe_load(raw) or {}
    except yaml.YAMLError as exc:
        return False, [f"Invalid YAML: {exc}"]

    defaults = config_dict.get("defaults")
    if defaults is None:
        errors.append("Missing 'defaults' section")
    else:
        _validate_defaults(defaults, errors)

    # Hydra composition check
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
