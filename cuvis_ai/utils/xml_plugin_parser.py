"""Shared XML parsing helpers for cuvis user-plugin files."""

from __future__ import annotations

from pathlib import Path

from defusedxml import ElementTree as ET


def xml_local_name(tag: str) -> str:
    """Return local XML tag name, stripping any namespace prefix."""
    return tag.split("}", 1)[1] if "}" in tag else tag


def read_xml_inputs(xml_path: Path) -> dict[str, str]:
    """Parse all ``<input>`` elements from a cuvis user-plugin XML into a dict.

    Each ``<input id="...">text</input>`` becomes ``{id: text}``.
    """
    root = ET.parse(xml_path).getroot()
    values: dict[str, str] = {}
    for node in root.iter():
        if xml_local_name(node.tag) != "input":
            continue
        key = (node.attrib.get("id") or "").strip()
        if not key:
            continue
        values[key] = (node.text or "").strip()
    return values


def parse_numeric_text(text: str | None, *, label: str) -> float:
    """Parse a numeric XML text payload with descriptive errors."""
    payload = (text or "").strip()
    if not payload:
        raise ValueError(f"{label} is empty")
    try:
        return float(payload)
    except ValueError as exc:
        raise ValueError(f"{label} must be numeric, got '{payload}'") from exc
