"""One table for the rotation values the spatial nodes accept.

``ToVideoNode``, ``ToImage`` and ``SpatialRotateNode`` all take a rotation in
degrees, accept the same eight spellings and apply it with :func:`torch.rot90`.
This module holds that knowledge once: :data:`ROTATION_QUARTER_TURNS` maps every
accepted value to its quarter-turn count, :func:`normalize_rotation` validates a
constructor argument and returns the canonical form (``None``, ``90``, ``-90`` or
``180``), and :func:`quarter_turns` gives ``torch.rot90`` its ``k``.
"""

from __future__ import annotations

# Accepted constructor value -> quarter turns for ``torch.rot90`` (anticlockwise).
ROTATION_QUARTER_TURNS: dict[int | None, int] = {
    None: 0,
    0: 0,
    90: 1,
    -270: 1,
    -90: -1,
    270: -1,
    180: 2,
    -180: 2,
}

_CANONICAL_BY_TURNS: dict[int, int | None] = {0: None, 1: 90, -1: -90, 2: 180}

_ACCEPTED_SPELLINGS = "None, 0, 90, -90, 180, -180, 270, -270"


def normalize_rotation(value: int | None, *, name: str = "rotation") -> int | None:
    """Return the canonical rotation for an accepted ``value``.

    ``None`` and ``0`` mean no rotation; ``270`` / ``-270`` and ``-180`` fold onto
    ``-90`` / ``90`` and ``180``. Any other value raises ``ValueError`` naming the
    parameter (``name``) and the accepted spellings.
    """
    if value not in ROTATION_QUARTER_TURNS:
        raise ValueError(f"{name} must be one of {_ACCEPTED_SPELLINGS}, got {value!r}")
    return _CANONICAL_BY_TURNS[ROTATION_QUARTER_TURNS[value]]


def quarter_turns(rotation: int | None) -> int:
    """Quarter turns for ``torch.rot90`` of a value :func:`normalize_rotation` returned."""
    return ROTATION_QUARTER_TURNS[rotation]
