"""Tests for the shared rotation table."""

from __future__ import annotations

import re

import pytest

from cuvis_ai.utils.rotation import ROTATION_QUARTER_TURNS, normalize_rotation, quarter_turns


@pytest.mark.parametrize(
    ("value", "canonical", "turns"),
    [
        (None, None, 0),
        (0, None, 0),
        (90, 90, 1),
        (-270, 90, 1),
        (-90, -90, -1),
        (270, -90, -1),
        (180, 180, 2),
        (-180, 180, 2),
    ],
)
def test_every_accepted_value_normalizes_and_turns(value, canonical, turns):
    assert normalize_rotation(value) == canonical
    assert quarter_turns(canonical) == turns
    assert ROTATION_QUARTER_TURNS[value] == turns


@pytest.mark.parametrize("value", [45, 360, -45, "90", 90.5])
def test_other_values_are_rejected_naming_the_parameter(value):
    expected = (
        f"frame_rotation must be one of None, 0, 90, -90, 180, -180, 270, -270, got {value!r}"
    )
    with pytest.raises(ValueError, match="^" + re.escape(expected) + "$"):
        normalize_rotation(value, name="frame_rotation")
    with pytest.raises(ValueError, match="^rotation must be one of "):
        normalize_rotation(value)


def test_table_covers_exactly_the_documented_spellings():
    assert set(ROTATION_QUARTER_TURNS) == {None, 0, 90, -90, 180, -180, 270, -270}
