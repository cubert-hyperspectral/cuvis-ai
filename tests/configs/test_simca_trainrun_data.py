"""The packaged SIMCA trainrun fits on frames without foreign objects.

A class model learns what normal looks like from its train split, so a frame with a stone in
it widens the limits without any warning. The shared ``lentils`` data config trains on such
frames; the SIMCA trainrun therefore has its own.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import cuvis_ai

pytestmark = pytest.mark.unit

CONFIGS = Path(cuvis_ai.__file__).resolve().parent / "configs"

# Measurements of the Lentils_000 demo recording whose annotation file marks a stone or brown
# lentils. Frames 0, 4 and 6 show black lentils and background only; 9, 10 and 12 carry no
# annotation and are left out of every split.
FOREIGN_OBJECT_FRAMES = {1, 2, 3, 5, 7, 8, 11, 13}


def _ids(data: dict, split: str) -> set[int]:
    return {index for selector in data["splits"][split] for index in selector["ids"]}


def test_simca_trainrun_uses_the_normal_only_data_config() -> None:
    path = CONFIGS / "trainrun" / "simca_statistical.yaml"
    trainrun = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert {"/data@data": "lentils_normals"} in trainrun["defaults"]


def test_normal_only_data_config_trains_on_frames_without_foreign_objects() -> None:
    path = CONFIGS / "data" / "lentils_normals.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    train = _ids(data, "train")
    assert train
    assert not train & FOREIGN_OBJECT_FRAMES
    assert not train & (_ids(data, "val") | _ids(data, "test"))
    # The held-out frames contain what the model has to find.
    assert _ids(data, "val") <= FOREIGN_OBJECT_FRAMES
    assert _ids(data, "test") <= FOREIGN_OBJECT_FRAMES
