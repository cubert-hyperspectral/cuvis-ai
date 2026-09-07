"""The shipped data manifests point at files that exist in the public datasets they name.

Every ``configs/data/*.yaml`` selects frames of a dataset ``uv run dataset download <name>
--data-dir data`` lays out under ``data/<target_dir>/``. The selectors' sources (and the
``params`` paths) must start with that prefix, contain no ``..``, and name a file the dataset
holds at its pinned revision, recorded in ``fixtures/dataset_files.json``; declared ids must
be unique and the train / val / test splits disjoint. The shipped universe CSVs are checked the
same way against the dataset they are copied into.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import yaml

from cuvis_ai_core.data.public_datasets import PublicDatasets

pytestmark = pytest.mark.unit

_CONFIGS = Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs"
_DATA_MANIFESTS = sorted((_CONFIGS / "data").glob("*.yaml"))
_UNIVERSES = sorted((_CONFIGS / "data" / "universes").glob("*.csv"))
_FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "dataset_files.json").read_text(encoding="utf-8")
)
_BY_TARGET_DIR = {spec.target_dir: spec for spec in PublicDatasets.list_specs()}


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _dataset_relative(path: str) -> tuple[str, str]:
    """Split ``data/<target_dir>/<rest>`` into the registry name and the path inside the dataset."""
    parts = Path(path).as_posix().split("/")
    assert parts[0] == "data" and len(parts) >= 3, (
        f"{path!r} must start with data/<dataset folder>/"
    )
    assert ".." not in parts, f"{path!r} must not contain '..'"
    assert parts[1] in _BY_TARGET_DIR, f"{parts[1]!r} is not the folder of a public dataset"
    return _BY_TARGET_DIR[parts[1]].name, "/".join(parts[2:])


def _assert_in_dataset(path: str) -> None:
    name, rel = _dataset_relative(path)
    assert rel in _FIXTURE[name]["files"], f"{rel!r} is not a file of {name} at its pinned revision"


def _selectors(doc: dict):
    for split in ("train", "val", "test", "predict"):
        for selector in (doc.get("splits") or {}).get(split) or []:
            yield split, selector


def _frames(selector: dict) -> list[tuple[str, int]]:
    if selector["kind"] == "file_indices":
        return [(selector["source"], i) for i in selector["ids"]]
    if selector["kind"] == "files":
        return [(p, 0) for p in selector["paths"]]
    pytest.fail(f"unexpected selector kind {selector['kind']!r}")


@pytest.mark.parametrize("path", _DATA_MANIFESTS, ids=lambda p: p.name)
def test_sources_exist_in_the_pinned_dataset(path: Path) -> None:
    doc = _load(path)
    selectors = list(_selectors(doc))
    assert selectors, "a data manifest declares at least one split selector"
    for _, selector in selectors:
        if selector["kind"] == "file_indices":
            _assert_in_dataset(selector["source"])
            ids = selector["ids"]
            assert ids and ids == sorted(set(ids)) and ids[0] >= 0, (path.name, selector["source"])
        else:
            for file_path in selector["paths"]:
                _assert_in_dataset(file_path)
    params = doc.get("params") or {}
    for key in ("cu3s_file_path", "annotation_json_path"):
        if key in params:
            _assert_in_dataset(params[key])
    if "universe_csv" in params:
        # The universe is copied beside the data, so it is not a file of the HF repo itself.
        name, rel = _dataset_relative(params["universe_csv"])
        assert rel == "universe.csv", (path.name, params["universe_csv"])


@pytest.mark.parametrize("path", _DATA_MANIFESTS, ids=lambda p: p.name)
def test_train_val_test_are_disjoint(path: Path) -> None:
    owner: dict[tuple[str, int], str] = {}
    for split, selector in _selectors(_load(path)):
        if split == "predict":
            continue
        for frame in _frames(selector):
            assert owner.setdefault(frame, split) == split, f"{frame} in {owner[frame]} and {split}"


@pytest.mark.parametrize("csv_path", _UNIVERSES, ids=lambda p: p.name)
def test_universe_csv_rows_exist_in_the_dataset_they_are_copied_into(csv_path: Path) -> None:
    manifest = _CONFIGS / "data" / f"{csv_path.stem}.yaml"
    assert manifest.exists(), f"{csv_path.name} has no manifest of the same name"
    name, _ = _dataset_relative(_load(manifest)["params"]["universe_csv"])
    files = set(_FIXTURE[name]["files"])
    with csv_path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and {"source", "index"} <= set(rows[0])
    seen = set()
    for row in rows:
        assert row["source"] in files, row["source"]
        assert row["source"] not in seen, f"duplicate universe row {row['source']}"
        seen.add(row["source"])
        if row.get("annotation"):
            assert row["annotation"] in files, row["annotation"]
