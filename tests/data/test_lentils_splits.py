import csv

from cuvis_ai.data.lentils_splits import (
    FrameRecord,
    _extract_frame_number,
    assign_groups,
    compute_stratification_keys,
    stratified_group_split,
    write_splits_csv,
)


def _make_records(day: str, *, categories_per_frame: list[list[int]]) -> list[FrameRecord]:
    records: list[FrameRecord] = []
    for i, cat_ids in enumerate(categories_per_frame):
        records.append(
            FrameRecord(
                day=day,
                cu3s_path=f"/fake/{day}/{i:06d}.cu3s",
                annotation_json=f"/fake/{day}/Auto_000.json",
                image_id=i,
                has_annotation=any(c != 0 for c in cat_ids),
                category_ids=cat_ids,
            )
        )
    return records


def test_extract_frame_number() -> None:
    assert _extract_frame_number("Auto_000_000123") == 123
    assert _extract_frame_number("no_digits") == 0


def test_assign_groups_sets_group_ids_and_indices() -> None:
    day = "day2"
    records = _make_records(day, categories_per_frame=[[0], [0], [0], [0], [0], [0], [0], [0]])
    groups = assign_groups(records, day)

    assert len(groups) == 2
    assert groups[0].group_id == "day2_g000000"
    assert groups[1].group_id == "day2_g000001"

    # Group-level fields are written back into FrameRecord instances
    assert records[0].group_id == "day2_g000000"
    assert records[0].group_index == 0
    assert records[3].group_index == 3
    assert records[4].group_id == "day2_g000001"
    assert records[4].group_index == 0


def test_compute_stratification_keys() -> None:
    day = "day2"
    # First group: only normal category 0 -> _none
    # Second group: anomaly category 2 -> _2
    records = _make_records(
        day,
        categories_per_frame=[[0], [0], [0], [0], [0, 2], [2], [0, 2], [2]],
    )
    groups = assign_groups(records, day)
    compute_stratification_keys(groups)

    assert groups[0].strat_key == "day2_none"
    assert groups[1].strat_key == "day2_2"


def test_stratified_group_split_is_deterministic() -> None:
    day = "day2"
    categories_per_frame = [[0], [2], [2], [0], [2], [0], [2], [0]]

    records1 = _make_records(day, categories_per_frame=categories_per_frame)
    groups1 = assign_groups(records1, day)
    compute_stratification_keys(groups1)
    stratified_group_split(groups1, train_ratio=0.5, val_ratio=0.0, seed=123)
    mapping1 = {g.group_id: f.split for g in groups1 for f in g.frames[:1]}

    records2 = _make_records(day, categories_per_frame=categories_per_frame)
    groups2 = assign_groups(records2, day)
    compute_stratification_keys(groups2)
    stratified_group_split(groups2, train_ratio=0.5, val_ratio=0.0, seed=123)
    mapping2 = {g.group_id: f.split for g in groups2 for f in g.frames[:1]}

    assert mapping1 == mapping2
    assert set(mapping1.values()).issubset({"train", "val", "test"})


def test_write_splits_csv_writes_expected_header(tmp_path) -> None:
    day = "day2"
    records = _make_records(day, categories_per_frame=[[0], [2], [0], [2]])
    groups = assign_groups(records, day)
    compute_stratification_keys(groups)
    stratified_group_split(groups, train_ratio=1.0, val_ratio=0.0, seed=1)

    out_path = tmp_path / "splits.csv"
    write_splits_csv(records, out_path)

    with out_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    assert reader.fieldnames == [
        "day",
        "group_id",
        "group_index",
        "split",
        "cu3s_path",
        "annotation_json",
        "image_id",
        "has_annotation",
        "category_labels",
    ]
    assert len(rows) == 4
    assert {r["split"] for r in rows}.issubset({"train", "val", "test"})
