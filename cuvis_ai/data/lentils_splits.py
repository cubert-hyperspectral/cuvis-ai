#!/usr/bin/env python3
"""Generate stratified train/val/test splits for lentils multi-file data.

Groups every 4 consecutive frames (same arrangement, different lighting) into
an atomic unit that must stay together in the same split. Then performs stratified
splitting at the group level to preserve category balance across splits.

Usage:
    uv run python -m cuvis_ai.data.lentils_splits \
        --data-root /mnt/data \
        --output splits.csv \
        --train-ratio 0.70 --val-ratio 0.15 --seed 42

Output CSV columns:
    day, group_id, group_index, split, cu3s_path, annotation_json, image_id,
    has_annotation, category_labels
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

DAY_NAMES = ("day2", "day3", "day4")
GROUP_SIZE = 4
NORMAL_CLASS_IDS = frozenset({0})


@dataclass
class FrameRecord:
    """A single lentils frame record used for group-aware split generation."""

    day: str
    cu3s_path: str
    annotation_json: str
    image_id: int
    has_annotation: bool
    category_ids: list[int] = field(default_factory=list)
    group_id: str = ""
    group_index: int = -1
    split: str = ""


def _extract_frame_number(stem: str) -> int:
    """Extract the trailing integer from a filename stem (e.g. ``Auto_000_000123``)."""
    m = re.search(r"(\d+)$", stem)
    return int(m.group(1)) if m else 0


def scan_day(day: str, data_root: Path) -> list[FrameRecord]:
    """Discover .cu3s frames and their annotations for one day."""
    day_dir = data_root / f"{day}_reflectance_all"
    ann_json = day_dir / "Auto_000.json"

    cu3s_files = sorted(
        day_dir.glob("Auto_000_*.cu3s"), key=lambda p: _extract_frame_number(p.stem)
    )
    if not cu3s_files:
        print(f"WARNING: No .cu3s files found in {day_dir}", file=sys.stderr)
        return []

    anns_by_image: dict[int, list[int]] = defaultdict(list)
    if ann_json.is_file():
        with ann_json.open(encoding="utf-8") as f:
            data = json.load(f)
        for ann in data.get("annotations", []):
            cid = int(ann.get("category_id", 0))
            anns_by_image[int(ann["image_id"])].append(cid)

    records: list[FrameRecord] = []
    for cu3s in cu3s_files:
        image_id = _extract_frame_number(cu3s.stem)
        cat_ids = sorted(set(anns_by_image.get(image_id, [])))
        anomaly_cats = [c for c in cat_ids if c not in NORMAL_CLASS_IDS]
        records.append(
            FrameRecord(
                day=day,
                cu3s_path=str(cu3s),
                annotation_json=str(ann_json) if ann_json.is_file() else "",
                image_id=image_id,
                has_annotation=len(anomaly_cats) > 0,
                category_ids=cat_ids,
            )
        )

    return records


@dataclass
class Group:
    """A quartet/group of consecutive frames that must stay in the same split."""

    group_id: str
    day: str
    frames: list[FrameRecord] = field(default_factory=list)
    strat_key: str = ""


def assign_groups(records: list[FrameRecord], day: str) -> list[Group]:
    """Group consecutive frames into quartets."""
    groups: list[Group] = []
    for i in range(0, len(records), GROUP_SIZE):
        chunk = records[i : i + GROUP_SIZE]
        gid = f"{day}_g{i // GROUP_SIZE:06d}"
        g = Group(group_id=gid, day=day)
        for j, rec in enumerate(chunk):
            rec.group_id = gid
            rec.group_index = j
            g.frames.append(rec)
        groups.append(g)
    return groups


def compute_stratification_keys(groups: list[Group]) -> None:
    """Assign a stratification key to each group based on its anomaly content."""
    for g in groups:
        all_anomaly_cats: set[int] = set()
        for f in g.frames:
            for c in f.category_ids:
                if c not in NORMAL_CLASS_IDS:
                    all_anomaly_cats.add(c)

        if not all_anomaly_cats:
            g.strat_key = f"{g.day}_none"
        else:
            g.strat_key = f"{g.day}_{'_'.join(str(c) for c in sorted(all_anomaly_cats))}"


def stratified_group_split(
    groups: list[Group],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    """Assign each group to train/val/test with proportional stratification.

    Falls back to random assignment when a stratum is too small for 3-way split.
    """
    import random

    rng = random.Random(seed)

    strata: dict[str, list[Group]] = defaultdict(list)
    for g in groups:
        strata[g.strat_key].append(g)

    for key in sorted(strata.keys()):
        bucket = strata[key]
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))
        # Ensure at least 1 for test if we have enough
        if n - n_train - n_val < 1 and n >= 3:
            n_val = max(0, n_val - 1)

        for i, g in enumerate(bucket):
            if i < n_train:
                split = "train"
            elif i < n_train + n_val:
                split = "val"
            else:
                split = "test"
            for f in g.frames:
                f.split = split


def write_splits_csv(all_frames: list[FrameRecord], output_path: Path) -> None:
    """Write split assignments and metadata to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )
        for rec in all_frames:
            writer.writerow(
                [
                    rec.day,
                    rec.group_id,
                    rec.group_index,
                    rec.split,
                    rec.cu3s_path,
                    rec.annotation_json,
                    rec.image_id,
                    int(rec.has_annotation),
                    ";".join(str(c) for c in rec.category_ids),
                ]
            )


def print_split_summary(all_frames: list[FrameRecord]) -> None:
    """Print a human-readable summary of the split distribution and annotation counts."""
    split_counts: Counter[str] = Counter()
    split_annotated: Counter[str] = Counter()
    split_day: dict[str, Counter[str]] = defaultdict(Counter)
    split_cats: dict[str, Counter[int]] = defaultdict(Counter)

    for f in all_frames:
        split_counts[f.split] += 1
        if f.has_annotation:
            split_annotated[f.split] += 1
        split_day[f.split][f.day] += 1
        for c in f.category_ids:
            if c not in NORMAL_CLASS_IDS:
                split_cats[f.split][c] += 1

    total = len(all_frames)
    print(f"\n{'=' * 60}")
    print(f"Split Summary  ({total} total frames)")
    print(f"{'=' * 60}")
    for s in ("train", "val", "test"):
        n = split_counts[s]
        ann = split_annotated[s]
        pct = 100.0 * n / total if total else 0
        print(
            f"  {s:6s}: {n:4d} frames ({pct:5.1f}%), {ann} annotated, per-day: {dict(split_day[s])}"
        )
        if split_cats[s]:
            print(f"          cats: {dict(sorted(split_cats[s].items()))}")
    print(f"{'=' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate lentils train/val/test splits")
    parser.add_argument("--data-root", type=Path, default=Path("/mnt/data"))
    parser.add_argument("--output", type=Path, default=Path("lentils_splits.csv"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--days", nargs="+", default=list(DAY_NAMES))
    args = parser.parse_args()

    all_groups: list[Group] = []
    all_frames: list[FrameRecord] = []

    for day in args.days:
        records = scan_day(day, args.data_root)
        if not records:
            continue
        groups = assign_groups(records, day)
        all_groups.extend(groups)
        for g in groups:
            all_frames.extend(g.frames)
        print(f"{day}: {len(records)} frames -> {len(groups)} groups")

    compute_stratification_keys(all_groups)
    stratified_group_split(
        all_groups,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print_split_summary(all_frames)
    write_splits_csv(all_frames, args.output)
    print(f"Saved splits to {args.output}")


if __name__ == "__main__":
    main()
