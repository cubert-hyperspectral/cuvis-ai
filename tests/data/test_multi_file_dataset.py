import json
from pathlib import Path
from typing import Any

import numpy as np

import cuvis_ai.data.multi_file_dataset as md


def _write_coco_json(path: Path) -> None:
    coco: dict[str, Any] = {
        "categories": [{"id": 1, "name": "anomaly"}],
        "annotations": [
            {
                "image_id": 0,
                "category_id": 1,
                # COCO-style polygon list: one polygon with 3 vertices.
                "segmentation": [[1, 1, 1, 3, 3, 3]],
            }
        ],
    }
    path.write_text(json.dumps(coco), encoding="utf-8")


def test_extract_frame_number() -> None:
    assert md._extract_frame_number("Auto_000_000123") == 123
    assert md._extract_frame_number("no_digits") == 0


def test_parse_coco_json(tmp_path) -> None:
    ann_path = tmp_path / "ann.json"
    _write_coco_json(ann_path)

    parsed = md._parse_coco_json(ann_path)
    assert parsed["cat_map"][1] == "anomaly"
    assert 0 in parsed["anns_by_image"]
    assert parsed["anns_by_image"][0][0]["category_id"] == 1


def test_build_category_mask() -> None:
    # Triangle polygon in a small image should mark some pixels with cat_id=1.
    anns = [{"category_id": 1, "segmentation": [[1, 1, 1, 3, 3, 3]]}]
    mask = md._build_category_mask(anns, h=5, w=5)
    assert mask.dtype == np.int32
    assert (mask == 1).any()
    assert (mask == 0).any()


def test_data_module_load_records(tmp_path) -> None:
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "day,group_id,group_index,split,cu3s_path,annotation_json,image_id,has_annotation,category_labels\n"
        "day2,g0,0,train,/fake/a,/fake/ann.json,0,1,2\n"
        "day2,g1,0,val,/fake/b,/fake/ann.json,1,0,\n"
        "day2,g2,0,test,/fake/c,/fake/ann.json,2,0,\n",
        encoding="utf-8",
    )

    dm = md.MultiFileCu3sDataModule(splits_csv=splits_csv, batch_size=2)
    records = dm._load_records()
    assert len(records["train"]) == 1
    assert len(records["val"]) == 1
    assert len(records["test"]) == 1
    assert records["train"][0]["image_id"] == 0


def test_dataset_getitem_with_mocked_cuvis(tmp_path, monkeypatch) -> None:
    # Prepare a COCO annotation json file.
    ann_path = tmp_path / "ann.json"
    _write_coco_json(ann_path)

    h, w, c = 5, 6, 3
    cube_array = np.random.rand(h, w, c).astype(np.float32)
    wavelengths = [400, 410, 420]

    class FakeCube:
        def __init__(self) -> None:
            self.array = cube_array
            self.wavelength = wavelengths
            self.channels = c

    class FakeMeasurement:
        def __init__(self) -> None:
            self.cube = FakeCube()
            self.data = {"cube": True}

    class FakeSessionFile:
        def __init__(self, _path: str) -> None:
            self._path = _path

        def get_measurement(self, _idx: int) -> FakeMeasurement:
            return FakeMeasurement()

    class FakeProcessingContext:
        def __init__(self, _session: FakeSessionFile) -> None:
            self.processing_mode = None

        def apply(self, mesu: FakeMeasurement) -> FakeMeasurement:
            return mesu

    class FakeProcessingMode:
        Raw = "Raw"
        Reflectance = "Reflectance"

    monkeypatch.setattr(md.cuvis, "SessionFile", FakeSessionFile)
    monkeypatch.setattr(md.cuvis, "ProcessingContext", FakeProcessingContext)
    monkeypatch.setattr(md.cuvis, "ProcessingMode", FakeProcessingMode)

    records = [
        {"cu3s_path": "/fake/a.cu3s", "annotation_json": str(ann_path), "image_id": 0},
    ]
    ds = md.MultiFileCu3sDataset(records, processing_mode="Reflectance")

    item = ds[0]
    assert item["cube"].shape == (h, w, c)
    assert item["wavelengths"].shape == (c,)
    assert item["mesu_index"] == 0
    assert item["mask"].shape == (h, w)
    assert (item["mask"] == 1).any()
