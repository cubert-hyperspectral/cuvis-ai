# Task: Extend `DataConfig` in `cuvis-ai-schemas` to Support Multi-File Datasets

## Summary

The `DataConfig` schema in `cuvis-ai-schemas` currently hard-requires `cu3s_file_path: str`
and uses `extra="forbid"`, which prevents multi-file dataset workflows (e.g. our lentils
day2/day3/day4 reflectance data where each frame is a separate `.cu3s` file with a shared
COCO annotation JSON per day, and splits are defined in a CSV).

We need to extend `DataConfig` so that `TrainRunConfig` can be saved/loaded cleanly for
both single-file and multi-file data sources, maintaining full reproducibility.

**Current workaround:** A local `MultiFileDataConfig` subclass in `cuvis-ai` that overrides
`cu3s_file_path` as optional and adds `splits_csv`. This lives in
`cuvis_ai/training/schemas.py` and is temporary until the upstream change lands.

---

## Acceptance Criteria

- [ ] `DataConfig` supports two mutually exclusive data source modes:
  - **Single-file:** `cu3s_file_path` (existing behavior)
  - **Multi-file:** `splits_csv` (new — path to a CSV defining train/val/test splits with
    per-frame `.cu3s` paths and annotation JSON references)
- [ ] A Pydantic `model_validator` enforces that exactly one of `cu3s_file_path` or
      `splits_csv` is provided (never both, never neither)
- [ ] All existing single-file configs and tests continue to pass without modification
- [ ] `TrainRunConfig.save_to_file()` / `load_from_file()` round-trips correctly for both modes
- [ ] Proto serialization (`to_proto` / `from_proto`) works for both modes (no proto schema
      change needed — `DataConfig` uses opaque `config_bytes` transport)
- [ ] New unit tests cover: multi-file creation, mutual exclusivity validation, round-trip
      save/load for multi-file `TrainRunConfig`
- [ ] `cuvis-ai` workaround (`MultiFileDataConfig` / `MultiFileTrainRunConfig` subclasses)
      is removed and replaced with the upstream `DataConfig`

---

## Implementation Plan

### PR 1: `cuvis-ai-schemas` (repo: `cubert-hyperspectral/cuvis-ai-schemas`)

**Branch:** `feature/multi-file-data-config` (rename to JIRA task ID once assigned)

**File:** `cuvis_ai_schemas/training/data.py`

```python
from pydantic import Field, model_validator

class DataConfig(BaseSchemaModel):
    __proto_message__: ClassVar[str] = "DataConfig"

    cu3s_file_path: str | None = Field(
        default=None, description="Path to single .cu3s file"
    )
    splits_csv: str | None = Field(
        default=None, description="Path to splits CSV for multi-file datasets"
    )
    annotation_json_path: str | None = Field(
        default=None, description="Path to annotation JSON (optional)"
    )
    train_ids: list[int] = Field(default_factory=list)
    val_ids: list[int] = Field(default_factory=list)
    test_ids: list[int] = Field(default_factory=list)
    train_split: float | None = Field(default=None, ge=0.0, le=1.0)
    val_split: float | None = Field(default=None, ge=0.0, le=1.0)
    shuffle: bool = Field(default=True)
    batch_size: int = Field(default=1, ge=1)
    processing_mode: str = Field(default="Reflectance")

    @model_validator(mode="after")
    def _validate_data_source(self) -> DataConfig:
        if not self.cu3s_file_path and not self.splits_csv:
            raise ValueError(
                "Either 'cu3s_file_path' or 'splits_csv' must be provided"
            )
        if self.cu3s_file_path and self.splits_csv:
            raise ValueError(
                "Provide only one of 'cu3s_file_path' (single-file) "
                "or 'splits_csv' (multi-file)"
            )
        return self
```

**File:** `tests/test_training.py` — add tests:

- `test_data_config_multi_file` — create with `splits_csv`, verify fields
- `test_data_config_requires_one_source` — neither field → `ValidationError`
- `test_data_config_rejects_both_sources` — both fields → `ValidationError`
- `test_trainrun_save_and_load_multi_file` — round-trip with `splits_csv`

**Version bump:** minor (e.g. `0.2.0` → `0.3.0`)

### PR 2: `cuvis-ai` (repo: `cubert-hyperspectral/cuvis-ai`)

**After schemas PR is merged and released to PyPI:**

1. Bump dependency: `cuvis-ai-schemas[full]>=0.3.0`
2. Delete `cuvis_ai/training/schemas.py` (local override)
3. Update imports in:
   - `examples/adaclip/lentils_concrete_gradient_training.py`
   - `examples/adaclip/lentils_drcnn_gradient_training.py`
4. Replace `MultiFileTrainRunConfig` usage with upstream `TrainRunConfig`
5. Run smoke tests to confirm clean `TrainRunConfig` export

---

## Context

- **Upstream schema location:** `cuvis_ai_schemas/training/data.py` in
  `github.com/cubert-hyperspectral/cuvis-ai-schemas`
- **Proto:** `DataConfig` uses opaque `bytes config_bytes = 1` — no proto change needed
- **`BaseSchemaModel`:** uses `ConfigDict(extra="forbid", validate_assignment=True)`
- **Current installed version:** `cuvis-ai-schemas==0.2.0` from PyPI
- **Multi-file data layout:**
  ```
  /mnt/data/day{2,3,4}_reflectance_all/
      Auto_000_*.cu3s       (one frame per file, uint16 reflectance)
      Auto_000.json         (COCO annotations, shared per day)
  ```
- **Splits CSV** generated by `cuvis_ai.data.lentils_splits` with columns:
  `day, group_id, group_index, split, cu3s_path, annotation_json, image_id, has_annotation, category_labels`

---

## Related Files

| File | Description |
|------|-------------|
| `cuvis_ai_schemas/training/data.py` | `DataConfig` schema (upstream, needs change) |
| `cuvis_ai_schemas/training/run.py` | `TrainRunConfig` (uses `DataConfig`, no change) |
| `cuvis_ai/training/schemas.py` | Local override (temporary, to be deleted) |
| `cuvis_ai/data/multi_file_dataset.py` | `MultiFileCu3sDataModule` consumer |
| `examples/adaclip/lentils_*_gradient_training.py` | Training scripts using multi-file data |
