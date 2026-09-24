"""Guards on the cu3s reader levers a shipped trainrun may switch on.

cuvis-ai-dataloader 0.7.0 added ``read_threads``, ``sdk_cuda`` and ``cuda_cubes`` to the cu3s
data module, 0.8.0 ``read_ahead`` (frames read ahead of the model step at batch 1). Its option
parser refuses combinations that cannot work (``cuda_cubes`` needs the SDK on the GPU, the
threaded, device-resident and read-ahead readers need ``num_workers: 0``, ``read_ahead`` is
capped), and an older plugin dies on the unknown keys. The CuvisNEXT training child composes the
plugin from the manifest tag, so a trainrun that sets a lever must ship with a manifest at that
lever's release or later. These checks run against the packaged trainruns and the manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.version import Version

import cuvis_ai

pytestmark = pytest.mark.unit

CONFIGS = Path(cuvis_ai.__file__).resolve().parent / "configs"
TRAINRUNS = sorted((CONFIGS / "trainrun").glob("*.yaml"))
DATALOADER_MANIFEST = CONFIGS / "plugins" / "cuvis_ai_dataloader.yaml"
LEVERS = ("read_threads", "sdk_cuda", "cuda_cubes", "read_ahead")
# The dataloader release that introduced each lever; an older plugin dies on the keyword.
LEVER_FLOORS = {
    "read_threads": Version("0.7.0"),
    "sdk_cuda": Version("0.7.0"),
    "cuda_cubes": Version("0.7.0"),
    "read_ahead": Version("0.8.0"),
}
LEVERS_FLOOR = min(LEVER_FLOORS.values())
MAX_READ_AHEAD = 8  # every frame in flight is a whole cube; the dataloader refuses more
# The dataloader release that masks every frame of a recording without a labels file (all zeros);
# an older plugin's batch carries no mask and the metric nodes die on it (missing 'targets').
LABEL_FREE_MASK_FLOOR = Version("0.8.1")


def _dataloader_manifest_version() -> Version:
    manifest = yaml.safe_load(DATALOADER_MANIFEST.read_text(encoding="utf-8"))
    return Version(str(manifest["tag"]).lstrip("v"))


def _reader_flag_violations(trainrun: dict[str, Any], dataloader_version: Version) -> list[str]:
    """Why a trainrun's cu3s reader levers cannot work as written; empty when they can."""
    data = trainrun.get("data") or {}
    params = data.get("params") or {}
    set_levers = {k: params[k] for k in LEVERS if k in params}
    if not set_levers:
        return []
    problems: list[str] = []
    needed = max(LEVER_FLOORS[lever] for lever in set_levers)
    if dataloader_version < needed:
        late = sorted(k for k in set_levers if LEVER_FLOORS[k] > dataloader_version)
        problems.append(
            f"sets {late} but the dataloader manifest pins v{dataloader_version}, "
            f"older than v{needed}: the child dies on unknown reader options"
        )
    if data.get("num_workers", 0) != 0:
        problems.append(
            f"num_workers={data.get('num_workers')}: the reader levers replace worker processes "
            "and need num_workers 0"
        )
    if params.get("cuda_cubes") and params.get("sdk_cuda") is False:
        problems.append("cuda_cubes needs the SDK on the GPU; sdk_cuda must stay unset or true")
    read_ahead = int(params.get("read_ahead") or 0)
    if not 0 <= read_ahead <= MAX_READ_AHEAD:
        problems.append(
            f"read_ahead={params['read_ahead']} is outside 0 and {MAX_READ_AHEAD}: every frame "
            "in flight is a whole cube in memory, the dataloader refuses more"
        )
    if params.get("read_threads", 0) and data.get("batch_size", 1) < 2 and read_ahead == 0:
        problems.append(
            f"read_threads={params['read_threads']} with batch_size {data.get('batch_size', 1)} "
            "and no read_ahead: reader parallelism is bounded by the batch size, the threads "
            "would idle"
        )
    return problems


def _trainrun(**data: Any) -> dict[str, Any]:
    return {"data": {"data_module": "cu3s", "num_workers": 0, "batch_size": 1, **data}}


class TestChecker:
    def test_no_levers_means_no_problems(self) -> None:
        assert _reader_flag_violations(_trainrun(params={}), Version("0.6.4")) == []

    def test_cuda_cubes_at_batch_one_is_fine(self) -> None:
        assert _reader_flag_violations(_trainrun(params={"cuda_cubes": True}), LEVERS_FLOOR) == []

    def test_threads_with_batching_is_fine(self) -> None:
        run = _trainrun(batch_size=4, params={"read_threads": 4, "cuda_cubes": True})
        assert _reader_flag_violations(run, LEVERS_FLOOR) == []

    def test_old_manifest_is_flagged(self) -> None:
        problems = _reader_flag_violations(_trainrun(params={"cuda_cubes": True}), Version("0.6.4"))
        assert any("older than v0.7.0" in p for p in problems)

    def test_workers_are_flagged(self) -> None:
        run = _trainrun(num_workers=2, params={"read_threads": 4}, batch_size=4)
        problems = _reader_flag_violations(run, LEVERS_FLOOR)
        assert any("num_workers" in p for p in problems)

    def test_device_cubes_on_a_host_sdk_are_flagged(self) -> None:
        run = _trainrun(params={"cuda_cubes": True, "sdk_cuda": False})
        problems = _reader_flag_violations(run, LEVERS_FLOOR)
        assert any("sdk_cuda" in p for p in problems)

    def test_threads_at_batch_one_are_flagged(self) -> None:
        problems = _reader_flag_violations(_trainrun(params={"read_threads": 4}), LEVERS_FLOOR)
        assert any("bounded by the batch size" in p for p in problems)

    def test_read_ahead_needs_the_dataloader_that_knows_it(self) -> None:
        # The DataModule's keyword signature is explicit: an older plugin dies with TypeError.
        problems = _reader_flag_violations(_trainrun(params={"read_ahead": 2}), Version("0.7.0"))
        assert any("read_ahead" in p and "0.8.0" in p for p in problems)

    def test_read_ahead_at_batch_one_is_fine_on_a_new_enough_dataloader(self) -> None:
        run = _trainrun(params={"read_ahead": 2})
        assert _reader_flag_violations(run, Version("0.8.0")) == []

    def test_threads_at_batch_one_are_fine_with_a_read_ahead(self) -> None:
        run = _trainrun(params={"read_threads": 2, "read_ahead": 2})
        assert _reader_flag_violations(run, Version("0.8.0")) == []

    def test_read_ahead_outside_its_range_is_flagged(self) -> None:
        for bad in (-1, 9):
            problems = _reader_flag_violations(
                _trainrun(params={"read_ahead": bad}), Version("0.8.0")
            )
            assert any("read_ahead" in p and "0 and 8" in p for p in problems), (bad, problems)

    def test_read_ahead_refuses_worker_processes(self) -> None:
        run = _trainrun(num_workers=2, params={"read_ahead": 2})
        problems = _reader_flag_violations(run, Version("0.8.0"))
        assert any("num_workers" in p for p in problems)


def test_dataloader_manifest_is_at_or_above_the_levers_floor() -> None:
    assert _dataloader_manifest_version() >= LEVERS_FLOOR


def test_dataloader_manifest_hands_label_free_frames_a_mask() -> None:
    """The wizard's split designer puts recordings without a labels file in val and test."""
    assert _dataloader_manifest_version() >= LABEL_FREE_MASK_FLOOR


@pytest.mark.parametrize("trainrun_yaml", TRAINRUNS, ids=lambda p: p.stem)
def test_shipped_trainrun_reader_flags_can_work(trainrun_yaml: Path) -> None:
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8")) or {}
    problems = _reader_flag_violations(trainrun, _dataloader_manifest_version())
    assert problems == [], f"{trainrun_yaml.name}: " + "; ".join(problems)


# The dataloader's cu3s DataModule defaults; the parser wants every reader lever named.
_PARSER_DEFAULTS: dict[str, Any] = {
    "max_open_sessions": 4,
    "read_threads": 0,
    "source_coherent_batches": False,
    "sdk_cuda": True,
    "cuda_cubes": False,
    "read_ahead": 0,
}

_PLUGIN_MODULE = "cuvis_ai_dataloader.data._extras"
_PLUGIN_REASON = "cuvis-ai-dataloader is not installed here (CI does not install the plugin)"


def test_read_ahead_cap_matches_the_dataloader() -> None:
    """The table's cap is a copy of the plugin's; where the plugin is present they must agree."""
    extras = pytest.importorskip(_PLUGIN_MODULE, reason=_PLUGIN_REASON)
    assert MAX_READ_AHEAD == extras.MAX_READ_AHEAD


@pytest.mark.parametrize("trainrun_yaml", TRAINRUNS, ids=lambda p: p.stem)
def test_dataloader_parser_accepts_what_the_table_accepts(
    trainrun_yaml: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cross-check, not verdict: the table above decides (it runs without the plugin and treats
    the parser's one warning-only case as a refusal). Where the plugin is installed, its own
    ``parse_cu3s_reader_options`` must not refuse a shipped trainrun the table passed, so a rule the
    dataloader tightens shows up here before a training child dies on it.
    """
    extras = pytest.importorskip(_PLUGIN_MODULE, reason=_PLUGIN_REASON)
    # The parser records the SDK device process-wide; this check must not touch it.
    monkeypatch.setattr(extras, "configure_cuvis_sdk", lambda **_: None)
    trainrun = yaml.safe_load(trainrun_yaml.read_text(encoding="utf-8")) or {}
    assert _reader_flag_violations(trainrun, _dataloader_manifest_version()) == []
    data = trainrun.get("data") or {}
    params = data.get("params") or {}
    levers = {**_PARSER_DEFAULTS, **{k: v for k, v in params.items() if k in _PARSER_DEFAULTS}}
    options = extras.parse_cu3s_reader_options(
        num_workers=data.get("num_workers", 0), batch_size=data.get("batch_size", 1), **levers
    )
    assert options.read_ahead == int(levers["read_ahead"])
