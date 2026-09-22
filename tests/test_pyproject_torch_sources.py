"""Guard the torch index forks in ``pyproject.toml`` and ``uv.lock``.

The ``cuda`` dependency group pins torch and torchvision to a PyTorch wheel index. aarch64
Linux (Jetson Thor, JetPack 7) needs the cu130 index: the cu128 index serves an SBSA aarch64
wheel whose kernels stop at sm_120, so a checkout synced from it installs cleanly and fails at
the first CUDA kernel with "no kernel image is available for execution on the device". Every
other platform keeps cu128. Both entries stay scoped to the ``cuda`` group, so a git or path
consumer (the ``cuvis_ai_builtin`` manifest inside a composed child environment) inherits no
index pin. torchcodec links against torch's ABI: 0.11 pairs with torch 2.11 only while 0.12+
accepts torch >= 2.11, so the floor must admit the newer torch the aarch64 fork resolves.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from packaging.markers import Marker
from packaging.requirements import Requirement
from packaging.version import Version

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"

CU128 = "https://download.pytorch.org/whl/cu128"
CU130 = "https://download.pytorch.org/whl/cu130"
FORKED = ("torch", "torchvision")
ENVIRONMENTS = {
    "jetson": {"sys_platform": "linux", "platform_machine": "aarch64"},
    "linux-x86_64": {"sys_platform": "linux", "platform_machine": "x86_64"},
    "windows": {"sys_platform": "win32", "platform_machine": "AMD64"},
}
EXPECTED_INDEX = {"jetson": CU130, "linux-x86_64": CU128, "windows": CU128}


@pytest.fixture(scope="module")
def pyproject() -> dict:
    """The parsed project file."""
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def lock() -> dict:
    """The parsed lock file."""
    return tomllib.loads(LOCK.read_text(encoding="utf-8"))


def _indexes(pyproject: dict) -> dict[str, dict]:
    return {entry["name"]: entry for entry in pyproject["tool"]["uv"].get("index", [])}


@pytest.mark.parametrize("package", FORKED)
def test_torch_source_is_a_group_scoped_marker_fork(pyproject: dict, package: str) -> None:
    """Two entries, both in the cuda group, each naming a declared explicit index."""
    sources = pyproject["tool"]["uv"]["sources"][package]
    assert isinstance(sources, list) and len(sources) == 2, f"{package}: expected two entries"
    indexes = _indexes(pyproject)
    for entry in sources:
        assert entry.get("group") == "cuda", f"{package}: {entry} must stay in the cuda group"
        assert entry["index"] in indexes, f"{package}: index {entry['index']!r} is not declared"
        assert indexes[entry["index"]].get("explicit") is True
        # A malformed marker (PEP 508 has no `not`) raises InvalidMarker here.
        Marker(entry["marker"])


@pytest.mark.parametrize("package", FORKED)
@pytest.mark.parametrize("platform", sorted(ENVIRONMENTS))
def test_exactly_one_index_matches_each_platform(
    pyproject: dict, package: str, platform: str
) -> None:
    """The markers partition the platforms: Jetson -> cu130, everything else -> cu128."""
    indexes = _indexes(pyproject)
    matching = [
        indexes[entry["index"]]["url"]
        for entry in pyproject["tool"]["uv"]["sources"][package]
        if Marker(entry["marker"]).evaluate(ENVIRONMENTS[platform])
    ]
    assert matching == [EXPECTED_INDEX[platform]], f"{package} on {platform}: {matching}"


def test_cuda_group_lists_exactly_the_forked_packages(pyproject: dict) -> None:
    """The group carries only the index pins and is installed by default."""
    assert sorted(pyproject["dependency-groups"]["cuda"]) == sorted(FORKED)
    assert "cuda" in pyproject["tool"]["uv"]["default-groups"]


def test_torchcodec_floor_admits_the_aarch64_torch(pyproject: dict) -> None:
    """torchcodec 0.11 pairs with torch 2.11 only; the floor must exclude it."""
    floors = {
        Requirement(dep).name: Requirement(dep) for dep in pyproject["project"]["dependencies"]
    }
    specifier = floors["torchcodec"].specifier
    assert specifier.contains("0.16.0")
    assert not specifier.contains("0.11.1")


def _packages(lock: dict, name: str) -> list[dict]:
    return [pkg for pkg in lock["package"] if pkg["name"] == name]


@pytest.mark.parametrize("package", FORKED)
def test_lock_forks_torch_between_cu128_and_cu130(lock: dict, package: str) -> None:
    """The lock carries one cu128 entry (x86_64 / Windows unchanged) and one cu130 aarch64 entry."""
    by_registry = {pkg["source"]["registry"]: pkg for pkg in _packages(lock, package)}
    assert set(by_registry) == {CU128, CU130}, f"{package}: registries {sorted(by_registry)}"
    assert by_registry[CU128]["version"].endswith("+cu128")
    cu130 = by_registry[CU130]
    assert cu130["version"].endswith("+cu130")
    assert any("manylinux_2_28_aarch64" in wheel["url"] for wheel in cu130["wheels"])


def test_lock_torchcodec_accepts_both_torch_forks(lock: dict) -> None:
    """One torchcodec for both forks, from the range that accepts torch >= 2.11."""
    entries = _packages(lock, "torchcodec")
    assert len(entries) == 1
    assert Version(entries[0]["version"]) >= Version("0.12")


@pytest.mark.parametrize("package", FORKED)
def test_base_requirements_are_split_along_the_fork_markers(pyproject: dict, package: str) -> None:
    """uv assigns one index per fork, so the base requirement must live inside the forks too."""
    requirements = [
        Requirement(dep)
        for dep in pyproject["project"]["dependencies"]
        if Requirement(dep).name == package
    ]
    assert len(requirements) == 2, f"{package}: expected one requirement per fork"
    assert len({str(req.specifier) for req in requirements}) == 1, f"{package}: floors differ"
    source_markers = {entry["marker"] for entry in pyproject["tool"]["uv"]["sources"][package]}
    assert {str(req.marker) for req in requirements} == {str(Marker(m)) for m in source_markers}
