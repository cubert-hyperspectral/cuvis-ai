"""Guard the torch index forks in ``pyproject.toml`` and ``uv.lock``.

The ``cuda`` dependency group pins torch and torchvision to a PyTorch wheel index. aarch64
Linux (Jetson Thor, JetPack 7) needs the cu130 index: the cu128 index serves an SBSA aarch64
wheel whose kernels stop at sm_120, so a checkout synced from it installs cleanly and fails at
the first CUDA kernel with "no kernel image is available for execution on the device". Every
other platform keeps cu128. Both entries stay scoped to the ``cuda`` group, so a git or path
consumer (the ``cuvis_ai_builtin`` manifest inside a composed child environment) inherits no
index pin. torchcodec is forked the same way: 0.11 is the release built against torch 2.11 and its
x86_64 wheel needs no CUDA 13 runtime, while every 0.12+ Linux wheel links libcudart / libnvrtc 13
(libnvjpeg 13 too) and so imports only beside the cu130 torch of the aarch64 fork.
"""

from __future__ import annotations

import ctypes.util
import os
import sys
import tomllib
from glob import glob
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
PYTHON_MINORS = ("3.11", "3.12", "3.13")
PYTHON_FULL_VERSIONS = {"3.11": "3.11.9", "3.12": "3.12.7", "3.13": "3.13.7"}


def _lock_environment(platform: str, minor: str) -> dict[str, str]:
    """A marker environment for one platform and one supported CPython minor."""
    return {
        **ENVIRONMENTS[platform],
        "python_version": minor,
        "python_full_version": PYTHON_FULL_VERSIONS[minor],
    }


def _applies(pkg: dict, environment: dict[str, str]) -> bool:
    """Whether a lock entry is selected in ``environment`` (no markers = every environment)."""
    markers = pkg.get("resolution-markers")
    return markers is None or any(Marker(m).evaluate(environment) for m in markers)


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


def _requirements(pyproject: dict, name: str) -> list[Requirement]:
    return [
        Requirement(dep)
        for dep in pyproject["project"]["dependencies"]
        if Requirement(dep).name == name
    ]


def test_torchcodec_is_forked_with_torch(pyproject: dict) -> None:
    """0.11.x beside the cu128 torch 2.11, 0.16+ beside the cu130 torch 2.14, same markers.

    Every 0.12+ torchcodec wheel on Linux links the CUDA 13 runtime, so a single 0.16 floor
    fails at import next to a cu128 torch (CI caught it); children inherit these markers.
    """
    by_marker = {str(req.marker): req.specifier for req in _requirements(pyproject, "torchcodec")}
    source_markers = {str(Marker(e["marker"])) for e in pyproject["tool"]["uv"]["sources"]["torch"]}
    assert set(by_marker) == source_markers, sorted(by_marker)
    for platform, environment in ENVIRONMENTS.items():
        (specifier,) = [s for m, s in by_marker.items() if Marker(m).evaluate(environment)]
        if platform == "jetson":
            assert specifier.contains("0.16.0") and not specifier.contains("0.11.1"), specifier
        else:
            assert specifier.contains("0.11.1") and not specifier.contains("0.16.0"), specifier


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


@pytest.mark.parametrize("platform", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("minor", PYTHON_MINORS)
def test_lock_torchcodec_follows_the_torch_fork(lock: dict, platform: str, minor: str) -> None:
    """One PyPI torchcodec per environment: 0.11.x beside cu128, 0.12+ beside the cu130 torch."""
    environment = _lock_environment(platform, minor)
    selected = [pkg for pkg in _packages(lock, "torchcodec") if _applies(pkg, environment)]
    assert len(selected) == 1, [pkg["version"] for pkg in selected]
    assert selected[0]["source"]["registry"] == "https://pypi.org/simple"
    version = Version(selected[0]["version"])
    if platform == "jetson":
        assert version >= Version("0.12"), version
    else:
        assert Version("0.11.1") <= version < Version("0.12"), version


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


@pytest.mark.parametrize("package", FORKED)
@pytest.mark.parametrize("platform", sorted(ENVIRONMENTS))
@pytest.mark.parametrize("minor", PYTHON_MINORS)
def test_lock_resolution_markers_select_one_fork_per_environment(
    lock: dict, package: str, platform: str, minor: str
) -> None:
    """Exactly one lock entry serves each platform x Python minor, from the expected index."""
    environment = _lock_environment(platform, minor)
    selected = [pkg for pkg in _packages(lock, package) if _applies(pkg, environment)]
    versions = [pkg["version"] for pkg in selected]
    assert len(selected) == 1, f"{package} on {platform} / Python {minor}: {versions}"
    assert selected[0]["source"]["registry"] == EXPECTED_INDEX[platform], versions


def _wheel_names(pkg: dict) -> list[str]:
    return [wheel["url"].rsplit("/", 1)[-1] for wheel in pkg["wheels"]]


@pytest.mark.parametrize("package", FORKED)
@pytest.mark.parametrize("minor", PYTHON_MINORS)
def test_lock_forks_ship_a_wheel_per_supported_python(lock: dict, package: str, minor: str) -> None:
    """cu130: one aarch64 wheel per CPython; cu128: one x86_64 Linux and one Windows wheel each."""
    by_registry = {pkg["source"]["registry"]: pkg for pkg in _packages(lock, package)}
    abi = f"cp{minor.replace('.', '')}-cp{minor.replace('.', '')}-"
    cu130 = [name for name in _wheel_names(by_registry[CU130]) if abi in name]
    assert sum("manylinux_2_28_aarch64" in name for name in cu130) == 1, cu130
    cu128 = [name for name in _wheel_names(by_registry[CU128]) if abi in name]
    assert sum("manylinux_2_28_x86_64" in name for name in cu128) == 1, cu128
    assert sum("win_amd64" in name for name in cu128) == 1, cu128


def _ffmpeg_shared_libraries_present() -> bool:
    """torchcodec dlopens FFmpeg at import; without the shared libraries the import says nothing."""
    if sys.platform == "win32":
        return any(
            glob(os.path.join(entry, "avcodec-*.dll"))
            for entry in os.environ.get("PATH", "").split(os.pathsep)
            if entry
        )
    return ctypes.util.find_library("avcodec") is not None


def test_torchcodec_loads_against_the_installed_torch(lock: dict) -> None:
    """torchcodec's shared library is built per torch ABI; a mismatch fails at import time."""
    import torch
    import torchcodec
    from torchcodec.decoders import VideoDecoder

    if not _ffmpeg_shared_libraries_present():
        pytest.skip("no FFmpeg shared libraries on this machine; CI installs them")
    import cuvis_ai  # noqa: F401  # Windows: registers the FFmpeg DLL directory first

    assert VideoDecoder is not None
    # The running interpreter is one of the lock's environments: the installed build must be the
    # entry the fork selects for it (0.11.x beside a cu128 torch, 0.16+ beside cu130).
    (locked,) = [pkg for pkg in _packages(lock, "torchcodec") if _applies(pkg, {})]
    installed = Version(torchcodec.__version__.split("+")[0])
    assert installed == Version(locked["version"]), (
        f"torchcodec {torchcodec.__version__} next to torch {torch.__version__}, lock says {locked['version']}"
    )
