"""Plugin manifests that the shipped configs use resolve on any machine.

The plugin catalog ships inside the wheel. A manifest with a local ``path:`` source resolves only
on its author's machine; ``cuvis_ai_builtin.yaml`` once shipped that way and failed at compose for
every user while it looked fine in a developer checkout. A manifest that no shipped config uses
may keep a local path (a plugin still under development). Once a shipped pipeline lists it, or a
shipped trainrun reads one of its data modules, it needs a ``repo`` and a ``tag``.
"""

from collections import defaultdict
from pathlib import Path

from cuvis_ai_schemas.pipeline.config import PipelineConfig
from cuvis_ai_schemas.plugin import GitPluginSource, PluginManifest, load_plugin_manifest

from tests.configs.test_trainrun_configs_valid import _TRAINRUNS, _load_trainrun

_CONFIGS = Path(__file__).resolve().parents[2] / "cuvis_ai" / "configs"


def _pipeline_plugin_uses(pipeline_dir: Path) -> dict[str, set[str]]:
    """Plugin name to the pipelines (relative to ``pipeline_dir``) that list it."""
    uses: dict[str, set[str]] = defaultdict(set)
    for path in sorted(pipeline_dir.rglob("*.yaml")):
        for name in PipelineConfig.load_from_file(path).plugins or []:
            uses[name].add(path.relative_to(pipeline_dir).as_posix())
    return dict(uses)


def _trainrun_data_module_uses() -> dict[str, set[str]]:
    """Data module name to the shipped trainruns that read it."""
    uses: dict[str, set[str]] = defaultdict(set)
    for path in _TRAINRUNS:
        data = _load_trainrun(path).data
        if data is not None and data.data_module:
            uses[data.data_module].add(path.name)
    return dict(uses)


def _unresolvable_uses(
    plugin_uses: dict[str, set[str]],
    data_module_uses: dict[str, set[str]],
    plugins_dir: Path,
) -> list[str]:
    """One line per used plugin or data module whose manifest is local or missing."""
    by_name: dict[str, tuple[str, PluginManifest]] = {}
    by_data_module: dict[str, tuple[str, PluginManifest]] = {}
    for path in sorted(plugins_dir.glob("*.yaml")):
        manifest = load_plugin_manifest(path)
        by_name[manifest.name] = (path.name, manifest)
        for capability in manifest.capabilities:
            data_module_name = getattr(capability, "data_module_name", None)
            if data_module_name:
                by_data_module[data_module_name] = (path.name, manifest)

    problems = []
    for kind, uses, index in (
        ("plugin", plugin_uses, by_name),
        ("data module", data_module_uses, by_data_module),
    ):
        for name, configs in sorted(uses.items()):
            users = ", ".join(sorted(configs))
            if name not in index:
                problems.append(f"{kind} {name!r} has no manifest (used by {users})")
                continue
            file_name, manifest = index[name]
            if not isinstance(manifest, GitPluginSource):
                problems.append(
                    f"{file_name} has a local path source, but {kind} {name!r} is used by {users}"
                )
    return problems


def test_manifests_used_by_shipped_configs_are_git_sourced() -> None:
    """Every manifest a shipped pipeline or trainrun needs declares ``repo`` and ``tag``."""
    plugin_uses = _pipeline_plugin_uses(_CONFIGS / "pipeline")
    data_module_uses = _trainrun_data_module_uses()

    # The collection must see the catalog, or the check below passes on nothing.
    assert "cuvis_ai_builtin" in plugin_uses
    assert "cu3s" in data_module_uses

    problems = _unresolvable_uses(plugin_uses, data_module_uses, _CONFIGS / "plugins")
    assert not problems, "\n".join(problems)


_GIT_MANIFEST = """\
name: {name}
repo: "https://example.com/{name}.git"
tag: "v1.0.0"
capabilities:
- class_name: {name}_pkg.node.Thing
"""
_LOCAL_MANIFEST = """\
name: {name}
path: "D:/somewhere/{name}"
capabilities:
- class_name: {name}_pkg.node.Thing
"""
_PIPELINE = """\
metadata:
  name: {name}
plugins: {plugins}
nodes: []
connections: []
"""


def test_only_a_used_local_manifest_or_a_missing_one_is_reported(tmp_path: Path) -> None:
    """A local manifest is reported once a pipeline uses it; an unused one is left alone."""
    plugins = tmp_path / "plugins"
    plugins.mkdir()
    (plugins / "a.yaml").write_text(_GIT_MANIFEST.format(name="used_git"), encoding="utf-8")
    (plugins / "b.yaml").write_text(_LOCAL_MANIFEST.format(name="used_local"), encoding="utf-8")
    (plugins / "c.yaml").write_text(_LOCAL_MANIFEST.format(name="unused_local"), encoding="utf-8")
    pipelines = tmp_path / "pipeline" / "group"
    pipelines.mkdir(parents=True)
    (pipelines / "p.yaml").write_text(
        _PIPELINE.format(name="p", plugins=["used_git", "used_local", "ghost"]), encoding="utf-8"
    )

    uses = _pipeline_plugin_uses(tmp_path / "pipeline")
    problems = _unresolvable_uses(uses, {}, plugins)

    assert uses == {
        "used_git": {"group/p.yaml"},
        "used_local": {"group/p.yaml"},
        "ghost": {"group/p.yaml"},
    }
    assert len(problems) == 2
    assert any("b.yaml" in line and "group/p.yaml" in line for line in problems)
    assert any("'ghost'" in line and "no manifest" in line for line in problems)
    assert not any("unused_local" in line or "c.yaml" in line for line in problems)
