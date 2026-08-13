"""The builtin manifest must parse to the same source wherever the catalog is installed.

``path: "../../.."`` resolved against the manifest's own directory, which is the checkout
root in a source tree but ``<site-packages>`` in a wheel install, where there is no project
to build: the composer rejected it with "has no pyproject.toml". Any location-dependent
source reintroduces that split, so the invariant is tested rather than the specific source
kind.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from cuvis_ai_schemas.plugin import load_plugin_manifest

pytestmark = pytest.mark.unit

MANIFEST = Path("cuvis_ai/configs/plugins/cuvis_ai_builtin.yaml")


def test_builtin_manifest_source_does_not_depend_on_install_location(tmp_path: Path) -> None:
    relocated = tmp_path / "elsewhere" / MANIFEST.name
    relocated.parent.mkdir(parents=True)
    shutil.copyfile(MANIFEST, relocated)

    assert load_plugin_manifest(MANIFEST).model_dump() == load_plugin_manifest(
        relocated
    ).model_dump(), (
        f"{MANIFEST.name} resolves differently depending on where it is installed, so a "
        "wheel install and a source checkout compose different child environments"
    )
