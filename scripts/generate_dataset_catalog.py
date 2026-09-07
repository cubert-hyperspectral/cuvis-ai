"""Generate the datasets catalog index at ``catalogs/datasets/index.md``.

The summary table is rendered from cuvis-ai-core's public-dataset registry
(``cuvis_ai_core.data.public_datasets.PublicDatasets``): the same rows ``uv run dataset
list`` prints and CuvisNEXT's Settings › Cuvis.AI › Datasets tab shows, so the page
cannot drift from what is downloadable. The per-dataset pages are the mirrored
Hugging Face READMEs written by ``scripts/sync_hf_readmes.py``; the card blurbs in
``CARDS`` are the only hand-written part, and a registry row without a card fails the
docs build on purpose.
"""

from __future__ import annotations

import mkdocs_gen_files

from cuvis_ai_core.data.public_datasets import DatasetSpec, PublicDatasets

# Registry name -> (docs page slug, Material icon, card blurb). The slug is the
# ``display_slug`` of ``scripts/sync_hf_readmes.py``.
CARDS: dict[str, tuple[str, str, str]] = {
    "Blood_Perfusion": (
        "XMR_Demo_Blood_Perfusion",
        ":material-water:",
        "Tissue oxygenation visualised via two-band differential. Pairs with the "
        "[Blood Perfusion tutorial](../../tutorials/statistical/blood-perfusion.md). "
        "No training manifest: a statistical demo.",
    ),
    "Demo_Industrial_FOD_Lentils": (
        "XMR_Demo_Industrial_FOD_Lentils",
        ":material-food:",
        "Lentil-conveyor scene with stones, stems, aluminium shards, and flies. Pairs with the "
        "[AdaCLIP tutorial](../../tutorials/gradient/adaclip.md) and is the inference showcase "
        "of the trained Dinomaly pipelines on its companion model repo. No training manifest: "
        "the README declares no split.",
    ),
    "Demo_Object_Tracking": (
        "XMR_Demo_Object_Tracking",
        ":material-account-group:",
        "Crowded bus-station scene with passive and active (invisible-ink) tracking modes. "
        "No training manifest: the tracking nodes do not train.",
    ),
    "Lentils": (
        "XMR_Lentils",
        ":material-database:",
        "One lentil-conveyor session with COCO annotations, the recording the test suite and "
        "`configs/data/lentils.yaml` use.",
    ),
    "Industrial_FOD_Lentils": (
        "XMR_Industrial_FOD_Lentils",
        ":material-factory:",
        "Fifteen merged conveyor sessions over three days, 1,136 frames (696 annotated), seven "
        "foreign-object classes, with baked train / val / test selectors "
        "(`configs/data/industrial_fod_lentils.yaml`, `industrial_fod_lentils_normals.yaml`); "
        "the demo's trained Dinomaly pipelines were fitted on it.",
    ),
    "Industrial_FOD_Bedding": (
        "X4_SWIR_Industrial_FOD_Bedding",
        ":material-bed:",
        "252 VIS + SWIR frames of bedding: 193 normal training frames and a 59-frame validation "
        "split holding the 51 annotated foreign-object frames "
        "(`configs/data/industrial_fod_bedding.yaml`).",
    ),
}

CAMERA_NOTES = {
    "XMR": "[Ultris XMR](https://cubert-hyperspectral.com/de/ultris-xmr/) (61 bands, 430–910 nm)",
    "X4 SWIR": "Ultris X4 + SWIR rig (three visible and three short-wave-infrared bands)",
}


def _size(num_bytes: int) -> str:
    return f"{num_bytes / 1e9:.1f} GB" if num_bytes >= 1e9 else f"{num_bytes / 1e6:.0f} MB"


def _camera_sentence(specs: list[DatasetSpec]) -> str:
    parts = []
    for camera, note in CAMERA_NOTES.items():
        count = sum(1 for spec in specs if spec.camera == camera)
        if count:
            parts.append(f"{count} with a Cubert {note}")
    return "Captured " + "; ".join(parts) + "."


def _table(specs: list[DatasetSpec]) -> str:
    lines = [
        "| Dataset | Camera | Size | Files | Task | Licence | Download |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec in specs:
        slug = CARDS[spec.name][0]
        lines.append(
            f"| [{spec.display_name}]({slug}.md) | {spec.camera} | {_size(spec.size_bytes)} | "
            f"{spec.file_count} | {', '.join(spec.tags)} | {spec.license} | "
            f"`uv run dataset download {spec.name}` |"
        )
    return "\n".join(lines)


def _cards(specs: list[DatasetSpec]) -> str:
    blocks = []
    for spec in specs:
        slug, icon, blurb = CARDS[spec.name]
        blocks.append(f"-   {icon} **[{slug}]({slug}.md)**\n\n    ---\n\n    {blurb}\n")
    return '<div class="grid cards" markdown>\n\n' + "\n".join(blocks) + "\n</div>\n"


def render(specs: list[DatasetSpec]) -> str:
    """Render the catalog index for ``specs`` (every row needs a ``CARDS`` entry)."""
    missing = [spec.name for spec in specs if spec.name not in CARDS]
    if missing:
        raise KeyError(f"datasets without a catalog card: {missing}")
    return (
        "# Datasets Catalog\n\n"
        "Public datasets for cuvis-ai, hosted on "
        "[Hugging Face](https://huggingface.co/cubert-gmbh) under the `cubert-gmbh` "
        "organisation. Each page below mirrors the dataset's Hugging Face README; the HF page "
        "is the authoritative source.\n\n"
        f"{_camera_sentence(specs)}\n\n"
        "The table is generated from the registry in cuvis-ai-core (`uv run dataset list`). "
        "`uv run dataset download <name> --data-dir data` fetches a dataset into "
        "`data/<folder>` and records a `.cuvis-dataset.json` marker beside it; "
        "`uv run dataset status --data-dir data` reports what is present. In CuvisNEXT the same "
        "list lives under Settings › Cuvis.AI › Datasets.\n\n"
        f"{_table(specs)}\n\n"
        "## Available datasets\n\n"
        f"{_cards(specs)}"
    )


def main() -> None:
    """Write ``catalogs/datasets/index.md`` into the generated docs tree."""
    specs = sorted(PublicDatasets.list_specs(), key=lambda spec: spec.display_name.lower())
    with mkdocs_gen_files.open("catalogs/datasets/index.md", "w") as fh:
        fh.write(render(specs))


if __name__ in {"__main__", "<run_path>"}:
    main()
