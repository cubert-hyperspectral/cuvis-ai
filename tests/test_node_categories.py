from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

from cuvis_ai_schemas.enums import NodeTag

from cuvis_ai_core.utils.node_registry import NodeRegistry

REGISTRY_PACKAGES = ("cuvis_ai.node", "cuvis_ai.node.anomaly", "cuvis_ai.node.deciders")

EXPECTED_PRESENT: set[str] = {
    "ROIZoomNode",
    "MaskRobustifier",
    "MaskToBBoxKalman",
    "MaskedMeanSpectrum",
    "SpectrumPlotNode",
}

EXPECTED_MIN_REGISTRY_SIZE = 105


@contextmanager
def _isolated_builtin_registry() -> Iterator[None]:
    snapshot = dict(NodeRegistry._builtin_registry)  # noqa: SLF001
    try:
        NodeRegistry.clear()
        for package_name in REGISTRY_PACKAGES:
            NodeRegistry.auto_register_package(package_name)
        yield
    finally:
        NodeRegistry.clear()
        NodeRegistry._builtin_registry.update(snapshot)  # noqa: SLF001


def test_registry_is_populated():
    with _isolated_builtin_registry():
        registered = set(NodeRegistry.list_builtin_nodes())
        assert len(registered) >= EXPECTED_MIN_REGISTRY_SIZE
        missing = EXPECTED_PRESENT - registered
        assert not missing, f"Expected classes are not registered: {sorted(missing)}"


def test_all_builtin_nodes_have_explicit_category():
    """Every class must declare _category in its own __dict__, not inherit it."""
    with _isolated_builtin_registry():
        missing_explicit = [
            name
            for name in NodeRegistry.list_builtin_nodes()
            if "_category" not in NodeRegistry.get_builtin_class(name).__dict__
        ]
        assert not missing_explicit, (
            f"Classes with no explicit _category (may be inheriting): {missing_explicit}"
        )


def test_all_builtin_nodes_have_explicit_tags():
    """Every class must declare _tags in its own __dict__, not inherit it."""
    with _isolated_builtin_registry():
        missing_explicit = [
            name
            for name in NodeRegistry.list_builtin_nodes()
            if "_tags" not in NodeRegistry.get_builtin_class(name).__dict__
        ]
        assert not missing_explicit, (
            f"Classes with no explicit _tags (may be inheriting): {missing_explicit}"
        )


def test_every_node_has_at_least_one_modality_or_lifecycle_tag():
    modality_or_lifecycle = {
        NodeTag.IMAGE,
        NodeTag.VIDEO,
        NodeTag.RGB,
        NodeTag.MULTISPECTRAL,
        NodeTag.HYPERSPECTRAL,
        NodeTag.POINT_CLOUD,
        NodeTag.DEPTH,
        NodeTag.MASK,
        NodeTag.BBOX,
        NodeTag.KEYPOINTS,
        NodeTag.TEXT,
        NodeTag.AUDIO,
        NodeTag.TABULAR,
        NodeTag.TIME_SERIES,
        NodeTag.METADATA,
        NodeTag.EMBEDDING,
        NodeTag.PREPROCESSING,
        NodeTag.POSTPROCESSING,
        NodeTag.AUGMENTATION,
        NodeTag.CALIBRATION,
        NodeTag.NORMALIZATION,
        NodeTag.TRAINING,
        NodeTag.EVALUATION,
        NodeTag.INFERENCE,
    }
    with _isolated_builtin_registry():
        offenders = [
            name
            for name in NodeRegistry.list_builtin_nodes()
            if not (NodeRegistry.get_builtin_class(name).get_tags() & modality_or_lifecycle)
        ]
        assert not offenders, f"Nodes with no modality / lifecycle tag: {offenders}"


def test_no_category_dominates():
    """Sanity: no single category should account for > 70% of the catalog."""
    with _isolated_builtin_registry():
        names = list(NodeRegistry.list_builtin_nodes())
        total = len(names)
        counts = Counter(NodeRegistry.get_builtin_class(n).get_category() for n in names)
        for cat, count in counts.items():
            assert count / total <= 0.70, (
                f"Category {cat} dominates: {count}/{total} ({count / total:.0%})"
            )
