"""Lifecycle tags, categories and execution stages must agree for every builtin node.

The rules read the class declarations (``EXECUTION_STAGES``, ``_category``, ``_tags``): a
node's stages are exactly its class declaration, which is also what plugin manifests and
gRPC ``NodeInfo`` expose.

- R0: category ``LOSS`` or ``REGULARIZER`` never runs at inference. Category decides here
  because a loss has no other purpose; for ``METRIC`` / ``SINK`` / ``VISUALIZER`` the
  category is too coarse (video writers are sinks that exist for inference,
  ``DistinctLabelCount`` is a metric that deliberately runs at inference), so the lifecycle
  tags below carry the per-node intent.
- R1: a node that never runs at inference is tagged ``TRAINING`` or ``EVALUATION``.
- R2: a node tagged ``TRAINING`` or ``EVALUATION`` but not ``INFERENCE`` never runs at
  inference. ``AUGMENTATION`` nodes are exempt: the occlusion family sits inline
  (``cube -> occlusion -> model``), so pruning it by stage would leave the model's input
  unsatisfied; it keeps ``ALWAYS`` until it gains a stage-aware pass-through.
- R3: a node tagged ``INFERENCE`` runs at inference.
"""

from __future__ import annotations

from cuvis_ai_schemas.enums import ExecutionStage, NodeCategory, NodeTag

from cuvis_ai_core.utils.node_registry import NodeRegistry
from tests.test_node_categories import _isolated_builtin_registry

TRAINING_ONLY_CATEGORIES = {NodeCategory.LOSS, NodeCategory.REGULARIZER}
LIFECYCLE_TRAINING_TAGS = {NodeTag.TRAINING, NodeTag.EVALUATION}


def _runs_at_inference(cls: type) -> bool:
    stages = cls.get_execution_stages()
    return ExecutionStage.ALWAYS in stages or ExecutionStage.INFERENCE in stages


def _describe(name: str, cls: type) -> str:
    stages = sorted(stage.value for stage in cls.get_execution_stages())
    tags = sorted(tag.value for tag in cls.get_tags())
    return f"{name}: category={cls.get_category().value}, stages={stages}, tags={tags}"


def _builtin_classes() -> list[tuple[str, type]]:
    return [
        (name, NodeRegistry.get_builtin_class(name)) for name in NodeRegistry.list_builtin_nodes()
    ]


def test_r0_losses_and_regularizers_never_run_at_inference():
    with _isolated_builtin_registry():
        violations = [
            _describe(name, cls)
            for name, cls in _builtin_classes()
            if cls.get_category() in TRAINING_ONLY_CATEGORIES and _runs_at_inference(cls)
        ]
    assert not violations, "loss/regularizer nodes running at inference:\n" + "\n".join(violations)


def test_r1_training_only_nodes_carry_a_lifecycle_tag():
    with _isolated_builtin_registry():
        violations = [
            _describe(name, cls)
            for name, cls in _builtin_classes()
            if not _runs_at_inference(cls) and not (cls.get_tags() & LIFECYCLE_TRAINING_TAGS)
        ]
    assert not violations, (
        "nodes pruned at inference without a TRAINING/EVALUATION tag:\n" + "\n".join(violations)
    )


def test_r2_lifecycle_tagged_nodes_do_not_run_at_inference():
    with _isolated_builtin_registry():
        violations = []
        for name, cls in _builtin_classes():
            tags = cls.get_tags()
            if not (tags & LIFECYCLE_TRAINING_TAGS):
                continue
            if NodeTag.INFERENCE in tags or NodeTag.AUGMENTATION in tags:
                continue
            if _runs_at_inference(cls):
                violations.append(_describe(name, cls))
    assert not violations, (
        "TRAINING/EVALUATION-tagged nodes that still run at inference "
        "(add the INFERENCE tag if that is intended):\n" + "\n".join(violations)
    )


def test_r3_inference_tagged_nodes_run_at_inference():
    with _isolated_builtin_registry():
        violations = [
            _describe(name, cls)
            for name, cls in _builtin_classes()
            if NodeTag.INFERENCE in cls.get_tags() and not _runs_at_inference(cls)
        ]
    assert not violations, "INFERENCE-tagged nodes pruned at inference:\n" + "\n".join(violations)


def test_stage_declarations_are_frozensets_of_stages():
    """The core normalizes class declarations; a stray set literal or string would slip
    past the rules above, so check the type once for the whole registry."""
    with _isolated_builtin_registry():
        bad = [
            name
            for name, cls in _builtin_classes()
            if not isinstance(cls.get_execution_stages(), frozenset)
            or not all(isinstance(s, ExecutionStage) for s in cls.get_execution_stages())
        ]
    assert not bad, f"non-normalized EXECUTION_STAGES: {bad}"
