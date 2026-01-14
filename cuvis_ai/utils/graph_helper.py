"""Graph output restructuring utilities for efficient node access."""

from typing import Any


def restructure_output_to_node_dict(
    outputs: dict[tuple[str, str], Any],
) -> dict[str, dict[str, Any]]:
    """Transform graph outputs from flat structure to nested structure.

    Transforms:
        dict[(node_name, port_name), value]
    To:
        dict[node_name, dict[port_name, value]]

    This enables O(1) lookup per node instead of O(n_outputs) iteration,
    which is critical for efficient loss and metric collection.

    Parameters
    ----------
    outputs : dict[tuple[str, str], Any]
        Graph outputs keyed by (node_name, port_name) tuples

    Returns
    -------
    dict[str, dict[str, Any]]
        Restructured outputs with node_name as primary key

    Examples
    --------
    >>> outputs = {
    ...     ("loss_node", "loss"): tensor(0.5),
    ...     ("metric_node", "metrics"): [Metric(...), Metric(...)],
    ... }
    >>> node_dict = restructure_output_to_node_dict(outputs)
    >>> # Access loss: node_dict["loss_node_123"]["loss"]
    >>> # O(1) instead of filtering entire outputs dict
    """
    structured = {}
    for (node_id, port_name), value in outputs.items():
        if node_id not in structured:
            structured[node_id] = {}
        structured[node_id][port_name] = value
    return structured
