import numpy as np

from cuvis_ai.node import Node


def traverse(obj: dict, route: list[str]):
    for r in route:
        if r not in obj.keys():
            return None
        obj = obj[r]
    return obj


def get_route(name) -> list[str]:
    return name.split("__")


def get_forward_metadata(node: Node, metadata: dict):
    requested_meta = node.get_forward_requested_meta()
    return get_requested_metadata(requested_meta, metadata)


def get_fit_metadata(node: Node, metadata: dict):
    requested_meta = node.get_fit_requested_meta()
    return get_requested_metadata(requested_meta, metadata)


def get_requested_metadata(requested: dict[str, bool], metadata: dict):
    additional_meta = dict()
    for k in requested.keys():
        additional_meta[k] = list()

    if len(requested) > 0 and metadata is None:
        raise RuntimeError("Requested metadata but no metadata supplied")

    if len(requested) == 0:
        return additional_meta

    for idx in range(len(metadata)):
        for k, v in requested.items():
            if not v:
                continue
            retrieved = traverse(metadata[idx], get_route(k))
            if retrieved is None:
                raise RuntimeError(
                    f"Could not find requested metadata {'/'.join(get_route(k))}"
                )  # nopep8

            additional_meta[k].append(retrieved)

    for k in requested.keys():
        if isinstance(additional_meta[k][0], np.ndarray):
            additional_meta[k] = np.concatenate(additional_meta[k], axis=0)

    return additional_meta
