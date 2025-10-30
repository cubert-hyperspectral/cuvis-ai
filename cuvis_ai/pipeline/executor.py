from collections.abc import Sequence
from typing import Any

import networkx as nx
import torch
from torch import Tensor

from cuvis_ai.node.consumers import CubeConsumer, LabelConsumer
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.meta_routing import get_fit_metadata, get_forward_metadata

LabelLike = Tensor | Sequence[Any] | None
MetaLike = Any
NodeOutputs = tuple[Tensor, LabelLike, MetaLike]


class MemoryExecutor:
    def __init__(self, graph: nx.DiGraph, nodes: dict[str, Node], entry_point: str):
        self.graph = graph
        self.nodes = nodes
        self.entry_point = entry_point

    @staticmethod
    def _ensure_tensor(value: Any) -> Tensor:
        if not torch.is_tensor(value):
            raise TypeError(
                f"Expected torch.Tensor input, received {type(value)!r}. "
                "Ensure all nodes emit torch tensors."
            )
        return value

    @staticmethod
    def _ensure_bhwc(value: Any) -> Tensor:
        tensor = MemoryExecutor._ensure_tensor(value)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 4:
            raise ValueError(
                f"Input tensors must follow BHWC format (B, H, W, C). Received shape {tuple(tensor.shape)}."
            )
        return tensor

    @staticmethod
    def _maybe_label(value: Any) -> LabelLike:
        if value is None:
            return None
        if torch.is_tensor(value):
            return value
        if hasattr(value, "__array__"):
            return torch.as_tensor(value)
        return value

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(MemoryExecutor._is_missing(item) for item in value)
        return False

    def forward(self, x: Any, y: Any = None, m: Any = None) -> NodeOutputs:
        """Execute the graph in inference mode using BHWC torch tensors end-to-end."""
        sorted_graph = list(nx.topological_sort(self.graph))
        assert sorted_graph[0] == self.entry_point

        xs = self._ensure_bhwc(x)
        ys = self._maybe_label(y)
        ms = m

        intermediary: dict[str, Tensor] = {}
        intermediary_labels: dict[str, LabelLike] = {}
        intermediary_metas: dict[str, MetaLike] = {}

        (
            intermediary[self.entry_point],
            intermediary_labels[self.entry_point],
            intermediary_metas[self.entry_point],
        ) = self.forward_node(self.nodes[self.entry_point], xs, ys, ms)

        for node in sorted_graph[1:]:
            self._forward_helper(node, intermediary, intermediary_labels, intermediary_metas)

        last_node = sorted_graph[-1]
        return (
            intermediary[last_node],
            intermediary_labels[last_node],
            intermediary_metas[last_node],
        )

    def _forward_helper(
        self,
        current: str,
        intermediary: dict[str, Tensor],
        intermediary_labels: dict[str, LabelLike],
        intermediary_metas: dict[str, MetaLike],
    ) -> None:
        """Aggregate parent outputs and propagate tensors through a single node."""
        parent_nodes = list(self.graph.predecessors(current))

        parent_products = [self._ensure_tensor(intermediary[p]) for p in parent_nodes]
        use_prods = torch.cat(parent_products, dim=-1)

        base_label = intermediary_labels[parent_nodes[0]]
        if self._is_missing(base_label):
            use_labels: LabelLike = None
        elif torch.is_tensor(base_label):
            use_labels = torch.cat(
                [self._ensure_tensor(intermediary_labels[p]) for p in parent_nodes],
                dim=-1,
            )
        else:
            use_labels = [intermediary_labels[p] for p in parent_nodes]

        base_meta = intermediary_metas[parent_nodes[0]]
        if self._is_missing(base_meta):
            use_metas: MetaLike = None
        elif torch.is_tensor(base_meta):
            use_metas = torch.cat(
                [self._ensure_tensor(intermediary_metas[p]) for p in parent_nodes],
                dim=-1,
            )
        else:
            use_metas = [intermediary_metas[p] for p in parent_nodes]

        (
            intermediary[current],
            intermediary_labels[current],
            intermediary_metas[current],
        ) = self.forward_node(self.nodes[current], use_prods, use_labels, use_metas)

        if self._not_needed_anymore(current, intermediary):
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)

    def forward_node(
        self, node: Node, data: Any, labels: LabelLike, metadata: MetaLike
    ) -> NodeOutputs:
        """Run a single node forward pass using torch tensors."""
        tensor_data = self._ensure_tensor(data)
        additional_meta = get_forward_metadata(node, metadata)

        out = node.forward(
            tensor_data,
            y=labels,
            m=metadata,
            **additional_meta,
        )

        if isinstance(out, tuple):
            data_out, labels_out, metadata_out = out
        else:
            data_out, labels_out, metadata_out = out, labels, metadata

        return self._ensure_tensor(data_out), self._maybe_label(labels_out), metadata_out

    def _not_needed_anymore(self, node_id: str, intermediary: dict[str, Tensor]) -> bool:
        """Return True if all successors already computed their outputs."""
        successors = list(self.graph.successors(node_id))
        return all(successor in intermediary for successor in successors) and len(successors) > 0

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
    ) -> None:
        """Fit and evaluate the graph using PyTorch dataloaders."""
        if not isinstance(train_dataloader, torch.utils.data.DataLoader) or not isinstance(
            test_dataloader, torch.utils.data.DataLoader
        ):
            raise TypeError("train or test dataloader argument is not a pytorch DataLoader!")

        for batch in iter(train_dataloader):
            if isinstance(batch, Sequence):
                if len(batch) == 3:
                    x, y, m = batch
                elif len(batch) == 2:
                    x, y = batch
                    m = None
                else:
                    raise ValueError("Train dataloader must yield (x, y) or (x, y, m).")
            else:
                raise ValueError("Train dataloader must yield a sequence.")
            self.fit(x, y, m, warm_start=True)

        for batch in iter(test_dataloader):
            if isinstance(batch, Sequence):
                if len(batch) == 3:
                    x, y, m = batch
                elif len(batch) == 2:
                    x, y = batch
                    m = None
                else:
                    raise ValueError("Test dataloader must yield (x, y) or (x, y, m).")
            else:
                raise ValueError("Test dataloader must yield a sequence.")
            self.forward(x, y, m)

    def fit(
        self,
        X: Any,
        Y: Any = None,
        M: Any = None,
        warm_start: bool = False,
    ) -> None:
        """Fit all nodes in the graph using BHWC torch tensors."""
        sorted_graph = list(nx.topological_sort(self.graph))
        assert sorted_graph[0] == self.entry_point

        data = self._ensure_bhwc(X)
        labels = self._maybe_label(Y)
        metadata = M

        intermediary: dict[str, Tensor] = {}
        intermediary_labels: dict[str, LabelLike] = {}
        intermediary_metas: dict[str, MetaLike] = {}

        (
            intermediary[self.entry_point],
            intermediary_labels[self.entry_point],
            intermediary_metas[self.entry_point],
        ) = self.fit_node(
            self.nodes[self.entry_point], data, labels, metadata, warm_start=warm_start
        )

        for node in sorted_graph[1:]:
            self._fit_helper(
                node, intermediary, intermediary_labels, intermediary_metas, warm_start
            )

    def _fit_helper(
        self,
        current: str,
        intermediary: dict[str, Tensor],
        intermediary_labels: dict[str, LabelLike],
        intermediary_metas: dict[str, MetaLike],
        warm_start: bool = False,
    ) -> None:
        """Fit a single node by aggregating parent outputs."""
        parent_nodes = list(self.graph.predecessors(current))

        parent_products = [self._ensure_tensor(intermediary[p]) for p in parent_nodes]
        use_prods = torch.cat(parent_products, dim=-1)

        base_label = intermediary_labels[parent_nodes[0]]
        if self._is_missing(base_label):
            use_labels: LabelLike = None
        elif torch.is_tensor(base_label):
            use_labels = torch.cat(
                [self._ensure_tensor(intermediary_labels[p]) for p in parent_nodes],
                dim=-1,
            )
        else:
            use_labels = [intermediary_labels[p] for p in parent_nodes]

        base_meta = intermediary_metas[parent_nodes[0]]
        if self._is_missing(base_meta):
            use_metas: MetaLike = None
        elif torch.is_tensor(base_meta):
            use_metas = torch.cat(
                [self._ensure_tensor(intermediary_metas[p]) for p in parent_nodes],
                dim=-1,
            )
        else:
            use_metas = [intermediary_metas[p] for p in parent_nodes]

        (
            intermediary[current],
            intermediary_labels[current],
            intermediary_metas[current],
        ) = self.fit_node(
            self.nodes[current], use_prods, use_labels, use_metas, warm_start=warm_start
        )

        if self._not_needed_anymore(current, intermediary):
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)

    def fit_node(
        self,
        node: Node,
        data: Any,
        labels: LabelLike,
        metadata: MetaLike,
        warm_start: bool = False,
    ) -> NodeOutputs:
        """Fit a single node and return its forward outputs."""
        tensor_data = self._ensure_tensor(data)
        label_input = self._maybe_label(labels)

        node_input: list[Any] = []
        if isinstance(node, CubeConsumer):
            node_input.append(tensor_data)
        if isinstance(node, LabelConsumer):
            node_input.append(label_input)

        if len(node_input) == 0:
            raise RuntimeError(f"Node {node} invalid, does not indicate input data type!")

        additional_meta = get_fit_metadata(node, metadata)
        if not node.freezed:
            if len(additional_meta) > 0:
                node.fit(*node_input, **additional_meta, warm_start=warm_start)
            else:
                node.fit(*node_input, warm_start=warm_start)

        return self.forward_node(node, tensor_data, label_input, metadata)


class HummingBirdExecutor:
    def __init__(self, graph: nx.DiGraph, nodes: dict[str, Node], entry_point: str):
        self.graph = graph
        self.nodes = nodes
        self.entry_point = entry_point
        self.sorted_nodes = list(nx.topological_sort(self.graph))
