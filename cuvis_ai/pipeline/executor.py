import networkx as nx
import numpy as np
import torch

from cuvis_ai.node.consumers import CubeConsumer, LabelConsumer
from cuvis_ai.node.node import Node
from cuvis_ai.pipeline.meta_routing import get_fit_metadata, get_forward_metadata


class MemoryExecutor:
    def __init__(self, graph: nx.DiGraph, nodes: dict[str, Node], entry_point: str):
        self.graph = graph
        self.nodes = nodes
        self.entry_point = entry_point

    def forward(
        self, X: np.ndarray, Y: np.ndarray | list | None = None, M: np.ndarray | list | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pass data through the graph by starting at the root node and flowing through all
        intermediary stages.

        Parameters
        ----------
        X : np.ndarray
            Input data
        Y : Optional[Union[np.ndarray, List]], optional
            Label data
        M : Optional[Union[np.ndarray, List]], optional
            Metadata by default None

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Residuals of processed X, Y, and M
        """
        sorted_graph = list(nx.topological_sort(self.graph))
        assert sorted_graph[0] == self.entry_point

        xs = X
        ys = Y or [None] * len(xs)
        ms = M or [None] * len(xs)

        intermediary = {}
        intermediary_labels = {}
        intermediary_metas = {}
        (
            intermediary[self.entry_point],
            intermediary_labels[self.entry_point],
            intermediary_metas[self.entry_point],
        ) = self.forward_node(self.nodes[self.entry_point], xs, ys, ms)

        for node in sorted_graph[1:]:
            self._forward_helper(node, intermediary, intermediary_labels, intermediary_metas)

        results = intermediary[sorted_graph[-1]]
        return results

    def _forward_helper(
        self, current: str, intermediary: dict, intermediary_labels: dict, intermediary_metas: dict
    ):
        """Helper function to aggregate inputs and calculate products from a given node.

        Parameters
        ----------
        current : str
            id for current node
        intermediary : dict
            Dictionary containing intermediary products with key as id of node
        intermediary_labels : np.ndarray
            Dictionary containing intermediary labels with key as id of node
        intermediary_metas : np.ndarray
            Dictionary containing intermediary metadata with key as id of node
        """
        p_nodes = list(self.graph.predecessors(current))
        # TODO how to concat multiple input data from multiple nodes
        use_prods = np.concatenate([intermediary[p] for p in p_nodes], axis=-1)

        no_labels = intermediary_labels[p_nodes[0]] is None
        if not no_labels:
            if isinstance(intermediary_labels[p_nodes[0]], np.ndarray):
                use_labels = np.concatenate([intermediary_labels[p] for p in p_nodes], axis=-1)
            else:
                use_labels = [intermediary_labels[p] for p in p_nodes]
        else:
            use_labels = []

        no_metas = intermediary_metas[p_nodes[0]] is None
        if not no_metas:
            use_metas = [intermediary_metas[p] for p in p_nodes]
        else:
            use_metas = []

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = (
            self.forward_node(self.nodes[current], use_prods, use_labels, use_metas)
        )

        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)

    def forward_node(
        self, node: Node, data: np.ndarray, labels: np.ndarray, metadata: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pass data through a node which has already been trained/fit.

        Parameters
        ----------
        node : Node
            Node within the graph
        data : np.ndarray
            Data to pass through the nodes
        labels : np.ndarray
            Labels associated with input data
        metadata : np.ndarray
            Metadata needed for forward pass

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]
            Output data, output labels, output metadata
        """
        additional_meta = get_forward_metadata(node, metadata)

        if len(additional_meta) > 0:
            out = node.forward(data, **additional_meta)
        else:
            out = node.forward(data)
        if isinstance(out, tuple):
            return out
        else:
            return out, labels, metadata

    def _not_needed_anymore(self, id: str, intermediary: list[Node]) -> bool:
        """Private function to determine if a node products are still needed or can be safely removed.

        Parameters
        ----------
        id : str
            Alphanumeric identifier for node to check
        intermediary : list[Node]
            List node nodes for which the current node's data is an intermediary

        Returns
        -------
        bool
           If all successors are already present in intermediary, it will return True
        """
        return (
            all([succs in intermediary for succs in self.graph.successors(id)])
            and len(list(self.graph.successors(id))) > 0
        )  # Do not remove a terminal nodes data

    def train(
        self,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
    ):
        """Train a graph use a dataloader to iteratively pass data through the graph

        Parameters
        ----------
        train_dataloader : torch.utils.data.DataLoader
            Training dataloader
        test_dataloader : torch.utils.data.DataLoader
            Test dataloader

        Raises
        ------
        TypeError
            Raises error if dataloaders passed to train function are not pytorch dataloaders
        """
        if not isinstance(train_dataloader, torch.utils.data.DataLoader) or not isinstance(
            test_dataloader, torch.utils.data.DataLoader
        ):
            raise TypeError("train or test dataloader argument is not a pytorch DataLoader!")

        for x, y, m in iter(train_dataloader):
            self.fit(np.squeeze(x), np.squeeze(y), m, warm_start=True)

        # test stage
        test_results = []
        for x, y, m in iter(test_dataloader):
            test_results.append(self.forward(x, y, m))
            # do some metrics

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray | list | None = None,
        M: np.ndarray | list | None = None,
        warm_start=False,
    ):
        """Take a graph of uninitialized nodes and fit then given a set of inputs and outputs

        Parameters
        ----------
        X : np.ndarray
            Input data
        Y : Optional[Union[np.ndarray, List]], optional
            Input labels, by default None
        M : Optional[Union[np.ndarray, List]], optional
            Input metadata, by default None
        """
        # training stage
        sorted_graph = list(nx.topological_sort(self.graph))
        assert sorted_graph[0] == self.entry_point

        intermediary = {}
        intermediary_labels = {}
        intermediary_metas = {}

        (
            intermediary[self.entry_point],
            intermediary_labels[self.entry_point],
            intermediary_metas[self.entry_point],
        ) = self.fit_node(self.nodes[self.entry_point], X, Y, M, warm_start=warm_start)

        for node in sorted_graph[1:]:
            self._fit_helper(node, intermediary, intermediary_labels, intermediary_metas)

    def _fit_helper(
        self,
        current: str,
        intermediary: dict,
        intermediary_labels: dict,
        intermediary_metas: dict,
        warm_start=False,
    ):
        """Private helper function to fit an individual node.

        Parameters
        ----------
        current : str
            id of current node in graph
        intermediary : str
            Dictionary containing intermediary products
        intermediary_labels : np.ndarray
            Dictionary containing intermediary labels
        intermediary_metas : np.ndarray
            Dictionary containing intermediary metadata
        """
        p_nodes = list(self.graph.predecessors(current))

        # TODO how to concat multiple input data from multiple nodes
        use_prods = np.concatenate([intermediary[p] for p in p_nodes], axis=-1)

        no_labels = intermediary_labels[p_nodes[0]] is None
        if not no_labels:
            use_labels = np.concatenate([intermediary_labels[p] for p in p_nodes], axis=-1)
        else:
            use_labels = None

        no_metas = intermediary_metas[p_nodes[0]] is None
        if not no_metas:
            use_metas = np.concatenate([intermediary_metas[p] for p in p_nodes], axis=-1)
        else:
            use_metas = None

        intermediary[current], intermediary_labels[current], intermediary_metas[current] = (
            self.fit_node(self.nodes[current], use_prods, use_labels, use_metas)
        )

        if self._not_needed_anymore(current, intermediary):
            # Free memory that is not needed for the current passthrough anymore
            intermediary.pop(current)
            intermediary_labels.pop(current)
            intermediary_metas.pop(current)

    def fit_node(
        self,
        node: Node,
        data: np.ndarray,
        labels: np.ndarray,
        metadata: np.ndarray,
        warm_start=False,
    ) -> np.ndarray:
        """Private function wrapper to call the fit function for an individual node

        Parameters
        ----------
        node : Node
            Graph node that will be fit
        data : np.ndarray
            Training data
        labels : np.ndarray
            Training labels
        metadata : np.ndarray
            Training metadata

        Returns
        -------
        np.ndarray
            Results of passing data through the fit node

        Raises
        ------
        RuntimeError
            Data is empty (length 0)
        """
        node_input = []

        if isinstance(node, CubeConsumer):
            node_input.append(data)
        if isinstance(node, LabelConsumer):
            node_input.append(labels)

        if len(node_input) == 0:
            raise RuntimeError(f"Node {node} invalid, does not indicate input data type!")

        additional_meta = get_fit_metadata(node, metadata)
        if node.freezed == False:
            if len(additional_meta) > 0:
                node.fit(*node_input, **additional_meta, warm_start=warm_start)
            else:
                node.fit(*node_input, warm_start=warm_start)

        return self.forward_node(node, data, labels, metadata)


class HummingBirdExecutor:
    def __init__(self, graph: nx.DiGraph, nodes: dict[str, Node], entry_point: str):
        self.graph = graph
        self.nodes = nodes
        self.entry_point = entry_point
        self.sorted_nodes = list(nx.topological_sort(self.graph))
