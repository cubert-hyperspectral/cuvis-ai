import importlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator
from copy import copy, deepcopy
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pkg_resources  # part of setuptools
import torch

from cuvis_ai.node import Node
from cuvis_ai.node.consumers import *
from cuvis_ai.node.sklearn import SklearnWrapped
from cuvis_ai.node.skorch import SkorchWrapped
from cuvis_ai.node.wrap import make_node
from cuvis_ai.pipeline.executor import MemoryExecutor
from cuvis_ai.utils.dependencies import get_installed_packages_str
from cuvis_ai.utils.filesystem import change_working_dir
from cuvis_ai.utils.numpy import check_array_shape
from cuvis_ai.utils.serializer import YamlSerializer


def maybe_wrap_node(node):
    if isinstance(node, Node):
        return node
    return make_node(node)


class Graph:
    """Main class for connecting nodes in a CUVIS.AI processing graph"""

    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.entry_point = None
        self.name = name

    def add_node(self, node: Node, parent: list[Node] | Node = None) -> None:
        """Add a new node into the graph structure

        Parameters
        ----------
        node : Node
            CUVIS.AI type node
        parent : list[Node] | Node, optional
           Node(s) that the child node should be connected to,
           with data flowing from parent(s) to child, by default None.

        Raises
        ------
        ValueError
            If no parent is provided, node is assumed to be the base node of the graph.
            This event will raise an error to prevent base from being overwritten.
        ValueError
            If parent(s) do not already belong to the graph.
        ValueError
            If parent(s) and child nodes are mismatched in expected data size.

        """
        if parent is None:
            # this is the first Node of the graph
            if self.entry_point is not None:
                raise ValueError("Graph already has base node")
            self.entry_point = node.id
            parent = []

        if isinstance(parent, Node):
            parent = [parent]

        # Check if operation is valid
        if not all([self.graph.has_node(p.id) for p in parent]):
            raise ValueError("Not all parents are part of the Graph")

        if not all([check_array_shape(p.output_dim, node.input_dim) for p in parent]):
            raise ValueError("Unsatisfied dimensionality constraint!")

        self.graph.add_node(node.id)

        for p in parent:
            self.graph.add_edge(p.id, node.id)

        self.nodes[node.id] = node

        # Remove if verify fails
        if not self._verify():
            self.delete_node(node)

    def add_base_node(self, node: Node) -> None:
        """Adds new node into the graph by creating the first entry point.

        Parameters
        ----------
        node : Node
            CUVIS.AI node to add to the graph
        """
        node = maybe_wrap_node(node)

        self.graph.add_node(node.id)
        self.nodes[node.id] = node
        self.entry_point = node.id

    def add_edge(self, node: Node, node2: Node) -> None:
        """Adds sequential nodes to create a directed edge.
        At least one of the nodes should already be in the graph.

        Parameters
        ----------
        node : Node
            Parent node.
        node2 : Node
            Child node.
        """

        node = maybe_wrap_node(node)
        node2 = maybe_wrap_node(node2)

        self.graph.add_edge(node.id, node2.id)
        self.nodes[node.id] = node
        self.nodes[node2.id] = node2
        if not self._verify():
            # TODO Issue: This could potentially leave the graph in an invalid state
            # Delete nodes and connection
            del self.nodes[node.id]
            del self.nodes[node2.id]
            # Remove the nodes from the graph as a whole
            self.graph.remove_nodes_from([node.id, node2.id])

    def custom_copy(self):
        # Create a new instance of the class
        new_instance = self.__class__.__new__(self.__class__)

        new_instance.name = deepcopy(self.name)
        new_instance.graph = deepcopy(self.graph)  # Deep copy
        new_instance.nodes = copy(self.nodes)  # Shallow copy
        new_instance.entry_point = deepcopy(self.entry_point)

        return new_instance

    def __rshift__(self, other: Node):
        """Compose with *other*.

        Example:
            t = a >> b >> c
        """
        new_graph = self.custom_copy()
        if new_graph.entry_point == None:
            new_graph.add_base_node(other)
            return new_graph

        # Get all nodes without successors
        sink_nodes = [
            new_graph.nodes[node]
            for node in new_graph.graph.nodes
            if new_graph.graph.out_degree(node) == 0
        ]
        if len(sink_nodes) == 1:
            new_graph.add_edge(sink_nodes[0], other)
        return new_graph

    def __repr__(self) -> str:
        res = self.name + ":\n"
        for node in self.nodes:
            res += f"{node}\n"
        return res

    def _verify_input_outputs(self) -> bool:
        """Private function to validate the integrity of data passed between nodes.

        Returns
        -------
        bool
            Inputs and outputs of all nodes are congruent.
        """
        all_edges = list(self.graph.edges)
        for start, end in all_edges:
            # TODO: Issue what if multiple Nodes feed into the same successor Node, how would the shape look like?
            if not check_array_shape(self.nodes[start].output_dim, self.nodes[end].input_dim):
                # TODO reenable this, for now skip
                print("Unsatisfied dimensionality constraint!")
                # return True
        return True

    def _verify(self) -> bool:
        """Private function to verify the integrity of the processing graph.

        Returns
        -------
        bool
            Graph meets/does not meet requirements for ordered and error-free flow of data.
        """
        if len(self.nodes.keys()) == 0:
            print("Empty graph!")
            return True
        elif len(self.nodes.keys()) == 1:
            print("Single stage graph!")
            return True
        # Check that no cycles exist
        if len(list(nx.simple_cycles(self.graph))) > 0:
            return False
        # Get all edges in the graph
        if not self._verify_input_outputs():
            return False

        return True

    def delete_node(self, id: Node | str) -> None:
        """Removes a node from the graph.
        To successfully remove a node, it must not have successors.


        Parameters
        ----------
        id : Node | str
            UUID for target node to delete, or a copy of the node itself.

        Raises
        ------
        ValueError
            Node to delete contains successors in the graph.
        ValueError
            Node does not exist in the graph.
        """
        if isinstance(id, Node):
            id = id.id

        # Check if operation is valid
        if not len(list(self.graph.successors(id))) == 0:
            raise ValueError(
                "The node does have successors, removing it would invalidate the Graph structure"
            )

        if id not in self.nodes:
            raise ValueError("Cannot remove node, it no longer exists")

        self.graph.remove_edges_from([id])
        del self.nodes[id]

    def serialize(self, data_dir: Path) -> dict:
        """Convert graph structure and all contained nodes to a serializable YAML format.
        Numeric data and fit models will be stored in zipped directory named with current time.
        """
        from importlib.metadata import version

        data_dir = Path(data_dir)
        nodes_data = {}
        for key, node in self.nodes.items():
            serialized = node.serialize(data_dir)
            node_data = {
                "__node_module__": str(node.__module__),
                "__node_class__": str(node.__class__.__name__),
            }

            # maybe serialize source code
            if "code" in serialized.keys():
                import cuvis_ai.utils.inspect as ins

                cls = serialized.pop("code")
                node_code = ins.get_src(cls)
                with open(data_dir / f"{cls.__name__}.py", "w") as f:
                    f.writelines(node_code)
                node_data["__node_code__"] = f"{cls.__name__}.py"

            node_data |= serialized
            nodes_data[key] = node_data

        edges_data = [{"from": start, "to": end} for start, end in list(self.graph.edges)]

        output = {
            "edges": edges_data,
            "nodes": nodes_data,
            "name": self.name,
            "entry_point": self.entry_point,
            "version": version("cuvis_ai"),
            "packages": get_installed_packages_str(),
        }

        return output

    def load(self, structure: dict, data_dir: Path) -> None:
        data_dir = Path(data_dir)
        self.name = structure.get("name")

        installed_cuvis_version = pkg_resources.require("cuvis_ai")[0].version
        serialized_cuvis_version = structure.get("version")

        if installed_cuvis_version != serialized_cuvis_version:
            raise ValueError(
                f"Incorrect version of cuvis_ai package. Installed {installed_cuvis_version} but serialized with {serialized_cuvis_version}"
            )  # nopep8
        if not structure.get("nodes"):
            print("No node information available!")

        LOAD_SOURCE_FILES = True

        for key, params in structure.get("nodes").items():
            node_module = params.get("__node_module__")
            node_class = params.get("__node_class__")
            node_code = params.get("__node_code__", None)

            if node_code is not None and LOAD_SOURCE_FILES:
                spec = importlib.util.spec_from_file_location(node_module, data_dir / node_code)
                module = importlib.util.module_from_spec(spec)
                sys.modules[node_module] = module
                spec.loader.exec_module(module)
                cls = getattr(module, node_class)
            else:
                cls = getattr(importlib.import_module(node_module), node_class)
            if not issubclass(cls, Node):
                cls = make_node(cls)

            if "params" in params.keys():
                stage = cls(**params["params"])
            else:
                stage = cls()
            stage.load(params, data_dir)
            self.nodes[key] = stage

        # Set the entry point
        self.entry_point = structure.get("entry_point")
        # Create the graph instance
        self.graph = nx.DiGraph()
        # Handle base case where there is only one node
        if len(structure.get("nodes")) > 1:
            # Graph has at least one valid edge
            for edge in structure.get("edges"):
                self.graph.add_edge(edge.get("from"), edge.get("to"))
        else:
            # Only single node exists, add it into the graph
            self.add_base_node(list(self.nodes.values())[0])

    def save_to_file(self, filepath) -> None:
        filepath = Path(filepath)

        os.makedirs(filepath.parent, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmpDir:
            with change_working_dir(tmpDir):
                graph_data = self.serialize(".")

                serial = YamlSerializer(tmpDir, "main")
                serial.serialize(graph_data)

            shutil.make_archive(f"{str(filepath)}", "zip", tmpDir)
            print(f"Project saved to {str(filepath)}")

    @classmethod
    def load_from_file(cls, filepath: str) -> None:
        """Reconstruct the graph from a file path defining the location of a zip archive.

        Parameters
        ----------
        filepath : str
            Location of zip archive
        """
        new_graph = cls("Loaded")
        with tempfile.TemporaryDirectory() as tmpDir:
            shutil.unpack_archive(filepath, tmpDir)

            with change_working_dir(tmpDir):
                serial = YamlSerializer(tmpDir, "main")
                graph_data = serial.load()

                new_graph.load(graph_data, ".")
        return new_graph

    def forward(
        self,
        X: np.ndarray,
        Y: np.ndarray | list | None = None,
        M: np.ndarray | list | None = None,
        backend: str = "memory",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if backend == "memory":
            executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        elif backend == "hummingbird":
            from copy import copy

            from hummingbird.ml import convert

            def convert_node(node):
                new_node = copy(node)
                if "_wrapped" in node.__dict__ and isinstance(node, SklearnWrapped):
                    new_node._wrapped = convert(node._wrapped, "torch")
                return new_node

            nodes = {k: convert_node(v) for k, v in self.nodes.items()}

            executor = MemoryExecutor(self.graph, nodes, self.entry_point)
        else:
            raise ValueError("Unknown Backend")
        return executor.forward(X, Y, M)

    def fit(
        self, X: np.ndarray, Y: np.ndarray | list | None = None, M: np.ndarray | list | None = None
    ):
        executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        executor.fit(X, Y, M)

    def train(self, train_dl: torch.utils.data.DataLoader, test_dl: torch.utils.data.DataLoader):
        executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        executor.train(train_dl, test_dl)

    @property
    @lru_cache(maxsize=128)
    def torch_layers(self) -> list[torch.nn.Module]:
        """Get a list with all pytorch layers in the Graph."""
        layers = []
        for key, node in self.nodes.items():
            if isinstance(node, SkorchWrapped):
                layers.append(node.net.model_)
        return layers

    def parameters(self) -> Iterator:
        """Iterate over all (pytorch-) parameters in all layers contained in the Graph."""
        for layer in self.torch_layers:
            yield from layer.parameters()

    def freeze(self):
        for node in self.nodes.values():
            if node.initialized:
                node.freezed = True
            else:
                raise RuntimeError("Tried freezing a uninitialized node")
