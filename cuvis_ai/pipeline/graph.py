from __future__ import annotations

import importlib
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from copy import copy, deepcopy
from functools import cached_property
from pathlib import Path
from typing import Any

import networkx as nx
import pkg_resources  # part of setuptools
import torch
from torch import Tensor, nn

from cuvis_ai.node import Node
from cuvis_ai.node.consumers import *
from cuvis_ai.pipeline.executor import MemoryExecutor
from cuvis_ai.utils.filesystem import change_working_dir
from cuvis_ai.utils.serializer import YamlSerializer
from cuvis_ai.utils.torch import check_array_shape


class Graph:
    """Main class for connecting nodes in a CUVIS.AI processing graph"""

    def __init__(self, name: str) -> None:
        self.graph = nx.DiGraph()
        self.nodes: dict[str, Node] = {}
        self.leaf_nodes: dict[str, dict] = {}  # Leaf nodes (loss, viz, metrics)
        self.monitoring_plugins: list = []  # Monitoring backends (WandB, TB, etc.)
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

        if not all(
            [check_array_shape(p.output_dim, node.input_dim, p.id.split("-")[0]) for p in parent]
        ):
            raise ValueError("Unsatisfied dimensionality constraint!")

        self.graph.add_node(node.id)

        for p in parent:
            self.graph.add_edge(p.id, node.id)

        self.nodes[node.id] = node

        # Remove if verify fails
        if not self._verify():
            self.delete_node(node)

        # Invalidate torch_layers cache since we added a node
        if 'torch_layers' in self.__dict__:
            del self.__dict__['torch_layers']

    def add_base_node(self, node: Node) -> None:
        """Adds new node into the graph by creating the first entry point.

        Parameters
        ----------
        node : Node
            CUVIS.AI node to add to the graph
        """
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
        x: Tensor,
        y: Tensor | Sequence[Any] | None = None,
        m: Any = None,
    ) -> tuple[Tensor, Any, Any]:
        executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        return executor.forward(x, y, m)

    def fit(self, x: Tensor, y: Tensor | Sequence[Any] | None = None, m: Any = None):
        executor = MemoryExecutor(self.graph, self.nodes, self.entry_point)
        executor.fit(x, y, m)

    def set_parent(self, node: Node | str, new_parent: Node | str | None) -> None:
        """Change the parent of an existing node in the graph.
        
        This allows dynamic re-parenting of nodes, which is useful for:
        - Inserting selector nodes into existing pipelines
        - Swapping preprocessing steps
        - Modifying graph structure during experiments
        
        Parameters
        ----------
        node : Node | str
            Node or node ID to re-parent
        new_parent : Node | str | None
            New parent node, parent ID, or None for entry point
            
        Raises
        ------
        ValueError
            If node not found in graph
        ValueError
            If new_parent not found in graph (unless None)
        ValueError
            If re-parenting would create a cycle
        ValueError
            If dimension constraints are violated
            
        Examples
        --------
        >>> # Insert a selector between normalizer and PCA
        >>> graph = Graph("test")
        >>> normalizer = MinMaxNormalizer()
        >>> pca = TrainablePCA(n_components=3)
        >>> graph.add_node(normalizer)
        >>> graph.add_node(pca, parent=normalizer)
        >>> 
        >>> # Now insert selector between them
        >>> selector = SoftChannelSelector(n_select=15)
        >>> graph.add_node(selector, parent=normalizer)
        >>> graph.set_parent(pca, selector)  # PCA now receives from selector
        """
        # Resolve node ID
        node_id = node.id if isinstance(node, Node) else node
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' not found in graph")

        node_obj = self.nodes[node_id]

        # Resolve new parent
        if new_parent is None:
            # Make this node the entry point
            old_parents = list(self.graph.predecessors(node_id))
            for old_parent_id in old_parents:
                self.graph.remove_edge(old_parent_id, node_id)

            # Update entry point if needed
            if node_id != self.entry_point:
                # This node becomes the new entry point
                old_entry = self.entry_point
                self.entry_point = node_id
                # If old entry point has no children, warn
                if old_entry and self.graph.out_degree(old_entry) == 0:
                    import warnings
                    warnings.warn(
                        f"Old entry point '{old_entry}' now has no children. "
                        "You may want to delete it or add it as a child of the new entry point."
                    )
            return

        new_parent_id = new_parent.id if isinstance(new_parent, Node) else new_parent
        if new_parent_id not in self.nodes:
            raise ValueError(f"New parent '{new_parent_id}' not found in graph")

        new_parent_obj = self.nodes[new_parent_id]

        # Validate dimension constraints
        if not check_array_shape(
            new_parent_obj.output_dim,
            node_obj.input_dim,
            new_parent_obj.id.split("-")[0]
        ):
            raise ValueError(
                f"Dimension constraint violated: {new_parent_obj.id} output "
                f"{new_parent_obj.output_dim} incompatible with {node_obj.id} input "
                f"{node_obj.input_dim}"
            )

        # Remove old parent edges
        old_parents = list(self.graph.predecessors(node_id))
        for old_parent_id in old_parents:
            self.graph.remove_edge(old_parent_id, node_id)

        # Add new parent edge
        self.graph.add_edge(new_parent_id, node_id)

        # Check for cycles
        if list(nx.simple_cycles(self.graph)):
            # Rollback: remove new edge and restore old edges
            self.graph.remove_edge(new_parent_id, node_id)
            for old_parent_id in old_parents:
                self.graph.add_edge(old_parent_id, node_id)
            raise ValueError(
                f"Re-parenting {node_id} to {new_parent_id} would create a cycle"
            )

        # Verify the graph is still valid
        if not self._verify():
            # Rollback: remove new edge and restore old edges
            self.graph.remove_edge(new_parent_id, node_id)
            for old_parent_id in old_parents:
                self.graph.add_edge(old_parent_id, node_id)
            raise ValueError(
                f"Re-parenting {node_id} to {new_parent_id} would create an invalid graph"
            )

    def add_leaf_node(self, node, parent: Node | str, node_type=None) -> None:
        """Add a leaf node (loss, visualization, metric, or monitoring) to the graph.
        
        Leaf nodes extend the graph with auxiliary functionality without being part
        of the main computational path.
        
        Parameters
        ----------
        node : LeafNode
            Leaf node instance (LossNode, VisualizationNode, MetricNode, MonitoringNode)
        parent : Node | str
            Parent node or parent node ID to attach to
        node_type : type, optional
            Expected leaf node type for validation. Falls back to node.__class__
            
        Raises
        ------
        ValueError
            If parent node not found in graph
        TypeError
            If leaf node type validation fails
        """
        import inspect

        from cuvis_ai.training.leaf_nodes import (
            LeafNode,
            LossNode,
            MetricNode,
            MonitoringNode,
            VisualizationNode,
        )

        # Resolve parent
        parent_id = parent.id if isinstance(parent, Node) else parent
        if parent_id not in self.nodes:
            raise ValueError(f"Parent node '{parent_id}' not found in graph")

        parent_node = self.nodes[parent_id]

        # Validate parent compatibility
        node.validate_parent(parent_node)

        # Resolve node type
        resolved_type = node_type or node.__class__
        if not inspect.isclass(resolved_type) or not issubclass(resolved_type, LeafNode):
            raise TypeError(f"node_type must be a LeafNode subclass, got {resolved_type}")
        if not isinstance(node, resolved_type):
            raise ValueError(
                f"Leaf node type mismatch: parameter '{resolved_type.__name__}' is not "
                f"a superclass of {node.__class__.__name__}"
            )

        # Determine family
        family = None
        for family_cls in (LossNode, MetricNode, VisualizationNode, MonitoringNode):
            if issubclass(resolved_type, family_cls):
                family = family_cls
                break

        if family is None:
            raise ValueError(
                f"Unsupported leaf node family for {resolved_type.__name__}; "
                "expected subclass of LossNode, MetricNode, VisualizationNode, or MonitoringNode"
            )

        # Set parent reference on the leaf node
        node.parent = parent_node

        # Register leaf node
        leaf_id = f"{parent_id}_{resolved_type.__name__}_{len(self.leaf_nodes)}"
        self.leaf_nodes[leaf_id] = {
            "node": node,
            "parent": parent_id,
            "type": resolved_type,
            "family": family,
            "parent_cls": parent_node.__class__.__name__
        }

    def register_monitor(self, monitor) -> None:
        """Register a monitoring plugin (WandB, TensorBoard, etc.).
        
        Parameters
        ----------
        monitor : MonitoringNode
            Monitoring backend implementation
            
        Raises
        ------
        TypeError
            If monitor doesn't implement MonitoringNode protocol
        """
        from cuvis_ai.training.leaf_nodes import MonitoringNode
        if not isinstance(monitor, MonitoringNode):
            raise TypeError(
                f"monitor must implement MonitoringNode, got {monitor.__class__.__name__}"
            )
        self.monitoring_plugins.append(monitor)

    def _initialize_statistical_nodes(self, datamodule) -> None:
        """Initialize nodes that require statistical fitting before training.
        
        This is Phase 1 of the training process. Nodes with requires_initial_fit=True
        are initialized in topological order, with each receiving data transformed
        through its parent nodes.
        
        Parameters
        ----------
        datamodule : LightningDataModule
            Data module providing train_dataloader()
        """
        from loguru import logger

        # Ensure datamodule is set up
        if hasattr(datamodule, 'setup'):
            datamodule.setup(stage="fit")

        # Find nodes requiring initialization
        stat_nodes = [
            (node_id, node) for node_id, node in self.nodes.items()
            if node.requires_initial_fit
        ]

        if not stat_nodes:
            logger.info("No statistical nodes to initialize")
            return

        logger.info(f"Phase 1: Initializing {len(stat_nodes)} statistical nodes...")

        # Sort topologically to respect dependencies
        sorted_node_ids = list(nx.topological_sort(self.graph))
        stat_nodes_sorted = [
            (nid, node) for nid in sorted_node_ids
            for (node_id, node) in stat_nodes if nid == node_id
        ]

        # Initialize each node
        for node_id, node in stat_nodes_sorted:
            logger.info(f"  Initializing {node_id.split('-')[0]}...")

            # Get data stream transformed through parents
            data_stream = self._get_transformed_data_stream(node, datamodule.train_dataloader())

            # Initialize from data
            node.initialize_from_data(data_stream)

            # Prepare for training or freeze
            if node.is_trainable:
                node.prepare_for_train()
                logger.info(f"    {node_id.split('-')[0]} prepared for training")
            else:
                node.freeze()
                logger.info(f"    {node_id.split('-')[0]} frozen")

    def _get_device(self) -> torch.device:
        """Get the device where graph modules are located."""
        try:
            # Get device from first parameter we find
            return next(iter(self.torch_layers.parameters())).device
        except StopIteration:
            # No parameters found, default to CPU
            return torch.device('cpu')

    def _get_target_device_for_init(self, training_config) -> torch.device | None:
        """Determine target device for statistical initialization.
        
        Parameters
        ----------
        training_config : TrainingConfig
            Training configuration
            
        Returns
        -------
        torch.device or None
            Target device for initialization, or None if CPU should be used
        """
        accelerator = training_config.trainer.accelerator

        # Check if GPU/CUDA requested (case-insensitive, partial match)
        if accelerator is None or accelerator.lower() == 'auto':
            # Auto mode: use GPU if available
            if torch.cuda.is_available():
                accelerator = 'gpu'
            else:
                return None

        # Check if accelerator contains 'gpu' or 'cuda' (case-insensitive)
        accelerator_lower = accelerator.lower()
        is_gpu = any(keyword in accelerator_lower for keyword in ['gpu', 'cuda'])

        if not is_gpu:
            return None  # CPU training

        # Determine which device to use
        devices = training_config.trainer.devices

        if devices is None or devices == 'auto':
            # Default to first available GPU
            device_id = 0
        elif isinstance(devices, int):
            # Single device specified (e.g., devices=1 means use 1 GPU)
            # Lightning uses this to mean "use 1 GPU" (typically cuda:0)
            device_id = 0
        elif isinstance(devices, (list, tuple)):
            # Multiple devices specified (e.g., [0, 2])
            # Use the first one for Phase 1
            device_id = devices[0]
        elif isinstance(devices, str):
            # String format (e.g., "0,2" or "0")
            device_ids = [int(d.strip()) for d in devices.split(',')]
            device_id = device_ids[0]
        else:
            device_id = 0

        return torch.device(f'cuda:{device_id}')

    def _get_transformed_data_stream(self, target_node: Node, dataloader):
        """Yield batches transformed through parent nodes up to (but not including) target.
        
        Parameters
        ----------
        target_node : Node
            Target node that will receive the transformed data
        dataloader : DataLoader
            Source dataloader (must yield dictionaries)
            
        Yields
        ------
        tuple
            (x, y, m) tuples where x is transformed through parents
        """
        device = self._get_device()


        # Find path from entry to target's parents
        target_id = target_node.id
        parent_ids = list(self.graph.predecessors(target_id))

        if not parent_ids:
            # Target is entry point, yield raw data from dict batches
            for batch in dataloader:
                x = batch.get("cube") if "cube" in batch else batch.get("x")
                x = x.to(device)  # ← Move to correct device
                y = batch.get("mask") if "mask" in batch else batch.get("labels")
                if y is not None and torch.is_tensor(y):
                    y = y.to(device)  # ← Also move labels if tensor
                m = {k: v for k, v in batch.items() if k not in ["cube", "x", "mask", "labels"]}
                yield (x, y, m)
        else:
            # Transform through parents
            for batch in dataloader:
                x = batch.get("cube") if "cube" in batch else batch.get("x")
                x = x.to(device)  # ← Move to correct device
                y = batch.get("mask") if "mask" in batch else batch.get("labels")
                if y is not None and torch.is_tensor(y):
                    y = y.to(device)

                m = {k: v for k, v in batch.items() if k not in ["cube", "x", "mask", "labels"]}

                # Transform only through ancestors that feed the target node
                ancestor_ids = nx.ancestors(self.graph, target_id)
                relevant_ids = ancestor_ids.union({target_id})
                sorted_ids = list(nx.topological_sort(self.graph.subgraph(relevant_ids)))
                for node_id in sorted_ids:
                    if node_id == target_id:
                        break
                    if node_id in self.nodes:
                        node = self.nodes[node_id]
                        # Handle different node forward signatures
                        if isinstance(result := node.forward(x), dict):
                            # Node returns dict (e.g., RXGlobal)
                            x = result.get("out", result.get("cube", result.get("x", x)))
                        elif isinstance(result, tuple):
                            # Node returns (x, y, m) tuple
                            x, y, m = result if len(result) == 3 else (*result, {})
                        else:
                            # Node returns tensor directly
                            x = result

                yield (x, y, m)

    def train(
        self,
        datamodule,
        training_config=None,
        **trainer_kwargs
    ):
        """Train the graph using PyTorch Lightning.
        
        This method orchestrates the complete training pipeline:
        1. Phase 1: Statistical initialization of nodes with requires_initial_fit=True
        2. Phase 2: Gradient-based training using PyTorch Lightning
        
        Parameters
        ----------
        datamodule : pl.LightningDataModule or GraphDataModule
            Data module providing train_dataloader() and optionally val_dataloader()
        training_config : TrainingConfig, optional
            Training configuration. If None, uses defaults.
        **trainer_kwargs
            Additional keyword arguments passed to pl.Trainer (overrides config)
            
        Returns
        -------
        pl.Trainer
            The Lightning trainer instance (useful for accessing logs, checkpoints, etc.)
            
        Examples
        --------
        >>> from cuvis_ai.training.config import TrainingConfig, TrainerConfig
        >>> config = TrainingConfig(
        ...     seed=42,
        ...     trainer=TrainerConfig(max_epochs=10, accelerator="gpu")
        ... )
        >>> graph.train(datamodule=my_datamodule, training_config=config)
        """
        import pytorch_lightning as pl
        from loguru import logger

        from cuvis_ai.training.config import OptimizerConfig, TrainerConfig, TrainingConfig
        from cuvis_ai.training.lightning_module import CuvisLightningModule

        # Use default config if not provided
        if training_config is None:
            training_config = TrainingConfig(
                seed=42,
                trainer=TrainerConfig(max_epochs=0),  # Statistical only by default
                optimizer=OptimizerConfig(),
            )

        # Set seed for reproducibility
        if training_config.seed is not None:
            pl.seed_everything(training_config.seed, workers=True)
            logger.info(f"Set random seed to {training_config.seed}")

        # Move graph to GPU before Phase 1 if GPU training requested
        target_device = self._get_target_device_for_init(training_config)
        if target_device is not None:
            if torch.cuda.is_available():
                self.to(target_device)
                logger.info(f"Moved graph to {target_device} for statistical initialization")
            else:
                logger.warning(
                    f"GPU requested ({training_config.trainer.accelerator}) but CUDA not available. "
                    f"Falling back to CPU for statistical initialization."
                )

        # Phase 1: Statistical initialization
        logger.info("=" * 60)
        logger.info("PHASE 1: Statistical Initialization")
        logger.info("=" * 60)
        self._initialize_statistical_nodes(datamodule)
        logger.info("Phase 1 complete")

        # Check if gradient training is needed
        if training_config.trainer.max_epochs == 0:
            logger.info("No gradient training requested (max_epochs=0). Training complete.")
            return None

        # Sanity check if there are trainable parameters
        trainable_params = list(self.parameters(require_grad=True))
        has_visualizations = any(
            leaf_info["family"].__name__ == "VisualizationNode"
            for leaf_info in self.leaf_nodes.values()
        )

        if not trainable_params:
            if has_visualizations:
                logger.info("No trainable parameters found, but running validation to generate visualizations...")
            else:
                logger.info("No trainable parameters found in graph. Skipping gradient training.")
                return None
        else:
            logger.info(f"Found {len(trainable_params)} trainable parameters")

        # Phase 2: Gradient-based training or validation-only
        logger.info("=" * 60)
        if trainable_params:
            logger.info("PHASE 2: Gradient-Based Training")
        else:
            logger.info("PHASE 2: Validation (Visualization Generation)")
        logger.info("=" * 60)

        # Create Lightning module
        lightning_module = CuvisLightningModule(
            graph=self,
            training_config=training_config,
        )

        # Merge trainer config with kwargs (kwargs take precedence)
        from dataclasses import asdict
        trainer_config_dict = asdict(training_config.trainer)
        # Remove None values that Lightning doesn't handle well
        trainer_config_dict = {k: v for k, v in trainer_config_dict.items() if v is not None}
        trainer_config_dict.update(trainer_kwargs)

        # Create Lightning trainer
        trainer = pl.Trainer(**trainer_config_dict)

        # Train
        logger.info(f"Starting training for {training_config.trainer.max_epochs} epochs...")
        trainer.fit(
            model=lightning_module,
            datamodule=datamodule,
        )

        logger.info("Training complete!")
        return trainer

    @cached_property
    def torch_layers(self) -> nn.ModuleList:
        """Torch modules stored in the graph's nodes, packaged as an nn.ModuleList."""
        # If you'll mutate self.nodes after construction, delete this cache:
        #   del self.__dict__['torch_layers']
        modules = [node for node in self.nodes.values() if isinstance(node, nn.Module)]
        return nn.ModuleList(modules)

    def to(self, *args: Any, **kwargs: Any) -> Graph:
        """Move all torch-backed nodes to the requested device/dtype."""
        self.torch_layers.to(*args, **kwargs)
        return self

    def cuda(self, device: int | torch.device | None = None) -> Graph:
        """CUDA convenience wrapper mirroring nn.Module.cuda."""
        self.torch_layers.cuda(device=device)
        return self

    def cpu(self) -> Graph:
        """CPU convenience wrapper mirroring nn.Module.cpu."""
        self.torch_layers.cpu()
        return self

    def _iter_all_parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """
        Yield parameters from all layers (optionally recursing into submodules),
        de-duplicated across shared modules/parameters.
        """
        seen_params: set[int] = set()
        for layer in self.torch_layers:
            # nn.Module.parameters already includes submodules if recurse=True
            iterator = layer.parameters() if recurse else layer.parameters(recurse=False)
            for p in iterator:
                pid = id(p)
                if pid not in seen_params:
                    seen_params.add(pid)
                    yield p

    def parameters(
        self, *, recurse: bool = True, require_grad: bool | None = None
    ) -> Iterator[nn.Parameter]:
        """
        Iterate over unique parameters across the graph.

        Args:
            recurse: include parameters from child modules of each layer.
            require_grad: if set, filter by p.requires_grad == require_grad.
        """
        for p in self._iter_all_parameters(recurse=recurse):
            if require_grad is None or p.requires_grad is require_grad:
                yield p

    def named_parameters(
        self, *, recurse: bool = True, require_grad: bool | None = None, sep: str = "."
    ) -> Iterator[tuple[str, nn.Parameter]]:
        """
        Like torch.nn.Module.named_parameters, but across your graph.
        Names are prefixed with the node key to avoid collisions.
        """
        seen_params: set[int] = set()
        for key, node in self.nodes.items():
            if not isinstance(node, nn.Module):
                continue
            iterator = node.named_parameters(recurse=recurse)
            for local_name, p in iterator:
                pid = id(p)
                if pid in seen_params:
                    continue
                seen_params.add(pid)
                if require_grad is None or p.requires_grad is require_grad:
                    yield f"{key}{sep}{local_name}", p
