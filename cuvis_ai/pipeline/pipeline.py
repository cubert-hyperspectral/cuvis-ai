from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import networkx as nx
import torch
import yaml
from loguru import logger
from networkx.classes.reportviews import NodeView
from torch import nn

from cuvis_ai.node import Node
from cuvis_ai.pipeline.ports import InputPort, OutputPort, PortCompatibilityError
from cuvis_ai.utils.types import Context, ExecutionStage

if TYPE_CHECKING:
    from cuvis_ai.training.config import PipelineConfig, PipelineMetadata


class CuvisPipeline:
    """Main class for connecting nodes in a CUVIS.AI processing graph"""

    def __init__(self, name: str, strict_runtime_io_validation: bool = True) -> None:
        from cuvis_ai.training.config import PipelineMetadata

        self._graph = nx.MultiDiGraph()
        self._metadata = PipelineMetadata(name=name)
        self.strict_runtime_io_validation = strict_runtime_io_validation
        self._validation_cache: dict[tuple[str, frozenset, int | None], None] = {}

    @property
    def name(self) -> str:
        """Pipeline name (delegates to metadata)."""
        return self._metadata.name

    @name.setter
    def name(self, value: str) -> None:
        """Set pipeline name (updates metadata)."""
        self._metadata.name = value

    @property
    def metadata(self) -> PipelineMetadata:
        """Pipeline metadata (name, description, tags, etc.)."""
        return self._metadata

    @metadata.setter
    def metadata(self, value: PipelineMetadata) -> None:
        """Set pipeline metadata."""
        self._metadata = value

    def connect(self, *connections) -> None:
        """Connect output ports to input ports with validation.

        Automatically adds nodes to the graph if they are not already present.
        This allows for cleaner code where explicit add_node() calls are optional.

        Supports two syntaxes::

            graph.connect(output_port, input_port)
            graph.connect((output_port1, input_port1), (output_port2, input_port2))

        Parameters
        ----------
        *connections : tuple[OutputPort, InputPort] | OutputPort | InputPort
            Either a pair of OutputPort/InputPort objects or tuples of them.

        Raises
        ------
        ValueError
            If the connection payload has an unexpected structure.
        TypeError
            If provided objects are not OutputPort/InputPort instances.
        PortCompatibilityError
            If ports are incompatible according to their specifications.
        """
        if len(connections) == 2 and isinstance(connections[0], OutputPort):
            connections = [(connections[0], connections[1])]

        for conn in connections:
            if not isinstance(conn, tuple) or len(conn) != 2:
                raise ValueError(
                    f"Each connection must be an (OutputPort, InputPort) tuple, got {conn!r}"
                )

            source_port, target_port = conn

            if not isinstance(source_port, OutputPort):
                raise TypeError(f"Source must be OutputPort, got {type(source_port)!r}")
            if not isinstance(target_port, InputPort):
                raise TypeError(f"Target must be InputPort, got {type(target_port)!r}")

            source_node = source_port.node
            target_node = target_port.node

            is_valid, message = self._validate_connection(source_port, target_port)
            if not is_valid:
                raise PortCompatibilityError(
                    f"Cannot connect {source_node.name}.{source_port.name} "
                    f"to {target_node.name}.{target_port.name}: {message}"
                )

            # AUTO-ADD: Add source node if not in graph
            if source_node not in self._graph:
                self._assign_counter_and_add_node(source_node)

            # AUTO-ADD: Add target node if not in graph
            if target_node not in self._graph:
                self._assign_counter_and_add_node(target_node)

            # Add edge between Node objects (not IDs)
            self._graph.add_edge(
                source_node,
                target_node,
                from_port=source_port.name,
                to_port=target_port.name,
            )

            # Invalidate caches when graph structure changes
            if "_sorted_nodes" in self.__dict__:
                del self.__dict__["_sorted_nodes"]
            # Clear validation cache since graph structure has changed
            self._validation_cache.clear()

    def _assign_counter_and_add_node(self, node: Node) -> None:
        """Assign counter to node based on existing nodes with same base name.

        Uses the highest counter value + 1 to prevent filling gaps when nodes are removed.
        This ensures counter sequences are monotonically increasing, preserving gaps.

        Example:
            Existing: normalizer (0), normalizer-2 (2)  [gap at 1]
            New node: normalizer-3 (3)  [gap preserved]
        """
        # Find the highest counter value for this base name
        max_counter = -1
        for existing in self._graph.nodes():
            if existing._name == node._name:
                max_counter = max(max_counter, existing._pipeline_counter)

        # Assign next counter (gaps are preserved)
        node._pipeline_counter = max_counter + 1
        self._graph.add_node(node)

    def _validate_connection(
        self,
        source_port: OutputPort,
        target_port: InputPort,
    ) -> tuple[bool, str]:
        """Validate that two ports can be connected."""

        return source_port.spec.is_compatible_with(
            target_port.spec,
            source_port.node,
            target_port.node,
        )

    def custom_copy(self) -> CuvisPipeline:
        # Create a new instance of the class
        new_instance = self.__class__.__new__(self.__class__)

        new_instance.name = deepcopy(self.name)
        new_instance._graph = deepcopy(self._graph)  # Deep copy

        return new_instance

    def __repr__(self) -> str:
        res = self.name + ":\n"
        for node in self._graph.nodes():
            res += f"{node}\n"
        res += "Use `pipeline.visualize()` for Graphviz/Mermaid output.\n"
        return res

    def visualize(
        self,
        *,
        format: str = "graphviz",
        output_path: str | Path | None = None,
        **kwargs,
    ) -> str | Path:
        """Visualize the pipeline using built-in Graphviz/Mermaid helpers."""

        from cuvis_ai.pipeline.visualizer import visualize_pipeline

        return visualize_pipeline(self, format=format, output_path=output_path, **kwargs)

    def verify(self) -> None:
        """Verify the integrity of the processing graph.

        Checks for:
        - No cycles in the graph
        - All port connections are compatible (correct dtype and shape)

        Raises
        ------
        ValueError
            If the graph contains cycles.
        PortCompatibilityError
            If any port connection is incompatible.
        """
        # Check that no cycles exist
        if len(list(nx.simple_cycles(self._graph))) > 0:
            raise ValueError("Graph contains cycles!")

        # Verify all port connections
        for start_node, end_node, edge_data in self._graph.edges(data=True):
            # Extract port names from edge data
            from_port_name = edge_data.get("from_port")
            to_port_name = edge_data.get("to_port")

            # Get the actual port objects from nodes
            source_port = start_node._output_ports[from_port_name]
            target_port = end_node._input_ports[to_port_name]

            # Validate the connection
            is_valid, message = self._validate_connection(source_port, target_port)
            if not is_valid:
                raise PortCompatibilityError(
                    f"Invalid connection: {start_node.name}.{from_port_name} "
                    f"-> {end_node.name}.{to_port_name}: {message}"
                )

    @staticmethod
    def _convert_numpy_to_native(obj: Any) -> Any:
        """
        Recursively convert numpy arrays and types to native Python types.

        This ensures YAML serialization produces readable output instead of
        pickle-based binary blobs.

        Args:
            obj: Object to convert (can be nested dict/list/numpy)

        Returns:
            Converted object with numpy types replaced by Python natives
        """
        import numpy as np

        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle numpy scalar types
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        # Handle numpy bool
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Recursively handle dictionaries
        if isinstance(obj, dict):
            return {key: CuvisPipeline._convert_numpy_to_native(val) for key, val in obj.items()}

        # Recursively handle lists and tuples
        if isinstance(obj, (list, tuple)):
            converted = [CuvisPipeline._convert_numpy_to_native(item) for item in obj]
            return converted if isinstance(obj, list) else tuple(converted)

        # Return unchanged for other types
        return obj

    def serialize(self) -> PipelineConfig:
        """
        Serialize pipeline structure to PipelineConfig.

        Returns complete typed configuration that can reconstruct the pipeline,
        including nodes, connections, and metadata.

        Returns:
            PipelineConfig: Typed pipeline configuration with:
                - version: Schema version
                - metadata: Pipeline metadata
                - nodes: List of node configurations
                - connections: List of connection specifications
        """
        from cuvis_ai.training.config import PipelineConfig

        node_configs: list[dict[str, Any]] = []
        for node in self.nodes:
            params = {}
            if hasattr(node, "get_hparams"):
                params = node.get_hparams()  # type: ignore[attr-defined]
            elif hasattr(node, "hparams"):
                params = node.hparams

            # Convert numpy arrays to native Python types for clean YAML serialization
            params = self._convert_numpy_to_native(params or {})

            node_configs.append(
                {
                    "name": node.name,
                    "class": f"{node.__class__.__module__}.{node.__class__.__name__}",
                    "params": params,
                }
            )

        connection_configs: list[dict[str, str]] = []
        for source_node, target_node, edge_data in self._graph.edges(data=True):
            from_port = edge_data.get("from_port")
            to_port = edge_data.get("to_port")
            if from_port is None or to_port is None:
                continue
            connection_configs.append(
                {
                    "from": f"{source_node.name}.outputs.{from_port}",
                    "to": f"{target_node.name}.inputs.{to_port}",
                }
            )

        # Use stored metadata and update created timestamp
        # Create a copy to avoid modifying the original
        from copy import copy

        metadata = copy(self._metadata)
        metadata.created = datetime.now().isoformat()

        return PipelineConfig(
            metadata=metadata,
            nodes=node_configs,
            connections=connection_configs,
        )

    def save_to_file(
        self,
        config_path: str | Path,
        validate_nodes: bool = True,
        include_optimizer: bool = False,
        include_scheduler: bool = False,
        metadata: PipelineMetadata | None = None,
    ) -> None:
        """
        Save pipeline configuration and weights to files.

        Creates two files:
        1. YAML config file (structure, metadata)
        2. .pt weights file (all node state_dicts)

        Args:
            config_path: Path for YAML config file
            validate_nodes: If True, validate all nodes support serialization before saving
            include_optimizer: Whether to save optimizer state
            include_scheduler: Whether to save scheduler state
            metadata: Pipeline metadata (PipelineMetadata instance with description, tags, etc.)

        Raises:
            RuntimeError: If validate_nodes=True and any node doesn't support serialization

        Example:
            >>> from cuvis_ai.training.config import PipelineMetadata
            >>> pipeline.save_to_file(
            ...     "outputs/trained_pipeline.yaml",
            ...     validate_nodes=True,
            ...     metadata=PipelineMetadata(
            ...         name="my_pipeline",
            ...         description="Trained model",
            ...         tags=["statistical", "rx_detector"]
            ...     )
            ... )
        """
        # Validate nodes before saving
        if validate_nodes:
            invalid_nodes = []
            for node in self.nodes:
                if hasattr(node, "validate_serialization_support"):
                    is_valid, message = node.validate_serialization_support()
                    if not is_valid:
                        invalid_nodes.append((node.name, message))
                else:
                    # Node doesn't have validation method - check basics
                    if not hasattr(node, "state_dict"):
                        invalid_nodes.append((node.name, "Missing state_dict() method"))
                    elif not hasattr(node, "load_state_dict"):
                        invalid_nodes.append((node.name, "Missing load_state_dict() method"))

            if invalid_nodes:
                raise RuntimeError(
                    f"Cannot save pipeline: {len(invalid_nodes)} node(s) "
                    f"don't support serialization:\n"
                    + "\n".join(f"  - {name}: {msg}" for name, msg in invalid_nodes)
                )

        config_path = Path(config_path)

        # Get PipelineConfig and convert to dict
        pipeline_config = self.serialize()
        config_dict = pipeline_config.to_dict()

        # Add weights file reference (not part of pipeline structure, but needed for loading)

        # Update metadata if provided
        if metadata:
            config_dict["metadata"].update(metadata.to_dict())

        state_dict: dict[str, Any] = {}
        for node in self.nodes:
            if hasattr(node, "state_dict"):
                state_dict[node.name] = node.state_dict()

        checkpoint: dict[str, Any] = {
            "state_dict": state_dict,
            "metadata": config_dict["metadata"],
        }

        if include_optimizer and hasattr(self, "optimizer"):
            checkpoint["optimizer_state"] = self.optimizer.state_dict()  # type: ignore[attr-defined]

        if include_scheduler and hasattr(self, "scheduler"):
            checkpoint["scheduler_state"] = self.scheduler.state_dict()  # type: ignore[attr-defined]

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

        weights_path = config_path.with_suffix(".pt")
        torch.save(checkpoint, weights_path)

        logger.info(f"Pipeline saved: Config={config_path}, Weights={weights_path}")

    @classmethod
    def _build_pipeline_from_config(
        cls,
        config_path: str | Path,
        config_overrides: list[str] | dict[str, Any] | None = None,
        device: str | None = None,
    ) -> CuvisPipeline:
        """Build pipeline structure from YAML configuration.

        This method loads the YAML config, applies any overrides, and uses PipelineBuilder
        to reconstruct the pipeline structure.

        Args:
            config_path: Path to YAML config file
            config_overrides: Optional config overrides (list of strings or dict)
            device: Device to move pipeline to

        Returns:
            CuvisPipeline: Reconstructed pipeline with structure only (no weights)

        Raises:
            RuntimeError: If node class not found or instantiation fails
        """
        from omegaconf import OmegaConf

        from cuvis_ai.pipeline.pipeline_builder import PipelineBuilder

        config_path = Path(config_path)

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # Apply config overrides if provided
        if config_overrides:
            # Convert config to OmegaConf DictConfig
            config_omega = OmegaConf.create(config)

            # Apply overrides based on input type
            if isinstance(config_overrides, list):
                # Process each override string manually to handle list indices
                for override in config_overrides:
                    if "=" not in override:
                        raise ValueError(
                            f"Invalid override format: {override}. Expected format: key=value"
                        )

                    key, value = override.split("=", 1)

                    # Use OmegaConf.update() which handles list indices in bracket notation
                    # Convert dot notation with numeric indices to bracket notation
                    # e.g., "nodes.10.params.output_dir" -> "nodes[10].params.output_dir"
                    key_parts = key.split(".")
                    formatted_key = ""
                    for i, part in enumerate(key_parts):
                        if part.isdigit():
                            # Add bracket notation for numeric indices
                            formatted_key += f"[{part}]"
                        else:
                            # Add dot before non-numeric parts (except at the start)
                            if i > 0:
                                formatted_key += "."
                            formatted_key += part

                    OmegaConf.update(config_omega, formatted_key, value)
            elif isinstance(config_overrides, dict):
                # Merge dict directly
                override_omega = OmegaConf.create(config_overrides)
                config_omega = OmegaConf.merge(config_omega, override_omega)

            # Convert back to plain dict for builder
            config = OmegaConf.to_container(config_omega, resolve=True)  # type: ignore[assignment]

        # Build pipeline structure with error handling
        try:
            builder = PipelineBuilder()
            # Config should be dict[str, Any] at this point
            pipeline = builder.build_from_config(config)  # type: ignore[arg-type]
        except (ImportError, KeyError) as e:
            raise RuntimeError(
                f"Failed to load pipeline: Node class not found.\n"
                f"This likely indicates the node was removed or renamed.\n"
                f"Original error: {e}"
            ) from e
        except TypeError as e:
            raise RuntimeError(
                f"Failed to instantiate node: Parameter mismatch.\n"
                f"This likely indicates the node's __init__ signature has changed.\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"Failed to build pipeline from config.\nError: {e}") from e

        # Load metadata from config and set it on pipeline
        from cuvis_ai.training.config import PipelineMetadata

        metadata_dict = config.get("metadata", {})  # type: ignore[union-attr]
        if metadata_dict:
            pipeline._metadata = PipelineMetadata.from_dict(metadata_dict)

        if device:
            pipeline.to(device)

        return pipeline

    def _restore_weights_from_checkpoint(
        self,
        weights_path: str | Path,
        strict_weight_loading: bool = True,
        device: str | None = None,
    ) -> None:
        """Restore node weights from checkpoint.

        Args:
            weights_path: Path to .pt weights file
            strict_weight_loading: If True, raise error on state_dict key mismatch
            device: Device to load weights to

        Raises:
            FileNotFoundError: If weights file doesn't exist
            RuntimeError: If strict loading fails
        """
        weights_path = Path(weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")

        checkpoint = torch.load(
            weights_path, map_location=device if device else None, weights_only=False
        )

        state_dict = checkpoint.get("state_dict", {})
        missing_keys = []

        for node in self.nodes:
            if node.name not in state_dict:
                missing_keys.append(node.name)
                continue

            if not hasattr(node, "load_state_dict"):
                continue

            result = node.load_state_dict(state_dict[node.name], strict=strict_weight_loading)

            # Log any key mismatches
            if hasattr(result, "missing_keys") and result.missing_keys:
                logger.warning(f"Node '{node.name}' missing keys: {result.missing_keys}")
            if hasattr(result, "unexpected_keys") and result.unexpected_keys:
                logger.warning(f"Node '{node.name}' unexpected keys: {result.unexpected_keys}")

        # Report nodes without saved weights
        if missing_keys:
            logger.info(
                f"Nodes without saved weights: {missing_keys}. "
                f"These nodes will use initialized weights."
            )

    @classmethod
    def load_from_file(
        cls,
        config_path: str | Path,
        weights_path: str | Path | None = None,
        strict_weight_loading: bool = True,
        device: str | None = None,
        config_overrides: list[str] | dict[str, Any] | None = None,
    ) -> CuvisPipeline:
        """Load pipeline from configuration and optionally restore weights.

        The method is divided into two sections:
        1. Build Pipeline: Reconstruct pipeline structure from YAML
        2. Restore Weights: Optionally load trained weights (if weights_path is provided)

        Args:
            config_path: Path to YAML config file
            weights_path: Path to .pt weights file (optional, if None weights are not loaded)
            strict_weight_loading: If True, raise error on state_dict key mismatch
            device: Device to load weights to
            config_overrides: Optional config overrides. Can be either:
                - List of strings in dot notation (e.g., ["nodes.10.params.output_dir=outputs/my_tb"])
                - Dictionary with override values

        Returns:
            CuvisPipeline: Loaded pipeline instance

        Raises:
            RuntimeError: If node class not found or instantiation fails
            FileNotFoundError: If weights file missing when weights_path is provided
        """
        # ===== SECTION 1: BUILD CANVAS =====
        pipeline = cls._build_pipeline_from_config(
            config_path=config_path,
            config_overrides=config_overrides,
            device=device,
        )

        # ===== SECTION 2: RESTORE WEIGHTS (OPTIONAL) =====
        if weights_path is not None:
            pipeline._restore_weights_from_checkpoint(
                weights_path=weights_path,
                strict_weight_loading=strict_weight_loading,
                device=device,
            )

        return pipeline

    @staticmethod
    def load_trainer_state(
        checkpoint_path: str | Path,
        device: str | None = None,
    ) -> dict[str, Any]:
        """Load optimizer/scheduler state and metadata from checkpoint.

        Returns a dict with at least a ``metadata`` key. Optimizer and scheduler
        state are included when present in the checkpoint file.
        """
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device if device else None,
            weights_only=False,
        )

        state: dict[str, Any] = {"metadata": checkpoint.get("metadata", {})}

        if "optimizer_state" in checkpoint:
            state["optimizer_state"] = checkpoint["optimizer_state"]

        if "scheduler_state" in checkpoint:
            state["scheduler_state"] = checkpoint["scheduler_state"]

        return state

    @cached_property
    def _sorted_nodes(self) -> list[Node]:
        """Cached topological sort of graph nodes.

        This cache is invalidated when connect() is called to add new edges.
        Delete this cache manually if you modify the graph structure directly.
        """
        return list(nx.topological_sort(self._graph))

    def forward(
        self,
        batch: dict | None = None,
        stage: ExecutionStage = ExecutionStage.INFERENCE,
        upto_node: Node | None = None,
        context: Context | None = None,
    ) -> dict[tuple[str, str], Any]:
        """Execute graph with context-aware filtering and port-based routing.

        Parameters
        ----------
        batch : dict, optional
            Batch from dataloader (keys = port names). These keys are distributed
            to all nodes that request them in their INPUT_SPECS.
        stage : str
            Execution stage: "train", "val", "test", "inference"
        upto_node : Node, optional
            Node to stop execution at (exclusive). Only ancestors of this
            node will execute. Useful for statistical initialization and debugging.
        context : Context, optional
            Execution context with epoch, batch_idx, and global_step. If not provided,
            a default Context will be created with the specified stage.
        **entry_inputs : Any
            Additional entry inputs keyed by "node_name.port_name"
            These override batch keys if there's a conflict.

        Returns
        -------
        dict[tuple[str, str], Any]
            All outputs keyed by (node_name, port_name) tuples
        """

        # Create or use provided context
        if context is None:
            # No context provided, create one with the stage parameter
            context = Context(stage=stage)
        # If context is provided, use it as-is (it already has the correct stage)

        # Use the stage from context for execution
        execution_stage = context.stage

        # Determine execution scope
        if upto_node is not None:
            # Validate node is in graph
            if upto_node not in self._graph:
                raise ValueError(
                    f"Node '{upto_node.name}' not found in graph. "
                    f"Available nodes: {[n.name for n in self._graph.nodes()]}"
                )

            # Execute only ancestors of upto_node (nodes that feed into it)
            ancestors = nx.ancestors(self._graph, upto_node)
            executable_nodes = [
                node
                for node in self._sorted_nodes
                if node in ancestors and node.should_execute(execution_stage)
            ]
        else:
            # Execute all nodes for this stage
            executable_nodes = [
                node for node in self._sorted_nodes if node.should_execute(execution_stage)
            ]

        port_data: dict[tuple[str, str], Any] = {}
        batch = batch or {}

        # Create cache key for validation
        batch_keys = frozenset(batch.keys()) if batch else frozenset()
        upto_node_id = id(upto_node) if upto_node is not None else None
        cache_key = (execution_stage, batch_keys, upto_node_id)

        # Check if validation has already been performed for this configuration
        if cache_key not in self._validation_cache:
            # Perform validation and cache result
            self._validate_graph_inputs(batch, execution_stage, executable_nodes)
            self._validation_cache[cache_key] = None

        # Execute nodes in topological order
        for node in executable_nodes:
            # Gather inputs from both batch and connections
            node_inputs = self._gather_node_inputs(node, port_data, batch)

            # Validate inputs (if enabled)
            if self.strict_runtime_io_validation:
                self._validate_runtime_inputs(node, node_inputs)

            # Execute node with context
            outputs = node.forward(**node_inputs, context=context)

            if not isinstance(outputs, dict):
                raise TypeError(f"Node '{node.name}' must return dict, got {type(outputs)!r}")

            # Validate outputs (if enabled)
            if self.strict_runtime_io_validation:
                self._validate_runtime_outputs(node, outputs)

            # Store outputs
            for port_name, value in outputs.items():
                key = (node.name, port_name)
                if key in port_data:
                    raise ValueError(f"Duplicate output key detected: {key}")
                port_data[key] = value

        return port_data

    def _validate_graph_inputs(
        self,
        batch: dict,
        _stage: str,
        executable_nodes: list[Node],
    ) -> None:
        """Validate that executable nodes have their required inputs satisfied.

        Raises RuntimeError for nodes that will execute with missing inputs.

        Parameters
        ----------
        batch : dict
            Batch data from dataloader
        _stage : str
            Execution stage (unused; kept for interface compatibility)
        executable_nodes : list[Node]
            Nodes that will execute in this stage
        """
        for node in executable_nodes:
            # Check which required inputs are missing
            missing_inputs = []
            for port_name, spec in getattr(node, "INPUT_SPECS", {}).items():
                # Handle list-based specs (variadic ports)
                if isinstance(spec, list):
                    spec = spec[0]

                # Skip optional ports
                if getattr(spec, "optional", False):
                    continue

                # Check if input can be satisfied
                can_be_satisfied = False

                # Check if available in batch
                if port_name in batch:
                    can_be_satisfied = True

                # Check if connected from a predecessor
                for predecessor_node in self._graph.predecessors(node):
                    for _, edge_data in self._graph[predecessor_node][node].items():
                        if edge_data["to_port"] == port_name:
                            can_be_satisfied = True
                            break
                    if can_be_satisfied:
                        break

                if not can_be_satisfied:
                    missing_inputs.append(port_name)

            if missing_inputs:
                raise RuntimeError(f"Node '{node.name}' missing required inputs: {missing_inputs}")

    def _gather_node_inputs(
        self,
        node: Node,
        port_data: dict[tuple[str, str], Any],
        batch: dict,
    ) -> dict[str, Any]:
        """Gather inputs for a node from batch and predecessor connections.

        Parameters
        ----------
        node : Node
            Node to gather inputs for
        port_data : dict[tuple[str, str], Any]
            Accumulated port data from executed nodes
        batch : dict
            Batch data from dataloader

        Returns
        -------
        dict[str, Any]
            Gathered inputs keyed by port name
        """
        node_inputs: dict[str, Any] = {}

        # Get from batch
        for port_name in getattr(node, "INPUT_SPECS", {}):
            if (node.name, port_name) in port_data:
                node_inputs[port_name] = port_data[(node.name, port_name)]
            elif port_name in batch:
                node_inputs[port_name] = batch[port_name]

        # Get from predecessor connections
        for predecessor_node in self._graph.predecessors(node):
            for _, edge_data in self._graph[predecessor_node][node].items():
                from_port = edge_data["from_port"]
                to_port = edge_data["to_port"]

                if (predecessor_node.name, from_port) in port_data:
                    source_data = port_data[(predecessor_node.name, from_port)]

                    port_spec = getattr(node, "INPUT_SPECS", {}).get(to_port)
                    is_variadic = isinstance(port_spec, list)
                    if is_variadic:
                        # Variadic port - collect into list
                        if to_port not in node_inputs:
                            node_inputs[to_port] = []
                        node_inputs[to_port].append(source_data)
                    else:
                        node_inputs[to_port] = source_data

        return node_inputs

    def _has_required_inputs(self, node: Node, node_inputs: dict) -> bool:
        """Check if all required inputs are present.

        Parameters
        ----------
        node : Node
            Node to check inputs for
        node_inputs : dict
            Gathered inputs for the node

        Returns
        -------
        bool
            True if all required inputs are present
        """
        for port_name, spec in getattr(node, "INPUT_SPECS", {}).items():
            # Handle list-based specs (variadic ports)
            if isinstance(spec, list):
                spec = spec[0]

            if not getattr(spec, "optional", False) and port_name not in node_inputs:
                return False
        return True

    def _is_optional_port(self, node: Node, port_name: str) -> bool:
        """Check if a port is optional.

        Parameters
        ----------
        node : Node
            Node to check
        port_name : str
            Name of the port

        Returns
        -------
        bool
            True if port is optional
        """
        spec = getattr(node, "INPUT_SPECS", {}).get(port_name)
        if isinstance(spec, list):
            spec = spec[0]
        return getattr(spec, "optional", False)

    def _validate_runtime_inputs(self, node: Node, inputs: dict[str, Any]) -> None:
        """Validate input values match INPUT_SPECS at runtime.

        Parameters
        ----------
        node : Node
            Node whose inputs are being validated
        inputs : dict[str, Any]
            Input values keyed by port name

        Raises
        ------
        RuntimeError
            If a required input is missing
        TypeError
            If an input dtype or shape doesn't match the spec
        """

        for port_name, spec in getattr(node, "INPUT_SPECS", {}).items():
            # Check if variadic port
            is_variadic = isinstance(spec, list)
            if is_variadic:
                spec = spec[0]  # Get element spec

            if port_name not in inputs:
                if not getattr(spec, "optional", False):
                    raise RuntimeError(f"Node '{node.name}' missing required input '{port_name}'")
                continue

            value = inputs[port_name]

            # Handle variadic ports: validate each element
            if is_variadic:
                if not isinstance(value, list):
                    raise TypeError(
                        f"Node '{node.name}' variadic input '{port_name}' must receive a "
                        f"list, got {type(value)}"
                    )
                for i, item in enumerate(value):
                    self._validate_value_against_spec(
                        item, spec, node, f"{port_name}[{i}]", "input"
                    )
            else:
                self._validate_value_against_spec(value, spec, node, port_name, "input")

    def _validate_runtime_outputs(self, node: Node, outputs: dict[str, Any]) -> None:
        """Validate output values match OUTPUT_SPECS at runtime.

        Parameters
        ----------
        node : Node
            Node whose outputs are being validated
        outputs : dict[str, Any]
            Output values keyed by port name

        Raises
        ------
        RuntimeError
            If a required output is missing
        PortCompatibilityError
            If an output dtype or shape doesn't match the spec and fails downstream connections
        """

        # Check all required outputs are present and validate them
        for port_name, spec in getattr(node, "OUTPUT_SPECS", {}).items():
            # Check if variadic port (though outputs are typically not variadic)
            is_variadic = isinstance(spec, list)
            if is_variadic:
                spec = spec[0]  # Get element spec

            if port_name not in outputs:
                # Skip optional outputs if not present
                if getattr(spec, "optional", False):
                    continue
                raise RuntimeError(
                    f"Node '{node.name}' did not produce required output '{port_name}'"
                )

            value = outputs[port_name]

            # Handle variadic ports: validate each element
            if is_variadic:
                if not isinstance(value, list):
                    raise TypeError(
                        f"Node '{node.name}' variadic output '{port_name}' must be a "
                        f"list, got {type(value)}"
                    )
                for i, item in enumerate(value):
                    self._validate_value_against_spec(
                        item, spec, node, f"{port_name}[{i}]", "output"
                    )
            else:
                self._validate_value_against_spec(value, spec, node, port_name, "output")

    def _validate_value_against_spec(
        self, value: Any, spec: Any, node: Node, port_name: str, port_type: str
    ) -> None:
        """Validate a runtime value against its PortSpec using connection-time logic.

        Parameters
        ----------
        value : Any
            The actual runtime value
        spec : PortSpec
            The specification to validate against
        node : Node
            The node this port belongs to
        port_name : str
            Name of the port
        port_type : str
            Either "input" or "output"

        Raises
        ------
        PortCompatibilityError
            If value doesn't match spec dtype or shape, with full connection details
        """

        # Create a temporary PortSpec from the actual value
        actual_spec = self._infer_spec_from_value(value)

        # Use the SAME compatibility check as connection time
        is_compatible, message = actual_spec.is_compatible_with(spec, node, node)

        if not is_compatible:
            # For outputs, find all downstream connections to provide detailed error
            if port_type == "output":
                connections = self._get_output_connections(node, port_name)
                if connections:
                    # Format connection details
                    connection_details = []
                    for target_node, target_port in connections:
                        connection_details.append(f"  → {target_node.name}.{target_port}")

                    raise PortCompatibilityError(
                        f"Runtime validation failed for connection(s):\n"
                        f"  Source: {node.name}.{port_name}\n"
                        f"  Target(s):\n" + "\n".join(connection_details) + "\n"
                        f"  Error: {message}"
                    )

            # Fallback for inputs or when no connections found
            raise PortCompatibilityError(
                f"Node '{node.name}' {port_type} '{port_name}' validation failed: {message}"
            )

    def _get_output_connections(self, source_node: Node, port_name: str) -> list[tuple[Node, str]]:
        """Get all downstream connections for a node's output port.

        Parameters
        ----------
        source_node : Node
            Source node
        port_name : str
            Output port name

        Returns
        -------
        list[tuple[Node, str]]
            List of (target_node, target_port_name) tuples
        """
        connections = []
        for target_node in self._graph.successors(source_node):
            for _, edge_data in self._graph[source_node][target_node].items():
                if edge_data.get("from_port") == port_name:
                    connections.append((target_node, edge_data["to_port"]))
        return connections

    def _infer_spec_from_value(self, value: Any) -> Any:
        """Infer a PortSpec from an actual runtime value.

        Parameters
        ----------
        value : Any
            Runtime value to infer spec from

        Returns
        -------
        PortSpec
            Inferred specification
        """
        from cuvis_ai.pipeline.ports import PortSpec

        if hasattr(value, "dtype") and hasattr(value, "shape"):
            # It's a tensor-like object
            return PortSpec(
                dtype=value.dtype, shape=tuple(value.shape), description="Runtime value"
            )
        else:
            # Non-tensor type
            return PortSpec(dtype=type(value), shape=(), description="Runtime value")

    @cached_property
    def torch_layers(self) -> nn.ModuleList:
        """Torch modules stored in the graph's nodes, packaged as an nn.ModuleList."""
        # If you'll mutate self.nodes after construction, delete this cache:
        #   del self.__dict__['torch_layers']
        modules = [node for node in self._graph.nodes() if isinstance(node, nn.Module)]
        return nn.ModuleList(modules)

    def to(self, *args: Any, **kwargs: Any) -> CuvisPipeline:
        """Move all torch-backed nodes to the requested device/dtype."""
        self.torch_layers.to(*args, **kwargs)
        return self

    def cuda(self, device: int | torch.device | None = None) -> CuvisPipeline:
        """CUDA convenience wrapper mirroring nn.Module.cuda."""
        self.torch_layers.cuda(device=device)
        return self

    def cpu(self) -> CuvisPipeline:
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
        Names are prefixed with the node name to avoid collisions.
        """
        seen_params: set[int] = set()
        for node in self._graph.nodes():
            if not isinstance(node, nn.Module):
                continue
            iterator = node.named_parameters(recurse=recurse)
            for local_name, p in iterator:
                pid = id(p)
                if pid in seen_params:
                    continue
                seen_params.add(pid)
                if require_grad is None or p.requires_grad is require_grad:
                    yield f"{node.name}{sep}{local_name}", p

    def get_input_specs(self) -> dict[str, dict]:
        """Get input specifications for pipeline entry points.

        Returns specifications for ports that are not connected from any predecessor
        (i.e., they need to be provided in the batch input).

        Time Complexity: O(N + E) where N=nodes, E=edges

        Returns
        -------
        dict[str, dict]
            Dictionary mapping port names to their specifications.
            Each spec contains: name, dtype, shape, required

        Example
        -------
        >>> pipeline.get_input_specs()
        {
            'cube': {
                'name': 'cube',
                'dtype': 'float32',
                'shape': [-1, -1, -1, -1],  # [B, H, W, C]
                'required': True
            }
        }
        """
        # Step 1: Build set of connected input ports - O(E)
        connected_inputs: set[tuple[Node, str]] = set()
        for _source_node, target_node, edge_data in self._graph.edges(data=True):
            to_port = edge_data.get("to_port")
            if to_port:
                connected_inputs.add((target_node, to_port))

        # Step 2: Find unconnected input ports - O(N * avg_ports)
        input_specs = {}
        for node in self._graph.nodes():
            for port_name, port_spec in getattr(node, "INPUT_SPECS", {}).items():
                # Handle variadic ports
                if isinstance(port_spec, list):
                    port_spec = port_spec[0]

                # O(1) lookup in set
                if (node, port_name) not in connected_inputs:
                    spec_dict = {
                        "name": port_name,
                        "dtype": self._dtype_to_string(port_spec.dtype),
                        "shape": list(port_spec.shape) if hasattr(port_spec, "shape") else [-1],
                        "required": not getattr(port_spec, "optional", False),
                    }
                    input_specs[port_name] = spec_dict

        return input_specs

    def get_output_specs(self) -> dict[str, dict]:
        """Get output specifications for pipeline exit points.

        Returns specifications for ports that are not connected to any successor
        (i.e., they are final outputs of the pipeline).

        Time Complexity: O(N + E) where N=nodes, E=edges

        Returns
        -------
        dict[str, dict]
            Dictionary mapping output keys ("node.port") to their specifications.
            Each spec contains: name, dtype, shape, required

        Example
        -------
        >>> pipeline.get_output_specs()
        {
            'selector.selected': {
                'name': 'selector.selected',
                'dtype': 'float32',
                'shape': [-1, -1, -1, -1],
                'required': False
            },
            'decider.decisions': {
                'name': 'decider.decisions',
                'dtype': 'bool',
                'shape': [-1],
                'required': False
            }
        }
        """
        # Step 1: Build set of connected output ports - O(E)
        connected_outputs: set[tuple[Node, str]] = set()
        for source_node, _target_node, edge_data in self._graph.edges(data=True):
            from_port = edge_data.get("from_port")
            if from_port:
                connected_outputs.add((source_node, from_port))

        # Step 2: Find unconnected output ports - O(N * avg_ports)
        output_specs = {}
        for node in self._graph.nodes():
            for port_name, port_spec in getattr(node, "OUTPUT_SPECS", {}).items():
                # Handle variadic ports
                if isinstance(port_spec, list):
                    port_spec = port_spec[0]

                # O(1) lookup in set
                if (node, port_name) not in connected_outputs:
                    output_key = f"{node.name}.{port_name}"
                    spec_dict = {
                        "name": output_key,
                        "dtype": self._dtype_to_string(port_spec.dtype),
                        "shape": list(port_spec.shape) if hasattr(port_spec, "shape") else [-1],
                        "required": False,
                    }
                    output_specs[output_key] = spec_dict

        return output_specs

    def _dtype_to_string(self, dtype) -> str:
        """Convert dtype to string representation.

        Time Complexity: O(1)

        Parameters
        ----------
        dtype : type or torch.dtype or np.dtype
            Data type to convert

        Returns
        -------
        str
            String representation like "float32", "int32", etc.
        """
        import numpy as np

        # Handle PyTorch dtypes
        if isinstance(dtype, torch.dtype):
            dtype_map = {
                torch.float32: "float32",
                torch.float64: "float64",
                torch.int32: "int32",
                torch.int64: "int64",
                torch.uint8: "uint8",
                torch.bool: "bool",
                torch.float16: "float16",
            }
            return dtype_map.get(dtype, "float32")

        # Handle numpy dtypes
        if isinstance(dtype, type) and issubclass(dtype, np.generic):
            return dtype.__name__

        # Handle Python types
        if dtype is float:
            return "float32"
        if dtype is int:
            return "int64"
        if dtype is bool:
            return "bool"

        # Default
        return "float32"

    @property
    def nodes(self) -> NodeView:
        return self._graph.nodes

    def unfreeze_nodes_by_name(self, node_names: list[str]) -> None:
        """Unfreeze specific nodes in this pipeline by their names.

        This method validates that all requested nodes exist in the pipeline and
        unfreezes them for gradient training. Nodes must have an 'unfreeze' method.

        Parameters
        ----------
        node_names : list[str]
            List of node names to unfreeze. Names must match exactly.

        Raises
        ------
        ValueError
            If any specified node name is not found in the pipeline

        Examples
        --------
        >>> pipeline.unfreeze_nodes_by_name(["SoftChannelSelector", "RXLogitHead"])
        """
        if not node_names:
            return

        # Build mapping of available nodes
        available_nodes = {node.name: node for node in self.nodes()}

        # Validate all requested nodes exist
        missing_nodes = [name for name in node_names if name not in available_nodes]
        if missing_nodes:
            raise ValueError(
                f"Trainable nodes not found in pipeline: {missing_nodes}. "
                f"Available nodes: {list(available_nodes.keys())}"
            )

        # Unfreeze requested nodes
        for node_name in node_names:
            node = available_nodes[node_name]
            if hasattr(node, "unfreeze"):
                node.unfreeze()
