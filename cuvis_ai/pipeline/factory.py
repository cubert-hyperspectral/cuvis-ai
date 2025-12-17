"""Factory helpers for constructing pipelines from configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

from cuvis_ai.pipeline.pipeline import CuvisPipeline
from cuvis_ai.utils.node_registry import NodeRegistry


class PipelineBuilder:
    """
    Build ``CuvisPipeline`` instances from YAML configuration files or dictionaries.

    Supports:
    - NodeRegistry for node instantiation
    - OmegaConf interpolation between configs
    - Counter-based node naming
    - Connection validation

    Example:
        builder = PipelineBuilder()
        pipeline = builder.build_from_config(
            pipeline_config="configs/pipeline.yaml",
        )
    """

    def __init__(
        self, node_registry: type | None = None, default_pipeline_dir: str = "configs/pipeline"
    ) -> None:
        """
        Initialize PipelineBuilder.

        Args:
            node_registry: Registry class to use (defaults to NodeRegistry)
            default_pipeline_dir: Default directory for pipeline configs when using short names
                               (relative to current working directory)
        """
        self.node_registry = node_registry or NodeRegistry
        self.default_pipeline_dir = default_pipeline_dir

    def build_from_config(
        self,
        pipeline_config: str | Path | dict[str, Any],
    ) -> CuvisPipeline:
        """
        Build pipeline from YAML configuration.

        Args:
            pipeline_config: Path to pipeline YAML or dict config of connections and nodes

        Returns:
            Constructed CuvisPipeline instance

        Example:
            pipeline = builder.build_from_config(
                "configs/channel_selector.yaml",
                "configs/lentils_experiment.yaml"
            )
        """
        # Load configurations
        pipeline_cfg = self._load_config(pipeline_config)

        # Extract pipeline name from metadata or use default
        pipeline_name = pipeline_cfg.get("metadata", {}).get("name", "Pipeline")

        # Create pipeline instance
        pipeline = CuvisPipeline(name=pipeline_name)

        # Instantiate nodes
        nodes = self._instantiate_nodes(
            pipeline_cfg.nodes,
        )

        # Build connections
        self._build_connections(pipeline, nodes, pipeline_cfg.connections)

        return pipeline

    def _load_config(self, config: str | Path | dict[str, Any]) -> DictConfig:
        """
        Load configuration from file or dict.

        Supports short names (e.g., "rx_statistical") which will be resolved to
        "{default_pipeline_dir}/{short_name}.yaml".

        Args:
            config: Path to YAML file, short name, or dict

        Returns:
            OmegaConf DictConfig

        Raises:
            FileNotFoundError: If config file not found
        """
        if isinstance(config, (str, Path)):
            config_path = self._resolve_config_path(config)

            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        else:
            config_dict = config

        return OmegaConf.create(config_dict)

    def _resolve_config_path(self, config: str | Path) -> Path:
        """
        Resolve configuration path with support for short names.

        Resolution strategy:
        1. If path exists as-is, use it
        2. If it looks like a short name (no path separators), try default pipeline dir
        3. Add .yaml extension if missing and retry

        Args:
            config: Path string or Path object

        Returns:
            Resolved Path object

        Examples:
            "rx_statistical" -> "configs/pipeline/rx_statistical.yaml"
            "configs/pipeline/rx_statistical.yaml" -> "configs/pipeline/rx_statistical.yaml"
            "/abs/path/to/config.yaml" -> "/abs/path/to/config.yaml"
        """
        config_path = Path(config)

        # Strategy 1: Path exists as-is
        if config_path.exists():
            return config_path

        # Strategy 2: Short name detection and resolution
        # A short name has no path separators (/, \) and may not have .yaml extension
        config_str = str(config)
        is_short_name = "/" not in config_str and "\\" not in config_str

        if is_short_name:
            # Try resolving in default pipeline directory
            # Add .yaml extension if not present
            if not config_str.endswith(".yaml") and not config_str.endswith(".yml"):
                config_str = f"{config_str}.yaml"

            resolved_path = Path(self.default_pipeline_dir) / config_str
            if resolved_path.exists():
                return resolved_path

        # Strategy 3: Try adding .yaml extension to original path
        if not config_path.suffix:
            with_yaml = config_path.with_suffix(".yaml")
            if with_yaml.exists():
                return with_yaml

        # Return original path (will trigger FileNotFoundError in _load_config)
        return config_path

    def _instantiate_nodes(
        self,
        node_configs: Any,
    ) -> list:
        """
        Instantiate all nodes from configuration.

        Args:
            node_configs: List of node configurations

        Returns:
            List of tuples (base_name, node_instance)
        """
        nodes = []

        for node_cfg in node_configs:
            # Resolve OmegaConf interpolations
            resolved_cfg = OmegaConf.to_container(node_cfg, resolve=True)

            # Type check - should be dict after resolution
            if not isinstance(resolved_cfg, dict):
                raise TypeError(
                    f"Expected dict after resolving node config, got {type(resolved_cfg)}"
                )

            # Get node class from registry
            node_class = self.node_registry.get(resolved_cfg["class"])

            # Extract parameters
            params = resolved_cfg.get("params", {})

            # Instantiate node (all nodes accept **kwargs for name, execution_stages, etc.)
            node = node_class(**params)

            # Store with base name (counter will be assigned by pipeline.connect)
            base_name = resolved_cfg["name"]
            nodes.append((base_name, node))

        return nodes

    def _build_connections(
        self, pipeline: CuvisPipeline, nodes: list, connection_configs: list
    ) -> None:
        """
        Build connections between nodes.

        Args:
            pipeline: Pipeline instance to connect nodes to
            nodes: List of tuples (base_name, node_instance)
            connection_configs: List of connection specifications
        """
        # Track node instances by base name
        # For duplicate base names, we store only the first occurrence for connections
        node_instances = {}

        # First pass: add all nodes to pipeline to establish names
        for base_name, node in nodes:
            # Set the node's base name (before pipeline assigns counter)
            node._name = base_name

            # Add node to pipeline (this assigns the actual name with counter)
            pipeline._assign_counter_and_add_node(node)

            # Store mapping of base_name to node instance
            # Only store the first occurrence for each base_name (for connections)
            if base_name not in node_instances:
                node_instances[base_name] = node

        # Second pass: wire up connections using actual names
        for conn_cfg in connection_configs:
            self._connect_ports(pipeline, node_instances, conn_cfg)

    def _connect_ports(
        self, pipeline: CuvisPipeline, node_instances: dict[str, Any], conn_cfg: dict[str, str]
    ) -> None:
        """
        Connect two ports based on configuration.

        Args:
            pipeline: Pipeline instance
            node_instances: Mapping of base names to node instances
            conn_cfg: Connection config with 'from' and 'to' keys
        """
        # Parse connection specification
        # Format: "node_name.outputs.port_name"
        from_spec = conn_cfg["from"]
        to_spec = conn_cfg["to"]

        # Extract components
        from_parts = from_spec.split(".")
        to_parts = to_spec.split(".")

        if len(from_parts) != 3 or from_parts[1] != "outputs":
            raise ValueError(
                f"Invalid 'from' specification: {from_spec}. "
                f"Expected format: 'node_name.outputs.port_name'"
            )

        if len(to_parts) != 3 or to_parts[1] != "inputs":
            raise ValueError(
                f"Invalid 'to' specification: {to_spec}. "
                f"Expected format: 'node_name.inputs.port_name'"
            )

        from_node_name = from_parts[0]
        from_port_name = from_parts[2]
        to_node_name = to_parts[0]
        to_port_name = to_parts[2]

        # Get actual node instances
        if from_node_name not in node_instances:
            raise ValueError(f"Source node not found: {from_node_name}")
        if to_node_name not in node_instances:
            raise ValueError(f"Target node not found: {to_node_name}")

        from_node = node_instances[from_node_name]
        to_node = node_instances[to_node_name]

        # Get ports
        from_port = getattr(from_node.outputs, from_port_name, None)
        to_port = getattr(to_node.inputs, to_port_name, None)

        if from_port is None:
            raise ValueError(f"Output port '{from_port_name}' not found on node '{from_node_name}'")
        if to_port is None:
            raise ValueError(f"Input port '{to_port_name}' not found on node '{to_node_name}'")

        # Wire the connection (pipeline will validate)
        pipeline.connect(from_port, to_port)
