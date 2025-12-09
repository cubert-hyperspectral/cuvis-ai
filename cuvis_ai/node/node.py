from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from types import SimpleNamespace
from typing import Any

from torch import nn

from cuvis_ai.pipeline.ports import InputPort, OutputPort, PortSpec
from cuvis_ai.utils.serializer import Serializable
from cuvis_ai.utils.types import ExecutionStage, InputStream


class Node(nn.Module, ABC, Serializable):
    """
    Abstract class for data preprocessing.

    Node Serialization Requirements
    ================================

    All nodes MUST support serialization for pipeline save/load:

    1. Nodes inheriting from Node (nn.Module):
       - Already have state_dict() and load_state_dict() from PyTorch
       - Ensure all learnable parameters are registered properly
       - Use register_buffer() for non-trainable state

    2. Stateless nodes:
       - state_dict() returns empty dict {} (default PyTorch behavior)
       - load_state_dict() is no-op (default PyTorch behavior)

    3. Statistical nodes (requires_initial_fit=True):
       - Store fitted parameters as buffers (register_buffer())
       - Unfreeze converts buffers to nn.Parameters
       - Both states serialize correctly via state_dict()

    4. Custom serialization needs:
       - Override state_dict() to include custom state
       - Override load_state_dict() to restore custom state
       - Must call super().state_dict() / super().load_state_dict()

    Example - Custom State:
        >>> class CustomNode(Node):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.custom_data = {"key": "value"}
        ...
        ...     def state_dict(self):
        ...         state = super().state_dict()
        ...         state["custom_data"] = self.custom_data
        ...         return state
        ...
        ...     def load_state_dict(self, state_dict):
        ...         self.custom_data = state_dict.pop("custom_data", {})
        ...         super().load_state_dict(state_dict)
    """

    INPUT_SPECS: dict[str, PortSpec | list[PortSpec]] = {}
    OUTPUT_SPECS: dict[str, PortSpec | list[PortSpec]] = {}

    def __init__(
        self,
        name: str | None = None,
        execution_stages: set[ExecutionStage] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initialize node with execution stage control.

        Parameters
        ----------
        name : str, optional
            Custom name for the node. If not provided, uses class name.
            Useful for loss/metric nodes to enable semantic logging names.
        execution_stages : set[ExecutionStage]
            When to execute this node:
            - ExecutionStage.ALWAYS: Execute in all stages (default)
            - ExecutionStage.TRAIN: Only during training
            - ExecutionStage.VAL: Only during validation
            - ExecutionStage.TEST: Only during testing
            - ExecutionStage.INFERENCE: Only during inference
            - {ExecutionStage.TRAIN, ExecutionStage.VAL}: Multiple stages
        """
        # Allow subclasses to forward name/execution_stages inside kwargs without duplication
        if kwargs:
            name = kwargs.pop("name", name)
            execution_stages = kwargs.pop("execution_stages", execution_stages)
        if execution_stages is None:
            execution_stages = {ExecutionStage.ALWAYS}
        # Initialize Serializable first to capture hparams
        Serializable.__init__(self, *args, **kwargs)
        # Then initialize nn.Module without any args/kwargs
        nn.Module.__init__(self)

        if name is None:
            name = type(self).__name__
        # Store custom name
        self._name = name
        self._pipeline_counter: int | None = None

        # Execution stages
        self.execution_stages = set(execution_stages)

        self._statistically_initialized = False
        self.freezed = False
        self._input_ports: dict[str, InputPort] = {}
        self._output_ports: dict[str, OutputPort] = {}
        self._create_ports()

    @property
    def name(self) -> str:
        """Get node name (base name with optional counter suffix)."""
        if self._pipeline_counter is None:
            return self._name
        if self._pipeline_counter == 0:
            return self._name
        return f"{self._name}-{self._pipeline_counter}"

    @name.setter
    def name(self, value: str) -> None:  # noqa: D401
        """Disallow mutation of node names to avoid breaking graph keys."""
        raise AttributeError(f"Node name is immutable. Cannot change '{self.name}' to '{value}'.")

    @property
    def requires_initial_fit(self) -> bool:
        """Auto-detect whether the node implements statistical initialization."""
        override = getattr(self, "_requires_initial_fit_override", None)
        if override is not None:
            return bool(override)

        impl = getattr(self.__class__, "statistical_initialization", None)
        return callable(impl) and impl is not Node.statistical_initialization

    def statistical_initialization(self, input_stream: InputStream) -> None:
        """Statistical initialization from a port-based input stream."""
        return None

    def fit(self, input_stream: InputStream) -> None:
        """Backward-compatible alias for `statistical_initialization`."""
        warnings.warn(
            "Node.fit() is deprecated; use statistical_initialization() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        impl = getattr(self.__class__, "statistical_initialization", None)
        if self.requires_initial_fit and (impl is None or impl is Node.statistical_initialization):
            raise NotImplementedError(
                f"{self.__class__.__name__} requires statistical_initialization() implementation"
            )
        self.statistical_initialization(input_stream)

    def unfreeze(self) -> None:
        """Enable gradient computation for this node's parameters.

        For statistical nodes, this method should be overridden to convert
        buffers to nn.Parameters. The base implementation enables gradients
        for any existing parameters.

        After statistical initialization with fit(), nodes store their learned
        values as buffers. Call unfreeze() to convert them to trainable parameters
        for gradient-based optimization.

        Example
        -------
        >>> node.fit(input_stream)  # Statistical initialization -> buffers
        >>> node.unfreeze()  # Convert buffers -> nn.Parameters
        >>> # Now node can be trained with gradient descent
        """
        self.freezed = False
        self.requires_grad_(True)

    def freeze(self) -> None:
        """Disable gradient computation for this node's parameters.

        This disables requires_grad for all parameters in the node, preventing
        gradient updates during training. Use this after statistical initialization
        if you want to keep the node frozen, or to freeze a previously unfrozen node.

        Example
        -------
        >>> node.fit(input_stream)  # Statistical initialization
        >>> node.freeze()  # Keep frozen (already frozen by default)
        >>> # Or after unfreezing:
        >>> node.unfreeze()  # Enable gradients
        >>> node.freeze()  # Disable gradients again
        """
        self.freezed = True
        self.requires_grad_(False)

    def should_execute(self, stage: ExecutionStage | str) -> bool:
        """Check if node should execute in given stage.

        Parameters
        ----------
        stage : ExecutionStage | str
            Execution stage (enum or string): "train", "val", "test", "inference"

        Returns
        -------
        bool
            True if node should execute in this stage
        """
        # Convert string to enum if needed
        if isinstance(stage, str):
            try:
                stage = ExecutionStage(stage)
            except ValueError:
                return False

        return ExecutionStage.ALWAYS in self.execution_stages or stage in self.execution_stages

    def validate_serialization_support(self) -> tuple[bool, str]:
        """Validate that node can be serialized.

        Checks:
        - Has state_dict() method
        - Has load_state_dict() method
        - state_dict() returns a dict
        - No exceptions during state_dict() call

        Returns
        -------
        tuple[bool, str]
            Tuple of (is_valid, error_message)

        Example
        -------
        >>> node = MinMaxNormalizer()
        >>> is_valid, message = node.validate_serialization_support()
        >>> assert is_valid, message
        """
        # Check if state_dict exists
        if not hasattr(self, "state_dict"):
            return False, f"Node {self.name} missing state_dict() method"

        # Check if load_state_dict exists
        if not hasattr(self, "load_state_dict"):
            return False, f"Node {self.name} missing load_state_dict() method"

        # Try to call state_dict (should not raise)
        try:
            state = self.state_dict()
            if not isinstance(state, dict):
                return False, f"Node {self.name} state_dict() must return dict, got {type(state)}"
        except Exception as e:
            return False, f"Node {self.name} state_dict() failed: {e}"

        return True, "OK"

    @abstractmethod
    def forward(self, **inputs: Any) -> dict[str, Any]:
        """Execute node computation returning a dictionary of named outputs."""
        raise NotImplementedError

    @staticmethod
    def consume_base_kwargs(
        kwargs: dict[str, Any], default_stages: set[ExecutionStage] | None = None
    ) -> tuple[str | None, set[ExecutionStage] | None]:
        """Extract base Node kwargs centrally to avoid double-passing.

        Subclasses can pop name/execution_stages here before calling super().__init__.
        """
        name = kwargs.pop("name", None)
        execution_stages = kwargs.pop("execution_stages", default_stages)
        return name, execution_stages

    def _create_ports(self) -> None:
        """Create port proxy objects from class-level specifications."""
        for port_name, port_spec in self.INPUT_SPECS.items():
            if port_name in self._input_ports:
                raise AttributeError(
                    f"Cannot create input port '{port_name}'; attribute already exists."
                )
            port = InputPort(self, port_name, port_spec)
            self._input_ports[port_name] = port

        for port_name, port_spec in self.OUTPUT_SPECS.items():
            if port_name in self._output_ports:
                raise AttributeError(
                    f"Cannot create output port '{port_name}'; attribute already exists."
                )
            port = OutputPort(self, port_name, port_spec)
            self._output_ports[port_name] = port

    @property
    def inputs(self) -> SimpleNamespace:
        """Access input ports: node.inputs.portname"""
        return SimpleNamespace(**self._input_ports)

    @property
    def outputs(self) -> SimpleNamespace:
        """Access output ports: node.outputs.portname"""
        return SimpleNamespace(**self._output_ports)

    # def __getattr__(self, name):
    #     # Prevent infinite recursion during initialization
    #     if name.startswith('_'):
    #         raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    #     # Check if it's a unique port name
    #     in_inputs = name in self._input_ports
    #     in_outputs = name in self._output_ports

    #     if in_inputs and in_outputs:
    #         # Conflict - require explicit namespace
    #         raise AttributeError(
    #             f"Port '{name}' exists in both inputs and outputs. "
    #             f"Use {self.name}.inputs.{name} or {self.name}.outputs.{name}"
    #         )
    #     elif in_inputs:
    #         return self._input_ports[name]
    #     elif in_outputs:
    #         return self._output_ports[name]
    #     else:
    #         # Not a port, let normal AttributeError propagate
    #         raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def __getattr__(self, name: str) -> Any:
        # First, let PyTorch's nn.Module handle its own attributes
        # (parameters, buffers, modules, etc.) before checking for ports.
        # This is critical because nn.Module stores parameters in _parameters dict
        # and expects __getattr__ to not interfere with that lookup.

        # Try to get from nn.Module's dictionaries first (parameters, buffers, modules)
        # This avoids interfering with PyTorch's parameter management
        try:
            # Check _parameters, _buffers, _modules in order (like nn.Module does)
            modules = object.__getattribute__(self, "__dict__")
            if "_parameters" in modules and name in modules["_parameters"]:
                return modules["_parameters"][name]
            if "_buffers" in modules and name in modules["_buffers"]:
                return modules["_buffers"][name]
            if "_modules" in modules and name in modules["_modules"]:
                return modules["_modules"][name]
        except AttributeError:
            pass

        # Prevent infinite recursion for non-PyTorch underscore attributes
        if name.startswith("_"):
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

        # Now check if it's a port
        try:
            input_ports = object.__getattribute__(self, "_input_ports")
            output_ports = object.__getattribute__(self, "_output_ports")
        except AttributeError as err:
            # Ports not initialized yet
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'") from err

        # Check if it's a unique port name
        in_inputs = name in input_ports
        in_outputs = name in output_ports

        if in_inputs and in_outputs:
            # Conflict - require explicit namespace
            raise AttributeError(
                f"Port '{name}' exists in both inputs and outputs. "
                f"Use {self.name}.inputs.{name} or {self.name}.outputs.{name}"
            )
        elif in_inputs:
            return input_ports[name]
        elif in_outputs:
            return output_ports[name]
        else:
            # Not a port or parameter, let normal AttributeError propagate
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")

    def load_state_dict(self, state_dict, strict: bool = True) -> Any:
        result = super().load_state_dict(state_dict, strict=strict)
        self._infer_fitted_state_from_loaded_weights(state_dict)
        return result

    def _infer_fitted_state_from_loaded_weights(self, state_dict) -> None:
        """Infer statistical initialization status from loaded weights."""
        if not state_dict:
            return

        if self.requires_initial_fit:
            if hasattr(self, "_statistically_initialized"):
                self._statistically_initialized = True
