from __future__ import annotations

import uuid
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
        if execution_stages is None:
            execution_stages = {ExecutionStage.ALWAYS}
        # Initialize Serializable first to capture hparams
        Serializable.__init__(self, *args, **kwargs)
        # Then initialize nn.Module without any args/kwargs
        nn.Module.__init__(self)
        self.uuid = str(uuid.uuid4())

        if name is None:
            name = type(self).__name__
        # Store custom name
        self._name = name

        # Execution stages
        self.execution_stages = set(execution_stages)

        self._initialized = False
        self.freezed = False
        self._input_ports: dict[str, InputPort] = {}
        self._output_ports: dict[str, OutputPort] = {}
        self._create_ports()

    @property
    def name(self) -> str:
        """Get node name (custom or class name)."""
        return self._name

    @property
    def id(self) -> str:
        """Get node ID (same as node name)."""
        return f"{self.name}-{self.uuid}"

    @property
    def requires_initial_fit(self) -> bool:
        """Whether this node requires statistical initialization before training.

        Returns
        -------
        bool
            True if node needs fit() to be called before gradienttraining
        """
        return False

    def fit(self, input_stream: InputStream) -> None:
        """Train node with port-based input stream (for statistical nodes).

        This is the new port-based training API for Phase 4.7. Statistical nodes
        should implement this method instead of initialize_from_data().

        Parameters
        ----------
        input_stream : Iterator[dict[str, Any]]
            Iterator yielding dicts matching INPUT_SPECS.
            Keys are port names, values are tensors/data.

        Example
        -------
        >>> for inputs in input_stream:
        ...     features = inputs["features"]  # From INPUT_SPECS
        ...     self._accumulate_statistics(features)

        Notes
        -----
        Statistical nodes (requires_initial_fit=True) can implement this method
        to use the new port-based training API. After training, node should either
        unfreeze() or freeze().

        Raises
        ------
        NotImplementedError
            If the node requires training but doesn't implement this method
        """
        if self.requires_initial_fit:
            raise NotImplementedError(
                f"{self.__class__.__name__} requires training but does not implement train() method"
            )

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

    @abstractmethod
    def forward(self, **inputs: Any) -> dict[str, Any]:
        """Execute node computation returning a dictionary of named outputs."""
        raise NotImplementedError

    # @abstractmethod
    def serialize(self, serial_dir: str) -> dict:
        """
        Convert the class into a serialized representation
        """
        # ...
        return {**self.hparams}

    @abstractmethod
    def load(self, params: dict, serial_dir: str) -> None:
        """
        Load from serialized format into an object
        """
        ...

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
