from __future__ import annotations

from abc import abstractmethod

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.i_port import IPort
from .node import Node


class IONode(ioc.IONode, Node):
    """Abstract base class for input/output nodes in the g.Pype pipeline.

    IONode combines the functionality of ioiocore.IONode and Node to provide a
    base class for all signal processing nodes that have input and output
    ports. This class handles the common validation and setup logic for port
    contexts including sampling rates, channel counts, frame sizes, and data
    types.

    All IONode subclasses must implement the step() method to define their
    specific signal processing behavior. The setup() method is called once
    before processing begins to validate and configure the node based on input
    port contexts.

    Attributes:
        Input and output ports are managed by the parent ioiocore.IONode class.
        Node-specific configuration is handled by the parent Node class.
    """

    def __init__(
        self,
        input_ports: list[ioc.IPort.Configuration] = None,
        output_ports: list[ioc.OPort.Configuration] = None,
        **kwargs,
    ):
        """Initialize the IONode with input and output port configurations.

        Args:
            input_ports: List of input port configurations. If None, default
                configuration will be used.
            output_ports: List of output port configurations. If None, default
                configuration will be used.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        ioc.IONode.__init__(
            self, input_ports=input_ports, output_ports=output_ports, **kwargs
        )
        Node.__init__(self, target=self)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the node before processing begins.

        This method is executed once right before the first step of the node.
        It validates that all input ports have consistent configurations and
        creates output port contexts based on the input configurations.

        The method performs the following validations:
        - All ports must have frame_size and channel_count in their context
        - All ports must have the same sampling rate
        - Channel counts must be compatible (either all same or broadcastable)
        - All ports must have the same frame size
        - Port types must be compatible

        Args:
            data: Dictionary mapping port names to numpy arrays containing
                initial data (typically not used in setup).
            port_context_in: Dictionary mapping input port names to their
                context dictionaries containing metadata like sampling_rate,
                channel_count, frame_size, and type.

        Returns:
            Dictionary mapping output port names to their context dictionaries.
            The output contexts are derived from the validated input contexts.

        Raises:
            ValueError: If any validation fails, including missing required
                keys, mismatched sampling rates, incompatible channel counts,
                different frame sizes, or incompatible types.
        """
        # Validate required keys are present in all input port contexts
        for context in port_context_in.values():
            if Constants.Keys.CHANNEL_COUNT not in context:
                raise ValueError("channel_count must be provided in context.")
            if Constants.Keys.FRAME_SIZE not in context:
                raise ValueError("frame_size must be provided in context.")

        # Validate sampling rates - all ports must have the same sampling rate
        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rates = [
            md.get(sr_key, None) for md in port_context_in.values()
        ]
        sampling_rates = [sr for sr in sampling_rates if sr is not None]
        if len(set(sampling_rates)) != 1:
            raise ValueError("All ports must have the same sampling rate.")

        # Validate and normalize channel counts
        # Allow broadcasting: single-channel ports can be broadcast to
        # multi-channel ports
        cc_key = Constants.Keys.CHANNEL_COUNT
        channel_counts = [
            md.get(cc_key, None) for md in port_context_in.values()
        ]
        channel_counts = [cc for cc in channel_counts if cc is not None]
        channel_counts.append(1)  # add single channel for comparison
        if len(set(channel_counts)) > 2:
            # More than 2 unique values means incompatible multi-channel ports
            raise ValueError("All ports must have the same channel count.")

        # Broadcast single channels to maximum channel count
        for md in port_context_in.values():
            if md.get(cc_key) is not None:
                md[cc_key] = max(channel_counts)  # set to max (broadcast)

        # Validate frame sizes - all ports must have the same frame size
        fsz_key = Constants.Keys.FRAME_SIZE
        frame_sizes = [
            md.get(fsz_key, None) for md in port_context_in.values()
        ]
        frame_sizes = [fsz for fsz in frame_sizes if fsz is not None]
        if len(set(frame_sizes)) != 1:
            raise ValueError("All ports must have the same frame size.")

        # Validate port types - all ports must have compatible types
        type_key = IPort.Configuration.Keys.TYPE
        types = [md.get(type_key, None) for md in port_context_in.values()]
        types = [tp for tp in types if (tp != "Any" and tp is not None)]
        if len(set(types)) > 1:
            raise ValueError("All ports must have the same type.")

        # Build output port contexts by merging input port contexts
        port_context_out: dict[str, dict] = {}
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        name_key = IPort.Configuration.Keys.NAME
        context = {}

        # Get all unique keys from all input port contexts
        all_keys = set().union(*port_context_in.values())

        # For each key, determine how to merge values from different ports
        for key in all_keys:
            values = {}
            for port, config in port_context_in.items():
                if key in config:
                    values[port] = config[key]

            value_list = list(values.values())

            # Case 1: Key exists in only one port
            # → use that value directly
            if len(values) == 1:
                context[key] = value_list[0]

            # Case 2: Key exists in multiple ports with identical values
            # → use the common value
            elif all(value == value_list[0] for value in value_list[1:]):
                import copy

                context[key] = copy.deepcopy(value_list[0])

            # Case 3: Key exists in multiple ports with different values
            # → keep per-port mapping
            else:
                context[key] = values

        # Apply the merged context to all output ports
        for op in self.config[op_key]:
            port_context_out[op[name_key]] = context
        return port_context_out

    @abstractmethod
    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process data at each discrete time step.

        This method is executed at each discrete time step during pipeline
        execution. Subclasses must implement this method to define their
        specific signal processing behavior.

        Args:
            data: Dictionary mapping input port names to numpy arrays
                containing the input data for this time step. The shape
                and content depend on the specific node's input configuration.

        Returns:
            Dictionary mapping output port names to numpy arrays containing
            the processed output data. The shape and content depend on the
            specific node's processing logic. May return None if no output
            is produced for this time step.

        Note:
            This is an abstract method that must be implemented by all
            IONode subclasses.
        """
        pass  # pragma: no cover
