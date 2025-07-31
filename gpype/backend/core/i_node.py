from __future__ import annotations

from abc import abstractmethod

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from .i_port import IPort
from .node import Node


class INode(ioc.INode, Node):
    """Abstract base class for input-only nodes in the g.Pype pipeline.

    INode combines the functionality of ioiocore.INode and Node to provide a
    base class for nodes that only have input ports (no outputs). These are
    typically sink nodes that consume data without producing output, such as
    file writers, displays, or data loggers.

    All INode subclasses must implement the step() method to define their
    specific data consumption behavior. The setup() method validates input
    port contexts and ensures required metadata is present.

    Attributes:
        Input ports are managed by the parent ioiocore.INode class.
        Node-specific configuration is handled by the parent Node class.
    """

    def __init__(
        self, input_ports: list[IPort.Configuration] = None, **kwargs
    ):
        """Initialize the INode with input port configurations.

        Args:
            input_ports: List of input port configurations. If None, default
                configuration will be used.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        # Initialize ioiocore input node functionality
        ioc.INode.__init__(self, input_ports=input_ports, **kwargs)
        # Initialize g.Pype node functionality
        Node.__init__(self, target=self)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the input node before processing begins.

        This method validates that all input ports have the required metadata
        and then delegates to the parent setup method for additional
        processing.

        Args:
            data: Dictionary mapping port names to numpy arrays containing
                initial data (typically not used in setup).
            port_context_in: Dictionary mapping input port names to their
                context dictionaries containing metadata like sampling_rate,
                channel_count, frame_size, and type.

        Returns:
            Dictionary mapping port names to their context dictionaries.
            For input-only nodes, this typically returns the input contexts
            unchanged.

        Raises:
            ValueError: If required keys (frame_size, channel_count) are
                missing from any input port context.
        """
        # Validate required metadata is present in all input contexts
        for context in port_context_in.values():
            if Constants.Keys.FRAME_SIZE not in context:
                raise ValueError("frame_size must be provided in context.")
            if Constants.Keys.CHANNEL_COUNT not in context:
                raise ValueError("channel_count must be provided in context.")

        # Delegate to parent class for additional setup processing
        return super().setup(data, port_context_in)

    @abstractmethod
    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process input data at each discrete time step.

        This method is executed at each discrete time step during pipeline
        execution. Input-only nodes typically consume data without producing
        output (e.g., saving to file, displaying, logging).

        Args:
            data: Dictionary mapping input port names to numpy arrays
                containing the input data for this time step.

        Returns:
            Dictionary mapping output port names to numpy arrays. For
            input-only nodes, this is typically None or an empty dictionary
            since they don't produce output data.

        Note:
            This is an abstract method that must be implemented by all
            INode subclasses to define their specific data consumption
            behavior.
        """
        pass  # pragma: no cover
