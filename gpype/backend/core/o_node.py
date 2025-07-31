from __future__ import annotations

from abc import abstractmethod

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from .node import Node
from .o_port import OPort


class ONode(ioc.ONode, Node):
    """Abstract base class for output-only nodes in the g.Pype pipeline.

    ONode combines the functionality of ioiocore.ONode and Node to provide a
    base class for nodes that only have output ports (no inputs). These are
    typically source nodes that generate data without consuming input, such as
    signal generators, file readers, or data acquisition devices.

    All ONode subclasses must implement the step() method to define their
    specific data generation behavior. The setup() method delegates to the
    parent class for output port context configuration.

    Attributes:
        Output ports are managed by the parent ioiocore.ONode class.
        Node-specific configuration is handled by the parent Node class.
    """

    def __init__(
        self, output_ports: list[OPort.Configuration] = None, **kwargs
    ):
        """Initialize the ONode with output port configurations.

        Args:
            output_ports: List of output port configurations. If None, default
                configuration will be used.
            **kwargs: Additional keyword arguments passed to parent classes.
        """
        # Initialize ioiocore output node functionality
        ioc.ONode.__init__(self, output_ports=output_ports, **kwargs)
        # Initialize g.Pype node functionality
        Node.__init__(self, target=self)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the output node before processing begins.

        This method delegates to the parent setup method to configure output
        port contexts. Output-only nodes typically don't have input contexts
        to validate, so this mainly handles output port initialization.

        Args:
            data: Dictionary mapping port names to numpy arrays containing
                initial data (typically empty for output-only nodes).
            port_context_in: Dictionary mapping input port names to their
                context dictionaries (typically empty for output-only nodes).

        Returns:
            Dictionary mapping output port names to their context dictionaries.
            Contains metadata like sampling_rate, channel_count, frame_size,
            and type for each output port.
        """
        # Delegate to parent class for output port context setup
        return super().setup(data=data, port_context_in=port_context_in)

    @abstractmethod
    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Generate output data at each discrete time step.

        This method is executed at each discrete time step during pipeline
        execution. Output-only nodes generate data without consuming input
        (e.g., generating signals, acquiring from hardware).

        Args:
            data: Dictionary mapping input port names to numpy arrays. For
                output-only nodes, this is typically empty or None since
                they don't have input ports.

        Returns:
            Dictionary mapping output port names to numpy arrays containing
            the generated output data for this time step. The data should
            match the configured output port specifications.

        Note:
            This is an abstract method that must be implemented by all
            ONode subclasses to define their specific data generation
            behavior.
        """
        pass  # pragma: no cover
