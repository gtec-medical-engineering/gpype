from __future__ import annotations

from abc import abstractmethod

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from .node import Node
from .o_port import OPort


class ONode(ioc.ONode, Node):
    """Abstract base class for output-only nodes in the g.Pype pipeline.

    Combines ioiocore.ONode and Node functionality for nodes that generate
    output data without consuming inputs (e.g., EEG devices, signal
    generators). Subclasses must implement the abstract step() method.
    """

    def __init__(
        self, output_ports: list[OPort.Configuration] = None, **kwargs
    ):
        """Initialize the ONode with output port configurations.

        Args:
            output_ports: List of output port configurations or None.
            **kwargs: Additional arguments passed to parent classes.
        """
        # Initialize ioiocore output node functionality
        ioc.ONode.__init__(self, output_ports=output_ports, **kwargs)
        # Initialize g.Pype node functionality
        Node.__init__(self, target=self)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the output node before processing begins.

        Delegates to parent setup method to configure output port contexts.

        Args:
            data: Dictionary mapping port names to numpy arrays.
            port_context_in: Dictionary mapping input port names to contexts.

        Returns:
            Dictionary mapping output port names to context dictionaries.
        """
        # Delegate to parent class for output port context setup
        return super().setup(data=data, port_context_in=port_context_in)

    @abstractmethod
    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Generate output data at each discrete time step.

        Abstract method that must be implemented by subclasses to define
        their specific data generation behavior.

        Args:
            data: Dictionary mapping input port names to numpy arrays.
                Typically empty for output-only nodes.

        Returns:
            Dictionary mapping output port names to numpy arrays containing
            the generated output data.
        """
        pass  # pragma: no cover
