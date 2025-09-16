from __future__ import annotations

from abc import abstractmethod

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from .i_port import IPort
from .node import Node


class INode(ioc.INode, Node):
    """Abstract base class for input-only nodes in the g.Pype pipeline.

    Combines ioiocore.INode and Node functionality for nodes that consume
    input data without producing outputs (e.g., file writers, displays).
    Subclasses must implement the abstract step() method.
    """

    def __init__(
        self, input_ports: list[IPort.Configuration] = None, **kwargs
    ):
        """Initialize the INode with input port configurations.

        Args:
            input_ports: List of input port configurations or None.
            **kwargs: Additional arguments passed to parent classes.
        """
        # Initialize ioiocore input node functionality
        ioc.INode.__init__(self, input_ports=input_ports, **kwargs)
        # Initialize g.Pype node functionality
        Node.__init__(self, target=self)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the input node before pipeline processing begins.

        Validates that all input ports have required metadata keys
        (frame_size, channel_count) then delegates to parent setup.

        Args:
            data: Dictionary mapping port names to numpy arrays.
            port_context_in: Dictionary mapping port names to context dicts.

        Returns:
            Dictionary mapping port names to validated context dictionaries.

        Raises:
            ValueError: If required metadata keys are missing.
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
        """Process input data at each pipeline time step.

        Abstract method that must be implemented by subclasses to define
        their specific data consumption behavior.

        Args:
            data: Dictionary mapping input port names to numpy arrays.

        Returns:
            Dictionary mapping output port names to numpy arrays.
            Typically None or empty dict for input-only nodes.
        """
        pass  # pragma: no cover
