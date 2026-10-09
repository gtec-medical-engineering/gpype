from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ...common.document import omit_unassigned_edge
from ...common.launch_config import check_edge_id
from ._private.placement import PlacingMixin
from .i_port import IPort
from .node import Node


class INode(PlacingMixin, ioc.INode, Node):
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
                A sink or a widget passes its ``edge_id`` here, and it
                is checked here because every one of them reaches this
                constructor; a node that only touches data passes none.

        Raises:
            ValueError: If an edge_id is empty or padded with whitespace.
            TypeError: If an edge_id is not a string.
        """
        check_edge_id(kwargs.get(Constants.Keys.EDGE_ID))
        # Initialize ioiocore input node functionality
        ioc.INode.__init__(self, input_ports=input_ports, **kwargs)
        # Initialize g.Pype node functionality
        Node.__init__(self, target=self)

    def serialize(self) -> dict:
        """Serialise the node, writing an unset edge id as no key.

        So a document that assigns no sink or widget to an edge is what
        it was before edge ids existed. See
        ``document.omit_unassigned_edge``.

        Returns:
            The serialised node.
        """
        return omit_unassigned_edge(super().serialize())

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
