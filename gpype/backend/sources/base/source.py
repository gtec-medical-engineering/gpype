from __future__ import annotations

import json
from typing import Optional, Union

import numpy as np

from ....common.constants import Constants
from ...core.o_node import ONode
from ...core.o_port import OPort

#: Default output port identifier
OUT_PORT = Constants.Defaults.PORT_OUT


class Source(ONode):
    """Base class for data source nodes in a pipeline.

    Provides foundation for all data source nodes that generate or acquire
    data. Sources have only output ports and serve as pipeline entry points.
    Handles validation of output ports, channel counts, and frame sizes.
    """

    class Configuration(ONode.Configuration):
        """Configuration class for Source parameters."""

        class Keys(ONode.Configuration.Keys):
            """Configuration keys for source-specific settings."""

            #: Configuration key for number of channels per port
            CHANNEL_COUNT = Constants.Keys.CHANNEL_COUNT
            #: Configuration key for samples per frame
            FRAME_SIZE = Constants.Keys.FRAME_SIZE

    def __init__(
        self,
        output_ports: Optional[list] = None,
        channel_count: Optional[Union[list, int]] = None,
        frame_size: Optional[Union[list, int]] = None,
        **kwargs,
    ):
        """Initialize source with output port configuration.

        Args:
            output_ports: List of output port configurations. Required.
            channel_count: Number of channels per port. Can be int (all ports)
                or list (per port). Defaults to 1. Must be >= 1 or INHERITED.
            frame_size: Samples per frame. Can be int (all ports) or list
                (per port). Defaults to 1. Must be >= 1 or INHERITED.
            **kwargs: Additional arguments for parent ONode class.

        Raises:
            ValueError: If validation fails or input_ports specified.
        """
        # Validate that output_ports is provided (required for sources)
        if output_ports is None:
            raise ValueError("output_ports must be defined.")

        # Validate and normalize channel_count parameter
        if channel_count is None:
            # Default to 1 channel per output port
            channel_count = [1] * len(output_ports)
        elif isinstance(channel_count, int):
            # Convert single int to list for all ports
            channel_count = [channel_count]

        # Validate channel_count values
        if not all(isinstance(c, int) for c in channel_count):
            raise ValueError("All elements of channel_count must be integers.")
        if not all(c == Constants.INHERITED or c >= 1 for c in channel_count):
            raise ValueError(
                "All elements of channel_count must be greater or equal 1."
            )
        if len(output_ports) != len(channel_count):
            raise ValueError(
                "output_ports and channel_count must have the same length."
            )

        # Validate and normalize frame_size parameter
        if frame_size is None:
            # Default to 1 sample per frame for all ports
            frame_size = [Constants.Defaults.FRAME_SIZE] * len(output_ports)
        elif isinstance(frame_size, int):
            # Convert single int to list for all ports
            frame_size = [frame_size] * len(output_ports)

        # Validate frame_size values
        if not all(isinstance(f, int) for f in frame_size):
            raise ValueError("All elements of frame_size must be integers.")
        if not all(f == Constants.INHERITED or f >= 1 for f in frame_size):
            raise ValueError(
                "All elements of frame_size must be greater or equal 1."
            )

        # Check frame_size consistency (all non-inherited values must be equal)
        non_inherited_frames = [
            fsz for fsz in set(frame_size) if fsz != Constants.INHERITED
        ]
        if len(non_inherited_frames) != 1:
            raise ValueError("All elements of frame_size must be equal.")
        if len(output_ports) != len(frame_size):
            raise ValueError(
                "output_ports and frame_size must have the same length."
            )

        # Sources cannot have input ports by design
        if "input_ports" in kwargs:
            raise ValueError("Source must not have input ports.")

        #: Timing delay in seconds for synchronization
        self._delay: float = 0

        # Initialize parent ONode with validated parameters
        ONode.__init__(
            self,
            output_ports=output_ports,
            channel_count=channel_count,
            frame_size=frame_size,
            **kwargs,
        )

    @property
    def delay(self) -> float:
        """Get the timing delay in seconds.

        Returns:
            Current delay value in seconds for timing synchronization.
        """
        return self._delay

    @delay.setter
    def delay(self, value: float):
        """Set the timing delay in seconds.

        Args:
            value: Delay value in seconds. Must be non-negative.

        Raises:
            ValueError: If value is negative.
        """
        if value < 0:
            raise ValueError("Delay must be non-negative.")
        self._delay = value

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with channel and frame information.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with channel_count and
            frame_size information for each output port.
        """
        # Call parent setup method to initialize base contexts
        port_context_out = super().setup(data, port_context_in)

        # Get configuration parameters
        channel_count = self.config[self.Configuration.Keys.CHANNEL_COUNT]
        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
        out_ports = self.config[self.Configuration.Keys.OUTPUT_PORTS]

        # Configure context for each output port
        for i in range(len(out_ports)):
            # Create context with channel and frame information
            context = {
                Constants.Keys.CHANNEL_COUNT: channel_count[i],
                Constants.Keys.FRAME_SIZE: frame_size[i],
            }

            # Get port name and update its context
            port_name = out_ports[i][OPort.Configuration.Keys.NAME]
            port_context_out[port_name].update(context)

        self.log(f"Source setup complete with {json.dumps(port_context_out)}")
        return port_context_out
