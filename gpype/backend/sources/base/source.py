from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ....common.constants import Constants
from ...core.o_node import ONode
from ...core.o_port import OPort

# Convenience constant for default output port name
OUT_PORT = Constants.Defaults.PORT_OUT


class Source(ONode):
    """Base class for data source nodes in the BCI pipeline.

    This abstract base class provides the foundation for all data source
    nodes that generate or acquire data in the g.Pype framework. Source
    nodes have only output ports and serve as the entry points for data
    into the processing pipeline.

    The class handles validation and configuration of output ports,
    channel counts, and frame sizes. It ensures that all sources provide
    consistent metadata about their output data streams.

    Features:
    - Multiple output port support with individual configuration
    - Channel count validation and configuration per port
    - Frame size validation with consistency checks
    - Delay property for timing adjustments
    - Automatic context propagation to connected nodes

    Attributes:
        _delay: Timing delay in seconds for synchronization purposes

    Note:
        Source nodes cannot have input ports by design. All data
        generation must happen internally within the source implementation.
    """

    class Configuration(ONode.Configuration):
        """Configuration class for Source parameters."""

        class Keys(ONode.Configuration.Keys):
            """Configuration keys for source-specific settings."""

            CHANNEL_COUNT = Constants.Keys.CHANNEL_COUNT  # 'channel_count'
            FRAME_SIZE = Constants.Keys.FRAME_SIZE  # 'frame_size'

    def __init__(
        self,
        output_ports: Optional[list] = None,
        channel_count: Optional[Union[list, int]] = None,
        frame_size: Optional[Union[list, int]] = None,
        **kwargs,
    ):
        """Initialize the source with output port configuration.

        Args:
            output_ports: List of output port configurations. Must be provided
                as source nodes require at least one output port.
            channel_count: Number of channels for each output port. Can be:
                - None: defaults to 1 channel per port
                - int: same channel count for all ports
                - list: individual channel count per port
                Must be >= 1 or Constants.INHERITED.
            frame_size: Number of samples per frame for each output port.
                Can be:
                - None: defaults to 1 sample per frame
                - int: same frame size for all ports
                - list: individual frame size per port
                Must be >= 1 or Constants.INHERITED, and all non-inherited
                values must be equal.
            **kwargs: Additional arguments passed to parent ONode class.

        Raises:
            ValueError: If output_ports is None, if channel_count or
                frame_size contain invalid values, if lengths don't match,
                or if input_ports is specified (sources cannot have inputs).
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
                "All elements of channel_count must be greater " "or equal 1."
            )
        if len(output_ports) != len(channel_count):
            raise ValueError(
                "output_ports and channel_count must have the " "same length."
            )

        # Validate and normalize frame_size parameter
        if frame_size is None:
            # Default to 1 sample per frame for all ports
            frame_size = [1] * len(output_ports)
        elif isinstance(frame_size, int):
            # Convert single int to list for all ports
            frame_size = [frame_size] * len(output_ports)

        # Validate frame_size values
        if not all(isinstance(f, int) for f in frame_size):
            raise ValueError("All elements of frame_size must be integers.")
        if not all(f == Constants.INHERITED or f >= 1 for f in frame_size):
            raise ValueError(
                "All elements of frame_size must be greater " "or equal 1."
            )

        # Check frame_size consistency (all non-inherited values must be equal)
        non_inherited_frames = [
            fsz for fsz in set(frame_size) if fsz != Constants.INHERITED
        ]
        if len(non_inherited_frames) != 1:
            raise ValueError("All elements of frame_size must be equal.")
        if len(output_ports) != len(frame_size):
            raise ValueError(
                "output_ports and frame_size must have the " "same length."
            )

        # Sources cannot have input ports by design
        if "input_ports" in kwargs:
            raise ValueError("Source must not have input ports.")

        # Initialize delay property for timing control
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
            Current delay value in seconds used for timing synchronization.

        Note:
            Delay can be used to synchronize multiple sources or compensate
            for processing latencies in the pipeline.
        """
        return self._delay

    @delay.setter
    def delay(self, value: float):
        """Set the timing delay in seconds.

        Args:
            value: Delay value in seconds. Must be non-negative as negative
                delays are not physically meaningful in real-time systems.

        Raises:
            ValueError: If value is negative.

        Note:
            Changing delay during operation may affect timing consistency.
            It's recommended to set delay before starting the pipeline.
        """
        if value < 0:
            raise ValueError("Delay must be non-negative.")
        self._delay = value

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with channel and frame information.

        This method configures the output port contexts by propagating
        channel count and frame size information to connected downstream
        nodes. Each output port receives its specific configuration.

        Args:
            data: Dictionary of input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with channel_count and
            frame_size information for each configured output port.

        Note:
            The setup method ensures that downstream nodes receive accurate
            metadata about the data streams they will process. This includes
            the number of channels and samples per frame for each port.
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

        return port_context_out
