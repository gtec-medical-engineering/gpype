from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.io_node import IONode

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class Hold(IONode):
    """Classical sample-and-hold node for arbitrary resampling.

    Implements sample-and-hold behavior where input is forwarded to output
    and remains asynchronously available for retrieval at any rate.
    """

    def __init__(self, **kwargs):
        """Initialize sample-and-hold node with asynchronous output.

        Args:
            **kwargs: Additional arguments for parent IONode.
        """
        output_ports = [ioc.OPort.Configuration(timing=Constants.Timing.ASYNC)]
        super().__init__(output_ports=output_ports, **kwargs)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup asynchronous output context for sample-and-hold operation.

        Removes sampling rate constraints from output context, enabling
        downstream nodes to access held values at any desired rate.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts containing channel count
                and frame size.

        Returns:
            Output port contexts with asynchronous timing and no sampling rate.

        Raises:
            ValueError: If channel count or frame size not provided, or if
                frame size is not 1.
        """

        md = port_context_in[PORT_IN]
        channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in context.")
        frame_size = md.get(Constants.Keys.FRAME_SIZE)
        if frame_size is None:
            raise ValueError("Frame size must be provided in context.")
        if frame_size != 1:
            raise ValueError("Frame size must be 1.")

        port_context_out = super().setup(data, port_context_in)

        timing_key = ioc.OPort.Configuration.Keys.TIMING
        port_context_out[PORT_OUT][timing_key] = Constants.Timing.ASYNC
        del port_context_out[PORT_OUT][Constants.Keys.SAMPLING_RATE]
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Forward input sample and hold for asynchronous retrieval.

        Implements the hold operation by making input data available on
        output where it remains accessible until the next update.

        Args:
            data: Input data dictionary containing sample to hold.

        Returns:
            Dictionary with held sample on output port.
        """
        return {PORT_OUT: data[PORT_IN]}
