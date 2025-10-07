from __future__ import annotations

import numpy as np

from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Decimator(IONode):
    """Decimator node for downsampling data streams.

    Reduces data rate by outputting only every Nth sample based on decimation
    factor. Adjusts sampling rate and frame size in output context accordingly.
    """

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            pass

    def __init__(self, decimation_factor: int = 1, **kwargs):
        """Initialize decimator with decimation factor.

        Args:
            decimation_factor: Factor by which to reduce data rate. Must be
                positive integer. Value of 1 means no decimation.
            **kwargs: Additional arguments for parent IONode.
        """
        super().__init__(decimation_factor=decimation_factor, **kwargs)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output context with adjusted sampling rate and frame size.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts containing frame size and
                sampling rate information.

        Returns:
            Output port contexts with decimated sampling rate.

        Raises:
            ValueError: If frame_size is not provided or doesn't match
                decimation_factor.
        """

        port_context_out = super().setup(data, port_context_in)
        frame_size = port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE]
        if frame_size is None:
            raise ValueError("frame_size must be provided in context.")

        M = self.config[self.Configuration.Keys.DECIMATION_FACTOR]
        if frame_size != 1 and frame_size != M:
            raise ValueError(
                f"frame_size {frame_size} must match " f"decimation_factor {M}"
            )
        port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE] = 1
        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rate_out = port_context_in[PORT_IN][sr_key] / M
        port_context_out[PORT_OUT][sr_key] = sampling_rate_out
        return port_context_out

    def step(self, data: dict):
        """Process one step of decimation.

        Args:
            data: Input data dictionary containing data to be decimated.

        Returns:
            Dictionary with last sample of input data if decimation step,
            None otherwise.
        """
        if self.is_decimation_step():
            return {PORT_OUT: data[PORT_IN][-1:, :]}
        else:
            return None
