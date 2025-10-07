from __future__ import annotations

from collections import deque

import numpy as np

from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Delay(IONode):
    """Introduces configurable N-sample delay to input signal.

    Uses efficient deque-based buffer. Output is zero-initialized until
    buffer is filled with sufficient samples.
    """

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            #: Configuration key for number of delay samples
            NUM_SAMPLES = "num_samples"

    def __init__(self, num_samples: int, **kwargs):
        """Initialize delay with specified number of samples.

        Args:
            num_samples: Number of samples to delay signal. Must be
                non-negative.
            **kwargs: Additional arguments for parent IONode.

        Raises:
            ValueError: If num_samples is negative.
        """
        if num_samples < 0:
            raise ValueError("Number of taps must be non-negative.")
        super().__init__(num_samples=num_samples, **kwargs)
        #: Circular buffer storing delayed samples
        self._buffer: deque[np.ndarray] = None
        #: Number of samples to delay (same as num_samples)
        self._taps: int = num_samples
        #: Zero-filled frame for initialization period
        self._zero_frame: np.ndarray = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup delay buffer and zero frame.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts containing channel count.

        Returns:
            Output port contexts from parent setup.

        Raises:
            ValueError: If channel count is not provided in context.
        """
        md = port_context_in[PORT_IN]
        channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in context.")
        self._buffer = deque(maxlen=self._taps)
        self._zero_frame = np.zeros((1, channel_count), dtype=np.float32)
        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one step of delay.

        Args:
            data: Input data dictionary containing signal to delay.

        Returns:
            Dictionary with delayed signal or zero frame if buffer not full.
        """
        data_in = data[PORT_IN]

        if self._taps == 0:
            return {PORT_OUT: data_in}

        # Append current sample
        self._buffer.append(data_in)

        # If not enough data collected yet, return zero
        if len(self._buffer) < self._taps:
            return {PORT_OUT: self._zero_frame}

        # Return delayed output
        return {PORT_OUT: self._buffer[0]}
