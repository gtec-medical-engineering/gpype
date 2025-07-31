from __future__ import annotations

from collections import deque

import numpy as np

from ...common.constants import Constants
from ..core.io_node import IONode

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class Delay(IONode):
    """
    Introduces a configurable N-sample delay to the input signal.

    Parameters
    ----------
    taps : int
        Number of samples to delay the signal. Must be non-negative.

    Notes
    -----
    - Output is zero-initialized until the buffer is filled.
    - Efficient deque-based implementation with O(1) operations.
    """

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            NUM_SAMPLES = "num_samples"

    def __init__(self, num_samples: int, **kwargs):
        if num_samples < 0:
            raise ValueError("Number of taps must be non-negative.")
        super().__init__(num_samples=num_samples, **kwargs)
        self._buffer: deque[np.ndarray] = None
        self._taps: int = num_samples
        self._zero_frame: np.ndarray = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        md = port_context_in[PORT_IN]
        channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in context.")
        self._buffer = deque(maxlen=self._taps)
        self._zero_frame = np.zeros((1, channel_count), dtype=np.float32)
        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
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
