from __future__ import annotations

import numpy as np

from ...common.constants import Constants
from ..core.io_node import IONode

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class Decimator(IONode):

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            pass

    def __init__(self, decimation_factor: int = 1, **kwargs):
        super().__init__(decimation_factor=decimation_factor, **kwargs)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:

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
        if self.is_decimation_step():
            return {PORT_OUT: data[PORT_IN][-1:, :]}
        else:
            return None
