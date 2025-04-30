from .source import Source
from ....common.constants import Constants
from ...core.o_port import OPort
from typing import Any, Dict
import numpy as np

PORT_OUT = Constants.Defaults.PORT_OUT


class AmplifierSource(Source):

    class Configuration(Source.Configuration):
        class Keys(Source.Configuration.Keys):
            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE  # 'sampling_rate'

        def __init__(self,
                     sampling_rate: float,
                     **kwargs):
            if sampling_rate <= 0:
                raise ValueError("sampling_rate must be greater than zero.")
            super().__init__(sampling_rate=sampling_rate,
                             **kwargs)

    _devices: list[Any]
    _device: Any

    def __init__(self,
                 sampling_rate: float,
                 channel_count: int,
                 **kwargs):

        # remove output_ports from kwargs and assign it to output_ports;
        # if not present, assign a default value. This avoids errors when
        # deserialization is performed.
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        output_ports: list[OPort.Configuration] = kwargs.pop(op_key,
                                                             [OPort.Configuration()])  # noqa: E501

        self._devices = []
        Source.__init__(self,
                        output_ports=output_ports,
                        sampling_rate=sampling_rate,
                        channel_count=channel_count,
                        **kwargs)

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        port_metadata_out = {}
        sampling_rate_key = AmplifierSource.Configuration.Keys.SAMPLING_RATE
        sampling_rate = self.config[sampling_rate_key]
        channel_count_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT
        channel_count = self.config[channel_count_key][0]

        # check if sampling rate and channel count are still not set
        if sampling_rate == Constants.INHERITED:
            raise ValueError("sampling_rate inheritance not implemented.")
        if channel_count == Constants.INHERITED:
            raise ValueError("channel_count inheritance not implemented.")

        port_metadata_out[PORT_OUT] = {sampling_rate_key: sampling_rate,
                                       channel_count_key: channel_count}
        return port_metadata_out
