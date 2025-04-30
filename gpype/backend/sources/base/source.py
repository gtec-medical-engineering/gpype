from ...core.o_node import ONode
from ...core.o_port import OPort
from ....common.constants import Constants
from typing import Dict
import numpy as np

OUT_PORT = Constants.Defaults.PORT_OUT


class Source(ONode):

    class Configuration(ONode.Configuration):
        class Keys(ONode.Configuration.Keys):
            CHANNEL_COUNT = Constants.Keys.CHANNEL_COUNT  # 'channel_count'

    def __init__(self,
                 output_ports: list[OPort.Configuration] = None,
                 channel_count: list[int] = None,
                 **kwargs):
        if channel_count is None:
            channel_count = [1]
        if isinstance(channel_count, int):
            channel_count = [channel_count]
        if output_ports is None:
            raise ValueError("output_ports must be defined.")
        if any([c < 1 for c in channel_count]):
            raise ValueError("All elements of channel_count must be greater or equal 1.")  # noqa: E501

        if len(output_ports) != len(channel_count):
            raise ValueError("output_ports and channel_count must have the same length.")  # noqa: E501

        if "input_ports" in kwargs:
            raise ValueError("Source must not have input ports.")

        ONode.__init__(self,
                       output_ports=output_ports,
                       channel_count=channel_count,
                       **kwargs)

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        port_metadata_out: Dict[str, dict] = {}

        cc_key = self.Configuration.Keys.CHANNEL_COUNT
        op_key = self.Configuration.Keys.OUTPUT_PORTS

        cc = self.config[cc_key]
        op = self.config[op_key]  # noqa: E501
        for n in range(len(op)):
            metadata = {}
            metadata[cc_key] = cc[n]
            port_metadata_out[op[n][self.Configuration.Keys.NAME]] = metadata
        return port_metadata_out
