from .source import Source
from ...core.o_port import OPort
from ....common.constants import Constants
from typing import Dict
import numpy as np

PORT_OUT = Constants.Defaults.PORT_OUT


class EventSource(Source):

    def __init__(self, **kwargs):
        # remove output_ports from kwargs and assign it to output_ports;
        # if not present, assign a default value. This avoids errors when
        # deserialization is performed.
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        output_ports: list[OPort.Configuration] = kwargs.pop(op_key,
                                                             [OPort.Configuration(timing=Constants.Timing.ASYNC)])  # noqa: E501
        Source.__init__(self,
                        output_ports=output_ports,
                        **kwargs)
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        cc_key = self.Configuration.Keys.CHANNEL_COUNT
        name_key = OPort.Configuration.Keys.NAME
        self._data = {}
        for op, cc in zip(self.config[op_key], self.config[cc_key]):
            self._data[op[name_key]] = np.zeros((1, cc),
                                                dtype=Constants.DATA_TYPE)

    def start(self):
        # derived classes implement start logic here
        Source.start(self)
        self.cycle()

    def stop(self):
        # derived classes implement stop logic here
        Source.stop(self)

    # derived classes implement step logic here
    def trigger(self, data):
        self._data = data
        self.cycle()
        self._data = None

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        return self._data
