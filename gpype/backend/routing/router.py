import ioiocore as ioc
from ..core.i_port import IPort
from ..core.o_port import OPort
from typing import Dict
from ...common.constants import Constants
from ..core.io_node import IONode
import numpy as np


class Router(IONode):

    ALL: int = -1  # take all channels

    _map: dict

    class Configuration(ioc.IONode.Configuration):
        class Keys(ioc.IONode.Configuration.Keys):
            INPUT_SELECTOR = "input_selector"
            OUTPUT_SELECTOR = "output_selector"

    def __init__(self,
                 input_selector: list[list[int]] = None,
                 output_selector: list[list[int]] = None,
                 **kwargs):
        if input_selector is None:
            input_selector = [Router.ALL]
        if output_selector is None:
            output_selector = [Router.ALL]
        if len(input_selector) == 1:
            input_ports = [IPort.Configuration(timing=Constants.Timing.INHERITED)]  # noqa: E501
        else:
            input_ports = [IPort.Configuration(name=f"in{i + 1}",
                                               timing=Constants.Timing.INHERITED)  # noqa: E501
                           for i in range(len(input_selector))]
        if len(output_selector) == 1:
            output_ports = [OPort.Configuration()]
        else:
            output_ports = [OPort.Configuration(name=f"out{i + 1}")
                            for i in range(len(output_selector))]
        self._map = {}
        IONode.__init__(self,
                        input_selector=input_selector,
                        output_selector=output_selector,
                        input_ports=input_ports,
                        output_ports=output_ports,
                        **kwargs)

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        cc_key = Constants.Keys.CHANNEL_COUNT
        name_key = IPort.Configuration.Keys.NAME

        input_map: list = []
        ip_key = Router.Configuration.Keys.INPUT_PORTS
        is_key = Router.Configuration.Keys.INPUT_SELECTOR
        for k in range(len(self.config[ip_key])):
            port = self.config[ip_key][k]
            sel = self.config[is_key][k]
            name = port[name_key]

            if sel == Router.ALL:
                sel = range(port_metadata_in[name][cc_key])
            input_map.extend([{name: n} for n in sel])
        op_key = Router.Configuration.Keys.OUTPUT_PORTS
        os_key = Router.Configuration.Keys.OUTPUT_SELECTOR
        for k in range(len(self.config[op_key])):
            port = self.config[op_key][k]
            sel = self.config[os_key][k]
            name = port[name_key]

            if sel == Router.ALL:
                sel = range(len(input_map))
            self._map[name] = [input_map[n] for n in sel]

        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rates = [md.get(sr_key, None) for md in port_metadata_in.values()]  # noqa: E501
        sampling_rates = [sr for sr in sampling_rates if sr is not None]

        if len(set(sampling_rates)) != 1:
            raise ValueError("All ports must have the same sampling rate.")
        sr = sampling_rates[0]

        port_metadata_out: Dict[str, dict] = {}
        cc_key = Constants.Keys.CHANNEL_COUNT
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        name_key = OPort.Configuration.Keys.NAME
        for op in self.config[op_key]:
            metadata = {}
            metadata[cc_key] = len(self._map[op[name_key]])
            metadata[sr_key] = sr
            port_metadata_out[op[name_key]] = metadata
        return port_metadata_out

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        data_out: dict = {}
        # if data['in2'] is None:
        #    data['in2'] = np.ndarray((1, 1))
        for key, mapping in self._map.items():
            data_out[key] = np.hstack([np.vstack([data[key][:, val]
                                                  for key, val in m.items()])
                                       for m in mapping])
        return data_out
