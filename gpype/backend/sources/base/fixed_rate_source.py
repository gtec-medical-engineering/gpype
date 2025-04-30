from .source import Source
import threading
import time
from typing import Dict
import numpy as np
from ...core.o_port import OPort
from ....common.constants import Constants

OUT_PORT = Constants.Defaults.PORT_OUT


class FixedRateSource(Source):

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

    def __init__(self,
                 sampling_rate: float,
                 output_ports: list[OPort.Configuration] = None,
                 **kwargs):
        Source.__init__(self,
                        sampling_rate=sampling_rate,
                        output_ports=output_ports,
                        **kwargs)
        self._running: bool = False
        self._thread: threading.Thread = None
        self._time_start: float = None

    def start(self):
        Source.start(self)
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._thread_function,
                                            daemon=True)
            self._thread.start()

    def stop(self):
        Source.stop(self)
        if self._running:
            self._running = False
            self._thread.join(500)

    def _thread_function(self):
        rate = self.config[FixedRateSource.Configuration.Keys.SAMPLING_RATE]  # noqa: E501
        if self._time_start is None:
            self._time_start = time.time()
        while self._running:
            t_next_sample = (self.get_counter() + 1) / rate + self._time_start
            t_sleep = max(t_next_sample - time.time(), 1e-10)
            if t_sleep > 0:
                time.sleep(t_sleep)
            self.cycle()

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        port_metadata_out = super().setup(data, port_metadata_in)

        op_key = self.Configuration.Keys.OUTPUT_PORTS
        sr_key = self.Configuration.Keys.SAMPLING_RATE

        sr = self.config[sr_key]
        op = self.config[op_key]
        for cur_op in op:
            name = cur_op[self.config.Keys.NAME]
            port_metadata_out[name][sr_key] = sr
        return port_metadata_out
