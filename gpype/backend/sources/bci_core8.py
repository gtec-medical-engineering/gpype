from .base.amplifier_source import AmplifierSource
from ...common.constants import Constants
import numpy as np
import time
from typing import Dict, TYPE_CHECKING
import gtec_ble as ble

PORT_OUT = Constants.Defaults.PORT_OUT
PORT_IN = Constants.Defaults.PORT_IN

MAX_NUM_CHANNELS = 8


class BCICore8(AmplifierSource):

    FINGERPRINT = "3858bbee5649b94edf4564cff73b0667"
    SCANNING_TIMEOUT_S = 10
    SAMPLING_RATE = 250

    _target_sn: str

    def __init__(self,
                 serial: str = None,
                 channel_count: int = None,
                 **kwargs):

        if channel_count is None:
            channel_count = MAX_NUM_CHANNELS
        channel_count = max(1, min(channel_count, MAX_NUM_CHANNELS))
        self._device = ble.Amplifier(serial=serial)
        self._device.set_data_callback(self._data_callback)

        super().__init__(channel_count=channel_count,
                         sampling_rate=self._device.sampling_rate,
                         **kwargs)

    def start(self) -> None:
        self._device.start()
        super().start()

    def stop(self):
        self._device.stop()
        super().stop()
        del self._device

    def step(self, data):
        return {PORT_OUT: data[PORT_IN]}

    def _data_callback(self, data: np.ndarray):
        cc_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT
        self.cycle(data={Constants.Defaults.PORT_IN:
                         data.copy()[np.newaxis, :self.config[cc_key][0]]})
