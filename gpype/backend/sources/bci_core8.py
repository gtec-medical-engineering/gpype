import gtecble as gble
from .base.amplifier_source import AmplifierSource
from ...utilities.constants import Constants
import numpy as np
import time
from typing import Dict

PORT_OUT = Constants.Defaults.PORT_OUT
PORT_IN = Constants.Defaults.PORT_IN


class BCICore8(AmplifierSource):

    FINGERPRINT = "448a1f2caf54bf498e5f12b3d147f3ab"
    SCANNING_TIMEOUT_S = 10
    SAMPLING_RATE = 250

    _device: gble.GtecBLE
    _target_sn: str

    def __init__(self, serial: str = None, **kwargs):

        super().__init__(sampling_rate=BCICore8.SAMPLING_RATE,
                         **kwargs)
        self._target_sn = serial

    def start(self) -> None:
        super().start()
        gble.GtecBLE.AddDevicesDiscoveredEventhandler(
            self._on_devices_discovered)
        gble.GtecBLE.StartScanning()
        t_start = time.time()
        while self._device is None:
            time.sleep(0.2)
            if time.time() - t_start > BCICore8.SCANNING_TIMEOUT_S:
                raise Exception("No device connected.")
        gble.GtecBLE.StopScanning()
        gble.GtecBLE.RemoveDevicesDiscoveredEventhandler()
        self._device.AddDataAvailableEventhandler(self._data_callback)

    def stop(self):
        self._device.RemoveDataAvailableEventhandler()
        super().stop()
        del self._device

    def step(self, data):
        return {PORT_OUT: data[PORT_IN]}

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        cc_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT
        channel_count = self.config[cc_key][0]
        port_metadata_out = {}
        sampling_rate_key = AmplifierSource.Configuration.Keys.SAMPLING_RATE
        channel_count_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT

        port_metadata_out[PORT_OUT] = {sampling_rate_key:
                                       self._device.SamplingRate,
                                       channel_count_key:
                                       channel_count}
        return port_metadata_out

    def _on_devices_discovered(self, devices: list[str]):
        self._devices = devices
        if len(devices) > 0:
            idx = 0
            if self._target_sn is not None:
                sn = self._target_sn
                idx = devices.index(sn) if sn in devices else None
                if idx is None:
                    raise ValueError(f"Device {sn} not found.")
            self._device = gble.GtecBLE(devices[idx])
            print(f"Device {self._device.SerialNumber} connected.")

    def _data_callback(self, device, data: np.ndarray):
        cc_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT
        self.cycle(data={Constants.Defaults.PORT_IN:
                         data.copy()[np.newaxis, :self.config[cc_key][0]]})
