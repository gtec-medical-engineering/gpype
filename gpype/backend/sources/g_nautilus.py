import sys
if sys.platform != "win32":
    raise NotImplementedError("This module is only supported on Windows.")

from .base.amplifier_source import AmplifierSource
from ...common.constants import Constants
import threading
import numpy as np
import gtec_gds as gds

PORT_OUT = Constants.Defaults.PORT_OUT
PORT_IN = Constants.Defaults.PORT_IN


class GNautilus(AmplifierSource):

    FINGERPRINT = "6eea0766ddf031b8b1c0a32893062ec5"

    class Configuration(AmplifierSource.Configuration):
        class Keys(AmplifierSource.Configuration.Keys):
            SENITIVITY = 'sensitivity'

    def __init__(self,
                 serial: str = None,
                 sampling_rate: float = None,
                 channel_count: int = None,
                 sensitivity: float = None,
                 **kwargs):
        self._z = np.ones(channel_count) * (-10)
        self._device = gds.GNautilus(serial=serial,
                                     sampling_rate=sampling_rate,
                                     channel_count=channel_count,
                                     sensitivity=sensitivity)
        channel_count = self._device.channel_count
        sensitivity = self._device.sensitivity

        self._device.set_data_callback(self._data_callback)

        super().__init__(sampling_rate=sampling_rate,
                         channel_count=channel_count,
                         sensitivity=sensitivity,
                         **kwargs)

        self._impedance_check_running = False
        self._impedance_fresh = True

    def start(self) -> None:
        self._device.start()
        super().start()

    def stop(self):
        self._device.stop()
        super().stop()

    def start_impedance_check(self) -> None:
        # Start the impedance retrieval in a background thread
        self._impedance_check_running = True
        self._impedance_thread = threading.Thread(target=self._get_z_thread,
                                                  daemon=True)
        self._impedance_thread.daemon = True
        self._impedance_thread.start()

    def stop_impedance_check(self):
        self._impedance_check_running = False
        if self._impedance_thread:
            self._impedance_thread.join()

    def get_impedance(self):
        imp_fresh = self._impedance_fresh
        self._impedance_fresh = False
        return self._z, imp_fresh

    def _data_callback(self, data: np.ndarray):
        self.cycle(data={Constants.Defaults.PORT_IN: data})

    def _get_z_thread(self):
        first = True
        while self._impedance_check_running:
            self._z = self._device.get_impedance(first)
            first = False
            self._impedance_fresh = True

    def step(self, data):
        return {PORT_OUT: data[PORT_IN]}
