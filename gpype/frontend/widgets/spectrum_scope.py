import threading
import time

import ioiocore as ioc
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QPalette

from ...backend.core.i_port import IPort
from ...common.constants import Constants
from .base.scope import Scope

PORT_IN = ioc.Constants.Defaults.PORT_IN


class SpectrumScope(Scope):

    DEFAULT_AMPLITUDE_LIMIT = 50
    DEFAULT_NUM_AVERAGES = 10

    class Configuration(Scope.Configuration):

        class Keys(Scope.Configuration.Keys):
            AMPLITUDE_LIMIT = "amplitude_limit"
            NUM_AVERAGES = "num_averages"

        class KeysOptional:
            HIDDEN_CHANNELS = "hidden_channels"

    def __init__(
        self,
        amplitude_limit: float = None,
        num_averages: int = None,
        hidden_channels: list = None,
        **kwargs,
    ):

        if amplitude_limit is None:
            amplitude_limit = self.DEFAULT_AMPLITUDE_LIMIT

        if num_averages is None:
            num_averages = self.DEFAULT_NUM_AVERAGES

        if amplitude_limit > 5e3 or amplitude_limit < 1:
            raise ValueError("amplitude_limit without reasonable range.")

        if hidden_channels is None:
            hidden_channels = []

        input_ports = [IPort.Configuration(name=PORT_IN)]

        Scope.__init__(
            self,
            input_ports=input_ports,
            amplitude_limit=amplitude_limit,
            name="Spectrum Scope",
            hidden_channels=hidden_channels,
            num_averages=num_averages,
            **kwargs,
        )
        self._max_points: int = None
        self._data_buffer: np.ndarray = None
        self._display_buffer: np.ndarray = None
        self._plot_index: int = 0
        self._buffer_full: bool = False
        self._sample_index: int = 0
        self._start_time = time.time()
        self._update_counts = 0
        self._step_counts = 0
        self._step_rate = 0
        self._lock = threading.Lock()
        self._new_data = False
        self._rate_label = None
        p = self.widget.palette()
        self._foreground_color = p.color(QPalette.ColorRole.WindowText)
        self._background_color = p.color(QPalette.ColorRole.Window)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        c = port_context_in[PORT_IN]

        sampling_rate = c.get(Constants.Keys.SAMPLING_RATE)
        if sampling_rate is None:
            raise ValueError("sampling rate must be provided.")
        channel_count = c.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("channel count must be provided.")
        frame_size = c.get(Constants.Keys.FRAME_SIZE)
        if frame_size is None:
            raise ValueError("frame size must be provided.")
        if frame_size <= 1:
            raise ValueError("frame size must be greater than 1.")
        self._f_vec = np.fft.rfftfreq(frame_size, 1 / sampling_rate)
        hidden_channels = self.config[
            self.Configuration.KeysOptional.HIDDEN_CHANNELS
        ]
        self._channel_vec = [
            i for i in range(channel_count) if i not in hidden_channels
        ]
        self._channel_count = len(self._channel_vec)
        self._frame_size = frame_size
        self._sampling_rate = sampling_rate
        self._data_buffer: list = []
        self._num_averages = self.config[self.Configuration.Keys.NUM_AVERAGES]
        self._display_buffer = np.zeros((frame_size, self._channel_count))
        self._new_data = False
        self._start_time = time.time()
        return super().setup(data, port_context_in)

    def _update(self):

        if not self._new_data:
            return

        # Set up UI elements. Note that this has to be done in the main Qt
        # thread (like this)
        ylim = (0, self._channel_count)
        if self._curves is None:

            # Create curves
            [self.add_curve() for _ in range(self._channel_count)]
            amp_lim = self.config[self.Configuration.Keys.AMPLITUDE_LIMIT]
            yl = f"EEG Amplitudes (0 ... {amp_lim} µV)"
            self.set_labels(x_label="Frequency (Hz)", y_label=yl)
            ticks = [
                (
                    self._channel_count - i - 0.5,
                    f"CH{self._channel_vec[i] + 1}",
                )
                for i in range(self._channel_count)
            ]
            self._plot_item.getAxis("left").setTicks([ticks])
            self._plot_item.setYRange(*ylim)

        with self._lock:
            if not self._data_buffer:
                return
            self._display_buffer = np.mean(
                np.stack(self._data_buffer, axis=2), axis=2
            )
            self._display_buffer = np.abs(self._display_buffer)
            self._new_data = False

        ch_lim_key = self.Configuration.Keys.AMPLITUDE_LIMIT
        ch_lim = self.config[ch_lim_key]
        for i in range(len(self._channel_vec)):
            d = self._channel_count - i - 0.5
            self._curves[i].setData(
                self._f_vec,
                self._display_buffer[:, self._channel_vec[i]] / ch_lim / 2 + d,
                antialias=False,
            )

        # update xlim
        fw = self._f_vec[-1]
        margin = fw * 0.0125
        xlim = (-margin, fw + margin)
        self._plot_item.setXRange(*xlim)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._lock:
            fft_input = data[PORT_IN]
            self._data_buffer.append(fft_input)
            if len(self._data_buffer) > self._num_averages:
                self._data_buffer.pop(0)
        self._new_data = True
