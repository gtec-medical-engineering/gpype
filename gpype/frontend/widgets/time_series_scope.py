import numpy as np
from typing import Dict
from .base.scope import Scope
import ioiocore as ioc
from ...utilities.constants import Constants


PORT_IN = ioc.Constants.Defaults.PORT_IN


class TimeSeriesScope(Scope):

    PAD_DURATION = 0.1  # zero padding ahead of cursor in seconds

    class Configuration(Scope.Configuration):

        class Keys(Scope.Configuration.Keys):
            TIME_WINDOW = 'time_window'
            AMPLITUDE_LIMIT = 'amplitude_limit'

    def __init__(self,
                 time_window: int = 10,
                 amplitude_limit: float = 50,
                 **kwargs):

        if time_window <= 1:
            raise ValueError("time_window must be longer than 1 second.")
        if time_window >= 120:
            raise ValueError("time_window must be shorter than "
                             "120 seconds.")
        time_window = round(time_window)

        if amplitude_limit > 5e3 or amplitude_limit < 1:
            raise ValueError("amplitude_limit without reasonable range.")

        Scope.__init__(self,
                       time_window=time_window,
                       amplitude_limit=amplitude_limit,
                       name="Time Series Scope",
                       **kwargs)
        self._max_points: int = None
        self._data: np.ndarray = None
        self._plot_index: int = 0
        self._buffer_full: bool = False
        self._buffer_index: int = 0
#         self._plot_item.setTitle("EEG Time Series")

        self._name = "Time Series Scope"

    def setup(self,
              data: Dict[str, np.ndarray],
              port_metadata_in: Dict[str, dict]) -> Dict[str, dict]:
        md = port_metadata_in[PORT_IN]
        sampling_rate = md.get(ioc.Constants.Keys.SAMPLING_RATE)
        if sampling_rate is None:
            raise ValueError("sampling rate must be provided.")
        channel_count = md.get(ioc.Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("channel count must be provided.")
        time_window = self.config[self.Configuration.Keys.TIME_WINDOW]
        self._max_points = int(round(time_window * sampling_rate))
        self.t_vec = np.arange(0, self._max_points) / sampling_rate
        self._data = np.zeros((self._max_points, channel_count))
        self._channel_count = channel_count
        self._sampling_rate = sampling_rate
        self._last_second = None
        pd = TimeSeriesScope.PAD_DURATION
        self._pad_count = int(round(pd * sampling_rate))
        return super().setup(data, port_metadata_in)

    def _update(self):

        # return if no data is available
        if self._data is None:
            return

        # set up self._curves, because this here is the main thread as required
        # by Qt. This is not the case in the setup method.
        if self._curves is None:
            [self.add_curve() for _ in range(self._channel_count)]
            self.set_labels(x_label='Time (s)', y_label='EEG Amplitudes')
            ticks = [(self._channel_count - i - 0.5, f'CH{i + 1}')
                     for i in range(self._channel_count)]
            self._plot_item.getAxis('left').setTicks([ticks])
            ylim = (0, self._channel_count)
            self._plot_item.setYRange(*ylim)

        # update x-axis ticks
        cur_second = int(np.ceil(self.get_counter() / self._sampling_rate))
        if cur_second != self._last_second:
            tw_key = TimeSeriesScope.Configuration.Keys.TIME_WINDOW
            time_window = self.config[tw_key]
            if cur_second > time_window:
                ticks = [(i, f'{np.mod(i - cur_second, time_window) + cur_second - time_window:.0f}')  # noqa: E501
                         for i in range(np.floor(time_window))]
            else:
                ticks = [(i, f'{i:.0f}' if i < cur_second else '')
                         for i in range(time_window)]
            self._plot_item.getAxis('bottom').setTicks([ticks])
            self._last_second = cur_second

        # update data
        ch_lim_key = TimeSeriesScope.Configuration.Keys.AMPLITUDE_LIMIT
        ch_lim = self.config[ch_lim_key]
        for i in range(self._data.shape[1]):
            d = self._channel_count - i - 0.5
            self._curves[i].setData(self.t_vec, self._data[:, i] / ch_lim + d)

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        self._data[self._buffer_index, :] = data[PORT_IN][:, :]
        pad_idx = (self._buffer_index + np.arange(1, self._pad_count))
        self._data[pad_idx % self._max_points, :] = 0
        self._buffer_index = (self.get_counter() + 1) % self._max_points
