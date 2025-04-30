import numpy as np
from typing import Dict
from .base.scope import Scope
import ioiocore as ioc
from ...common.constants import Constants
from ...backend.core.i_port import IPort


PORT_IN = ioc.Constants.Defaults.PORT_IN
PORT_SQ = 'sq'


class TimeSeriesScope(Scope):

    PAD_DURATION = 0.1  # zero padding ahead of cursor in seconds

    class Configuration(Scope.Configuration):

        class Keys(Scope.Configuration.Keys):
            TIME_WINDOW = 'time_window'
            AMPLITUDE_LIMIT = 'amplitude_limit'

        class KeysOptional:
            MARKERS = 'markers'
            HIDDEN_CHANNELS = 'hidden_channels'

    def __init__(self,
                 time_window: int = 10,
                 amplitude_limit: float = 50,
                 markers: list = None,
                 hidden_channels: list = None,
                 enable_sq: bool = False,
                 **kwargs):

        if time_window <= 1:
            raise ValueError("time_window must be longer than 1 second.")
        if time_window >= 120:
            raise ValueError("time_window must be shorter than "
                             "120 seconds.")
        time_window = round(time_window)

        if amplitude_limit > 5e3 or amplitude_limit < 1:
            raise ValueError("amplitude_limit without reasonable range.")

        if markers is None:
            markers = []

        if hidden_channels is None:
            hidden_channels = []

        input_ports = [IPort.Configuration(name=PORT_IN)]
        if enable_sq:
            input_ports.append(IPort.Configuration(name=PORT_SQ))

        Scope.__init__(self,
                       input_ports=input_ports,
                       time_window=time_window,
                       amplitude_limit=amplitude_limit,
                       name="Time Series Scope",
                       markers=markers,
                       hidden_channels=hidden_channels,
                       **kwargs)

        self._max_points: int = None
        self._data: np.ndarray = None
        self._plot_index: int = 0
        self._buffer_full: bool = False
        self._buffer_index: int = 0
        self._markers: dict = None
        self._name = "Time Series Scope"
        self._enable_sq = enable_sq
        if enable_sq:
            self._sq_indicator = []
            self._sq_labels = []

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
        self._sq = np.zeros((1, channel_count))
        self._channel_count = channel_count
        hidden_channels = self.config[self.Configuration.KeysOptional.HIDDEN_CHANNELS]  # noqa: E501
        self._channel_vec = [i for i in range(channel_count)
                             if i not in hidden_channels]
        self._channel_count = len(self._channel_vec)
        self._sampling_rate = sampling_rate
        self._last_second = None
        pd = TimeSeriesScope.PAD_DURATION
        self._pad_count = int(round(pd * sampling_rate))
        self._markers = {}
        return super().setup(data, port_metadata_in)

    def _update(self):

        import pyqtgraph as pg

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

            if self._enable_sq:
                self._sq_indicator = [[] for _ in range(self._channel_count)]
                self._sq_labels = [None for _ in range(self._channel_count)]
                for i in self._channel_vec:
                    y_pos = self._channel_count - i - 0.5
                    for color in ['g', 'orange', 'r']:
                        circ = pg.ScatterPlotItem(
                            [self.t_vec[0] - 0.2],
                            [y_pos],
                            size=10,
                            brush=pg.mkBrush(color),
                            pen=pg.mkPen('k'),
                            pxMode=True,
                            name=f'light_{i}_{color}',
                            opacity=0.1
                        )
                        self._plot_item.addItem(circ)
                        self._sq_indicator[i].append(circ)

                    from pyqtgraph import TextItem
                    text = TextItem(text="",
                                    anchor=(0, 0.5),
                                    color=(200, 200, 200))
                    self._plot_item.addItem(text)
                    text.setPos(self.t_vec[-1] + 0.4, y_pos)
                    self._sq_labels[i] = text

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
        for i in self._channel_vec:
            d = self._channel_count - i - 0.5
            self._curves[i].setData(self.t_vec, self._data[:, i] / ch_lim + d)

        # update xlim
        tw = self.config[self.Configuration.Keys.TIME_WINDOW]
        margin = tw * 0.0125
        xlim = (-margin, tw + margin)
        self._plot_item.setXRange(*xlim)

        # update markers
        mk_key = self.Configuration.KeysOptional.MARKERS
        markers: dict = {}
        for m in self.config[mk_key]:
            ch = m['channel']
            val = m['value']
            hit = np.where((self._data[1:, ch] == val) & (self._data[:-1, ch] != val))[0] + 1  # noqa: E501
            for h in hit:
                if (self._buffer_index - h) % self._max_points > self._max_points - self._pad_count:  # noqa: E501
                    continue
                id = hash(tuple([h, ch, val]))
                markers[id] = {'index': h, 'curve': None, **m}

        # add markers
        for k in {k: markers[k] for k in markers.keys()
                  if k not in self._markers.keys()}:
            m = markers[k]
            idx = m['index']
            text = pg.TextItem(text=m['label'],
                               anchor=(0, 1),
                               color=pg.mkColor(m['color']))
            self._plot_item.addItem(text)
            text.setPos(self.t_vec[idx], self._channel_count)
            curve = self._plot_item.plot(pen=pg.mkPen(pg.mkColor(m['color'])))
            view_box = self._plot_item.getViewBox()
            top = view_box.mapSceneToView(text.boundingRect().topLeft()).y()
            curve.setData(self.t_vec[[idx, idx]], np.array([0, top]))
            markers[k]['curve'] = curve
            markers[k]['text'] = text
            self._markers[k] = markers[k]

        # remove markers
        for k in {k: self._markers[k] for k in self._markers.keys()
                  if k not in markers.keys()}:
            m = self._markers[k]
            self._plot_item.removeItem(m['curve'])
            self._plot_item.removeItem(m['text'])
            del self._markers[k]

        if self._enable_sq:
            # update SQI
            for i in self._channel_vec:
                sq_val = self._sq[0, i]
                sqi = self._sq_indicator[i]
                if abs(sq_val) < 0.5:
                    active = 0  # green
                elif abs(sq_val) < 1:
                    active = 1  # orange
                else:
                    active = 2  # red
                for j, light in enumerate(sqi):
                    light.setOpacity(1.0 if j == active else 0.1)

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:  # noqa: E501
        self._data[self._buffer_index, :] = data[PORT_IN][:, :]
        pad_idx = (self._buffer_index + np.arange(1, self._pad_count))
        self._data[pad_idx % self._max_points, :] = 0
        self._buffer_index = (self.get_counter() + 1) % self._max_points
        if self._enable_sq:
            self._sq = data[PORT_SQ][-1:, :]
