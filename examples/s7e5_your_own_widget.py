"""Season 7, Episode 5: a widget of your own, drawn on the GUI thread.

Requires the gui extra.
"""

import threading

import numpy as np

import gpype as gp

KEYS = gp.Constants.Keys
PORT_IN = gp.Constants.Defaults.PORT_IN


class LevelBars(gp.Scope):
    """One bar per channel: its RMS over the last half second."""

    #: Seconds of signal each bar is computed over.
    WINDOW_SECONDS = 0.5

    class Configuration(gp.Scope.Configuration):
        class Keys(gp.Scope.Configuration.Keys):
            LIMIT = "limit"

    def __init__(self, limit: float = 20.0, **kwargs):
        input_ports = kwargs.pop(KEYS.INPUT_PORTS,
                                 [gp.IPort.Configuration(name=PORT_IN)])
        super().__init__(input_ports=input_ports, limit=limit, **kwargs)
        self._lock = threading.Lock()
        self._window = None
        self._labels = None
        self._bars = None
        #: How often the bars were redrawn.
        self.repaints = 0
        # gp.Scope built no plot where this process does not draw.
        if self.widget is None:
            return
        self._plot_item.setYRange(0, limit)
        self.set_labels("channel", "RMS")

    def setup(self, data, port_context_in):
        # Runs on the pipeline's thread: size the buffer, draw nothing.
        context = port_context_in[PORT_IN]
        samples = int(context[KEYS.SAMPLING_RATE] * self.WINDOW_SECONDS)
        count = context[KEYS.CHANNEL_COUNT]
        self._window = np.zeros((samples, count), dtype=np.float32)
        labels = context.get(KEYS.CHANNEL_LABELS)
        self._labels = labels or [f"CH{i + 1}" for i in range(count)]
        return super().setup(data, port_context_in)

    def step(self, data):
        frame = data[PORT_IN]
        with self._lock:
            kept = np.vstack((self._window, frame))
            self._window = kept[-len(self._window):]
        return {}

    def _update(self):
        # Called by the Qt timer, on the GUI thread: the one place a
        # graphics item may be touched.
        if self._window is None:
            return
        import pyqtgraph as pg

        with self._lock:
            levels = np.sqrt(np.mean(self._window ** 2, axis=0))
        if self._bars is None:
            ticks = list(enumerate(self._labels))
            self._plot_item.getAxis("bottom").setTicks([ticks])
            self._bars = pg.BarGraphItem(x=np.arange(len(levels)),
                                         height=levels, width=0.6,
                                         brush=self._pen.color())
            self._plot_item.addItem(self._bars)
        else:
            self._bars.setOpts(height=levels)
        self.repaints += 1


if __name__ == "__main__":

    app = gp.MainApp()
    p = gp.Pipeline()

    # Two channels of alpha beside two of noise.
    alpha = gp.Generator(sampling_rate=250, channel_count=2,
                         signal_frequency=10, signal_amplitude=10.0,
                         noise_amplitude=1.0, name="alpha")
    noise = gp.Generator(sampling_rate=250, channel_count=2,
                         noise_amplitude=3.0, name="noise")
    both = gp.Router(input_channels=[gp.Router.ALL, gp.Router.ALL])
    bars = LevelBars(limit=10.0, name="levels")

    p.connect(alpha, both["in1"])
    p.connect(noise, both["in2"])
    p.connect(both, bars)
    app.add_widget(bars)

    p.start()
    app.run()
    p.stop()
    p.close()
