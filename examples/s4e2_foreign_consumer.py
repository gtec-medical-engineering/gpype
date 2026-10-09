"""Season 4, Episode 2: a consumer that is not g.Pype reads the stream.

Nothing here imports gpype: raw pylsl and pyqtgraph, which is what
somebody else's application sees when it subscribes to a stream g.Pype
publishes. Takes the first stream of type EEG it finds, and waits
until one appears. Requires pylsl, pyqtgraph and PySide6.
"""

import sys

import numpy as np
import pyqtgraph as pg
from pylsl import StreamInlet, resolve_byprop
from pyqtgraph.Qt import QtCore, QtWidgets

TIME_WINDOW = 10  # seconds shown
AMPLITUDE_LIMIT = 50  # one lane spans twice this


class LslScope(QtWidgets.QMainWindow):
    """A time-series plot fed by an LSL inlet."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSL consumer")

        # The stream description carries the channel count and the rate
        streams = resolve_byprop("type", "EEG")
        self.inlet = StreamInlet(streams[0])
        info = self.inlet.info()
        self.channel_count = info.channel_count()
        self.rate = info.nominal_srate()
        self.max_points = int(TIME_WINDOW * self.rate)

        plot = pg.PlotWidget()
        self.setCentralWidget(plot)
        self.item = plot.getPlotItem()
        self.item.showGrid(x=True, y=True, alpha=0.3)
        self.item.getViewBox().setMouseEnabled(x=False, y=False)
        self.item.setLabels(left="Channels", bottom="Time (s)")
        self.item.setYRange(0, self.channel_count)
        self.item.setXRange(0, TIME_WINDOW)
        self.item.getAxis("left").setTicks([[
            (self.channel_count - i - 0.5, f"CH{i + 1}")
            for i in range(self.channel_count)]])
        self.curves = [self.item.plot(pen=pg.mkPen(width=1))
                       for _ in range(self.channel_count)]

        # A ring buffer the size of the window
        self.times = np.arange(self.max_points) / self.rate
        self.buffer = np.zeros((self.max_points, self.channel_count))
        self.index = 0

        # Redraw on a timer, independent of the stream's rate
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)

    def update_plot(self):
        """Drain the inlet into the ring buffer and redraw."""
        while True:
            sample, _ = self.inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            self.buffer[self.index % self.max_points, :] = sample
            self.index += 1
        step = max(1, self.max_points // max(1, self.width()))
        for i, curve in enumerate(self.curves):
            lane = self.channel_count - i - 0.5
            curve.setData(self.times[::step],
                          self.buffer[::step, i] / AMPLITUDE_LIMIT / 2 + lane)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = LslScope()
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec())
