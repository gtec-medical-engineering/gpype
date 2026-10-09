"""Season 4, Episode 3: a receiver that is not g.Pype reads the packets.

Nothing here imports gpype: a socket and a plot. A packet carries no
header, so the port, the channel count and the number format below
have to match the sending pipeline. Requires pyqtgraph and PySide6.
"""

import socket
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

UDP_IP = "127.0.0.1"
UDP_PORT = 56000
CHANNEL_COUNT = 9  # eight signals and the key codes
SAMPLING_RATE = 250
TIME_WINDOW = 10  # seconds shown
AMPLITUDE_LIMIT = 50  # one lane spans twice this
MAX_POINTS = TIME_WINDOW * SAMPLING_RATE


class UdpScope(QtWidgets.QMainWindow):
    """A time-series plot fed by a UDP socket."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UDP receiver")

        plot = pg.PlotWidget()
        self.setCentralWidget(plot)
        self.item = plot.getPlotItem()
        self.item.showGrid(x=True, y=True, alpha=0.3)
        self.item.getViewBox().setMouseEnabled(x=False, y=False)
        self.item.setLabels(left="Channels", bottom="Time (s)")
        self.item.setYRange(0, CHANNEL_COUNT)
        self.item.setXRange(0, TIME_WINDOW)
        self.item.getAxis("left").setTicks([[
            (CHANNEL_COUNT - i - 0.5, f"CH{i + 1}")
            for i in range(CHANNEL_COUNT)]])
        self.curves = [self.item.plot(pen=pg.mkPen(width=1))
                       for _ in range(CHANNEL_COUNT)]

        # A ring buffer the size of the window
        self.times = np.arange(MAX_POINTS) / SAMPLING_RATE
        self.buffer = np.zeros((MAX_POINTS, CHANNEL_COUNT))
        self.index = 0

        # A non-blocking socket, drained on a timer
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))
        self.sock.setblocking(False)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)

    def update_plot(self):
        """Drain the socket into the ring buffer and redraw."""
        try:
            while True:
                packet, _ = self.sock.recvfrom(65536)
                # One frame of 64-bit floats, sample by sample; how many
                # samples follows from the packet's size
                frame = np.frombuffer(packet, dtype=np.float64)
                frame = frame.reshape((-1, CHANNEL_COUNT))
                rows = (self.index + np.arange(len(frame))) % MAX_POINTS
                self.buffer[rows, :] = frame
                self.index += len(frame)
        except BlockingIOError:
            pass  # nothing more to read
        step = max(1, MAX_POINTS // max(1, self.width()))
        for i, curve in enumerate(self.curves):
            lane = CHANNEL_COUNT - i - 0.5
            curve.setData(self.times[::step],
                          self.buffer[::step, i] / AMPLITUDE_LIMIT / 2 + lane)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = UdpScope()
    window.resize(1000, 500)
    window.show()
    sys.exit(app.exec())
