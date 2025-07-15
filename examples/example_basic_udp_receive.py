"""
Basic UDP Receive Example - Network Data Reception and Visualization

This example demonstrates how to receive and visualize data transmitted over
UDP networks in real-time. It complements example_basic_udp_send.py by showing
the receiving side of UDP communication, creating a standalone visualization
application for UDP data streams.

What this example shows:
- UDP socket programming for real-time data reception
- Multi-channel time-series visualization using PyQtGraph
- Binary data unpacking from network packets
- EEG display with channel stacking
- Continuous visualization in real-time scope

Expected behavior:
When you run this example:
- Opens UDP socket on localhost:56000 for incoming data
- Displays real-time visualization window
- Shows incoming multi-channel data as scrolling time-series plots
- Updates display at ~25 Hz for smooth real-time visualization
- Handles network timing variations and packet buffering

Workflow with UDP Send example:
1. Run this script first (starts UDP receiver and visualization)
2. Run example_basic_udp_send.py (connects and streams data)
3. Press arrow keys in the send example to see event markers
4. Observe real-time data updates in the visualization

Network configuration:
- Protocol: UDP (User Datagram Protocol)
- IP Address: 127.0.0.1 (localhost)
- Port: 56000 (configurable)
- Data format: Binary packed float64 arrays
- Packet size: Configurable frame size × channel count × 8 bytes

Real-world applications:
- Real-time BCI data monitoring and quality assessment
- Network-based signal analysis and processing
- Distributed BCI systems with remote visualization
- Integration with custom data acquisition hardware
- Multi-computer BCI setups for specialized processing
- Remote data logging and backup systems

Usage:
    1. Run: python example_basic_udp_receive.py
    2. Visualization window opens and waits for UDP data
    3. Run example_basic_udp_send.py to start data transmission
    4. Close window to stop reception

Prerequisites:
    - pyqtgraph (pip install pyqtgraph)
    - PySide6 (pip install PySide6)
    - Active UDP data source on the configured port

Note:
    UDP provides fast, low-latency communication ideal for real-time BCI
    applications where speed is prioritized over guaranteed packet delivery.
"""

import socket
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PySide6.QtGui import QPalette, QColor
import sys

# Network and display configuration constants
UDP_IP = "127.0.0.1"      # Listen on localhost
UDP_PORT = 56000          # UDP port for incoming data
FRAME_SIZE = 1            # Samples per UDP packet
CHANNEL_COUNT = 9         # Total channels (8 signals + 1 events)
SAMPLING_RATE = 250       # Expected sampling rate in Hz
TIME_WINDOW = 10          # Seconds of data to display
AMPLITUDE_LIMIT = 50      # µV - amplitude scaling for display

# Calculated constants for data handling
MAX_POINTS = int(TIME_WINDOW * SAMPLING_RATE)  # Total points in buffer
BUFFER_SIZE = 65536       # UDP receive buffer size


class UDPTimeScope(QtWidgets.QMainWindow):
    """
    Real-time UDP data visualization application.

    Creates a time-series display for multi-channel data received
    via UDP. Implements circular buffering and real-time plotting
    with PyQtGraph for smooth visualization of streaming data.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UDP Time Series Scope")

        # Get system colors for professional appearance
        palette = self.palette()
        self.foreground_color = palette.color(QPalette.ColorRole.WindowText)
        self.background_color = palette.color(QPalette.ColorRole.Window)

        # Create main plot widget with scientific visualization styling
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)  # Subtle grid
        # Disable mouse interaction for fixed view
        self.plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self.plot_widget.setBackground(self.background_color)

        # Configure axes labels and ranges for EEG-style display
        self.plot_item.setLabels(left="Channels", bottom="Time (s)")
        self.plot_item.setYRange(0, CHANNEL_COUNT)  # Stack channels vertically

        # Create channel labels (CH1, CH2, etc.) positioned at channel centers
        self.plot_item.getAxis('left').setTicks([[
            (CHANNEL_COUNT - i - 0.5, f"CH{i + 1}") for i in range(CHANNEL_COUNT)  # noqa: E501
        ]])
        self.plot_item.setXRange(-0.5, TIME_WINDOW + 0.5)

        # Create individual plot curves for each channel
        self.curves = []
        for _ in range(CHANNEL_COUNT):
            # Each channel gets its own curve with consistent styling
            curve = self.plot_item.plot(pen=pg.mkPen(QColor(self.foreground_color), width=1))  # noqa: E501
            self.curves.append(curve)

        # Initialize data buffers for circular buffering
        self.t_vec = np.arange(MAX_POINTS) / SAMPLING_RATE  # Time vector
        # Circular buffer for multi-channel data storage
        self.data_buffer = np.zeros((MAX_POINTS, CHANNEL_COUNT))
        self.sample_index = 0  # Current position in circular buffer

        # Set up UDP socket for non-blocking data reception
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((UDP_IP, UDP_PORT))  # Bind to specified address/port
        self.sock.setblocking(False)  # Non-blocking for real-time operation

        # Set up timer for regular plot updates (independent of data rate)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)  # ~25 Hz refresh rate for smooth visualization

        self._last_second = None  # Track time axis updates

    def update_plot(self):
        """
        Main update loop called by timer for real-time visualization.

        Handles UDP packet reception, data buffering, and plot updates.
        Runs at ~25 Hz for smooth visualization independent of data rate.
        """
        # Receive and process all available UDP packets
        try:
            while True:
                # Non-blocking receive - get packet if available
                packet, _ = self.sock.recvfrom(BUFFER_SIZE)

                # Unpack binary data: convert bytes to float64 array
                frame = np.frombuffer(packet, dtype=np.float64)
                frame = frame.reshape((FRAME_SIZE, CHANNEL_COUNT))

                # Store in circular buffer with proper indexing
                idx = self.sample_index + np.arange(FRAME_SIZE)
                idx %= MAX_POINTS  # Wrap around for circular buffering
                self.data_buffer[idx, :] = frame
                self.sample_index += FRAME_SIZE

        except BlockingIOError:
            # No data available - continue with plot update
            pass

        # Update visualization with decimated data for performance
        # Decimate data based on window width to avoid oversampling display
        N = max(1, int(MAX_POINTS / self.width()))
        display = self.data_buffer[::N]  # Take every Nth sample
        t_disp = self.t_vec[::N]         # Corresponding time points

        # Update each channel curve with normalized and offset data
        for i, curve in enumerate(self.curves):
            # Stack channels vertically with amplitude scaling
            offset = CHANNEL_COUNT - i - 0.5  # Vertical position
            # Normalize amplitude and add channel offset
            curve.setData(t_disp, display[:, i] / AMPLITUDE_LIMIT / 2 + offset)

        # Update time axis labels for scrolling display
        cur_second = int(np.floor((self.sample_index % MAX_POINTS) / SAMPLING_RATE))  # noqa: E501
        if cur_second != self._last_second:
            time_window = TIME_WINDOW
            if self.sample_index > MAX_POINTS:
                # Scrolling mode: calculate proper time labels
                ticks = []
                for i in range(int(np.floor(time_window)) + 1):
                    tick_val = np.mod(i - (cur_second + 1), time_window) + cur_second + 1  # noqa: E501
                    offset = np.floor(self.sample_index / MAX_POINTS - 1) * time_window  # noqa: E501
                    tick_val += offset
                    tick_label = f'{tick_val:.0f}'
                    ticks.append((i, tick_label))
            else:
                # Initial filling mode: simple sequential labels
                ticks = [(i, f'{i:.0f}' if i <= cur_second else '')
                         for i in range(int(np.floor(time_window)) + 1)]

            # Apply new tick labels to time axis
            self.plot_item.getAxis('bottom').setTicks([ticks])
            self._last_second = cur_second


if __name__ == '__main__':
    # Create Qt application for GUI event loop
    app = QtWidgets.QApplication(sys.argv)

    # Create and configure main window
    window = UDPTimeScope()
    window.resize(1000, 500)  # Set reasonable window size
    window.show()

    # Start application event loop (blocks until window closes)
    sys.exit(app.exec())
