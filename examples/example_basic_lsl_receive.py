"""
Basic LSL Receive Example - Lab Streaming Layer Reception and Visualization

This example demonstrates how to receive and visualize data from Lab Streaming
Layer (LSL) streams in real-time. It complements example_basic_lsl_send.py by
showing the receiving side of LSL communication, creating a standalone
visualization application for LSL data streams.

What this example shows:
- Automatic discovery and connection to LSL streams on the network
- Real-time data reception from LSL protocol
- Multi-channel time-series visualization using PyQtGraph
- Professional EEG-style display with channel stacking
- Continuous scrolling visualization with proper time axis

Expected behavior:
When you run this example:
- Automatically discovers available LSL streams (type "EEG")
- Connects to the first available stream
- Opens a real-time visualization window
- Displays incoming data as scrolling time-series plots
- Shows channel names and time axis with proper scaling

Workflow with LSL Send example:
1. Run example_basic_lsl_send.py (creates LSL stream)
2. Run this script (connects to and visualizes the stream)
3. Press arrow keys in the send example to see event markers
4. Observe real-time data updates in the visualization

Real-world applications:
- Real-time monitoring of BCI experiments
- Quality assessment of incoming data streams
- Integration with third-party BCI software
- Network-based data visualization systems
- Multi-application BCI setups
- LSL ecosystem testing and debugging

Technical features:
- Automatic LSL stream discovery by type ("EEG")
- Dynamic channel count detection
- Efficient circular buffer for continuous data
- Adaptive plot decimation for smooth display
- Professional scientific visualization styling
- Scrolling time axis with proper labeling

Visualization details:
- Channel stacking: Each channel offset vertically for clarity
- Time window: 10 seconds of scrolling history
- Amplitude scaling: Automatic normalization for display
- Refresh rate: ~25 Hz for smooth real-time updates
- Grid lines: Optional grid overlay for easier reading

Usage:
    1. Ensure an LSL stream is available (run example_basic_lsl_send.py)
    2. Run: python example_basic_lsl_receive.py
    3. Visualization window opens automatically
    4. Close window to stop reception

Prerequisites:
    - pylsl library (pip install pylsl)
    - pyqtgraph (pip install pyqtgraph)
    - PySide6 (pip install PySide6)
    - Active LSL stream on the network
"""

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PySide6.QtGui import QPalette, QColor
from pylsl import StreamInlet, resolve_byprop
import sys

# Display configuration constants
SAMPLING_RATE = 250  # Expected sampling rate in Hz
TIME_WINDOW = 10  # Seconds of data to display
AMPLITUDE_LIMIT = 50  # µV - amplitude scaling for display

MAX_POINTS = int(TIME_WINDOW * SAMPLING_RATE)  # Circular buffer size


class LSLTimeScope(QtWidgets.QMainWindow):
    """Real-time LSL data visualization widget with multi-channel display."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("LSL Time Series Scope")

        # Apply system theme colors for professional appearance
        palette = self.palette()
        self.foreground_color = palette.color(QPalette.ColorRole.WindowText)
        self.background_color = palette.color(QPalette.ColorRole.Window)

        # Create main plot widget with scientific styling
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(x=True, y=True, alpha=0.3)  # Subtle grid
        self.plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        self.plot_widget.setBackground(self.background_color)

        # Discover and connect to LSL stream
        print("Resolving LSL stream...")
        streams = resolve_byprop("type", "EEG")  # Find EEG-type streams
        self.inlet = StreamInlet(streams[0])  # Connect to first available
        info = self.inlet.info()
        self.CHANNEL_COUNT = info.channel_count()  # Dynamic channel detection
        self.FRAME_SIZE = 1  # Pull one sample at a time

        # Configure plot layout and styling
        self.plot_item.setLabels(left="Channels", bottom="Time (s)")
        self.plot_item.setYRange(0, self.CHANNEL_COUNT)  # Fit all channels

        # Create channel labels (CH1, CH2, etc.) on Y-axis
        self.plot_item.getAxis("left").setTicks(
            [
                [
                    (self.CHANNEL_COUNT - i - 0.5, f"CH{i + 1}")
                    for i in range(self.CHANNEL_COUNT)
                ]
            ]
        )
        self.plot_item.setXRange(-0.5, TIME_WINDOW + 0.5)  # Time axis range

        # Create individual plot curves for each channel
        self.curves = []
        for i in range(self.CHANNEL_COUNT):
            # Each channel gets its own colored curve
            curve = self.plot_item.plot(
                pen=pg.mkPen(
                    QColor(self.foreground_color), width=1  # noqa: E501
                )
            )
            self.curves.append(curve)

        # Initialize data buffers for efficient circular storage
        self.t_vec = np.arange(MAX_POINTS) / SAMPLING_RATE  # Time vector
        self.data_buffer = np.zeros((MAX_POINTS, self.CHANNEL_COUNT))
        self.sample_index = 0  # Current sample position

        # Setup display refresh timer for smooth real-time updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(40)  # 25 Hz refresh rate for smooth animation

        self._last_second = None  # Track time axis updates

    def update_plot(self):
        """Update the real-time plot with new LSL data."""
        # Pull all available samples from LSL stream (non-blocking)
        while True:
            sample, timestamp = self.inlet.pull_sample(timeout=0.0)
            if sample is None:  # No more samples available
                break
            # Convert sample to numpy array and store in circular buffer
            frame = np.array(sample, dtype=np.float64).reshape(
                (1, self.CHANNEL_COUNT)
            )
            idx = self.sample_index % MAX_POINTS  # Circular buffer index
            self.data_buffer[idx, :] = frame
            self.sample_index += 1

        # Update plot curves with decimated data for performance
        N = max(1, int(MAX_POINTS / self.width()))  # Decimation factor
        t_disp = self.t_vec[::N]  # Decimated time vector
        for i, curve in enumerate(self.curves):
            # Stack channels vertically with offset for clear separation
            offset = self.CHANNEL_COUNT - i - 0.5
            curve.setData(
                t_disp, self.data_buffer[::N, i] / AMPLITUDE_LIMIT / 2 + offset
            )  # noqa: E501

        # Update time axis labels for scrolling display
        cur_second = int(
            np.floor((self.sample_index % MAX_POINTS) / SAMPLING_RATE)
        )  # noqa: E501
        if cur_second != self._last_second:
            time_window = TIME_WINDOW
            if self.sample_index > MAX_POINTS:
                # Scrolling mode: update time labels as data scrolls
                ticks = []
                for i in range(int(np.floor(time_window)) + 1):
                    tick_val = (
                        np.mod(i - (cur_second + 1), time_window)
                        + cur_second
                        + 1
                    )  # noqa: E501
                    offset = (
                        np.floor(self.sample_index / MAX_POINTS - 1)
                        * time_window
                    )  # noqa: E501
                    tick_val += offset
                    tick_label = f"{tick_val:.0f}"
                    ticks.append((i, tick_label))
            else:
                # Initial filling mode: simple incremental time labels
                ticks = [
                    (i, f"{i:.0f}" if i <= cur_second else "")
                    for i in range(int(np.floor(time_window)) + 1)
                ]
            self.plot_item.getAxis("bottom").setTicks([ticks])
            self._last_second = cur_second


if __name__ == "__main__":
    # Create Qt application and LSL visualization window
    app = QtWidgets.QApplication(sys.argv)
    window = LSLTimeScope()
    window.resize(1000, 500)  # Set reasonable window size
    window.show()
    sys.exit(app.exec())  # Start Qt event loop
