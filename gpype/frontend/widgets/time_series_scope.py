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

# Default input port for time series data
PORT_IN = ioc.Constants.Defaults.PORT_IN


class TimeSeriesScope(Scope):
    """
    Real-time oscilloscope widget for multi-channel time series visualization.

    Displays continuous time-series data from BCI pipelines with multiple
    channels in a stationary oscilloscope-style view. Features include
    configurable time windows, amplitude scaling, channel hiding, event
    markers, and real-time performance monitoring.

    The widget automatically handles data buffering, display scaling,
    time axis management, and provides interactive features like keyboard
    shortcuts for toggling performance information.

    Features:
        - Multi-channel time series visualization
        - Configurable time window and amplitude limits
        - Channel selection and hiding capabilities
        - Event marker support with custom colors and labels
        - Real-time performance monitoring (Alt+R)
        - Automated, internal data decimation for performance
        - Thread-safe data handling

    Args:
        time_window (int): Display window duration in seconds (1-120).
        amplitude_limit (float): Y-axis scale limit in microvolts.
        markers (list): Event markers configuration.
        hidden_channels (list): Channel indices to hide from display.
        **kwargs: Additional arguments passed to parent Scope class.
    """

    # Default configuration values
    DEFAULT_TIME_WINDOW: int = 10  # seconds
    DEFAULT_AMPLITUDE_LIMIT: float = 50  # microvolts

    class Markers(dict):
        """
        Container for event marker configuration.

        Stores marker properties including visual appearance (color),
        identification (label), channel association, and trigger value
        for event-based visualization in the time series plot.

        Args:
            color: Hex color code or color name for marker visualization.
            label: Text label for marker identification.
            channel: Channel index where marker should be displayed.
            value: Trigger value that activates the marker.
        """

        def __init__(self, color: str, label: str, channel: int, value: int):
            """Initialize marker configuration with display properties."""
            super().__init__()
            self["color"] = color
            self["label"] = label
            self["channel"] = channel
            self["value"] = value

    class Configuration(Scope.Configuration):
        """
        Configuration keys for TimeSeriesScope widget settings.

        Extends the base Scope configuration with time series specific
        parameters for display window, amplitude scaling, event markers,
        and channel visibility management.
        """

        class Keys(Scope.Configuration.Keys):
            """Required configuration parameter keys."""

            TIME_WINDOW = "time_window"  # Display duration in seconds
            AMPLITUDE_LIMIT = "amplitude_limit"  # Y-axis scale in microvolts

        class KeysOptional:
            """Optional configuration parameter keys."""

            MARKERS = "markers"  # Event marker configurations
            HIDDEN_CHANNELS = "hidden_channels"  # Channels to hide

    class KeyPressFilter(QObject):
        """
        Event filter for keyboard shortcuts in the time series scope.

        Captures and processes keyboard events to provide interactive
        functionality like toggling performance information display.
        Currently handles Alt+R for performance monitoring toggle.

        Args:
            callback: Function to call when target key combination is pressed.
        """

        def __init__(self, callback):
            """Initialize key press filter with callback function."""
            super().__init__()
            self.callback = callback

        def eventFilter(self, obj, event):
            """
            Filter keyboard events and trigger callbacks for shortcuts.

            Args:
                obj: Qt object that received the event.
                event: Qt event to process.

            Returns:
                bool: False to allow event propagation, True to consume event.
            """
            if event.type() == QEvent.KeyPress:
                if event.key() == Qt.Key_R:
                    if event.modifiers() & Qt.AltModifier:
                        self.callback()
            return False

    def __init__(
        self,
        time_window: int = None,
        amplitude_limit: float = None,
        markers: list = None,
        hidden_channels: list = None,
        **kwargs,
    ):
        """
        Initialize the time series oscilloscope widget.

        Sets up the plotting area, data buffers, timer configurations,
        and user interface elements for real-time time series visualization.
        Validates input parameters and configures the widget for optimal
        performance with multi-channel BCI data streams.

        Args:
            time_window: Display window duration in seconds (1-120).
                Uses DEFAULT_TIME_WINDOW if None.
            amplitude_limit: Y-axis scale limit in microvolts (1-5000).
                Uses DEFAULT_AMPLITUDE_LIMIT if None.
            markers: List of marker configurations for event visualization.
                Empty list if None.
            hidden_channels: List of channel indices to hide from display.
                Empty list if None.
            **kwargs: Additional arguments passed to parent Scope class.

        Raises:
            ValueError: If time_window is outside valid range (1-120).
            ValueError: If amplitude_limit is outside reasonable range.
        """
        # Set default values if not provided
        if time_window is None:
            time_window = self.DEFAULT_TIME_WINDOW

        if amplitude_limit is None:
            amplitude_limit = self.DEFAULT_AMPLITUDE_LIMIT

        # Validate time window parameters
        if time_window <= 1:
            raise ValueError("time_window must be longer than 1 second.")
        if time_window >= 240:
            raise ValueError("time_window must be shorter than 240 seconds.")
        time_window = round(time_window)

        # Validate amplitude limit parameters
        if amplitude_limit > 5e3 or amplitude_limit < 1:
            raise ValueError("amplitude_limit without reasonable range.")

        # Initialize marker and hidden channel lists
        if markers is None:
            markers = []

        if hidden_channels is None:
            hidden_channels = []

        # Configure input ports for data reception
        input_ports = [IPort.Configuration(name=PORT_IN)]

        # Initialize parent Scope class with configuration
        Scope.__init__(
            self,
            input_ports=input_ports,
            time_window=time_window,
            amplitude_limit=amplitude_limit,
            name="Time Series Scope",
            markers=markers,
            hidden_channels=hidden_channels,
            **kwargs,
        )

        # Data buffer management
        self._max_points: int = None  # Maximum displayable points
        self._data_buffer: np.ndarray = None  # Raw data storage buffer
        self._display_buffer: np.ndarray = None  # Processed display data
        self._plot_index: int = 0  # Current plot position
        self._buffer_full: bool = False  # Buffer overflow indicator
        self._sample_index: int = 0  # Current sample index

        # Performance monitoring
        self._start_time = time.time()  # Widget start timestamp
        self._update_counts = 0  # Display update counter
        self._step_counts = 0  # Processing step counter
        self._step_rate = 0  # Calculated step rate

        # Thread synchronization
        self._lock = threading.Lock()  # Data access synchronization
        self._new_data = False  # New data availability flag

        # UI components
        self._rate_label = None  # Performance display label

        # Theme and appearance setup
        p = self.widget.palette()
        self._foreground_color = p.color(QPalette.ColorRole.WindowText)
        self._background_color = p.color(QPalette.ColorRole.Window)

        # Interactive features
        self._show_rates = False  # Performance visibility toggle
        self._cursor = None  # Data cursor

        # Install keyboard event filter for shortcuts
        self._key_filter = self.KeyPressFilter(self._toggle_show_rates)
        self.widget.installEventFilter(self._key_filter)

    def _toggle_show_rates(self):
        """
        Toggle visibility of performance monitoring information.

        Shows or hides the performance statistics display that includes
        frame rates, processing rates, and timing information. This method
        is triggered by the Alt+R keyboard shortcut.
        """
        self._show_rates = not self._show_rates
        self._rate_label.setVisible(self._show_rates)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Initialize the widget with data stream parameters and allocate buffers.

        Sets up the time series scope based on incoming data characteristics
        including sampling rate, channel count, and frame size. Allocates
        appropriate buffer sizes and initializes display parameters.

        Args:
            data: Input data dictionary (not used in setup phase).
            port_context_in: Context information for input ports containing
                sampling rate, channel count, and frame size.

        Returns:
            dict: Updated port context for downstream components.

        Raises:
            ValueError: If required context parameters are missing.
        """
        c = port_context_in[PORT_IN]

        # Extract and validate required context parameters
        sampling_rate = c.get(Constants.Keys.SAMPLING_RATE)
        if sampling_rate is None:
            raise ValueError("sampling rate must be provided.")
        channel_count = c.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("channel count must be provided.")
        frame_size = c.get(Constants.Keys.FRAME_SIZE)
        if frame_size is None:
            raise ValueError("frame size must be provided.")

        # Calculate buffer dimensions based on time window and sampling rate
        time_window = self.config[self.Configuration.Keys.TIME_WINDOW]
        self._max_points = int(round(time_window * sampling_rate))
        self._t_vec = np.arange(0, self._max_points) / sampling_rate

        # Determine visible channels (exclude hidden ones)
        hidden_channels = self.config[
            self.Configuration.KeysOptional.HIDDEN_CHANNELS
        ]
        self._channel_vec = [
            i for i in range(channel_count) if i not in hidden_channels
        ]
        self._channel_count = len(self._channel_vec)

        # Store processing parameters
        self._frame_size = frame_size
        self._sampling_rate = sampling_rate
        self._last_second = None

        # Allocate data buffers
        # Raw data buffer holds all channels
        self._data_buffer = np.zeros((self._max_points, channel_count))
        # Display buffer holds only visible channels
        self._display_buffer = np.zeros(
            (self._max_points, self._channel_count)
        )

        # Initialize state variables
        self._new_data = False
        self._start_time = time.time()
        self._markers = {}

        return super().setup(data, port_context_in)

    def _update(self):
        """
        Update the visual display with new data from the buffer.

        This method is called by the Qt timer to refresh the plot with
        the latest data. It handles curve creation, data plotting,
        performance monitoring, and marker visualization. The method
        operates in the main Qt thread for UI safety.

        Note:
            This method only updates when new data is available
            (_new_data flag). UI elements are created lazily on first
            update call.
        """
        if not self._new_data:
            return

        # Set up UI elements. Note that this has to be done in the main Qt
        # thread (like this)
        ylim = (0, self._channel_count)
        if self._curves is None:

            # Create curves for each visible channel
            [self.add_curve() for _ in range(self._channel_count)]
            amp_lim = self.config[self.Configuration.Keys.AMPLITUDE_LIMIT]
            yl = f"EEG Amplitudes (-{amp_lim} ... +{amp_lim} µV)"
            self.set_labels(x_label="Time (s)", y_label=yl)

            # Configure channel labels on Y-axis
            ticks = [
                (
                    self._channel_count - i - 0.5,
                    f"CH{self._channel_vec[i] + 1}",
                )
                for i in range(self._channel_count)
            ]
            self._plot_item.getAxis("left").setTicks([ticks])
            self._plot_item.setYRange(*ylim)

            # Create cursor for time tracking
            col = QColor(self._foreground_color)  # makes a copy
            col.setAlpha(128)
            pen = pg.mkPen(self._pen)
            pen.setColor(col)
            self._cursor = pg.PlotCurveItem(pen=pen)
            self._plot_item.addItem(self._cursor)

            # Create performance monitoring labels
            x = self.config[self.Configuration.Keys.TIME_WINDOW]
            y = ylim[1]
            font_size = 7
            self._rate_label = pg.TextItem(
                text="", color=QColor(col), anchor=(1, 1)
            )
            self._plot_item.addItem(self._rate_label)
            self._rate_label.setPos(x, y)
            self._rate_label.setFont(QFont("Arial", font_size))

        # Update data with decimation for performance optimization
        # Decimation factor N reduces displayed points based on widget width
        N = np.maximum(
            1, int(np.floor(self._max_points / self.widget.width()))
        )
        with self._lock:
            # Copy decimated data from buffer for visible channels only
            self._display_buffer[::N, :] = self._data_buffer[
                ::N, self._channel_vec
            ]
            self._new_data = False
            sample_idx = self._sample_index - 1

        # Plot time cursor to show current data position
        t_cursor = (sample_idx % self._max_points) / self._sampling_rate
        self._cursor.setData([t_cursor] * 2, [*ylim], antialias=False)

        # Update x-axis ticks dynamically
        cur_second = int(
            np.floor((sample_idx % self._max_points) / self._sampling_rate)
        )
        if cur_second != self._last_second:
            time_window = self.config[self.Configuration.Keys.TIME_WINDOW]
            if sample_idx > self._max_points:
                ticks = []
                for i in range(int(np.floor(time_window)) + 1):
                    tick_val = (
                        np.mod(i - (cur_second + 1), time_window)
                        + cur_second
                        + 1
                    )
                    offset = (
                        np.floor(sample_idx / self._max_points - 1)
                        * time_window
                    )
                    tick_val += offset
                    tick_label = f"{tick_val:.0f}"
                    ticks.append((i, tick_label))
            else:
                ticks = [
                    (i, f"{i:.0f}" if i <= cur_second else "")
                    for i in range(int(np.floor(time_window)) + 1)
                ]
            self._plot_item.getAxis("bottom").setTicks([ticks])
            self._last_second = cur_second

        # Plot channel data with amplitude scaling and vertical offset
        ch_lim_key = TimeSeriesScope.Configuration.Keys.AMPLITUDE_LIMIT
        ch_lim = self.config[ch_lim_key]
        for i in range(len(self._channel_vec)):
            # Vertical offset: each channel gets its own "lane"
            d = self._channel_count - i - 0.5
            self._curves[i].setData(
                self._t_vec[::N],
                self._display_buffer[::N, i] / ch_lim / 2 + d,
                antialias=False,
            )

        # Update x-axis range with small margin
        tw = self.config[self.Configuration.Keys.TIME_WINDOW]
        margin = tw * 0.0125
        xlim = (-margin, tw + margin)
        self._plot_item.setXRange(*xlim)

        # Update event markers: detect state changes in marker channels
        mk_key = self.Configuration.KeysOptional.MARKERS
        markers: dict = {}
        for m in self.config[mk_key]:
            ch = m["channel"]
            val = m["value"]
            # Find rising edges where marker value appears
            hit = (
                np.where(
                    (self._data_buffer[1:, ch] == val)
                    & (self._data_buffer[:-1, ch] != val)
                )[0]
                + 1
            )
            for h in hit:
                id = hash(tuple([h, ch, val]))
                markers[id] = {"index": h, "curve": None, **m}

        # Add new markers to plot
        for k in {
            k: markers[k]
            for k in markers.keys()
            if k not in self._markers.keys()
        }:
            m = markers[k]
            idx = m["index"]
            # Create text label for marker
            text = pg.TextItem(
                text=m["label"], anchor=(0.5, 1), color=pg.mkColor(m["color"])
            )
            self._plot_item.addItem(text)
            text.setPos(self._t_vec[idx], self._channel_count)
            # Create vertical line for marker
            curve = self._plot_item.plot(pen=pg.mkPen(pg.mkColor(m["color"])))
            curve.setData(self._t_vec[[idx, idx]], np.array([*ylim]))
            markers[k]["curve"] = curve
            markers[k]["text"] = text
            self._markers[k] = markers[k]

        # Remove outdated markers from plot
        for k in {
            k: self._markers[k]
            for k in self._markers.keys()
            if k not in markers.keys()
        }:
            m = self._markers[k]
            self._plot_item.removeItem(m["curve"])
            self._plot_item.removeItem(m["text"])
            del self._markers[k]

        # Update performance monitoring display
        self._update_counts += 1
        if self._show_rates:
            update_rate = (
                self._update_counts / sample_idx * self._sampling_rate
            )
            self._rate_label.setText(
                f"data rate: {self._step_rate:.1f} Hz, "
                f"refresh rate: {update_rate:.1f} Hz"
            )

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Process incoming data frames and store them in the circular buffer.

        This method is called by the pipeline for each new data frame.
        It handles performance monitoring, circular buffer management,
        and thread-safe data storage for real-time visualization.

        Args:
            data: Dictionary containing input data arrays from connected ports.
                Expected to have PORT_IN key with shape (frame_size, channels).

        Returns:
            dict: Empty dictionary (this is a sink node with no outputs).
        """
        # Initialize performance monitoring on first call
        if self._step_counts == 0:
            self._start_time = time.time()
        self._step_counts += 1

        # Calculate data processing rate for performance monitoring
        t_el = time.time() - self._start_time
        self._step_rate = self._step_counts / t_el

        # Calculate circular buffer write indices
        write_idx = self._sample_index + np.arange(self._frame_size)
        write_idx %= self._max_points

        # Thread-safe data storage in circular buffer
        with self._lock:
            self._data_buffer[write_idx, :] = data[PORT_IN]
            self._sample_index = self._sample_index + self._frame_size

        # Signal that new data is available for display update
        self._new_data = True
