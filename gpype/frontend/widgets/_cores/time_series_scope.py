"""The widget half of TimeSeriesScope: everything that needs Qt.

Imported by TimeSeriesScope.create_internal_nodes only when the residency
actually builds a display. Kept out of the chain's own module so that a
server process -- which never builds one -- does not import Qt in order
to deserialize a document that names a scope.
"""

import threading
import time

import ioiocore as ioc
import numpy as np
import pyqtgraph as pg
from PySide6 import QtWidgets
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)

from ....backend.core.i_port import IPort
from ....common.constants import Constants
from .._validation import check_bounds
from ..base.scope import Scope
from ..time_series_scope import Markers, TimeSeriesScope

#: Default input port identifier for time series data
PORT_IN = ioc.Constants.Defaults.PORT_IN


class _TimeSeriesScopeCore(Scope):
    """Internal scope node for multi-channel time series visualization.

    Displays continuous time-series data from BCI pipelines with configurable
    time windows, amplitude scaling, channel hiding, and event markers.
    Wrapped by TimeSeriesScope for distributed operation.
    """

    #: Default display window duration in seconds
    DEFAULT_TIME_WINDOW: int = TimeSeriesScope.DEFAULT_TIME_WINDOW
    #: Default amplitude scale limit in microvolts
    DEFAULT_AMPLITUDE_LIMIT: float = TimeSeriesScope.DEFAULT_AMPLITUDE_LIMIT
    #: Repaints per second when the author does not choose.
    #:
    #: Read from the chain, like the two above, so there is one literal.
    #: It has to be restated *here* as well: Widget.__init__ falls back
    #: to ``self.DEFAULT_REFRESH_RATE``, which for this core resolves
    #: along the MRO to Widget's own 10 Hz -- the chain is not in that
    #: MRO. Without this line the chain would advertise 20 Hz to the
    #: catalog and to anyone reading it, and the widget that actually
    #: got built would repaint at 10.
    DEFAULT_REFRESH_RATE: float = TimeSeriesScope.DEFAULT_REFRESH_RATE

    #: Amplitude scale change per button press, in microvolts. An int, so
    #: that an authored int limit stays an int and the axis label does not
    #: gain a ".0" the moment the viewer touches it.
    AMPLITUDE_STEP: int = 5
    #: Time-window change per button press, in seconds.
    TIME_WINDOW_STEP: int = 5

    #: The public marker helper, which lives with the chain
    #: because gp.TimeSeriesScope.Markers is public API and
    #: must not require Qt to reach.
    Markers = Markers

    class Configuration(Scope.Configuration):
        """Configuration keys for TimeSeriesScope widget settings.

        Extends the base Scope configuration with time series specific
        parameters for display window, amplitude scaling, event markers,
        and channel visibility management.
        """

        class Keys(Scope.Configuration.Keys):
            """Required configuration parameter keys."""

            #: Configuration key for display duration in seconds
            TIME_WINDOW = "time_window"
            #: Configuration key for Y-axis scale in microvolts
            AMPLITUDE_LIMIT = "amplitude_limit"

        class EmptyableKeys:
            """Optional keys that may also be *empty*.

            Deliberately not ``OptionalKeys``. That name means one
            specific thing to ioiocore -- may be absent, but must not be
            empty when present (``Configuration.__init__``) -- and an
            empty list is a legitimate value for every key declared here:
            no markers, no hidden channels. Declaring them under the name
            the validator reads makes ``markers=[]`` raise.

            Measured, so that nobody has to rediscover it::

                OptionalKeys   markers=[]  -> ValueError, must not be empty
                EmptyableKeys  markers=[]  -> accepted

            The name is still the declaration a reader and a catalog
            generator look for; it is only kept out of the validator's
            emptiness rule. The underlying gap is ioiocore's: there is no
            way to say "optional, and empty is fine".
            """

            #: Configuration key for event marker configurations
            MARKERS = "markers"
            #: Configuration key for channels to hide from display
            HIDDEN_CHANNELS = "hidden_channels"

    class KeyPressFilter(QObject):
        """Event filter for keyboard shortcuts in the time series scope.

        Captures keyboard events to provide interactive functionality.
        Currently handles Alt+R for performance monitoring toggle.

        Args:
            callback: Function to call when target key combination is pressed.
        """

        def __init__(self, callback):
            """Initialize key press filter with callback function."""
            super().__init__()
            self.callback = callback

        def eventFilter(self, obj, event):
            """Filter keyboard events and trigger callbacks for shortcuts.

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
        show_amplitude_control: bool = True,
        show_time_window_control: bool = True,
        name: str = None,
        **kwargs,
    ):
        """Initialize the time series oscilloscope widget.

        Args:
            time_window: Display window duration in seconds; longer than 1 and
                shorter than 240.
                Uses DEFAULT_TIME_WINDOW if None.
            amplitude_limit: Y-axis scale limit in microvolts (1-5000).
                Uses DEFAULT_AMPLITUDE_LIMIT if None.
            markers: List of marker configurations for event visualization.
                Empty list if None.
            hidden_channels: List of channel indices to hide from display.
                Empty list if None.
            show_amplitude_control: Whether to offer the viewer buttons
                for the amplitude scale.
            show_time_window_control: Whether to offer the viewer buttons
                for the time window.
            **kwargs: Additional arguments passed to parent Scope class.

        Raises:
            ValueError: If time_window is not longer than 1 second and
                shorter than 240.
            ValueError: If amplitude_limit is outside reasonable range.
        """
        # Set default values if not provided
        if time_window is None:
            time_window = self.DEFAULT_TIME_WINDOW

        if amplitude_limit is None:
            amplitude_limit = self.DEFAULT_AMPLITUDE_LIMIT

        # The same function and the same constants the chain uses, so
        # there is one bound rather than two that happen to agree. The
        # chain is where a server reaches; this is where a directly
        # constructed core is guarded.
        check_bounds(
            "time_window",
            time_window,
            TimeSeriesScope.MIN_TIME_WINDOW,
            TimeSeriesScope.MAX_TIME_WINDOW,
            unit="s",
            inclusive=False,
        )
        time_window = round(time_window)
        check_bounds(
            "amplitude_limit",
            amplitude_limit,
            TimeSeriesScope.MIN_AMPLITUDE_LIMIT,
            TimeSeriesScope.MAX_AMPLITUDE_LIMIT,
            unit="uV",
        )

        # Initialize marker and hidden channel lists
        if markers is None:
            markers = []

        if hidden_channels is None:
            hidden_channels = []

        # Configure input ports for data reception
        ip_key = self.Configuration.Keys.INPUT_PORTS
        input_ports: list[IPort.Configuration] = kwargs.pop(
            ip_key, [IPort.Configuration(name=PORT_IN)]
        )

        # Set name if not provided
        if name is None:
            name = "Time Series Scope"

        # Initialize parent Scope class with configuration
        Scope.__init__(
            self,
            input_ports=input_ports,
            time_window=time_window,
            amplitude_limit=amplitude_limit,
            name=name,
            markers=markers,
            hidden_channels=hidden_channels,
            **kwargs,
        )

        # What the viewer is currently looking at, as opposed to what
        # the document asked for. An ioiocore Configuration is read-only
        # by construction -- "Configuration object is read-only. To store
        # user data, use contexts." -- and that is the right split: a
        # document should round-trip the pipeline that was authored, not
        # a scale somebody nudged mid-recording. Everything that draws
        # reads these two, so there is still one source per value.
        #: Amplitude scale currently drawn, in microvolts.
        self._amplitude_limit = amplitude_limit
        #: Time window currently drawn, in seconds.
        self._time_window = time_window

        # Interactive controls
        #: Whether to build the amplitude buttons.
        self._show_amplitude_control = show_amplitude_control
        #: Whether to build the time-window buttons.
        self._show_time_window_control = show_time_window_control
        #: Readout beside the amplitude buttons, built with them.
        self._amplitude_label = None
        #: Readout beside the time-window buttons, built with them.
        self._time_window_label = None

        # Continuity across a time-window change
        #: Samples written before the current buffer, banked whenever the
        #: window changes so that the time axis keeps counting up instead
        #: of restarting at zero.
        self._elapsed_samples = 0
        #: Whether tick labels need a decimal. A resize moves the sweep
        #: origin off whole seconds, so afterwards they do.
        self._fractional_ticks = False

        # Data buffer management
        #: Maximum number of displayable data points
        self._max_points: int = None
        #: Raw data storage buffer for all channels
        self._data_buffer: np.ndarray = None
        #: Processed display data for visible channels only
        self._display_buffer: np.ndarray = None
        #: X range currently applied to the view, so an unchanged one is
        #: not re-applied on every frame.
        self._xlim_drawn: tuple = None
        #: Current position index in circular buffer
        self._plot_index: int = 0
        #: Flag indicating circular buffer overflow status
        self._buffer_full: bool = False
        #: Global sample counter for data tracking
        self._sample_index: int = 0

        # Performance monitoring
        #: Widget initialization timestamp for rate calculations
        self._start_time = time.time()
        #: Counter for display update operations
        self._update_counts = 0
        #: Counter for data processing steps
        self._step_counts = 0
        #: Calculated data processing rate in Hz
        self._step_rate = 0

        # Thread synchronization
        #: Thread lock for safe data buffer access
        self._lock = threading.Lock()
        #: Flag indicating new data is available for display
        self._new_data = False
        #: Samples written since start, incremented under the lock. A
        #: monotone count rather than a flag, so the display can tell how
        #: much it has missed and a burst cannot be mistaken for silence.
        self._write_count = 0

        # UI components
        #: Label widget for displaying performance statistics
        self._rate_label = None

        # Theme and appearance setup
        p = self.widget.palette()
        #: Foreground color extracted from system theme
        self._foreground_color = p.color(QPalette.ColorRole.WindowText)
        #: Background color extracted from system theme
        self._background_color = p.color(QPalette.ColorRole.Window)

        # Interactive features
        #: Flag controlling performance statistics visibility
        self._show_rates = False
        #: Keyboard event filter for interactive shortcuts
        self._key_filter = self.KeyPressFilter(self._toggle_show_rates)
        self.widget.installEventFilter(self._key_filter)

    def _toggle_show_rates(self):
        """Toggle visibility of performance monitoring information.

        Shows or hides performance statistics including frame rates and
        processing rates. Triggered by the Alt+R keyboard shortcut.
        """
        self._show_rates = not self._show_rates
        self._rate_label.setVisible(self._show_rates)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Initialize the widget with data stream parameters and buffers.

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

        # Calculate buffer dimensions based on time window and sampling
        # rate. Taken from config rather than from the window currently
        # drawn, because the buffer geometry allocated here is cut to
        # it: a fresh run re-applies what the document asked for. The
        # amplitude scale is deliberately not treated the same way --
        # it allocates nothing, so there is nothing to reconcile, and
        # it stays where the viewer left it.
        time_window = self.config[self.Configuration.Keys.TIME_WINDOW]
        self._time_window = time_window
        self._refresh_time_window_label()
        self._max_points = int(round(time_window * sampling_rate))
        self._t_vec = np.arange(0, self._max_points) / sampling_rate

        # Determine visible channels (exclude hidden ones)
        hidden_channels = self.config[
            self.Configuration.EmptyableKeys.HIDDEN_CHANNELS
        ]
        self._channel_vec = [
            i for i in range(channel_count) if i not in hidden_channels
        ]
        self._channel_count = len(self._channel_vec)
        self._channel_labels = self._resolve_channel_labels(c)

        # Store processing parameters
        #: Number of samples per data frame from pipeline
        self._frame_size = frame_size
        #: Data acquisition sampling rate in Hz
        self._sampling_rate = sampling_rate
        #: Last displayed second for time axis tick updates
        self._last_second = None

        # Allocate data buffers
        # Raw data buffer holds all channels
        self._data_buffer = np.zeros((self._max_points, channel_count))
        # Display buffer holds only visible channels
        self._display_buffer = np.zeros(
            (self._max_points, self._channel_count)
        )
        # The x range follows the time window, which has just been
        # re-read, so the cached copy of it must not survive.
        self._xlim_drawn = None

        # Initialize state variables
        #: Flag indicating new data availability for display
        self._new_data = False
        #: Widget initialization timestamp
        self._start_time = time.time()
        #: Dictionary storing active event markers
        self._markers = {}

        # The time axis starts from zero again on a fresh run, so the
        # banked count and the fractional labels it forces go with it.
        self._elapsed_samples = 0
        self._fractional_ticks = False

        return super().setup(data, port_context_in)

    def _update(self):
        """Update the visual display with new data from the buffer.

        Called by the Qt timer to refresh the plot with latest data.
        Handles curve creation, data plotting, performance monitoring,
        and marker visualization. Only updates when new data is available.
        """
        if not self._new_data:
            return

        # Set up UI elements. Note that this has to be done in the main Qt
        # thread (like this)
        ylim = (0, self._channel_count)
        if self._curves is None:

            # Create curves for each visible channel
            [self.add_curve() for _ in range(self._channel_count)]
            self._refresh_axis_labels()

            # Configure channel labels on Y-axis
            ticks = [
                (
                    self._channel_count - i - 0.5,
                    self._channel_label(self._channel_vec[i]),
                )
                for i in range(self._channel_count)
            ]
            self._plot_item.getAxis("left").setTicks([ticks])
            self._plot_item.setYRange(*ylim)

            # Translucent foreground, for the rate overlay below.
            # A vertical cursor line used to be drawn here too, to
            # mark the write head; the NaN break in the curves
            # shows the same seam without a second item to keep in
            # step with it.
            col = QColor(self._foreground_color)  # makes a copy
            col.setAlpha(128)

            # Create performance monitoring labels
            x = self._time_window
            y = ylim[1]
            font_size = 7
            self._rate_label = pg.TextItem(
                text="", color=QColor(col), anchor=(1, 1)
            )
            self._plot_item.addItem(self._rate_label)
            self._rate_label.setPos(x, y)
            self._rate_label.setFont(QFont("Arial", font_size))

            # Interactive controls, built once and on the GUI
            # thread, for the same reason everything else here is.
            self._create_controls()

        # Update data with decimation for performance optimization.
        # Decimation factor N reduces displayed points based on widget
        # width -- which is 0 until the widget has been laid out, in
        # an unselected tab or a collapsed splitter, and dividing by
        # it raised ZeroDivisionError on the GUI thread.
        width = max(1, self.widget.width())
        N = max(1, int(self._max_points / width))
        with self._lock:
            # Copy decimated data from buffer for visible channels only
            self._display_buffer[::N, :] = self._data_buffer[
                ::N, self._channel_vec
            ]
            self._new_data = False
            sample_idx = self._sample_index - 1

        # Where the sweep is writing, in samples and in plot points
        cursor_pos = sample_idx % self._max_points
        break_idx = cursor_pos // N

        # Update x-axis ticks dynamically
        time_window = self._time_window
        cur_second = int(np.floor(cursor_pos / self._sampling_rate))
        if cur_second != self._last_second:
            # Seconds since the display started: at the cursor, then
            # at x=0 of the current sweep. Stated as one subtraction
            # rather than as modulo arithmetic over the sweep count,
            # which is what it replaced -- this form is checkable by
            # eye, and it is the one that survives a mid-run resize,
            # because _elapsed_samples carries what the previous
            # buffers held.
            elapsed = self._elapsed_samples + sample_idx
            base = (elapsed - cursor_pos) / self._sampling_rate
            digits = 1 if self._fractional_ticks else 0
            span = int(np.floor(time_window)) + 1
            if sample_idx > self._max_points:
                # The buffer has wrapped, so everything right of the
                # cursor is the previous sweep: one window older.
                ticks = [
                    (
                        i,
                        (
                            f"{base + i:.{digits}f}"
                            if i <= cur_second
                            else f"{base + i - time_window:.{digits}f}"
                        ),
                    )
                    for i in range(span)
                ]
            else:
                # First sweep: nothing right of the cursor has been
                # written, so label only what it has passed.
                ticks = [
                    (
                        i,
                        f"{base + i:.{digits}f}" if i <= cur_second else "",
                    )
                    for i in range(span)
                ]
            self._plot_item.getAxis("bottom").setTicks([ticks])
            self._last_second = cur_second

        # Plot channel data with amplitude scaling and vertical
        # offset, broken at the write head. Drawn as one run of
        # points the curve joins the newest sample straight back to
        # the oldest one, painting a full-scale vertical line across
        # every lane -- an artefact of the ring buffer that is
        # nowhere in the data. A NaN with connect="finite" leaves a
        # gap there instead.
        ch_lim = self._amplitude_limit
        t_plot = self._t_vec[::N]
        n_points = len(t_plot)
        gap_half = max(1, int(0.005 * time_window * self._sampling_rate / N))
        for i in range(len(self._channel_vec)):
            # Vertical offset: each channel gets its own "lane"
            d = self._channel_count - i - 0.5
            y_plot = self._display_buffer[::N, i] / ch_lim / 2 + d
            if 0 <= break_idx < n_points - 1:
                gap_from = max(0, break_idx - gap_half)
                gap_to = min(n_points - 1, break_idx + gap_half)
                self._curves[i].setData(
                    np.concatenate(
                        (
                            t_plot[:gap_from],
                            [np.nan],
                            t_plot[gap_to + 1 :],
                        )
                    ),
                    np.concatenate(
                        (
                            y_plot[:gap_from],
                            [np.nan],
                            y_plot[gap_to + 1 :],
                        )
                    ),
                    connect="finite",
                    antialias=False,
                )
            else:
                self._curves[i].setData(t_plot, y_plot, antialias=False)

        # Update x-axis range with small margin. Applied only when it
        # has actually moved: it follows the time window alone, so
        # setting it unconditionally asked the view to re-range to the
        # value it already held on every single frame.
        margin = time_window * 0.0125
        xlim = (-margin, time_window + margin)
        if xlim != self._xlim_drawn:
            self._plot_item.setXRange(*xlim)
            self._xlim_drawn = xlim

        # Update event markers: detect state changes in marker channels
        mk_key = self.Configuration.EmptyableKeys.MARKERS
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
            # max(1, ...): sample_idx is 0 on the very first frame of
            # a frame_size=1 stream, and this divided by it.
            update_rate = (
                self._update_counts / max(1, sample_idx) * self._sampling_rate
            )
            self._rate_label.setText(
                f"data rate: {self._step_rate:.1f} Hz, "
                f"refresh rate: {update_rate:.1f} Hz"
            )

        # Mark all layers dirty to force redraw.
        #
        # The rectangle is left null on purpose. Qt reads a null rect as
        # "the whole scene", which is exactly what sceneRect() was being
        # asked for -- but computing it walks every item in the scene,
        # and a PlotCurveItem answers by taking nanmin and nanmax over
        # the points it holds. So the old form scanned every sample of
        # every curve, twice per curve, to produce an argument whose
        # only effect was to mean "everything".
        #
        # Measured offscreen, 10 s at 250 Hz, 8 channels, 900 px
        # (2500 points, decimation 2): this one line was 0.475 s of the
        # 0.693 s spent in 300 repaints -- 69% of the redraw -- via 2400
        # boundingRect and 4800 dataBounds calls. Dropping the
        # sceneRect() argument takes _update() from 1.292 ms a call to
        # 0.415 ms, which is what pays for the default refresh rate
        # going from 10 Hz to 30: three times the repaints at 1.24% of
        # one core, against 1.29% for the old 10 Hz display.
        scene = self._plot_item.scene()
        scene.invalidate(layers=QtWidgets.QGraphicsScene.AllLayers)

    def _refresh_axis_labels(self) -> None:
        """Restate the axis labels for the amplitude currently drawn.

        The y label carries the scale, so it is rewritten whenever the
        viewer changes it. Left alone, the axis would keep announcing
        the authored scale while the curves were drawn to another one --
        a display that lies about its own units.
        """
        amp_lim = self._amplitude_limit
        self.set_labels(
            x_label="Time (s)",
            y_label=f"EEG Amplitudes (-{amp_lim} ... +{amp_lim} µV)",
        )

    def _refresh_amplitude_label(self) -> None:
        """Show the amplitude currently drawn on the control bar."""
        if self._amplitude_label is not None:
            self._amplitude_label.setText(f"±{self._amplitude_limit:g} µV")

    def _refresh_time_window_label(self) -> None:
        """Show the time window currently drawn on the control bar."""
        if self._time_window_label is not None:
            self._time_window_label.setText(f"{self._time_window:g} s")

    @staticmethod
    def _add_step_buttons(layout, on_decrease, on_increase) -> None:
        """Append a -/+ button pair to a control bar.

        Args:
            layout: Layout to append the buttons to.
            on_decrease: Handler for the "-" button.
            on_increase: Handler for the "+" button.
        """
        for text, handler in (("-", on_decrease), ("+", on_increase)):
            button = QPushButton(text)
            button.setFixedSize(24, 24)
            # Read by the dark style sheet: at this size its side
            # padding would leave no room for the glyph, and the
            # button renders blank. See gpype.frontend.theme.
            button.setProperty("compact", True)
            button.clicked.connect(handler)
            layout.addWidget(button)

    def _create_controls(self) -> None:
        """Build the amplitude and time-window control bar.

        Inserted above the plot and inside the group box: ``self._layout``
        *is* the group box's content layout -- the one the plot widget
        was added to -- so the bar needs no walk down the widget tree to
        find its place.

        Called from :meth:`_update`, which is the GUI thread, once.
        """
        if not (
            self._show_amplitude_control or self._show_time_window_control
        ):
            return

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(8, 3, 8, 3)
        layout.setSpacing(6)

        if self._show_amplitude_control:
            layout.addWidget(QLabel("Amplitude"))
            self._amplitude_label = QLabel()
            layout.addWidget(self._amplitude_label)
            self._add_step_buttons(
                layout, self._decrease_amplitude, self._increase_amplitude
            )
            self._refresh_amplitude_label()

        if self._show_amplitude_control and self._show_time_window_control:
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.VLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            layout.addWidget(separator)

        if self._show_time_window_control:
            layout.addWidget(QLabel("Time Window"))
            self._time_window_label = QLabel()
            layout.addWidget(self._time_window_label)
            self._add_step_buttons(
                layout,
                self._decrease_time_window,
                self._increase_time_window,
            )
            self._refresh_time_window_label()

        layout.addStretch()
        self._layout.insertWidget(0, container)

    def _increase_amplitude(self) -> None:
        """Step the amplitude scale up, within the declared range."""
        self._set_amplitude_limit(self._amplitude_limit + self.AMPLITUDE_STEP)

    def _decrease_amplitude(self) -> None:
        """Step the amplitude scale down, within the declared range."""
        self._set_amplitude_limit(self._amplitude_limit - self.AMPLITUDE_STEP)

    def _set_amplitude_limit(self, amplitude_limit) -> None:
        """Draw at a new amplitude scale, clamped to the declared range.

        Clamped against the same MIN/MAX the constructor enforces, not
        against the step size. The prototype this came from stopped one
        step above zero and had no upper bound at all, so a few clicks
        took the display to a scale the constructor would have refused.

        Args:
            amplitude_limit: Requested scale, in microvolts.
        """
        amplitude_limit = min(
            TimeSeriesScope.MAX_AMPLITUDE_LIMIT,
            max(TimeSeriesScope.MIN_AMPLITUDE_LIMIT, amplitude_limit),
        )
        if amplitude_limit == self._amplitude_limit:
            return
        self._amplitude_limit = amplitude_limit
        self._refresh_amplitude_label()
        self._refresh_axis_labels()
        # Redraw even though no frame has arrived: the scale changed, so
        # what is on screen is now drawn to the wrong one.
        self._new_data = True

    def _increase_time_window(self) -> None:
        """Step the time window up, within the declared range."""
        self._set_time_window(self._time_window + self.TIME_WINDOW_STEP)

    def _decrease_time_window(self) -> None:
        """Step the time window down, within the declared range."""
        self._set_time_window(self._time_window - self.TIME_WINDOW_STEP)

    def _set_time_window(self, time_window) -> None:
        """Draw a new time window, re-cutting the sweep buffers.

        MIN_TIME_WINDOW and MAX_TIME_WINDOW are *exclusive* -- the
        constructor rejects a window sitting on either -- so a step is
        clamped to the first whole second inside them rather than to the
        bound itself.

        Args:
            time_window: Requested window length, in seconds.
        """
        time_window = min(
            TimeSeriesScope.MAX_TIME_WINDOW - 1,
            max(TimeSeriesScope.MIN_TIME_WINDOW + 1, time_window),
        )
        if time_window == self._time_window:
            return
        self._time_window = time_window
        self._refresh_time_window_label()
        # None forces the tick labels to be rebuilt on the next frame,
        # which they otherwise would not be until the second changed.
        self._last_second = None
        if self._rate_label is not None:
            self._rate_label.setPos(time_window, self._channel_count)
        self._resize_buffers()
        self._new_data = True

    def _resize_buffers(self) -> None:
        """Re-cut the sweep buffers for the time window now drawn.

        Nothing to do before setup: the buffers do not exist yet, and
        setup sizes them from the window this leaves behind.

        The write index restarts, so the elapsed-sample count is banked
        first. That is what lets the time axis keep counting up across
        the change instead of jumping back to zero, and it is why the
        labels turn fractional afterwards -- the sweep origin no longer
        lands on a whole second.

        The whole reallocation is held under the lock, because step()
        writes into these very attributes from the pipeline thread.
        """
        if self._data_buffer is None:
            return
        with self._lock:
            self._elapsed_samples += self._sample_index
            max_points = int(round(self._time_window * self._sampling_rate))
            channel_count = self._data_buffer.shape[1]
            self._data_buffer = np.zeros((max_points, channel_count))
            self._display_buffer = np.zeros((max_points, self._channel_count))
            self._max_points = max_points
            self._t_vec = np.arange(0, max_points) / self._sampling_rate
            self._sample_index = 0
        # The x range follows the time window, which has just changed,
        # so the cached copy of it is stale.
        self._xlim_drawn = None
        self._fractional_ticks = True

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process incoming data frames and store them in the circular buffer.

        Called by the pipeline for each new data frame. Handles performance
        monitoring, circular buffer management, and thread-safe data storage.

        Args:
            data: Dictionary containing input data arrays from connected ports.
                Expected to have PORT_IN key with shape (frame_size, channels).

        Returns:
            dict: Empty dictionary (this is a sink node with no outputs).
        """
        self._step_counts += 1

        # Calculate data processing rate for performance monitoring
        t_el = time.time() - self._start_time + 1e-10
        self._step_rate = self._step_counts / t_el

        # The write length comes from the frame that actually arrived, not
        # from the frame size declared at setup. Sync stacks whatever it
        # has held into one output frame, so the two differ in normal
        # operation, and trusting the declaration raised a shape mismatch
        # that became a permanent error condition naming no cause.
        block = data[PORT_IN]

        # Thread-safe data storage in circular buffer. Everything
        # that reads _max_points or _sample_index is inside the
        # lock, because a time-window change replaces both buffers
        # and resets the index from the GUI thread. Computing the
        # write offset outside it meant a resize landing in between
        # wrote a frame at an offset into the previous buffer's
        # geometry -- and, when the window shrank, made `split`
        # negative and raised the very shape mismatch this method
        # had already been taught to avoid once.
        with self._lock:
            max_points = self._max_points
            n = block.shape[0]
            if n > max_points:
                block = block[-max_points:, :]
                n = max_points

            start = self._sample_index % max_points
            end = start + n

            if end <= max_points:
                self._data_buffer[start:end, :] = block
            else:
                split = max_points - start
                self._data_buffer[start:, :] = block[:split]
                self._data_buffer[: end - max_points, :] = block[split:]
            self._sample_index = self._sample_index + n
            # Counted inside the lock so the display cannot see a count
            # that describes samples it has not been shown yet.
            self._write_count += n

        # Signal that new data is available for display update
        self._new_data = True
