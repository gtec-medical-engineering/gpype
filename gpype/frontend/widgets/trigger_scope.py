"""TriggerScope, importable with no Qt installed.

A server constructs this node to describe the display an edge draws,
and may have no PySide6 at all, so every Qt symbol is reached from
inside a function. ``test_frontend_widgets_import_without_qt`` holds it.
"""

from __future__ import annotations

import colorsys
import threading
import time
from typing import Optional

import ioiocore as ioc
import numpy as np
from sympy import lambdify

from ...backend.core._private.chain_params import parse_plot_expressions
from ...backend.core.i_port import IPort
from ...backend.flow.trigger import Trigger
from ...common._private import channels
from ...common.constants import Constants
from ._validation import check_bounds
from .base.scope import Scope

#: Default input port identifier for trigger data
PORT_IN = ioc.Constants.Defaults.PORT_IN


class TriggerScope(Scope):
    """Event-triggered oscilloscope widget for analyzing signal epochs.

    Displays the epochs cut by an upstream Trigger as their average: every
    epoch received since the pipeline started, not each epoch on its own.
    One input port per free variable of the plot expressions, each
    bridged on its own stream where the pipeline is distributed.
    """

    #: Default Y-axis scale limit in microvolts.
    DEFAULT_AMPLITUDE_LIMIT: float = 50.0

    #: One input per plot expression, so the names come from `plots`.
    INPUT_PORTS_FROM = "plots"

    #: Caption when the author names the scope nothing.
    CAPTION = "Trigger Scope"

    #: Accepted range for ``amplitude_limit``, in microvolts. Named so
    #: that a form can offer a slider with the right bounds.
    MIN_AMPLITUDE_LIMIT: float = 1.0
    MAX_AMPLITUDE_LIMIT: float = 5_000.0

    class Configuration(Scope.Configuration):
        """Configuration keys for TriggerScope widget settings.

        Extends the base Scope configuration with trigger-specific
        parameters for amplitude scaling and mathematical plot expressions.
        """

        class Keys(Scope.Configuration.Keys):
            """Required configuration parameter keys."""

            #: Configuration key for Y-axis scale limit
            AMPLITUDE_LIMIT = "amplitude_limit"
            #: Configuration key for mathematical expression list
            PLOTS = "plots"

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

            #: Configuration key for channels to hide from display
            HIDDEN_CHANNELS = "hidden_channels"

    def __init__(
        self,
        amplitude_limit: float = DEFAULT_AMPLITUDE_LIMIT,
        plots: list = None,
        hidden_channels: list = None,
        refresh_rate: float = None,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the trigger scope widget.

        Sets up mathematical expression parsing, input port configuration,
        and display parameters for event-triggered signal visualization.

        Args:
            amplitude_limit: Y-axis scale limit in microvolts (1-5000).
            plots: List of mathematical expressions to evaluate and plot.
                Empty list defaults to [PORT_IN] for direct signal display.
            hidden_channels: List of channel indices to hide from display.
                Empty list if None.
            refresh_rate: Repaints per second. Defaults to
                DEFAULT_REFRESH_RATE.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments passed to parent Scope class.

        Raises:
            ValueError: If amplitude_limit is outside reasonable range.
        """
        # Validate amplitude limit parameters
        check_bounds(
            "amplitude_limit",
            amplitude_limit,
            self.MIN_AMPLITUDE_LIMIT,
            self.MAX_AMPLITUDE_LIMIT,
            unit="uV",
        )

        # Initialize configuration lists
        hidden_channels = channels.as_selection(
            [] if hidden_channels is None else hidden_channels,
            "hidden_channels",
        )

        if plots is None:
            plots = [Constants.Defaults.PORT_IN]

        # Initialize expression parsing storage
        #: List of compiled lambda functions for mathematical expressions
        self._plot_funcs = []
        #: Argument lists for each compiled function
        self._plot_args = []

        # The port names are declared on this node's input ports, and
        # the Link the pipeline puts in front of it reads them there.
        #: Required input port names extracted from expressions
        self._port_names, parsed = parse_plot_expressions(plots)
        for expression, variables in parsed:
            self._plot_funcs.append(
                lambdify(variables, expression, modules="numpy")
            )
            self._plot_args.append([str(var) for var in variables])

        # Configure input ports based on required variables. A stored
        # configuration wins: it already describes the ports that were
        # derived when the scope was first built, and passing them here
        # as well would be a duplicate keyword.
        #
        # ASYNC, because what this plots is epochs, and every node that
        # cuts them -- Trigger, EpochAverage -- emits them ASYNC. It was
        # INHERITED, which a Link cannot take ("OPort does not support
        # inherited timing"), so the Link a distributed pipeline puts in
        # front of this scope fell back to SYNC, and connecting a
        # Trigger to it was refused on the SERVER and on the EDGE.
        keys = self.Configuration.Keys
        input_ports = kwargs.pop(
            keys.INPUT_PORTS,
            [
                IPort.Configuration(
                    name=name,
                    type=np.ndarray.__name__,
                    timing=Constants.Timing.ASYNC,
                )
                for name in self._port_names
            ],
        )

        # Initialize parent Scope class with configuration
        Scope.__init__(
            self,
            input_ports=input_ports,
            amplitude_limit=amplitude_limit,
            plots=plots,
            hidden_channels=hidden_channels,
            refresh_rate=refresh_rate,
            edge_id=edge_id,
            **kwargs,
        )

        # Data buffer management
        #: Maximum number of displayable data points
        self._max_points: int = None
        #: Raw trigger epoch storage organized by port name
        self._data_buffer: dict = None
        #: Processed display data after expression evaluation
        self._display_buffer: np.ndarray = None
        #: Current position index in display buffer
        self._plot_index: int = 0
        #: Flag indicating buffer overflow condition
        self._buffer_full: bool = False
        #: Global sample counter for data tracking
        self._sample_index: int = 0

        # Performance monitoring
        #: Widget initialization timestamp for rate calculations
        self._start_time = time.time()
        #: Counter for display update operations
        self._update_counts = 0
        #: Counter for trigger epoch processing steps
        self._step_counts = 0
        #: Calculated trigger processing rate in Hz
        self._step_rate = 0

        # Thread synchronization
        #: Thread lock for safe data buffer access
        self._lock = threading.Lock()
        #: Flag indicating new trigger data is available
        self._new_data = False

        # UI components
        #: Label widget for displaying performance statistics
        self._rate_label = None
        #: Vertical cursor line marking trigger time (t=0)
        self._cursor = None
        #: Number of mathematical plot functions to evaluate
        self._no_plots = len(self._plot_funcs)

        # Everything past here reads a Qt palette or builds a legend on
        # the plot item, and a SERVER has neither -- see base/scope.py
        # for why a scope is constructed there at all.
        if self.widget is None:
            self._foreground_color = None
            self._background_color = None
            self._legend = None
            return

        from PySide6.QtGui import QBrush, QPalette, QPen

        # Theme and appearance setup
        p = self.widget.palette()
        #: Foreground color extracted from system theme
        self._foreground_color = p.color(QPalette.ColorRole.WindowText)
        #: Background color extracted from system theme
        self._background_color = p.color(QPalette.ColorRole.Window)

        # Setup legend for multi-plot visualization
        if len(self._port_names) > 1 or len(self._plot_funcs) > 1:
            from pyqtgraph.graphicsItems.LegendItem import LegendItem

            # Create custom legend positioned in top-left corner within plot
            self._legend = LegendItem(offset=(10, 10))
            self._legend.setParentItem(self._plot_item.getViewBox())
            self._legend.anchor((-0.1, 0), (0, 0))  # Anchor to top-left
            bg_color = self._background_color
            fg_color = self._foreground_color

            self._legend.setBrush(QBrush(bg_color))
            self._legend.setPen(QPen(fg_color))
            self._legend.setVisible(False)
        else:
            #: Legend widget for multi-plot identification (None if single)
            self._legend = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Initialize the widget with trigger parameters and allocate buffers.

        Sets up the trigger scope based on upstream trigger node configuration
        including timing, sampling parameters, and channel configuration.

        Args:
            data: Input data dictionary (not used in setup phase).
            port_context_in: Context information for input ports containing
                trigger timing and signal parameters.

        Returns:
            dict: Updated port context for downstream components.

        Raises:
            ValueError: If required context parameters are missing or
                       inconsistent across ports.
        """
        import pyqtgraph as pg

        # Extract and validate required context parameters
        keys = [
            Constants.Keys.SAMPLING_RATE,
            Constants.Keys.CHANNEL_COUNT,
            Constants.Keys.FRAME_SIZE,
            Trigger.Configuration.Keys.TIME_PRE,
            Trigger.Configuration.Keys.TIME_POST,
        ]
        context = {}

        # Ensure all ports have consistent parameter values
        for key in keys:
            values = {}
            for port, config in port_context_in.items():
                if key in config:
                    values[port] = config[key]

            value_list = list(values.values())

            if not all(value == value_list[0] for value in value_list[1:]):
                raise ValueError(f"{key} must be the same for all ports.")

            context[key] = value_list[0]

        # Extract validated parameters
        sampling_rate = context.get(Constants.Keys.SAMPLING_RATE)
        channel_count = context.get(Constants.Keys.CHANNEL_COUNT)
        frame_size = context.get(Constants.Keys.FRAME_SIZE)

        # Create time vector for trigger epochs
        self._t_vec = np.arange(0, frame_size) / sampling_rate
        time_pre = context[Trigger.Configuration.Keys.TIME_PRE]
        self._t_vec -= time_pre  # Adjust time vector for pre-trigger period

        # Determine visible channels (exclude hidden ones)
        hidden_channels = self.config[
            self.Configuration.EmptyableKeys.HIDDEN_CHANNELS
        ]
        self._channel_vec = [
            i for i in range(channel_count) if i not in hidden_channels
        ]
        self._channel_count = len(self._channel_vec)

        # Labels may arrive on any of the input ports.
        self._channel_labels = self._channel_labels_of_ports(port_context_in)

        # Store processing parameters
        #: Number of samples per trigger epoch
        self._frame_size = frame_size
        #: Data acquisition sampling rate in Hz
        self._sampling_rate = sampling_rate

        # Initialize data buffers for each input port
        self._data_buffer = {}
        for name in self._port_names:
            self._data_buffer[name] = []

        # Initialize display buffers for each plot function
        self._display_buffer = [
            np.zeros((frame_size, self._channel_count))
            for _ in range(self._no_plots)
        ]

        def generate_colors(n):
            """Generate visually distinct colors using HSV color space.

            One curve takes the theme's foreground. Several take hues
            spread around the circle, light on a dark background and
            deep on a light one: a saturated blue at 80 % value is
            near-invisible on the dark theme.
            """
            if n == 1:
                return [self._foreground_color]
            dark = self._background_color.lightness() < 128
            saturation, value = (0.65, 1.0) if dark else (0.9, 0.7)
            return [
                pg.mkColor(
                    *(
                        int(c * 255)
                        for c in colorsys.hsv_to_rgb(
                            ((i + 0.25) % n) / n, saturation, value
                        )
                    )
                )
                for i in range(n)
            ]

        #: Color palette for distinguishing multiple plot expressions
        self._curve_colors = generate_colors(self._no_plots)

        # Initialize state variables
        #: Flag indicating new trigger data availability
        self._new_data = False
        #: Widget initialization timestamp
        self._start_time = time.time()

        return super().setup(data, port_context_in)

    def _update(self):
        """Update the visual display with new trigger epoch data.

        Called by the Qt timer to refresh the plot with latest trigger epochs.
        Handles curve creation, mathematical expression evaluation, signal
        averaging, and multi-plot visualization with color-coded legends.
        Only updates when new data is available.
        """
        import pyqtgraph as pg

        if not self._new_data:
            return

        # Set up UI elements. Note that this has to be done in the main Qt
        # thread (like this)
        ylim = (0, self._channel_count)
        if self._curves is None:

            # Create curves for each channel and plot combination
            for j in range(self._channel_count):
                for k in range(self._no_plots):
                    self.add_curve(
                        pen=pg.mkPen(color=self._curve_colors[k], width=1.5)
                    )
                    # Add legend entry only once per plot (first channel)
                    if self._legend is not None and j == 0:
                        expr = self.config[self.Configuration.Keys.PLOTS][k]
                        self._legend.addItem(self._curves[-1], expr)
                        self._legend.setVisible(True)

            # Configure axis labels and limits
            amp_lim = self.config[self.Configuration.Keys.AMPLITUDE_LIMIT]
            yl = f"EEG Amplitudes (-{amp_lim} ... +{amp_lim} µV)"
            # Milliseconds, and no SI prefix: pyqtgraph would otherwise
            # label a sub-second axis "Time (s) (x0.001)"
            self.set_labels(x_label="Time (ms)", y_label=yl)
            self._plot_item.getAxis("bottom").enableAutoSIPrefix(False)

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

            # Create time cursor at trigger point (t=0)
            self._cursor = pg.PlotCurveItem(
                pen=self._foreground_color, width=3
            )
            self._plot_item.addItem(self._cursor)
            self._cursor.setData([0] * 2, [*ylim], antialias=False)

        # Update data: evaluate mathematical expressions and average epochs
        with self._lock:
            # Average all trigger epochs for each port
            data = {
                name: np.mean(
                    np.stack(self._data_buffer[name], axis=2), axis=2
                )
                for name in self._port_names
            }

            # Evaluate mathematical expressions for each plot
            for k in range(self._no_plots):
                func = self._plot_funcs[k]
                arg_names = self._plot_args[k]
                args = [data[name] for name in arg_names]
                self._display_buffer[k] = func(*args)
            self._new_data = False

        # Plot channel data with amplitude scaling and vertical offset
        ch_lim_key = TriggerScope.Configuration.Keys.AMPLITUDE_LIMIT
        ch_lim = self.config[ch_lim_key]
        j = 0
        for i in range(len(self._channel_vec)):
            # Vertical offset: each channel gets its own "lane"
            d = self._channel_count - i - 0.5
            for k in range(self._no_plots):
                self._curves[j].setData(
                    self._t_vec * 1000.0,
                    (
                        self._display_buffer[k][:, self._channel_vec[i]]
                        / ch_lim
                        / 2
                        + d
                    ),
                    antialias=False,
                )
                j += 1

        # Update x-axis range with small margin
        tw = 1000.0 * self._frame_size / self._sampling_rate
        margin = tw * 0.0125
        xlim = (
            1000.0 * self._t_vec[0] - margin,
            1000.0 * self._t_vec[-1] + margin,
        )
        self._plot_item.setXRange(*xlim)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process incoming trigger epoch data and store for visualization.

        Called by the pipeline for each new trigger epoch. Accumulates
        trigger epochs in buffers for averaging and mathematical expression
        evaluation.

        Args:
            data: Dictionary containing trigger epoch arrays from connected
                ports. Each array has shape (frame_size, channels)
                representing the pre/post trigger signal segment.

        Returns:
            dict: Empty dictionary (this is a sink node with no outputs).
        """
        # Accumulate trigger epochs for each input port
        for name in self._port_names:
            if name in data and data[name] is not None:
                self._data_buffer[name].append(data[name])
                # Signal that new data is available for display update
                self._new_data = True
