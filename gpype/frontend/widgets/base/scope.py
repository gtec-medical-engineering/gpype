from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import pyqtgraph as pg
from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QWidget

from ....backend.core.i_node import INode
from ....backend.core.i_port import IPort
from ....common._private import channels
from .widget import Widget, _require_qt_application


class _PlotRelay(QObject):
    """Carries "setup has run" from the pipeline thread to the GUI thread.

    :meth:`Scope.setup` runs on the first cycle, which under direct
    execution is the source's own thread, and a graphics item may only be
    touched from the thread that owns it. A signal emitted on an object
    created on the GUI thread is queued to that thread, which is the one
    hand-over that needs no locking of our own -- the same mechanism
    MainApp uses to put a pipeline failure in front of the user.

    ``QTimer.singleShot(0, ...)`` stood here and cannot do this job:
    singleShot starts a timer in the *calling* thread, and the pipeline
    thread runs no event loop, so the timeout never fired. Measured on
    Generator -> Bandpass -> TimeSeriesScope: 670 samples arrived, eight
    curves were built and filled, and the plot item stayed hidden for the
    whole run -- so every scope in the package displayed nothing, with
    nothing anywhere saying why.
    """

    #: Emitted from the pipeline thread once setup has run.
    ready = Signal()


class Scope(INode, Widget):
    """Base class for oscilloscope-style visualization widgets.

    Combines INode data processing with Widget visualization for real-time
    plotting of BCI signals. Uses PyQtGraph for high-performance plotting
    with configurable appearance and multiple curve support. Note that
    subclasses must implement the _update() method.
    """

    #: List of plot curves for multi-channel data display
    _curves: list[pg.PlotDataItem]
    #: Main plot item for PyQtGraph visualization
    _plot_item: pg.PlotItem

    class Configuration(INode.Configuration):
        """Configuration class for Scope parameters."""

        class Keys(INode.Configuration.Keys):
            """Configuration key constants for the Scope."""

            #: Configuration key for plot line color
            LINE_COLOR = "line_color"
            #: Configuration key for plot axis/background color
            AXIS_COLOR = "axis_color"

    def __init__(
        self,
        input_ports: list[IPort.Configuration] = None,
        line_color: tuple[int, int, int] = None,
        axis_color: tuple[int, int, int] = None,
        name="Scope",
        refresh_rate: float = None,
        **kwargs,
    ):
        """Initialize the Scope widget with plot setup and color configuration.

        Args:
            input_ports (list[IPort.Configuration], optional): Input port
                configurations for pipeline connections.
            line_color (tuple[int, int, int], optional): RGB values for
                plot line color. Uses system text color if None.
            axis_color (tuple[int, int, int], optional): RGB values for
                plot background color. Uses system window color if None.
            name (str): Widget group box title.
            **kwargs: Additional configuration passed to parent classes.
        """
        # Before the first Qt object, not after: Qt aborts the process
        # when a QWidget is built with no QApplication -- not an
        # exception, the interpreter dies with no traceback and no
        # message -- and this line is where that used to happen. The
        # guard in Widget.__init__ was too late to be reached.
        _require_qt_application(type(self).__name__)

        # Create the main plot widget
        widget = pg.PlotWidget()

        # Determine line color from system theme if not specified
        if line_color is None:
            palette = widget.palette()
            c = palette.color(QPalette.ColorRole.WindowText)
            line_color = (c.red(), c.green(), c.blue())

        # Determine axis/background color from system theme if not specified
        if axis_color is None:
            palette = widget.palette()
            c = palette.color(QPalette.ColorRole.Window)
            axis_color = (c.red(), c.green(), c.blue())

        # Initialize parent classes with configuration
        Widget.__init__(
            self, widget=QWidget(), name=name, refresh_rate=refresh_rate
        )
        INode.__init__(
            self,
            input_ports=input_ports,
            line_color=line_color,
            axis_color=axis_color,
            **kwargs,
        )

        # Configure plot appearance
        widget.setBackground(self.config[self.Configuration.Keys.AXIS_COLOR])
        self._plot_item = widget.getPlotItem()

        # Enable grid with transparency for better readability
        self._plot_item.showGrid(x=True, y=True, alpha=0.3)

        # Disable mouse interaction to prevent accidental zoom/pan
        self._plot_item.getViewBox().setMouseEnabled(x=False, y=False)

        # And the context menu, for the same reason. Disabling the mouse
        # left the right-click menu -- export, "View All", per-axis
        # autoscale -- which rescales the very axes the scope has just
        # set, so a stray right-click silently made the display lie
        # about its own amplitude scale.
        self._plot_item.setMenuEnabled(False)
        self._plot_item.getViewBox().setMenuEnabled(False)

        # And pyqtgraph's own auto-range button, which was the last
        # route left to the same outcome: it appears in the corner on
        # hover and calls autoRange() when clicked, rescaling the axes
        # the scope has just set. Nothing here needs it -- the scope
        # decides its own amplitude and time window -- so it is only a
        # way for the display to end up lying about its scale.
        self._plot_item.hideButtons()

        # Initially hide plot until setup is complete
        self._plot_item.setVisible(False)

        # Built here, on the GUI thread, so that an emit from the
        # pipeline thread is queued to this one. Held on the instance: a
        # relay that is collected takes its connection with it, and the
        # plot would never be shown.
        self._plot_relay = _PlotRelay()
        self._plot_relay.ready.connect(self._show_plot)

        # Initialize plot data structures
        #: Per-channel labels from the stream, or None when it names no
        #: channels. Resolved at setup; see _resolve_channel_labels.
        self._channel_labels = None
        #: List of plot curves for multi-channel display
        self._curves = None
        #: Current data buffer for plot updates
        self._data = None

        #: Default pen configuration for drawing curves
        self._pen = pg.mkPen(
            color=self.config[self.Configuration.Keys.LINE_COLOR], width=1
        )

        # Add the plot widget to the layout
        self._layout.addWidget(widget)

    def set_labels(self, x_label: str, y_label: str):
        """Set axis labels for the plot.

        Args:
            x_label (str): Label text for the x-axis (horizontal).
            y_label (str): Label text for the y-axis (vertical).
        """
        self._plot_item.setLabel("bottom", x_label)
        self._plot_item.setLabel("left", y_label)

    def add_curve(self, pen=None):
        """Add a new plot curve for displaying data.

        Args:
            pen: PyQtGraph pen object for curve styling. Uses default
                pen if None.

        Returns:
            pg.PlotCurveItem: The newly created curve item for data updates.
        """
        # Use default pen if none provided
        if pen is None:
            pen = self._pen

        # Initialize curves list if first curve
        if self._curves is None:
            self._curves = []

        # Create optimized curve for real-time plotting
        curve = pg.PlotCurveItem(
            pen=pen,
            skipFiniteCheck=True,
            antialias=True,  # Performance optimization
        )  # Smooth appearance

        # Add curve to the plot and store reference
        self._curves.append(curve)
        self._plot_item.addItem(curve)

        return curve

    @staticmethod
    def _resolve_channel_labels(context: dict) -> Optional[list]:
        """Return the stream's channel labels, or None if it has none.

        Channel names already travel: ChannelLabeler writes them into the
        port context, Sync carries them, and channels.labels_of reads
        them. No scope read them, so every display said CH1..CHn no
        matter what the electrodes were called -- the name was known at
        the source and thrown away at the one place a user looks.

        Asked through :func:`channels.has_labels` rather than taking the
        positional default :func:`channels.labels_of` falls back to. That
        default is "Ch01", and silently restyling every existing display
        from CH1 to Ch01 is not this change's business -- a label is
        adopted only where one was really supplied.

        Args:
            context: Port context of the port the scope draws.

        Returns:
            One label per channel, or None if the stream names none.
        """
        if not channels.has_labels(context):
            return None
        return channels.labels_of(context)

    @classmethod
    def _channel_labels_of_ports(cls, port_contexts: dict) -> Optional[list]:
        """Return the channel labels carried by any of several ports.

        For a scope fed by more than one port. Every port of one scope
        carries the same channels -- its own setup has already refused
        the graph otherwise -- so the first port that names them names
        them all, and a port that stays silent is not evidence of
        anything.

        Args:
            port_contexts: Port context per port name, as setup receives
                it.

        Returns:
            One label per channel, or None if no port names them.
        """
        for context in port_contexts.values():
            labels = cls._resolve_channel_labels(context)
            if labels is not None:
                return labels
        return None

    def _channel_label(self, channel: int) -> str:
        """Return the axis label for one channel of the source stream.

        Args:
            channel: Index into the *source* stream, not into the
                visible channels -- hidden channels leave gaps, and the
                label has to follow the channel rather than its slot.

        Returns:
            The stream's label for it, or the positional CH<n> every
            scope showed before labels were read.
        """
        labels = self._channel_labels
        if labels is not None and 0 <= channel < len(labels):
            return str(labels[channel])
        return f"CH{channel + 1}"

    def _show_plot(self) -> None:
        """Reveal the plot now that the stream's shape is known.

        Runs on the GUI thread -- see :class:`_PlotRelay`.
        """
        self._plot_item.setVisible(True)

    def setup(self, data: dict, port_context_in: dict):
        """Set up the scope widget and make the plot visible.

        Args:
            data (dict): Initial data dictionary for setup.
            port_context_in (dict): Input port context information.

        Returns:
            dict: Output port context from parent setup.
        """
        # Make the plot visible now that setup is complete. setup() runs
        # on the first cycle, which under direct execution is the source's
        # own thread, and a graphics item may only be touched from the GUI
        # thread. Hand the change over instead of making it here.
        self._plot_relay.ready.emit()

        # Complete setup with parent class
        return super().setup(data, port_context_in)

    @abstractmethod
    def _update(self):
        """Abstract method for implementing scope-specific update logic.

        Subclasses must implement this method to define how data is retrieved
        from the pipeline and displayed on the plot. Called periodically by
        the widget timer for real-time updates.
        """
        pass  # pragma: no cover
