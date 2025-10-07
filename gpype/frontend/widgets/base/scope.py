from __future__ import annotations

from abc import abstractmethod

import pyqtgraph as pg
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QWidget

from ....backend.core.i_node import INode
from ....backend.core.i_port import IPort
from .widget import Widget


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
        update_rule: "INode.UpdateRule" = None,
        line_color: tuple[int, int, int] = None,
        axis_color: tuple[int, int, int] = None,
        name="Scope",
        **kwargs,
    ):
        """Initialize the Scope widget with plot setup and color configuration.

        Args:
            input_ports (list[IPort.Configuration], optional): Input port
                configurations for pipeline connections.
            update_rule (INode.UpdateRule, optional): Node update rule.
            line_color (tuple[int, int, int], optional): RGB values for
                plot line color. Uses system text color if None.
            axis_color (tuple[int, int, int], optional): RGB values for
                plot background color. Uses system window color if None.
            name (str): Widget group box title.
            **kwargs: Additional configuration passed to parent classes.
        """
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
        Widget.__init__(self, widget=QWidget(), name=name)
        INode.__init__(
            self,
            input_ports=input_ports,
            update_rule=update_rule,
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

        # Initially hide plot until setup is complete
        self._plot_item.setVisible(False)

        # Initialize plot data structures
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

    def setup(self, data: dict, port_context_in: dict):
        """Set up the scope widget and make the plot visible.

        Args:
            data (dict): Initial data dictionary for setup.
            port_context_in (dict): Input port context information.

        Returns:
            dict: Output port context from parent setup.
        """
        # Make plot visible now that setup is complete
        self._plot_item.setVisible(True)

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
