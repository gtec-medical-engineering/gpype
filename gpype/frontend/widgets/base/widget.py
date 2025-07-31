from __future__ import annotations

from abc import abstractmethod

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QBoxLayout, QGroupBox, QHBoxLayout, QVBoxLayout,
                               QWidget)


class Widget:
    """
    Base class for BCI visualization widgets with automatic updates.

    Provides a common foundation for creating real-time visualization
    widgets in the g.Pype BCI framework. Handles automatic UI updates
    via a QTimer and provides a standardized layout structure with
    grouped content.

    The widget automatically wraps content in a QGroupBox with optional
    naming and configurable layout direction. Subclasses implement the
    _update() method to define their specific visualization logic.

    Features:
        - Automatic periodic updates for real-time visualization
        - Configurable update rate (default 60 FPS)
        - Standardized layout with optional grouping and naming
        - Timer-based lifecycle management
        - Abstract interface for custom update logic

    Args:
        widget (QWidget): The Qt widget to wrap and manage.
        name (str): Optional name for the group box title.
        layout (type[QBoxLayout]): Layout class for the content area
            (default: QVBoxLayout).

    Note:
        Subclasses must implement the _update() method to define
        their specific update behavior.
    """

    # Update interval for widget refresh in milliseconds
    # 16.67ms = 60 FPS, suitable for smooth real-time BCI visualization
    UPDATE_INTERVAL_MS: float = 16.67

    def __init__(
        self,
        widget: QWidget,
        name: str = "",
        layout: type[QBoxLayout] = QVBoxLayout,
    ):
        """
        Initialize the widget with layout and timer setup.

        Creates the widget structure with a group box container and
        sets up the automatic update timer. The widget is wrapped in
        a horizontal layout containing a named group box with the
        specified content layout.

        Args:
            widget (QWidget): The Qt widget to wrap and manage.
            name (str, optional): Title for the group box. Defaults to "".
            layout (type[QBoxLayout], optional): Layout class for organizing
                content within the group box. Defaults to QVBoxLayout.
        """
        # Store reference to the main widget
        self.widget = widget

        # Set up automatic update timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._update)

        # Create layout structure: HBox -> GroupBox -> Content Layout
        box_layout = QHBoxLayout()  # Main horizontal container
        box = QGroupBox(name)  # Named group box for content
        box_layout.addWidget(box)

        # Create and assign the content layout within the group box
        self._layout: QBoxLayout = layout(box)
        box.setLayout(self._layout)

        # Set the main layout on the widget
        self.widget.setLayout(box_layout)

    def run(self):
        """
        Start the automatic update timer for real-time visualization.

        Begins periodic updates at the configured interval (default 60 FPS).
        The _update() method will be called repeatedly until terminate()
        is called or the widget is destroyed.

        Note:
            Should be called after the widget is fully initialized and
            ready to display data.
        """
        self._timer.start(self.UPDATE_INTERVAL_MS)

    def terminate(self):
        """
        Stop the automatic update timer and cleanup resources.

        Stops the periodic updates and prepares the widget for cleanup.
        Should be called before the widget is destroyed to ensure
        proper resource management.

        Note:
            After calling terminate(), the widget will no longer update
            automatically until run() is called again.
        """
        self._timer.stop()

    @abstractmethod
    def _update(self):
        """
        Abstract method for implementing widget-specific update logic.

        This method is called periodically by the timer (at UPDATE_INTERVAL_MS
        intervals) to refresh the widget's visual content. Subclasses must
        implement this method to define their specific visualization behavior.

        The method should:
        - Update displayed data from the pipeline
        - Refresh visual elements (plots, text, indicators)
        - Handle any real-time visualization requirements
        - Be efficient to maintain smooth frame rates

        Note:
            This method runs on the main Qt thread, so heavy computations
            should be avoided or moved to background threads.
        """
        pass  # pragma: no cover
