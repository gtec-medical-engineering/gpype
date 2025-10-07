from __future__ import annotations

from abc import abstractmethod

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QBoxLayout, QGroupBox, QHBoxLayout, QVBoxLayout,
                               QWidget)


class Widget:
    """Base class for BCI visualization widgets with automatic updates.

    Provides foundation for real-time visualization widgets with automatic
    UI updates via QTimer and standardized layout structure with grouped
    content. Wraps content in QGroupBox with configurable layout.

    Args:
        widget (QWidget): The Qt widget to wrap and manage.
        name (str): Optional name for the group box title.
        layout (type[QBoxLayout]): Layout class for the content area
            (default: QVBoxLayout).

    Note:
        Subclasses must implement the _update() method.
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
        """Initialize the widget with layout and timer setup.

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
        """Start the automatic update timer for real-time visualization.

        Begins periodic updates at the configured interval (default 60 FPS).
        Should be called after the widget is fully initialized.
        """
        self._timer.start(self.UPDATE_INTERVAL_MS)

    def terminate(self):
        """Stop the automatic update timer and cleanup resources.

        Should be called before the widget is destroyed to ensure
        proper resource management.
        """
        self._timer.stop()

    @abstractmethod
    def _update(self):
        """Abstract method for implementing widget-specific update logic.

        Called periodically by the timer to refresh the widget's visual
        content. Subclasses must implement this method to define their
        specific visualization behavior.

        Note:
            Runs on the main Qt thread, so avoid heavy computations.
        """
        pass  # pragma: no cover
