from __future__ import annotations

import os
from pathlib import Path
from typing import List

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget

from .widgets.base.widget import Widget


class MainApp:
    """
    Main application class for g.Pype frontend applications.

    Provides a simple framework for creating PyQt6-based applications
    with widget management, window configuration, and lifecycle handling.
    Uses composition over inheritance for better flexibility and testability.

    The application manages a collection of widgets, handles window setup,
    and coordinates application startup and shutdown procedures.

    Features:
        - Automatic QApplication creation and management
        - Window configuration with icon and positioning
        - Widget lifecycle management (run/terminate)
        - Graceful application shutdown handling
        - Support for dependency injection (testing)

    Attributes:
        DEFAULT_POSITION: Default window geometry [x, y, width, height].
        ICON_PATH: Path to the application icon file.
    """

    # Default window geometry configuration
    DEFAULT_POSITION = [100, 100, 800, 600]  # [x, y, width, height]

    # Application icon path
    ICON_PATH = os.path.join(Path(__file__).parent, "resources", "gtec.ico")

    def __init__(self,
                 caption: str = "g.Pype Application",
                 position: List[int] = None,
                 app=None):
        """
        Initialize the main application with window and widget management.

        Sets up the QApplication, main window, layout system, and event
        handling for a complete g.Pype frontend application. Supports
        dependency injection for testing purposes.

        Args:
            caption: Window title text displayed in the title bar.
            position: Window geometry as [x, y, width, height] list.
                Uses DEFAULT_POSITION if None.
            app: Existing QApplication instance for testing or integration.
                Creates new QApplication if None.
        """
        # Create or use existing QApplication (composition over inheritance)
        # This allows for better testability and flexibility
        self._app = app or QApplication([])

        # Initialize widget collection for lifecycle management
        self._widgets: List[Widget] = []

        # Create and configure main window
        self._window = QMainWindow()
        self._window.setWindowTitle(caption)

        # Set application icon if file exists
        icon_path = MainApp.ICON_PATH
        if os.path.exists(icon_path):
            self._window.setWindowIcon(QIcon(icon_path))

        # Configure window geometry
        if position is None:
            position = MainApp.DEFAULT_POSITION
        self._window.setGeometry(*position)

        # Create central widget and layout system
        # QMainWindow requires a central widget to contain other widgets
        central_widget = QWidget()
        self._window.setCentralWidget(central_widget)

        # Create vertical layout for widget arrangement
        self._layout = QVBoxLayout()
        central_widget.setLayout(self._layout)

        # Connect cleanup handler for graceful shutdown
        self._app.aboutToQuit.connect(self._on_quit)

    def add_widget(self, widget: Widget):
        """
        Add a widget to the application layout and management system.

        Registers the widget for lifecycle management and adds it to the
        main window's vertical layout. The widget will be automatically
        started during run() and terminated during shutdown.

        Args:
            widget: Widget instance to add to the application.
                Must inherit from the base Widget class.
        """
        # Register widget for lifecycle management
        self._widgets.append(widget)

        # Add widget to the vertical layout for display
        self._layout.addWidget(widget.widget)

    def _on_quit(self):
        """
        Handle application shutdown cleanup.

        Called automatically when the QApplication is about to quit.
        Ensures all registered widgets are properly terminated to
        prevent resource leaks and data corruption.
        """
        # Terminate all widgets gracefully
        for widget in self._widgets:
            widget.terminate()

    def run(self) -> int:
        """
        Start the application and enter the main event loop.

        Shows the main window, starts all registered widgets, and enters
        the Qt event loop. This method blocks until the application is
        closed by the user or programmatically terminated.

        Returns:
            int: Application exit code. 0 indicates successful execution,
                non-zero values indicate errors or abnormal termination.
        """
        # Show the main window
        self._window.show()

        # Start all registered widgets
        for widget in self._widgets:
            widget.run()

        # Enter the Qt event loop (blocks until application closes)
        return self._app.exec()
