from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QGridLayout, QMainWindow, QWidget

from .widgets.base.widget import Widget


class MainApp:
    """Main application class for g.Pype frontend applications.

    Provides framework for creating PyQt6-based applications with widget
    management, window configuration, and lifecycle handling. Uses
    composition for better flexibility and testability.
    """

    #: Default window geometry configuration
    DEFAULT_POSITION = [100, 100, 700, 400]  # [x, y, width, height]

    #: Default grid size (rows, cols)
    DEFAULT_GRID_SIZE = [3, 3]

    #: Application icon path
    ICON_PATH = Path("resources") / "gtec.ico"

    def __init__(
        self,
        caption: str = "g.Pype Application",
        position: list[int] = None,
        grid_size: list[int] = None,
        app=None,
        prevent_sleep: bool = True,
    ):
        """Initialize the main application with window and widget management.

        Args:
            caption: Window title text displayed in the title bar.
            position: Window geometry as [x, y, width, height] list.
                Uses DEFAULT_POSITION if None.
            grid_size: Grid dimensions as [rows, cols] list.
                Uses DEFAULT_GRID_SIZE if None.
            app: Existing QApplication instance for testing or integration.
                Creates new QApplication if None.
            prevent_sleep: Whether to prevent system sleep/power saving.
                Default True for real-time applications.
        """
        # Create or use existing QApplication (composition over inheritance)
        # This allows for better testability and flexibility
        self._app = app or QApplication([])

        # Initialize widget collection for lifecycle management
        self._widgets: list[Widget] = []

        # Store configuration
        self._grid_size = grid_size or MainApp.DEFAULT_GRID_SIZE
        self._grid_rows, self._grid_cols = self._grid_size
        self._prevent_sleep = prevent_sleep
        self._sleep_prevention_active = False

        # Create and configure main window
        self._window = QMainWindow()
        self._window.setWindowTitle(caption)

        # Set application icon if file exists
        icon_path = Path(__file__).parent / MainApp.ICON_PATH
        if icon_path.exists():
            self._window.setWindowIcon(QIcon(str(icon_path)))

        # Configure window geometry
        if position is None:
            position = MainApp.DEFAULT_POSITION
        self._window.setGeometry(*position)

        # Create central widget and layout system
        # QMainWindow requires a central widget to contain other widgets
        central_widget = QWidget()
        self._window.setCentralWidget(central_widget)

        # Create grid layout for widget arrangement (MATLAB subplot-style)
        self._layout = QGridLayout()
        central_widget.setLayout(self._layout)

        # Connect cleanup handler for graceful shutdown
        self._app.aboutToQuit.connect(self._on_quit)

    def add_widget(self, widget: Widget, grid_positions: list[int] = None):
        """Add a widget to the application layout and management system.

        Registers the widget for lifecycle management and adds it to the
        main window's grid layout. Widget will be automatically started
        during run() and terminated during shutdown.

        Args:
            widget: Widget instance to add to the application.
                Must inherit from the base Widget class.
            grid_positions: List of grid positions (1-indexed) to span.
                For a 3x3 grid: [1,2,3] spans top row, [1,4,7] spans left col.
                If None, adds to next available position.
        """
        # Register widget for lifecycle management
        self._widgets.append(widget)

        if grid_positions is None:
            # Auto-placement: find next available position
            self._layout.addWidget(widget.widget)
        else:
            # Manual placement: convert positions to grid coordinates
            min_pos = min(grid_positions)
            max_pos = max(grid_positions)

            # Convert 1-indexed positions to 0-indexed row/col
            start_row = (min_pos - 1) // self._grid_cols
            start_col = (min_pos - 1) % self._grid_cols
            end_row = (max_pos - 1) // self._grid_cols
            end_col = (max_pos - 1) % self._grid_cols

            # Calculate span
            row_span = end_row - start_row + 1
            col_span = end_col - start_col + 1

            # Add widget to grid layout with specified span
            self._layout.addWidget(
                widget.widget, start_row, start_col, row_span, col_span
            )

    def _enable_sleep_prevention(self):
        """Enable system sleep prevention based on the current platform."""
        if not self._prevent_sleep or self._sleep_prevention_active:
            return

        system = platform.system()

        try:
            if system == "Windows":
                self._prevent_sleep_windows()
            elif system == "Darwin":  # macOS
                self._prevent_sleep_macos()
            else:
                print(f"Sleep prevention not implemented for {system}")
                return

            self._sleep_prevention_active = True
        except Exception as e:
            print(f"Failed to enable sleep prevention: {e}")

    def _disable_sleep_prevention(self):
        """Disable system sleep prevention and restore normal power mgmt."""
        if not self._sleep_prevention_active:
            return

        system = platform.system()

        try:
            if system == "Windows":
                self._restore_sleep_windows()
            elif system == "Darwin":  # macOS
                self._restore_sleep_macos()

            self._sleep_prevention_active = False
        except Exception as e:
            print(f"Failed to disable sleep prevention: {e}")

    def _prevent_sleep_windows(self):
        """Prevent sleep on Windows using SetThreadExecutionState."""
        try:
            import ctypes
            from ctypes import wintypes

            # Constants for SetThreadExecutionState
            ES_CONTINUOUS = 0x80000000
            ES_SYSTEM_REQUIRED = 0x00000001
            ES_DISPLAY_REQUIRED = 0x00000002
            ES_AWAYMODE_REQUIRED = 0x00000040

            # Prevent system sleep and display sleep
            execution_state = (
                ES_CONTINUOUS
                | ES_SYSTEM_REQUIRED
                | ES_DISPLAY_REQUIRED
                | ES_AWAYMODE_REQUIRED
            )

            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
            kernel32.SetThreadExecutionState.restype = wintypes.DWORD

            result = kernel32.SetThreadExecutionState(execution_state)
            if not result:
                raise RuntimeError("SetThreadExecutionState failed")

        except ImportError:
            raise RuntimeError("Windows API not available")

    def _restore_sleep_windows(self):
        """Restore normal sleep behavior on Windows."""
        try:
            import ctypes
            from ctypes import wintypes

            # ES_CONTINUOUS without other flags restores normal behavior
            ES_CONTINUOUS = 0x80000000

            kernel32 = ctypes.windll.kernel32
            kernel32.SetThreadExecutionState.argtypes = [wintypes.DWORD]
            kernel32.SetThreadExecutionState.restype = wintypes.DWORD

            kernel32.SetThreadExecutionState(ES_CONTINUOUS)

        except ImportError:
            pass  # Silently fail if Windows API not available

    def _prevent_sleep_macos(self):
        """Prevent sleep on macOS using caffeinate or IOKit."""
        try:
            import subprocess

            # Try to use caffeinate command (available on macOS 10.8+)
            self._caffeinate_process = subprocess.Popen(
                ["caffeinate", "-d", "-i", "-m", "-s"]
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            try:
                # Fallback to IOKit (requires pyobjc)
                self._prevent_sleep_macos_iokit()
            except ImportError:
                raise RuntimeError(
                    "macOS sleep prevention requires caffeinate command "
                    "or pyobjc library"
                )

    def _prevent_sleep_macos_iokit(self):
        """Prevent sleep on macOS using IOKit (requires pyobjc)."""
        try:
            import Cocoa  # noqa: F401
            from CoreFoundation import kCFStringEncodingUTF8  # noqa: F401

            # Create assertion to prevent sleep
            reason = Cocoa.CFStringCreateWithCString(
                None, "g.Pype Application", kCFStringEncodingUTF8
            )

            # Import IOKit functions
            from IOKit import IOPMAssertionCreateWithName  # noqa: F401
            from IOKit import kIOPMAssertionTypeNoDisplaySleep

            success, self._sleep_assertion_id = IOPMAssertionCreateWithName(
                kIOPMAssertionTypeNoDisplaySleep,
                255,  # kIOPMAssertionLevelOn
                reason,
                None,
            )

            if not success:
                raise RuntimeError("Failed to create IOKit assertion")

        except ImportError:
            raise ImportError("pyobjc library required for IOKit method")

    def _restore_sleep_macos(self):
        """Restore normal sleep behavior on macOS."""
        # Terminate caffeinate process if running
        if hasattr(self, "_caffeinate_process"):
            try:
                self._caffeinate_process.terminate()
                self._caffeinate_process.wait(timeout=5)
            except (AttributeError, subprocess.TimeoutExpired):
                try:
                    self._caffeinate_process.kill()
                except AttributeError:
                    pass
            finally:
                delattr(self, "_caffeinate_process")

        # Release IOKit assertion if created
        if hasattr(self, "_sleep_assertion_id"):
            try:
                from IOKit import IOPMAssertionRelease  # noqa: F401

                IOPMAssertionRelease(self._sleep_assertion_id)
            except ImportError:
                pass
            finally:
                delattr(self, "_sleep_assertion_id")

    def _on_quit(self):
        """Handle application shutdown cleanup.

        Called automatically when the QApplication is about to quit.
        Ensures all registered widgets are properly terminated and
        restores normal sleep behavior.
        """
        # Disable sleep prevention before terminating widgets
        self._disable_sleep_prevention()

        # Terminate all widgets gracefully
        for widget in self._widgets:
            widget.terminate()

    def run(self) -> int:
        """Start the application and enter the main event loop.

        Shows the main window, starts all registered widgets, enables
        sleep prevention if configured, and enters the Qt event loop.
        Blocks until the application is closed.

        Returns:
            int: Application exit code. 0 indicates successful execution,
                non-zero values indicate errors or abnormal termination.
        """
        # Enable sleep prevention if configured
        if self._prevent_sleep:
            self._enable_sleep_prevention()

        # Show the main window
        self._window.show()

        # Start all registered widgets
        for widget in self._widgets:
            widget.run()

        # Enter the Qt event loop (blocks until application closes)
        return self._app.exec()
