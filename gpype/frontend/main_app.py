from __future__ import annotations

import importlib.util
import os
import platform
import subprocess
import sys
import warnings
from pathlib import Path

from ..common.constants import Constants
from ..common.launch_config import LaunchConfig
from . import identity as _identity
from . import theme as _theme
from .widgets.base.widget import _weak_slot

#: What a window needs: Qt for MainApp itself, pyqtgraph for the plotting
#: scopes. Both come with the 'gui' extra.
_GUI_PACKAGES = ("PySide6", "pyqtgraph")


def _present(name: str) -> bool:
    """Whether ``import name`` would find a top-level package.

    As ``bundle._importable`` asks it: ``sys.modules`` first, since
    ``find_spec`` raises ValueError for an entry with no spec, such as a
    stand-in a host or a test suite put there; an entry of None fails the
    import, and so counts as absent.
    """
    if sys.modules.get(name) is not None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _require_gui() -> None:
    """Refuse, naming the 'gui' extra, when a window cannot be built.

    Without it a missing PySide6 surfaced as a bare ``No module named
    'PySide6'``, and a missing pyqtgraph got past ``MainApp()`` to fail
    inside a scope's construction. Both are checked here, before anything
    is built, as the backend's readers and writers already name the
    'formats' extra.

    Raises:
        ModuleNotFoundError: Naming every missing package and the
            install command; ``name`` is the first one missing.
    """
    missing = [name for name in _GUI_PACKAGES if not _present(name)]
    if missing:
        raise ModuleNotFoundError(
            "MainApp needs %s, provided by the 'gui' extra: "
            'pip install "gpype[gui]"' % " and ".join(missing),
            name=missing[0],
        )


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
    ICON_PATH = Path("resources") / "gpype.ico"

    #: Futures of the run() calls waiting in a Web Worker; end_run()
    #: resolves them.
    _waiting: list = []

    #: An end_run() that came before any run() was waiting, kept for the
    #: next one. A new MainApp -- the next script -- forgets it.
    _ended_early: bool = False

    @classmethod
    def end_run(cls) -> bool:
        """End a run() that is waiting in a browser's Web Worker.

        The host's counterpart of closing the window. A worker has none,
        so run() waits there until the page ends it -- its Stop button,
        through the worker's message handler -- and the script carries
        on with ``p.stop()`` as on a desktop. A Stop that comes before
        run() waits -- while the script awaits something on its way to
        it -- is kept, and that run() returns at once.

        Returns:
            bool: True if a run() was waiting and now returns; False if
            none was, and the next one will not wait.
        """
        # MainApp's names, not cls's: a subclass must end the same runs.
        waiting, MainApp._waiting = MainApp._waiting, []
        ended = False
        for future in waiting:
            if not future.done():
                future.set_result(None)
                ended = True
        MainApp._ended_early = not ended
        return ended

    def __init__(
        self,
        caption: str = "g.Pype Application",
        position: list[int] = None,
        grid_size: list[int] = None,
        app=None,
        prevent_sleep: bool = True,
        theme: str = _theme.DARK,
        icon: str | Path | None = None,
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
            theme: Application theme, one of
                gpype.frontend.theme.THEMES. "dark" applies the
                bundled dark palette and style sheet to the
                QApplication; "system" leaves Qt's own defaults
                alone, for a host application that brings its own
                look.
            icon: The application's icon file, for its windows and its
                taskbar button. If None, the one a build names in
                ``GPYPE_APP_ICON``, else g.Pype's. See
                gpype.frontend.identity.
        """
        config = LaunchConfig.get()
        self._is_server = config.residency == Constants.Residency.SERVER

        if self._is_server:
            # Server residency: no GUI — empty hull
            self._widgets = []
            # A Stop meant for a previous script's run() is not this one's.
            MainApp._ended_early = False
            return

        _require_gui()

        from PySide6.QtWidgets import (
            QApplication,
            QGridLayout,
            QMainWindow,
            QWidget,
        )

        from .widgets.base.widget import Widget

        # A script's taskbar button is otherwise Python's, whatever icon
        # the window has; named before the first window. A process that
        # has an ID already, its host's, keeps it (gpype.frontend.identity).
        _identity.name_process(_identity.app_id(caption))

        # Create or use existing QApplication (composition over inheritance)
        # This allows for better testability and flexibility
        self._app = app or QApplication([])

        # Theme the application before the first widget exists.
        # The scopes read their colours out of the palette when
        # they are constructed, so a theme installed later leaves
        # every plot on the system background -- see
        # gpype.frontend.theme for why the sheet alone is not
        # enough.
        _theme.apply(self._app, theme)

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

        # The application's icon, application-wide so that every window
        # and the taskbar button carry it; a host's QApplication keeps
        # an icon it set itself.
        qicon = _identity.window_icon(
            self._app, icon, default=Path(__file__).parent / self.ICON_PATH
        )
        if qicon is not None:
            self._window.setWindowIcon(qicon)
            if app is None or self._app.windowIcon().isNull():
                self._app.setWindowIcon(qicon)

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
        self._app.aboutToQuit.connect(_weak_slot(self._on_quit))

    def add_widget(self, widget, grid_positions: list[int] = None):
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
        if self._is_server:
            return

        # Nor on an edge the widget is not assigned to: that edge runs
        # no core for it and a scope there holds no Qt object, so the
        # window is the owning edge's to show. A script written for the
        # whole system runs unchanged on every edge this way.
        from ..backend.core._private import assembly

        if not assembly.builds_core_here(widget):
            return

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

    def _watch_pipelines(self) -> None:
        """Subscribe to every live pipeline's failures.

        The handler runs on ioiocore's monitoring thread, and Qt widgets
        may only be touched from the thread that owns them. So the
        report crosses over through a signal on an object created here,
        on the GUI thread: Qt queues a signal emitted from elsewhere to
        the receiver's thread, which is the one mechanism that is safe
        without any locking of our own.
        """
        from PySide6.QtCore import QObject, Signal

        from ..common._private import pipelines

        class _Relay(QObject):
            """Carries a failure from the monitor thread to the GUI."""

            failed = Signal(object)

        relay = _Relay()
        relay.failed.connect(_weak_slot(self._show_failure))
        # Kept on the instance: a relay that is collected takes the
        # connection with it, and the failure would go nowhere.
        self._failure_relay = relay

        for pipeline in pipelines.live():
            try:
                pipeline.add_error_handler(relay.failed.emit)
            except Exception:
                # A pipeline that cannot take a handler is not a reason
                # to refuse to start the application.
                pass

    def _show_failure(self, entry) -> None:
        """Show a failed pipeline to the user.

        Deliberately not modal. A modal dialog raised from a background
        failure blocks whatever the user was doing and, in a test run,
        blocks the run itself.

        Args:
            entry: The failing log entry.
        """
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import QMessageBox

        source = {}
        message = "The pipeline stopped."
        try:
            message = str(entry["message"])
            source = entry["source"] or {}
        except Exception:  # pragma: no cover
            pass

        detail = []
        node = source.get("instance")
        if node:
            detail.append(f"Node: {node}")
        where = source.get("summary")
        if where:
            detail.append(f"Raised in {where}")

        box = QMessageBox(getattr(self, "_window", None))
        # Given a parent, QMessageBox is window-modal by default, which
        # would lock the very window the user is trying to read. show()
        # alone is not enough -- the modality has to be cleared too.
        box.setWindowModality(Qt.NonModal)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("Pipeline stopped")
        box.setText(message)
        if detail:
            box.setInformativeText("\n".join(detail))
        box.setStandardButtons(QMessageBox.Close)
        # Held so it is not collected the moment this returns.
        self._failure_box = box
        box.show()

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
        if self._is_server:
            from ..backend.core._private import threads

            if not threads.available():
                return self._run_in_worker()
            # No GUI on SERVER — block until user presses Enter, then exit
            try:
                input("SERVER running. Press Enter to stop...\n")
            except EOFError:
                pass  # non-interactive environment (e.g. piped stdin)
            return 0

        # Put a pipeline failure in front of the user. Without this the
        # only report is printed to stdout, which a windowed application
        # may never show -- so the display freezes and nothing says why.
        self._watch_pipelines()

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

    def _run_in_worker(self) -> int:
        """Wait in a Web Worker until the host calls end_run().

        Blocking would stop the only thread, and with it every message
        the pipeline waits for. JSPI's ``run_sync`` suspends this call
        stack instead, and the worker's loop -- the page's frames, the
        pipeline, its broker -- runs on underneath, so an unchanged
        ``p.start(); app.run(); p.stop()`` behaves as on a desktop
        (D-CORE-130). Where nothing can suspend -- no JSPI, or a script
        not entered through ``runPythonAsync`` -- it returns at once.

        Returns:
            int: 0.
        """
        from ..backend.core._private import threads

        try:
            from pyodide.ffi import can_run_sync, run_sync
        except ImportError:
            can_run_sync = run_sync = None
        loop = threads.event_loop()
        if run_sync is None or loop is None or not can_run_sync():
            warnings.warn(
                "MainApp.run() returns at once where no thread can be "
                "started and this runtime cannot suspend a call (it needs "
                "JSPI, entered through runPythonAsync): the host owns the "
                "pipeline's lifetime and calls stop() itself.",
                stacklevel=3,
            )
            return 0
        if MainApp._ended_early:
            MainApp._ended_early = False
            return 0
        future = loop.create_future()
        MainApp._waiting.append(future)
        try:
            run_sync(future)
        finally:
            if future in MainApp._waiting:
                MainApp._waiting.remove(future)
        return 0
