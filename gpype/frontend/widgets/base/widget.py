from __future__ import annotations

from abc import abstractmethod

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QBoxLayout,
    QGroupBox,
    QHBoxLayout,
    QVBoxLayout,
    QWidget,
)

from ....common._private.naming import public_name


def _public_name(internal: str) -> str:
    """The name a user would have typed. See gpype.common._private.naming.

    Kept as a name in this module because the tests and the guard below
    both reach for it here, and because a file writer needs the same
    derivation -- so the implementation moved somewhere both halves can
    import rather than being copied.
    """
    return public_name(internal)


def _require_qt_application(what: str) -> None:
    """Refuse to build a widget before there is an application to hold it.

    Qt aborts the process when a QWidget is constructed with no
    QApplication -- not an exception: the interpreter dies, with no
    traceback, no message, and an exit code of 127 on this platform. So a
    script that built a scope before ``gp.MainApp()`` simply vanished,
    and there was nothing anywhere to read.

    Args:
        what: The class being constructed, for the message.

    Raises:
        RuntimeError: If no QApplication exists yet.
    """
    if QApplication.instance() is not None:
        return
    name = _public_name(what)
    raise RuntimeError(
        f"{name} needs a Qt application before it can be built. Create "
        "the app first:"
        "\n\n"
        "    app = gp.MainApp()\n"
        f"    widget = gp.{name}(...)\n"
        "    app.add_widget(widget)"
        "\n\n"
        "Qt terminates the process rather than raising when a widget is "
        "built without one, so this check exists to say so instead."
    )


class Widget:
    """Base class for main app visualization widgets with automatic updates.

    Provides foundation for real-time visualization widgets with automatic
    UI updates via QTimer and standardized layout structure with grouped
    content. Wraps content in QGroupBox with configurable layout.

    Args:
        widget (QWidget): The Qt widget to wrap and manage.
        name (str): Optional name for the group box title.
        layout (type[QBoxLayout]): Layout class for the content area
            (default: QVBoxLayout).

    Subclasses must implement the _update() method.
    """

    #: Default display refresh rate in Hz.
    #:
    #: Ten repaints a second is enough for a widget whose picture only
    #: changes when something arrives -- a spectrum, an epoch average, a
    #: numeric readout. A display whose picture moves continuously
    #: declares its own; see TimeSeriesScope, which asks for 20.
    #:
    #: Note that this rate was not actually being delivered. See the
    #: timer type in __init__: Qt coarsens any interval above 20 ms, so
    #: a widget asking for 10 Hz repainted at 9.1 Hz with 5.7 ms of
    #: jitter. The number here only became true when that was fixed.
    DEFAULT_REFRESH_RATE: float = 10.0
    #: Lowest accepted refresh rate in Hz.
    MIN_REFRESH_RATE: float = 1.0
    #: Highest accepted refresh rate in Hz.
    MAX_REFRESH_RATE: float = 60.0

    def __init__(
        self,
        widget: QWidget,
        name: str = "",
        layout: type[QBoxLayout] = QVBoxLayout,
        refresh_rate: float = None,
    ):
        """Initialize the widget with layout and timer setup.

        Args:
            widget (QWidget): The Qt widget to wrap and manage.
            name (str, optional): Title for the group box. Defaults to "".
            layout (type[QBoxLayout], optional): Layout class for organizing
                content within the group box. Defaults to QVBoxLayout.
            refresh_rate (float, optional): Repaints per second. Defaults
                to DEFAULT_REFRESH_RATE.

        Raises:
            ValueError: If refresh_rate is outside the accepted range.
        """
        _require_qt_application(type(self).__name__)

        if refresh_rate is None:
            refresh_rate = self.DEFAULT_REFRESH_RATE
        if not (
            self.MIN_REFRESH_RATE <= refresh_rate <= self.MAX_REFRESH_RATE
        ):
            raise ValueError(
                f"refresh_rate must be between {self.MIN_REFRESH_RATE} and "
                f"{self.MAX_REFRESH_RATE} Hz."
            )
        self._interval_ms = int(round(1000.0 / refresh_rate))

        # Store reference to the main widget
        self.widget = widget

        # Set up the automatic update timer.
        #
        # This was a CoarseTimer, on the reasoning that coarse timing is
        # enough for a display refresh and lets the platform coalesce
        # wake-ups. It is not enough: Qt coarsens any interval above
        # 20 ms by up to 5%, and the result is not the rate that was
        # asked for. Measured on this platform, requested rate against
        # what the timer actually delivered:
        #
        #     Hz   interval   CoarseTimer        PreciseTimer
        #     10     100 ms   9.1 Hz (sd 5.73)   10.0 Hz (sd 0.40)
        #     20      50 ms   16.0 Hz (sd 4.17)  20.0 Hz (sd 0.47)
        #     25      40 ms   21.4 Hz (sd 3.23)  25.0 Hz (sd 0.57)
        #     30      33 ms   21.3 Hz (sd 4.02)  30.3 Hz (sd 0.56)
        #     50      20 ms   50.0 Hz (sd 0.57)  50.0 Hz (sd 0.47)
        #
        # Two consequences, and the second is the trap. A scope asking
        # for 10 Hz was repainting at 9.1 Hz with 5.7 ms of jitter,
        # which is the stutter that gets reported as lag. And every rate
        # between 11 and 49 Hz collapsed onto something lower -- 20, 25
        # and 30 Hz all landed near 16-21 Hz -- so raising a default
        # without this line would have changed the number and not the
        # display. Intervals of 20 ms and below are not coarsened,
        # which is why 50 Hz was accurate all along.
        self._timer = QTimer()
        self._timer.setTimerType(Qt.TimerType.PreciseTimer)
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

        Begins periodic updates at the configured refresh rate.
        Should be called after the widget is fully initialized.
        """
        self._timer.start(self._interval_ms)

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
