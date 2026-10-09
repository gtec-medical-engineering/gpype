from __future__ import annotations

import threading
import weakref
from abc import abstractmethod
from typing import TYPE_CHECKING, Optional

from ....common._private.naming import public_name

if TYPE_CHECKING:  # annotations only; never imported at runtime
    from PySide6.QtWidgets import QBoxLayout, QWidget


#: Built once, on first use. See :func:`_tree_receiver_class`.
_TREE_RECEIVER: Optional[type] = None


def _weak_slot(method):
    """``method`` as a Qt slot that holds no reference to its object.

    Connected directly, a bound method makes PySide put a weak reference
    with a callback of its own on the method's object. When the object
    is freed, that callback runs on whichever thread freed it: it
    releases the interpreter lock, walks and changes PySide's
    process-wide connection table, which no lock guards, and calls
    ``QObject::disconnect``; Qt calls the sender's ``disconnectNotify``
    holding the sender's lock, and PySide's override waits there for the
    interpreter lock, which a thread connecting to the same sender holds
    while it waits for that lock. A widget freed by a collection on another
    thread (a pipeline thread's allocation can start one) then did that
    while the GUI thread ran its event loop and made connections of its
    own, and the process crashed or deadlocked: 9 of 100 runs of the
    widget-freed tests on 2026-10-09, and two Windows jobs of Make
    Release run 37889144404. No PySide release up to 6.12 changes it
    (D-NODE-79). Through this slot the object carries no such callback:
    the signal finds it gone and does nothing, and the sender is deleted
    on its own thread with the widget's tree (D-NODE-75).

    Passes as many of the signal's arguments as ``method`` takes, as a
    direct connection does.

    Args:
        method: A bound method.

    Returns:
        A function to connect in place of ``method``.
    """
    import inspect

    ref = weakref.WeakMethod(method)
    parameters = inspect.signature(method).parameters.values()
    if any(p.kind is p.VAR_POSITIONAL for p in parameters):
        count = None
    else:
        count = sum(
            p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
            for p in parameters
        )

    def slot(*args):
        bound = ref()
        if bound is None:
            return None
        return bound(*(args if count is None else args[:count]))

    return slot


def _tree_receiver_class() -> type:
    """Return the class that deletes a freed widget's tree, defining it once.

    Defined inside a function for the reason ``_plot_relay_class`` in
    ``scope.py`` gives: a Qt base class is evaluated at import, and this
    module must import without Qt.

    Returns:
        The ``_TreeReceiver`` class, cached after the first call.
    """
    global _TREE_RECEIVER
    if _TREE_RECEIVER is None:
        from PySide6.QtCore import QObject, Qt, Signal

        class _TreeReceiver(QObject):
            """Lives on the GUI thread and deletes the trees sent to it."""

            #: Emitted with the :class:`_QtTree` to delete, on whichever
            #: thread freed the widget; run from this thread's event loop.
            freed = Signal(object)

            def __init__(self):
                super().__init__()
                # Queued on this thread too, where AutoConnection would
                # call _delete inside whatever the collection interrupted.
                self.freed.connect(
                    self._delete, Qt.ConnectionType.QueuedConnection
                )

            def _delete(self, tree: _QtTree) -> None:
                if _QtTree.get_ident() == tree.thread:
                    tree.delete_here()
                else:
                    _QtTree.stranded.append(tree)

        _TREE_RECEIVER = _TreeReceiver
    return _TREE_RECEIVER


class _QtTree:
    """Deletes a freed widget's Qt objects on the thread they live on.

    D-NODE-75. A widget is in a reference cycle from construction, so the
    cyclic collector frees it and clears its wrappers in an order of its
    own. Clearing a layout-owned ``QSpacerItem`` of pyqtgraph's plot menu
    before its layout makes Shiboken delete the spacer twice, and the
    process dies (ci-matrix-flakes row 4, D-BUILD-75). Deleted from C++
    first, the layout deletes it once and every wrapper turns invalid.

    The widget's weak reference calls this as the widget is freed. Its
    tree, timer and the other Qt objects it registered are held here, so
    they are never the collector's garbage, and queued to the
    :class:`_TreeReceiver` on their own thread, which deletes them from
    its event loop. Never where the collection ran: on another thread (a
    pipeline thread's allocation can start one) because Qt objects may
    be deleted only on their own, and a key filter deleted elsewhere
    crashed the GUI thread's event processing; on their own because a
    collection there can run inside Qt's delivery of an event to the
    tree's root (the key filter runs Python for every one).
    """

    __slots__ = ("tree", "timer", "others", "thread")

    #: The weak references being watched, by ``id``. Each carries its
    #: ``_QtTree`` as the callback, and so keeps the objects alive.
    watched: dict = {}
    #: Objects freed on another thread with no application or receiver
    #: left to take them. Held, so the collector never meets them;
    #: PySide's own shutdown deletes them at exit.
    stranded: list = []
    #: The :class:`_TreeReceiver`, built on the GUI thread.
    receiver = None
    #: ``shiboken6.isValid``, ``shiboken6.delete`` and
    #: ``QCoreApplication.instance``, bound with the first tree: a widget
    #: freed at interpreter shutdown must not need an import.
    is_valid = None
    delete = None
    application = None
    get_ident = threading.get_ident

    def __init__(self, tree, timer):
        self.tree = tree
        self.timer = timer
        #: Qt objects the widget holds outside its tree; see
        #: Widget._delete_with_tree.
        self.others = []
        self.thread = threading.get_ident()

    @classmethod
    def watch(cls, widget, tree, timer) -> _QtTree:
        """Delete ``tree`` and ``timer`` once ``widget`` is freed.

        Called on the thread the tree lives on.

        Returns:
            The entry, to which the widget adds its other Qt objects.
        """
        import shiboken6
        from PySide6.QtCore import QCoreApplication

        if cls.is_valid is None:
            cls.is_valid = shiboken6.isValid
        if cls.delete is None:
            cls.delete = shiboken6.delete
        if cls.application is None:
            cls.application = QCoreApplication.instance
        if cls.receiver is None or not shiboken6.isValid(cls.receiver):
            cls.receiver = _tree_receiver_class()()
        entry = cls(tree, timer)
        ref = weakref.ref(widget, entry)
        cls.watched[id(ref)] = ref
        return entry

    def __call__(self, ref) -> None:
        """The widget is being freed, on whichever thread collected it."""
        cls = type(self)
        cls.watched.pop(id(ref), None)
        if not self.alive() or self._hand_over():
            return
        if cls.get_ident() == self.thread:
            # Nothing would run the hand-over; with no application, no
            # event is being delivered either.
            self.delete_here()
        else:
            cls.stranded.append(self)

    def _hand_over(self) -> bool:
        """Queue this entry to the receiver, if anything will run it.

        Returns:
            False when there is no application or no receiver: Qt's
            shutdown deleted it, or none was built.
        """
        cls = type(self)
        receiver = cls.receiver
        if receiver is None:
            return False
        try:
            if cls.application() is None or not cls.is_valid(receiver):
                return False
            receiver.freed.emit(self)
        except Exception:  # torn down between the check and the emit
            return False
        return True

    def _objects(self) -> list:
        """Timer first, tree last: nothing outlives what it acts on."""
        return [self.timer, *self.others, self.tree]

    def alive(self) -> bool:
        """Whether any of the objects still exists in C++."""
        is_valid = type(self).is_valid
        return any(
            obj is not None and is_valid(obj) for obj in self._objects()
        )

    def delete_here(self) -> int:
        """Delete every object still there; a timer stops as it goes.

        Runs on the thread they live on. Any may be gone already: the
        suite's teardown deletes the tree (D-BUILD-89), and a window a
        widget was added to deletes it with itself.

        Returns:
            How many objects were deleted.
        """
        cls = type(self)
        objects = self._objects()
        self.timer = self.tree = None
        self.others = []
        deleted = 0
        for obj in objects:
            if obj is not None and cls.is_valid(obj):
                cls.delete(obj)
                deleted += 1
        return deleted


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
    from PySide6.QtWidgets import QApplication

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

    Public as ``gp.Widget``, for a display that is not a plot: subclass
    it beside ``gp.INode``, as ``(INode, Widget)``, and the pipeline
    builds the chain a widget needs. For a plot, subclass ``gp.Scope``.
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
        widget: Optional[QWidget] = None,
        name: str = "",
        layout: Optional[type[QBoxLayout]] = None,
        refresh_rate: float = None,
    ):
        """Initialize the widget with layout and timer setup.

        Args:
            widget (QWidget, optional): The Qt widget to wrap and manage,
                or None where this process draws nothing -- a SERVER, or
                an edge the widget is not assigned to. A headless widget
                holds no Qt object, no timer and no layout, and
                `run`/`terminate` do nothing.
            name (str, optional): Title for the group box. Defaults to "".
            layout (type[QBoxLayout], optional): Layout class for
                organizing content. Defaults to QVBoxLayout when None;
                see the note at the assignment for why it is not the
                declared default. Originally documented as
                content within the group box. Defaults to QVBoxLayout.
            refresh_rate (float, optional): Repaints per second. Defaults
                to DEFAULT_REFRESH_RATE.

        Raises:
            ValueError: If refresh_rate is outside the accepted range.
        """
        # **A server builds no widget, and must not import Qt to find
        # that out.** Under `chain-assembly` step 4 the public name is
        # the core, so a document naming a scope is *constructed* on
        # every process that reads it -- including a SERVER, which has no
        # screen and, since 4.0.0's layered install, very possibly no
        # PySide6 at all (`docker/Dockerfile` sets `GPYPE_EXTRAS=""`).
        #
        # `assembly.widget_nodes` already declines to run a scope core
        # there. This is the same answer one step earlier: nothing Qt is
        # touched, so the node can exist as a description of what the
        # edge will draw without pretending it can draw it.
        #
        # Every consumer already expects this: `MainApp` builds no GUI
        # under SERVER, so nothing asks a server for a widget.
        if widget is None:
            self.widget = None
            self._timer = None
            self._layout = None
            self._interval_ms = None
            self._qt_tree = None
            return

        from PySide6.QtCore import Qt, QTimer
        from PySide6.QtWidgets import (
            QBoxLayout,
            QGroupBox,
            QHBoxLayout,
            QVBoxLayout,
            QWidget,
        )

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
        self._timer.timeout.connect(_weak_slot(self._update))

        # Deleted on this thread once the widget is freed, before the
        # collector can clear the tree's wrappers in a crashing order
        # (D-NODE-75). A server builds no tree and never gets here.
        self._qt_tree = _QtTree.watch(self, widget, self._timer)

        # Create layout structure: HBox -> GroupBox -> Content Layout
        box_layout = QHBoxLayout()  # Main horizontal container
        box = QGroupBox(name)  # Named group box for content
        box_layout.addWidget(box)

        # Create and assign the content layout within the group box.
        # None rather than QVBoxLayout as the default: a default
        # argument is evaluated when `def` runs, so naming a Qt class
        # there made this module unimportable without Qt. Resolved
        # here instead, where Qt is already required.
        self._layout: QBoxLayout = (layout or QVBoxLayout)(box)
        box.setLayout(self._layout)

        # Set the main layout on the widget
        self.widget.setLayout(box_layout)

    def _delete_with_tree(self, obj):
        """Delete a Qt object with the widget's tree once the widget is freed.

        For a parentless Qt object the widget holds outside its tree: it
        is then deleted on its own thread too, not wherever the widget
        happens to be collected (D-NODE-75). It must not hold the widget
        strongly, or the widget is never freed. Does nothing on a
        headless widget.

        Args:
            obj: The Qt object.

        Returns:
            ``obj``.
        """
        qt_tree = getattr(self, "_qt_tree", None)
        if qt_tree is not None:
            qt_tree.others.append(obj)
        return obj

    def run(self):
        """Start the automatic update timer for real-time visualization.

        Begins periodic updates at the configured refresh rate.
        Should be called after the widget is fully initialized.

        Does nothing on a headless widget -- see the constructor.
        """
        if self._timer is None:
            return
        self._timer.start(self._interval_ms)

    def terminate(self):
        """Stop the automatic update timer and cleanup resources.

        Should be called before the widget is destroyed to ensure
        proper resource management.

        Does nothing on a headless widget -- see the constructor.
        """
        if self._timer is None:
            return
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
