"""The widget half of ResultScope: everything that needs Qt.

Imported by ResultScope.create_internal_nodes only when the residency
actually builds a display. Kept out of the chain's own module so that a
server process -- which never builds one -- does not import Qt in order
to deserialize a document that names a scope.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import ioiocore as ioc
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QGridLayout, QLabel, QWidget

from ....backend.core.i_node import INode
from ....backend.core.i_port import IPort
from ....common._private import channels
from .._validation import check_bounds
from ..base.widget import Widget, _require_qt_application
from ..result_scope import ResultScope

#: Default input port identifier
PORT_IN = ioc.Constants.Defaults.PORT_IN

#: What separates one past value from the next in a history row.
_HISTORY_SEPARATOR = "  "


class _ResultScopeCore(INode, Widget):
    """Internal node for a numeric per-channel result readout.

    One row per channel: the channel's name, its current value, and --
    when ``history`` is positive -- the values before it. Wrapped by
    ResultScope for distributed operation.

    Derives from INode and Widget directly rather than from Scope.
    ``Scope.__init__`` builds a ``pg.PlotWidget`` and a curve list, of
    which a numeric readout inherits nothing: Scope is the *plot* base,
    and INode + Widget is the pair Scope itself composes.

    INode and not IONode is load-bearing. ``INode.setup`` requires only
    frame_size and channel_count, while ``gpype.IONode.setup`` raises
    "All ports must have the same sampling rate" when no input declares
    one -- ``len(set(sampling_rates)) != 1`` is true of the empty set. A
    result tapping a sparse decision stream has no rate, and this choice
    is what makes "continuously or once at the end, don't care" true
    rather than aspirational.
    """

    #: Digits after the decimal point, when the author names none.
    DEFAULT_DECIMALS: int = ResultScope.DEFAULT_DECIMALS
    #: Past values shown per channel, when the author names none.
    DEFAULT_HISTORY: int = ResultScope.DEFAULT_HISTORY

    class Configuration(INode.Configuration):
        """Configuration keys for the result readout."""

        class Keys(INode.Configuration.Keys):
            """Required configuration parameter keys."""

            #: Configuration key for the drawn decimal precision
            DECIMALS = "decimals"
            #: Configuration key for how many past values to draw
            HISTORY = "history"

    def __init__(
        self,
        decimals: int = None,
        history: int = None,
        name: str = None,
        refresh_rate: float = None,
        **kwargs,
    ):
        """Initialize the result readout widget.

        Args:
            decimals: Digits after the decimal point. Uses
                DEFAULT_DECIMALS if None.
            history: Past values shown per channel besides the current
                one. Uses DEFAULT_HISTORY if None.
            name: Group box caption. Defaults to "Result".
            refresh_rate: Repaints per second. Defaults to the Widget
                default.
            **kwargs: Additional arguments passed to parent classes.

        Raises:
            RuntimeError: If no Qt application exists yet.
            ValueError: If decimals or history is outside its range.
        """
        # Before the first Qt object, not after: the QWidget() below is
        # an *argument*, so it is constructed before Widget.__init__ can
        # check anything, and Qt aborts the process -- exit 127, no
        # traceback, no message -- when a QWidget is built with no
        # QApplication. base/scope.py:88-93 records the same thing.
        _require_qt_application(type(self).__name__)

        if decimals is None:
            decimals = self.DEFAULT_DECIMALS
        if history is None:
            history = self.DEFAULT_HISTORY

        # The same function and the same constants the chain uses, so
        # there is one bound rather than two that happen to agree. The
        # chain is where a server reaches; this is where a directly
        # constructed core is guarded.
        check_bounds(
            "decimals",
            decimals,
            ResultScope.MIN_DECIMALS,
            ResultScope.MAX_DECIMALS,
        )
        check_bounds(
            "history",
            history,
            ResultScope.MIN_HISTORY,
            ResultScope.MAX_HISTORY,
        )
        decimals = int(decimals)
        history = int(history)

        # Popped rather than passed alongside **kwargs: a core rebuilt
        # from a stored configuration carries its ports in kwargs, and
        # passing them here as well is a duplicate keyword.
        ip_key = self.Configuration.Keys.INPUT_PORTS
        input_ports: list[IPort.Configuration] = kwargs.pop(
            ip_key, [IPort.Configuration(name=PORT_IN)]
        )

        if name is None:
            name = "Result"

        Widget.__init__(
            self, widget=QWidget(), name=name, refresh_rate=refresh_rate
        )
        INode.__init__(
            self,
            input_ports=input_ports,
            decimals=decimals,
            history=history,
            **kwargs,
        )

        #: Digits drawn after the decimal point.
        self._decimals = decimals
        #: How many past values each row shows besides the current one.
        self._history_len = history

        #: Guards everything step() writes and _update() reads. step()
        #: runs on the pipeline thread, _update() on the GUI thread.
        self._lock = threading.Lock()
        #: Whether a frame has arrived since the last repaint.
        self._new_data = False
        #: The most recent sample, one value per channel, or None before
        #: the first frame. This and _past are the whole of what the
        #: frontend keeps: it owns what it draws and nothing else, the
        #: same relationship _TimeSeriesScopeCore has with its window.
        self._latest: Optional[np.ndarray] = None
        #: The samples before it, oldest first, or None when history is
        #: zero -- so "keeps nothing" is a state, not a length.
        self._past: Optional[deque] = (
            deque(maxlen=history) if history > 0 else None
        )

        #: One label per channel, resolved at setup.
        self._channel_labels: Optional[list] = None
        #: The labels the rows on screen were built from. Compared
        #: against _channel_labels on every repaint, because a pipeline
        #: may be started again after stop() -- and a restart against a
        #: stream of a different width or with different channel names
        #: would otherwise keep the previous run's captions beside this
        #: run's numbers, which is a display that lies about what it
        #: shows.
        self._drawn_labels: Optional[list] = None
        #: The grid holding the rows, kept so it can be discarded when
        #: the rows are rebuilt.
        self._rows: Optional[QWidget] = None
        #: Row captions, built on the GUI thread with the rest.
        self._name_labels: Optional[list] = None
        #: Current-value readouts, one per channel.
        self._value_labels: Optional[list] = None
        #: Past-value readouts, or None when history is zero.
        self._history_labels: Optional[list] = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Read the stream's shape, which is a channel count and nothing more.

        No sampling rate is read, and that is the point of the class: a
        classifier decision, a sparse label or an end-of-run summary has
        no rate, and demanding one would make this scope unusable for
        exactly the streams it exists for.

        Args:
            data: Input data dictionary (not used in setup phase).
            port_context_in: Context information for input ports.

        Returns:
            dict: Updated port context for downstream components.

        Raises:
            ValueError: If the input context declares no channel count.
        """
        c = port_context_in[PORT_IN]

        # channels.labels_of, not Scope._resolve_channel_labels. That
        # helper asks has_labels first and answers None otherwise, on
        # purpose: adopting the positional default would have silently
        # restyled every existing plot from CH1 to Ch01. This readout is
        # new, so it has nothing to restyle, and it needs a caption for
        # every row -- so it takes the sanctioned reader's own default
        # rather than inventing positional labels of its own.
        channels.channel_count(c)
        self._channel_labels = channels.labels_of(c)

        # A pipeline may be started again after stop(), and the previous
        # run's values are not this run's -- a history carried across the
        # restart would be drawn as though it had just arrived.
        with self._lock:
            self._latest = None
            if self._past is not None:
                self._past.clear()
            self._new_data = False

        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Keep the newest frame, and the tail of it that history shows.

        Args:
            data: Dictionary containing input data arrays from connected
                ports.

        Returns:
            dict: Empty dictionary (this is a sink node with no outputs).
        """
        frame = data.get(PORT_IN)
        if frame is None or frame.shape[0] == 0:
            # An ASYNC port carries nothing on a cycle with no event, and
            # a result stream is exactly the sparse kind that has such
            # cycles. Nothing arrived, so nothing changed.
            return {}

        # The length comes from the frame that actually arrived, not from
        # the frame size declared at setup. Sync stacks whatever it has
        # held into one output frame, so the two differ in normal
        # operation; trusting the declaration raised a shape mismatch in
        # _TimeSeriesScopeCore that became a permanent error condition
        # naming no cause.
        n = frame.shape[0]

        # Copied, and to float: the pipeline's arrays are float32 and may
        # be reused for the next frame, so a reference kept here would
        # start showing values that have not been drawn yet.
        with self._lock:
            self._latest = np.array(frame[n - 1, :], dtype=float)
            if self._past is not None:
                # Only the tail history can show. Appending every sample
                # of a 250 Hz frame into a deque of maxlen 5 discards
                # all but the last five anyway.
                for row in frame[max(0, n - self._history_len) :, :]:
                    self._past.append(np.array(row, dtype=float))
            self._new_data = True

        return {}

    def _update(self):
        """Repaint the readout from the newest frame.

        Called by the Qt timer, which is the GUI thread -- so this is
        also where the labels are built. A QWidget may only be created by
        the thread that owns the application, and step() runs on the
        pipeline's, so they cannot be built where the stream's shape
        becomes known.
        """
        if not self._new_data:
            return

        with self._lock:
            latest = self._latest
            past = list(self._past) if self._past is not None else []
            self._new_data = False

        if latest is None:
            return  # pragma: no cover

        # Rebuilt, not merely built once: see _drawn_labels.
        if self._drawn_labels != self._channel_labels:
            self._create_rows()

        # zip, not an index: it stops at the shorter of the two rather
        # than raising if a stream ever delivers a frame narrower than
        # the channel count its context declared.
        for label, value in zip(self._value_labels, latest):
            label.setText(self._format(value))

        if self._history_labels is not None:
            for channel, label in enumerate(self._history_labels):
                label.setText(
                    _HISTORY_SEPARATOR.join(
                        self._format(row[channel]) for row in past
                    )
                )

    def _create_rows(self) -> None:
        """Build one row per channel, inside the group box.

        ``self._layout`` *is* the group box's content layout, so the grid
        needs no walk down the widget tree to find its place.

        Any previous grid is discarded first. Left in place it would be
        drawn above the new one, so a restart against a different stream
        would show both runs' rows at once.
        """
        if self._rows is not None:
            self._layout.removeWidget(self._rows)
            self._rows.setParent(None)
            self._rows.deleteLater()

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(8, 3, 8, 3)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(2)

        # Fixed-pitch, and right-aligned: with a proportional font the
        # digits change width as the value changes, so a readout updating
        # ten times a second visibly jitters sideways.
        numbers = QFont()
        numbers.setStyleHint(QFont.StyleHint.Monospace)
        numbers.setFamily("Monospace")

        self._name_labels = []
        self._value_labels = []
        self._history_labels = [] if self._history_len > 0 else None

        for row, name in enumerate(self._channel_labels):
            caption = QLabel(str(name))
            grid.addWidget(caption, row, 0)
            self._name_labels.append(caption)

            value = QLabel("")
            value.setFont(numbers)
            value.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            grid.addWidget(value, row, 1)
            self._value_labels.append(value)

            if self._history_labels is not None:
                history = QLabel("")
                history.setFont(numbers)
                grid.addWidget(history, row, 2)
                self._history_labels.append(history)

        # The last column takes the slack, so the captions and the
        # current values stay together on the left.
        grid.setColumnStretch(2 if self._history_len > 0 else 1, 1)
        self._layout.addWidget(container)
        self._rows = container
        # Recorded last, so a failure above leaves the rows marked as
        # not yet drawn rather than as drawn for labels they never got.
        self._drawn_labels = list(self._channel_labels)

    def _format(self, value) -> str:
        """Return one value as the readout draws it.

        Args:
            value: A single channel's value.

        Returns:
            The value at the configured precision.
        """
        return f"{float(value):.{self._decimals}f}"
