from __future__ import annotations

from typing import Optional, Union

from ...common.constants import Constants
from ..core._private import assembly
from .base import raw
from .base.event_source import EventSource


class Marker(EventSource):
    """Marks events from the code that produced them.

    The stimulus is presented by the application, so the application is
    what knows when it happened. This is the sanctioned way to say so:

        marker = gp.Marker(codes={"target": 1, "nontarget": 2})
        ...
        marker.emit("target")

    The event is timestamped on the calling thread at the moment it is
    emitted, and placed onto the master timeline from that observation.
    Nothing is buffered or delayed, because a delay would move the event
    away from when it actually happened.
    """

    def __init__(
        self,
        codes: Optional[dict] = None,
        channel_count: int = 1,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the marker.

        Args:
            codes: Optional name to numeric code map, so a paradigm can
                mark events by name instead of by magic number.
            channel_count: Number of marker channels.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments for the parent EventSource.
        """
        self._codes = dict(codes) if codes else {}
        #: Whether the "this run is replaying" note has been made. Once
        #: per Marker, not once per trial: a paradigm calls emit() in a
        #: loop, and one line per event would bury the log it is in.
        self._declined_emit = False
        # Default into kwargs rather than passing alongside them: a stored
        # configuration arrives as kwargs, and an explicit value here would
        # collide with it on every rebuild.
        kwargs.setdefault("frame_size", 1)
        super().__init__(
            channel_count=channel_count,
            codes=self._codes,
            edge_id=edge_id,
            **kwargs,
        )

    def emit(self, value: Union[int, float, str]) -> None:
        """Emit one marker.

        Args:
            value: A numeric code, or a name declared in ``codes``.

        Raises:
            RuntimeError: Under server residency, where markers do not
                originate -- the stimulus is presented on the edge -- and
                on an edge this Marker is not assigned to.
            KeyError: If a name was given that is not in ``codes``.
        """
        # Residency first, and the order is load-bearing. On a server the
        # markers arrive over the Link from the edge, whatever
        # ``load_from`` says. Asking about replay first swallowed the call
        # and told the caller the recording had supplied it, which was
        # simply untrue: a paradigm mistakenly driven from the server
        # process lost every trial marker and was reassured about it.
        if not assembly.builds_core_here(self):
            raise RuntimeError(
                "This Marker has no local source to emit from; markers "
                "originate where the stimulus is presented -- on the "
                "edge the Marker is assigned to."
            )
        if raw.is_replaying():
            # A replayed run already carries the markers this paradigm
            # emitted when it was recorded. Accepting live ones too would
            # put each event in the stream twice, at two different
            # instants -- so the call is declined rather than refused: a
            # paradigm script is meant to run unchanged against a replay,
            # and raising here would make every one of them crash on the
            # first trial.
            if not self._declined_emit:
                self._declined_emit = True
                self.log(
                    "this run is replaying a recording, so markers come "
                    "from the recording and emit() does nothing. The "
                    "recorded markers are already in the stream; "
                    "emitting again would place each event twice.",
                    type=Constants.LogTypes.WARNING,
                )
            return
        if isinstance(value, str):
            if value not in self._codes:
                raise KeyError(
                    f"Unknown marker name '{value}'. Declared names: "
                    f"{sorted(self._codes)}"
                )
            value = self._codes[value]
        self.trigger(float(value))
