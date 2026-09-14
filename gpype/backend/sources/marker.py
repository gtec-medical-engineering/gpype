from __future__ import annotations

from typing import List, Optional, Union

import ioiocore as ioc

from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.event_source import EventSource


class _MarkerCore(EventSource):
    """Internal node emitting markers supplied by the application."""

    def __init__(
        self,
        codes: Optional[dict] = None,
        channel_count: int = 1,
        **kwargs,
    ):
        """Initialize the marker core.

        Args:
            codes: Optional name to numeric code map, so a paradigm can
                mark events by name instead of by magic number.
            channel_count: Number of marker channels.
            **kwargs: Additional arguments for the parent EventSource.
        """
        self._codes = dict(codes) if codes else {}
        # Default into kwargs rather than passing alongside them: a stored
        # configuration arrives as kwargs, and an explicit value here would
        # collide with it on every rebuild.
        kwargs.setdefault("frame_size", 1)
        super().__init__(
            channel_count=channel_count,
            codes=self._codes,
            **kwargs,
        )

    def emit(self, value: Union[int, float, str]) -> None:
        """Emit one marker.

        Args:
            value: A numeric code, or a name declared in ``codes``.

        Raises:
            KeyError: If a name was given that is not in ``codes``.
        """
        if isinstance(value, str):
            if value not in self._codes:
                raise KeyError(
                    f"Unknown marker name '{value}'. Declared names: "
                    f"{sorted(self._codes)}"
                )
            value = self._codes[value]
        self.trigger(float(value))


class Marker(ioc.OChain):
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
        **kwargs,
    ):
        """Initialize the marker chain.

        Args:
            codes: Optional name to numeric code map.
            channel_count: Number of marker channels.
            **kwargs: Additional arguments.
        """
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "codes": codes,
            "channel_count": channel_count,
        }
        self._core_params.update(strip_chain_keys(kwargs))
        self._core = None
        #: Whether the "this run is replaying" note has been made. Once
        #: per Marker, not once per trial: a paradigm calls emit() in a
        #: loop, and one line per event would bury the log it is in.
        self._declined_emit = False
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration(timing=Constants.Timing.ASYNC)],
        )
        ioc.OChain.__init__(
            self,
            codes=codes,
            channel_count=channel_count,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create the internal node chain.

        Returns:
            The head of the list is whatever stands in for this chain's
            core: the core itself, the core followed by a raw tap
            under ``save_as``, a replay core under ``load_from``,
            or nothing at all under SERVER residency. Then a Link
            where the pipeline is distributed, and Sync last.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        # Recorded, replaced, or simply built, as the launch
        # configuration says. Contributes nothing under server
        # residency: the core lives where the stimulus is presented, and
        # so does anything recording or replaying it.
        nodes.extend(
            raw.source_stage(
                self, self._build_core, timing=Constants.Timing.ASYNC
            )
        )
        if residency != Constants.Residency.STANDALONE:
            nodes.append(
                Link(
                    sender=Constants.Residency.EDGE,
                    receiver=Constants.Residency.SERVER,
                    stream_id=self._link_stream_id,
                    timing=Constants.Timing.ASYNC,
                )
            )
        nodes.append(Sync(timing=Constants.Timing.ASYNC))
        return nodes

    def _build_core(self) -> ioc.Node:
        """Build the marker core and keep the handle :meth:`emit` uses.

        A factory rather than a constructor call at the append site,
        because ``raw.source_stage`` decides whether a core is built at
        all -- and when it is not, ``self._core`` has to stay None so
        :meth:`emit` can say why.

        Returns:
            The marker core.
        """
        self._core = _MarkerCore(**self._core_params)
        return self._core

    def emit(self, value: Union[int, float, str]) -> None:
        """Emit one marker.

        Args:
            value: A numeric code, or a name declared in ``codes``.

        Raises:
            RuntimeError: If the marker has no local node to emit from,
                which is the case under server residency.
            KeyError: If a name was given that is not in ``codes``.
        """
        if self._core is None:
            # Residency first, and the order is load-bearing. Under
            # SERVER there is no core whatever ``load_from`` says --
            # ``source_stage`` returns nothing there, because the
            # markers arrive over the Link from the edge. Asking about
            # replay first swallowed the call and told the caller the
            # recording had supplied it, which was simply untrue: a
            # paradigm mistakenly driven from the server process lost
            # every trial marker and was reassured about it.
            from ...common.launch_config import LaunchConfig

            if LaunchConfig.get().residency == Constants.Residency.SERVER:
                raise RuntimeError(
                    "This Marker has no local source to emit from; "
                    "markers originate where the stimulus is presented."
                )
            if raw.is_replaying():
                # A replayed run already carries the markers this
                # paradigm emitted when it was recorded. Accepting live
                # ones too would put each event in the stream twice, at
                # two different instants -- so the call is declined
                # rather than refused: a paradigm script is meant to run
                # unchanged against a replay, and raising here would
                # make every one of them crash on the first trial.
                if not self._declined_emit:
                    self._declined_emit = True
                    self.log(
                        "this run is replaying a recording, so markers "
                        "come from the recording and emit() does "
                        "nothing. The recorded markers are already in "
                        "the stream; emitting again would place each "
                        "event twice.",
                        type=Constants.LogTypes.WARNING,
                    )
                return
            raise RuntimeError(
                "This Marker has no local source to emit from; markers "
                "originate where the stimulus is presented."
            )
        self._core.emit(value)
