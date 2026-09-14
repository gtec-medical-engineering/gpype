"""GtcReader: a `.gtc` recording as one batch block.

Written against the API requested in
``devdoc/archive/requests/2026-09-02_gtc_python_package.md``. The mapping from
that API to g.Pype's port context lives here and nowhere else, so when
the package lands, the parts that do not match are a failing test in
``test/test_backend_sources_gtc_reader.py`` rather than a discussion.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...common.constants import Constants
from .base.batch_source import BatchSource

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: GTC channel types mapped onto g.Pype channel roles. GTC's `CHAN`
#: carries a per-channel *type*; g.Pype's roles say how a channel must be
#: treated, which is the same question. An unrecognised type is read as
#: signal -- the format's own default reading for a data channel -- and
#: logged, rather than silently excluded from filtering. To be confirmed
#: against the real package's vocabulary.
_ROLE_OF_TYPE = {
    "signal": Constants.ChannelRoles.SIGNAL,
    "eeg": Constants.ChannelRoles.SIGNAL,
    "ecog": Constants.ChannelRoles.SIGNAL,
    "ieeg": Constants.ChannelRoles.SIGNAL,
    "trigger": Constants.ChannelRoles.TRIGGER,
    "quality": Constants.ChannelRoles.QUALITY,
    "auxiliary": Constants.ChannelRoles.AUXILIARY,
    "aux": Constants.ChannelRoles.AUXILIARY,
    "index": Constants.ChannelRoles.INDEX,
}


def _open(file_name: str):
    """Open a `.gtc` file, or say what is missing.

    Args:
        file_name: Path to the recording.

    Returns:
        An open GTC file handle.

    Raises:
        ImportError: If the ``gtc`` package is not installed. Imported
            here rather than at module scope so that a document naming
            GtcReader loads on a machine without it, the way the
            amplifier sources behave.
    """
    try:
        import gtc
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "GtcReader needs the 'gtc' package, which is not installed. "
            "Install it to read .gtc recordings."
        ) from error
    return gtc.open(file_name)


class GtcReader(BatchSource):
    """Reads a `.gtc` recording as one block.

    The format carries what CSV cannot: units and exact per-channel
    calibration, absolute start time, recording interruptions, markers,
    provenance, and whether the file was sealed. All of it is published
    into the port context, so it reaches a ``Result`` without a second
    metadata channel to keep in sync.

    Args:
        file_name: Path to the recording.
        channels: Channel names or indices to read. All by default.
            Reading a subset costs proportionally less -- the format is
            seekable per channel -- and the published metadata is
            narrowed to match.
        start: First sample to read, on the nominal grid. Zero by
            default.
        stop: One past the last sample to read. End of recording by
            default.
        **kwargs: Additional arguments for the parent BatchSource.

    Raises:
        ImportError: If the ``gtc`` package is not installed.
        ValueError: If no file name is given, or the requested range is
            empty or outside the recording.
    """

    def __init__(
        self,
        file_name: str,
        channels: Optional[list] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **kwargs,
    ):
        if file_name is None:
            raise ValueError("file_name must be provided.")

        self._handle = _open(file_name)
        self._channels = list(channels) if channels is not None else None

        total = int(self._handle.sample_count)
        first = 0 if start is None else int(start)
        last = total if stop is None else int(stop)
        if first < 0 or last > total:
            raise ValueError(
                f"the requested range [{first}, {last}) is outside "
                f"'{file_name}', which holds {total} samples."
            )
        if last <= first:
            raise ValueError(
                f"the requested range [{first}, {last}) is empty."
            )
        self._start = first
        self._stop = last

        described = list(self._handle.channels)
        if self._channels is not None:
            described = self._select(described, self._channels)
        self._described = described

        # An exact fraction on the way in, a float on the way into the
        # engine, and the fraction kept so a round trip does not turn
        # 512000/1001 into a decimal.
        self._rate_exact = self._handle.rate
        super().__init__(
            sampling_rate=float(self._rate_exact),
            channel_count=len(described),
            sample_count=self._stop - self._start,
            file_name=file_name,
            channels=self._channels,
            start=start,
            stop=stop,
            **kwargs,
        )

    @staticmethod
    def _select(described: list, wanted: list) -> list:
        """Return the described channels the caller asked for, in order.

        Args:
            described: Every channel the file describes.
            wanted: Names or indices, in the order to read them.

        Returns:
            The selected descriptions.

        Raises:
            ValueError: If a name or index does not exist.
        """
        by_name = {getattr(ch, "name", None): ch for ch in described}
        selected = []
        for item in wanted:
            if isinstance(item, int):
                if not 0 <= item < len(described):
                    raise ValueError(
                        f"channel index {item} is outside the "
                        f"recording's {len(described)} channels."
                    )
                selected.append(described[item])
            else:
                if item not in by_name:
                    raise ValueError(
                        f"no channel named {item!r} in this recording; "
                        f"it has {', '.join(sorted(str(n) for n in by_name))}."
                    )
                selected.append(by_name[item])
        return selected

    def read_all(self) -> np.ndarray:
        """Read the requested range as float32.

        Returns:
            Samples of shape ``(time, channel)``.
        """
        return np.asarray(
            self._handle.read(
                start=self._start,
                stop=self._stop,
                channels=self._channels,
                dtype=str(np.dtype(Constants.DATA_TYPE)),
            ),
            dtype=Constants.DATA_TYPE,
        )

    def describe(self) -> dict:
        """Publish what the file knows, as JSON-safe context entries.

        Returns:
            Context entries for the output port. A key is absent rather
            than defaulted when the file does not carry it.
        """
        described = self._described
        context: dict = {
            Constants.Keys.CHANNEL_LABELS: [
                str(getattr(ch, "name", index))
                for index, ch in enumerate(described)
            ],
            Constants.Keys.CHANNEL_ROLES: [
                self._role_of(ch) for ch in described
            ],
        }

        units = [getattr(ch, "unit", None) for ch in described]
        if any(unit is not None for unit in units):
            context[Constants.Keys.CHANNEL_UNITS] = [
                None if unit is None else str(unit) for unit in units
            ]

        # An exact fraction as [numerator, denominator], so a round trip
        # can restore it and JSON can carry it.
        rate = self._rate_exact
        numerator = getattr(rate, "numerator", None)
        denominator = getattr(rate, "denominator", None)
        if numerator is not None and denominator is not None:
            context[Constants.Keys.SAMPLING_RATE_EXACT] = [
                int(numerator),
                int(denominator),
            ]

        start_time = getattr(self._handle, "start_time", None)
        if start_time is not None:
            context[Constants.Keys.START_TIME] = (
                start_time.isoformat()
                if hasattr(start_time, "isoformat")
                else str(start_time)
            )

        trust = getattr(self._handle, "trust", None)
        if trust is not None:
            context[Constants.Keys.TRUST] = str(trust)

        gaps = getattr(self._handle, "gaps", None)
        if gaps:
            context[Constants.Keys.GAPS] = [
                [int(gap[0]), int(gap[1]), str(gap[2])]
                for gap in gaps
                if self._overlaps(int(gap[0]), int(gap[1]))
            ]

        markers = self._markers()
        if markers:
            context[Constants.Keys.MARKERS] = markers

        return context

    def _markers(self) -> list:
        """Return markers inside the requested range, grid-relative.

        Returns:
            ``[sample, duration, channel, label]`` per marker, with
            sample relative to the first sample read -- so a marker
            indexes the block that was emitted rather than the file it
            came from.
        """
        query = getattr(self._handle, "markers", None)
        if query is None:
            return []
        found = query(start=self._start, stop=self._stop)
        markers = []
        for marker in found or []:
            # Named fields where the marker has them, positional
            # otherwise. Written as a branch rather than as
            # ``getattr(marker, "sample", marker[0])``, because a
            # default argument is evaluated whether it is needed or
            # not -- so that form subscripts every marker, and raises
            # on exactly the objects it was meant to accommodate.
            if hasattr(marker, "sample"):
                sample = int(marker.sample)
                duration = int(getattr(marker, "duration", 0) or 0)
                channel = getattr(marker, "channel", None)
                label = getattr(marker, "label", "")
            else:
                sample, duration, channel, label = (
                    int(marker[0]),
                    int(marker[1]),
                    marker[2],
                    marker[3],
                )
            markers.append(
                [
                    sample - self._start,
                    int(duration),
                    None if channel is None else int(channel),
                    str(label),
                ]
            )
        return markers

    def _overlaps(self, first_missing: int, n_missing: int) -> bool:
        """Whether a gap falls inside the range being read."""
        return (
            first_missing < self._stop
            and first_missing + n_missing > self._start
        )

    def _role_of(self, channel) -> str:
        """Map a GTC channel type onto a g.Pype channel role."""
        declared = getattr(channel, "type", None)
        if declared is None:
            return Constants.ChannelRoles.SIGNAL
        role = _ROLE_OF_TYPE.get(str(declared).lower())
        if role is not None:
            return role
        self.log(
            f"Channel type '{declared}' is not one g.Pype knows, and is "
            f"treated as a signal channel -- so filters and spatial "
            f"operations will include it.",
            type=Constants.LogTypes.WARNING,
        )
        return Constants.ChannelRoles.SIGNAL

    def stop(self):
        """Stop the source and release the file handle."""
        super().stop()
        close = getattr(self._handle, "close", None)
        if callable(close):
            close()
