"""GtcReader: a `.gtc` recording as one batch block.

Written against the API requested in
``devdoc/archive/requests/2026-09-02_gtc_python_package.md``. The mapping from
that API to g.Pype's port context lives here and nowhere else, so when
the package lands, the parts that do not match are a failing test in
``test/test_backend_sources_gtc_reader.py`` rather than a discussion.
"""

from __future__ import annotations

import datetime
from fractions import Fraction
from typing import Optional

import numpy as np

from ...common._private import channels as selection
from ...common.constants import Constants
from ..core._private import assembly
from .base import file_shape, raw
from .base.batch_source import BatchSource
from .base.recording_reader import input_identity
from .base.source import ABSENT, Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: What a stored configuration carries that this reader derives from the
#: file rather than takes from its author: the rate, the channel count
#: and the sample count, stored as the frame size.
_DERIVED = file_shape.SHAPE_KEYS

#: The channel a GTC marker names when it is about the whole recording.
#: A u32 in the format; g.Pype's Event says None.
WHOLE_RECORDING = 0xFFFFFFFF

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

    The rate, channel count and frame size are the file's. Given one
    that disagrees, the reader refuses it; a document's copy is warned
    about and replaced by the file's.

    Where the node is not run -- on a server, on an edge it is not
    assigned to, under ``load_from`` -- it opens no file and imports no
    ``gtc``: a shape the document gives is used as given, and one it
    leaves to the file is stood in for.

    Args:
        file_name: Path to the recording.
        channels: Channel names or indices to read. All by default.
            Reading a subset costs proportionally less -- the format is
            seekable per channel -- and the published metadata is
            narrowed to match. An index may be a numpy integer; a
            boolean is refused rather than read as 0 or 1.
        start: First sample to read, on the nominal grid. Zero by
            default.
        stop: One past the last sample to read. End of recording by
            default.
        edge_id: Which edge runs this node, matched against the
            edge process's --edge-id. None, the default, is every
            edge; ignored when the pipeline is not distributed.
        **kwargs: Additional arguments for the parent BatchSource.

    Raises:
        ImportError: If the ``gtc`` package is not installed.
        ValueError: If no file name is given, a channel is neither a name
            nor an index, the requested range is empty or outside the
            recording, or a ``sampling_rate``, ``channel_count`` or
            ``frame_size`` passed by hand disagrees with the file.
    """

    def __init__(
        self,
        file_name: str,
        channels: Optional[list] = None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        if file_name is None:
            raise ValueError("file_name must be provided.")

        # Plain ints and strings, so the handle, the lookup and the
        # document all see the same selection (D-NODE-46).
        self._channels = (
            None
            if channels is None
            else selection.as_selection(channels, "channels", labels=True)
        )

        # The file's shape arrives from a stored configuration beside the
        # parameters that decide it, or from an author by hand. Passed
        # on, it collided with the value derived here. Where the file is
        # read it is the file's to answer, so it is checked against the
        # file rather than dropped (D-NODE-61).
        given = {key: kwargs.pop(key, ABSENT) for key in _DERIVED}

        node = file_shape.node_name(self, kwargs)

        # A refusal after the file is open closes it: nothing else holds
        # the handle, so it would stay open until the process ended.
        self._handle = None
        #: What `Constants.Keys.INPUT` publishes, or None where the file
        #: is not opened.
        self._input_identity = None
        try:
            # **Neither a server nor a replay opens the file**, and nor
            # does an edge the reader is not assigned to: none of them
            # runs the node (`raw.source_stage`). The shape a document
            # gives is used as given, and what it leaves to the file is
            # stood in for and written back as absent (D-CORE-70).
            if not assembly.builds_core_for(edge_id) or raw.is_replaying():
                shape = raw.stand_in(
                    self,
                    sampling_rate=(
                        given[Constants.Keys.SAMPLING_RATE],
                        raw.READER_STAND_IN_RATE,
                    ),
                    channel_count=(given[Constants.Keys.CHANNEL_COUNT], 1),
                    frame_size=(given[Constants.Keys.FRAME_SIZE], 1),
                )
                rate, channel_count, sample_count = map(Source.scalar, shape)
                self._described = []
                self._rate_exact = None
                self._start = 0 if start is None else int(start)
                self._stop = None if stop is None else int(stop)
            else:
                self._handle = _open(file_name)
                rate, channel_count, sample_count = self._read_shape(
                    file_name, start, stop, given, node
                )
                # The whole file, whatever range or channels are read:
                # it names the input, not the block.
                self._input_identity = input_identity(file_name)

            super().__init__(
                sampling_rate=float(rate),
                channel_count=int(channel_count),
                sample_count=int(sample_count),
                file_name=file_name,
                channels=self._channels,
                start=start,
                stop=stop,
                edge_id=edge_id,
                **kwargs,
            )
        except BaseException:
            self._release()
            raise

    def _read_shape(
        self,
        file_name: str,
        start: Optional[int],
        stop: Optional[int],
        given: dict,
        node: str,
    ) -> tuple:
        """Take the range, the channels and the rate from the open file.

        Args:
            file_name: Path to the recording, for the messages.
            start: First sample to read, or None.
            stop: One past the last sample to read, or None.
            given: What the reader was given for each of ``_DERIVED``.
            node: The reader as a message names it.

        Returns:
            ``(rate, channel_count, sample_count)``, the rate exact.

        Raises:
            ValueError: If the range is empty or outside the recording, a
                channel does not exist, or a given shape disagrees.
        """
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
        rate = self._rate_exact
        channel_count = len(described)
        sample_count = self._stop - self._start
        # Agreement is equality here, so the file's values are the ones
        # to build with; the rate stays the exact fraction.
        file_shape.resolve(
            file_name,
            given,
            {
                Constants.Keys.SAMPLING_RATE: rate,
                Constants.Keys.CHANNEL_COUNT: channel_count,
                Constants.Keys.FRAME_SIZE: sample_count,
            },
            node,
            "The rate is the file's, the channel count is what "
            "'channels' selects, and the frame size is the range 'start' "
            "to 'stop' reads: drop what disagrees.",
        )
        return rate, channel_count, sample_count

    @staticmethod
    def _select(described: list, wanted: list) -> list:
        """Return the described channels the caller asked for, in order.

        Args:
            described: Every channel the file describes.
            wanted: Names or plain int indices, in the order to read
                them, as ``channels.as_selection`` returns them.

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

        # GTC's calibration field set (D-BATCH-43): gain and offset as
        # exact fractions, never floats, so a round trip through JSON
        # and back restores them exactly; clipping limits; hardware
        # filters already applied. Each is absent unless at least one
        # channel states it, same rule as units.
        gains = [getattr(ch, "gain", None) for ch in described]
        if any(gain is not None for gain in gains):
            context[Constants.Keys.CHANNEL_GAINS] = [
                self._as_fraction_pair(gain) for gain in gains
            ]
        offsets = [getattr(ch, "offset", None) for ch in described]
        if any(offset is not None for offset in offsets):
            context[Constants.Keys.CHANNEL_OFFSETS] = [
                self._as_fraction_pair(offset) for offset in offsets
            ]
        clipping = [getattr(ch, "clipping", None) for ch in described]
        if any(limits is not None for limits in clipping):
            context[Constants.Keys.CHANNEL_CLIPPING] = [
                (
                    None
                    if limits is None
                    else [float(limits[0]), float(limits[1])]
                )
                for limits in clipping
            ]
        filters = [getattr(ch, "filters", None) for ch in described]
        if any(filters):
            context[Constants.Keys.CHANNEL_FILTERS] = [
                list(f) if f else [] for f in filters
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

        start_time = self._start_time()
        if start_time is not None:
            context[Constants.Keys.START_TIME] = start_time

        trust = getattr(self._handle, "trust", None)
        if trust is not None:
            context[Constants.Keys.TRUST] = str(trust)

        gaps = getattr(self._handle, "gaps", None)
        if gaps:
            context[Constants.Keys.GAPS] = self._gaps(gaps)

        markers = self._markers()
        if markers:
            context[Constants.Keys.MARKERS] = markers

        if self._input_identity is not None:
            context[Constants.Keys.INPUT] = dict(self._input_identity)

        return context

    def _start_time(self) -> Optional[str]:
        """Return the absolute time of the first sample read.

        Returns:
            ISO 8601: the file's start time, moved by ``start / rate`` on
            the nominal grid. None when the file records none, or when a
            read from ``start > 0`` cannot parse it to move it.
        """
        stamp = getattr(self._handle, "start_time", None)
        if stamp is None:
            return None
        if self._start == 0:
            return (
                stamp.isoformat()
                if hasattr(stamp, "isoformat")
                else str(stamp)
            )
        if not isinstance(stamp, datetime.datetime):
            try:
                stamp = datetime.datetime.fromisoformat(str(stamp))
            except ValueError:
                self.log(
                    f"The file's start time {str(stamp)!r} cannot be "
                    f"parsed, so a read from sample {self._start} "
                    f"reports none.",
                    type=Constants.LogTypes.WARNING,
                )
                return None
        # Exact until the last step: timedelta holds microseconds.
        offset = Fraction(self._start) / Fraction(self._rate_exact)
        shift = datetime.timedelta(microseconds=round(offset * 1_000_000))
        return (stamp + shift).isoformat()

    def _markers(self) -> list:
        """Return markers inside the requested range, grid-relative.

        Returns:
            ``[sample, duration, channel, label]`` per marker, with
            sample relative to the first sample read -- so a marker
            indexes the block that was emitted rather than the file it
            came from. A marker about the whole recording, channel
            ``WHOLE_RECORDING`` in the format, has channel None.
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
                    (
                        None
                        if channel is None or int(channel) == WHOLE_RECORDING
                        else int(channel)
                    ),
                    str(label),
                ]
            )
        return markers

    def _gaps(self, gaps) -> list:
        """Return gaps inside the requested range, on the block's grid.

        Args:
            gaps: ``(first_missing, n_missing, reason)`` per gap, on the
                file's grid.

        Returns:
            ``[first_missing, n_missing, reason]`` per gap, relative to
            the first sample read as markers are, and cut at the range's
            ends so ``n_missing`` counts only samples of this block.
        """
        found = []
        for gap in gaps:
            first, count = int(gap[0]), int(gap[1])
            if not self._overlaps(first, count):
                continue
            begin = max(first, self._start)
            end = min(first + count, self._stop)
            found.append([begin - self._start, end - begin, str(gap[2])])
        return found

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

    @staticmethod
    def _as_fraction_pair(value) -> Optional[list]:
        """A CHAN gain or offset as ``[numerator, denominator]``, or None.

        Args:
            value: A ``Fraction``, or None where the channel does not
                state one.

        Returns:
            The exact pair, JSON-safe, or None.
        """
        if value is None:
            return None
        return [int(value.numerator), int(value.denominator)]

    def stop(self):
        """Stop the source and release the file handle."""
        super().stop()
        self._release()

    def _release(self) -> None:
        """Close the file handle, where there is one."""
        close = getattr(self._handle, "close", None)
        if callable(close):
            close()
