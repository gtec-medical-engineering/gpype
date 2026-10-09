"""Shared machinery for a source that replays a saved recording.

``CsvReader`` (``csv_reader.py``) open-codes all of this today --
mode validation, the pacing/batch split, exhaustion, frame emission. This
module holds the same machinery once, so :class:`~gpype.MatReader`,
:class:`~gpype.HDF5Reader` and :class:`~gpype.EDFReader` do not each
reimplement it. Refactoring ``CsvReader`` onto this base is out of
scope for this change and is left as a follow-up.
"""

from __future__ import annotations

import datetime
import hashlib
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ....common._private import channels, events
from ....common.constants import Constants
from ...core._private import assembly
from ...core.o_port import OPort
from . import file_shape, raw
from .fixed_rate_source import FixedRateSource
from .source import ABSENT, Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: Bytes hashed per read by :func:`input_identity`.
_HASH_CHUNK = 1 << 20


def input_identity(path) -> Optional[dict]:
    """Say which file a reader read, for ``Constants.Keys.INPUT``.

    The basename, not the path: a path names somebody's directory tree
    and differs on every machine that reads the same file. The SHA-256
    of the whole file is the identity, so a copy is the same input
    under another name and mtime (D-BATCH-71).

    Args:
        path: The file the reader opened.

    Returns:
        ``{name, size, mtime, sha256}``, JSON-safe, or None when *path*
        names no regular file.
    """
    if not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    stat = os.stat(path)
    mtime = datetime.datetime.fromtimestamp(
        stat.st_mtime, tz=datetime.timezone.utc
    )
    return {
        "name": os.path.basename(os.fspath(path)),
        "size": int(stat.st_size),
        "mtime": mtime.isoformat(),
        "sha256": digest.hexdigest(),
    }


@dataclass
class Recording:
    """What a format-specific ``_load()`` hands back to the shared base.

    Attributes:
        samples: The recording, shape ``(time, channel)``, as
            ``Constants.DATA_TYPE``.
        labels: One label per channel, or None to leave channels unnamed.
            None rather than positional placeholders when the file
            itself does not name its channels -- inventing "Ch01" here
            would make a downstream file writer believe a montage said
            so.
        roles: One role per channel, or None to default every channel to
            ``SIGNAL``.
        sampling_rate: Rate in Hz, or None when nothing in the file could
            supply one and the caller's own ``sampling_rate`` argument is
            the only source left.
        extras: Ready-made port-context entries the reader already knows
            how to name -- ``Constants.Keys.DEVICE_SERIAL``,
            ``CHANNEL_UNITS``, ``START_TIME``, ``SAMPLING_RATE_EXACT``,
            ``MARKERS``, ``GAPS``, ``TRUST``. Markers and gaps index the
            recording's first sample. Merged into the output port
            context verbatim. The one exception is the key ``"mark"``:
            it is not a port-context key (nothing else in g.Pype has
            needed one), so the base class pulls it out and exposes it
            as the reader's own ``mark`` property instead of publishing
            it downstream.
        rate_from_times: Whether ``sampling_rate`` was derived from a
            time column rather than stated by the file. A given rate
            agrees with a derived one to ``file_shape``'s tolerance, and
            with a stated one exactly.
    """

    samples: np.ndarray
    labels: Optional[list] = None
    roles: Optional[list] = None
    sampling_rate: Optional[float] = None
    extras: dict = field(default_factory=dict)
    rate_from_times: bool = False


def rate_from_time_column(times) -> Optional[float]:
    """Derive a sampling rate from a recorded time column.

    Args:
        times: Time values in seconds, in sample order.

    Returns:
        The rate in Hz, or None if fewer than two samples are given or
        the derived step is not positive.

    The snap to the nearest integer below is not about recovering the
    rate -- for a float64 time column built the way ``FileWriter`` builds
    one (``np.arange(n) / fs``), the derived rate is already correct to
    about 1e-12 relative (measured: 250 Hz over 100000 samples comes back
    as 249.999999999917), five orders of magnitude better than the CSV
    text column this snap was originally written for. It is about
    publishing exactly ``250.0`` rather than ``249.999999999917`` into a
    port context that downstream filter design and rate comparisons
    read literally. Measured that the 1e-4 threshold correctly does not
    fire for a genuine fractional rate: 500/3 sits 2.0e-3 away from the
    nearest integer and 512000/1001 sits 9.6e-4 away, both comfortably
    outside the snap.
    """
    times = np.asarray(times, dtype=np.float64)
    if times.shape[0] < 2:
        return None
    step = float(np.median(np.diff(times)))
    if step <= 0:
        return None
    rate = 1.0 / step
    nearest = round(rate)
    if nearest > 0 and abs(rate - nearest) / nearest < 1e-4:
        rate = float(nearest)
    return rate


def _h5py():
    """Import h5py, or say what installs it.

    Returns:
        The h5py module.

    Raises:
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "reading this file needs h5py, provided by the 'formats' "
            'extra: pip install "gpype[formats]"'
        ) from error
    return h5py


def _read_gpype_meta(handle, module) -> Optional[dict]:
    """Read and parse ``/gpype_meta`` from an open HDF5/MAT handle.

    Args:
        handle: Open h5py File.
        module: The h5py module (passed in rather than re-imported, since
            the caller already resolved it).

    Returns:
        The parsed document, or None if the file carries none -- which is
        exactly what a file the untracked writer drafts produced looks
        like, and what a third-party MAT/HDF5 file looks like too.
    """
    from ...sinks.base import recording_meta

    if recording_meta.META_DATASET not in handle:
        return None
    node = handle[recording_meta.META_DATASET]
    if np.issubdtype(node.dtype, np.integer):
        # MatWriter's encoding: a (N, 1) column of uint16 character
        # codes, MATLAB's own convention for "this is really text".
        codes = np.asarray(node[()]).reshape(-1)
        blob = "".join(chr(int(code)) for code in codes)
    else:
        # HDF5Writer's encoding: an ordinary HDF5 string dataset.
        blob = node.asstr()[()]
    return recording_meta.parse(blob)


def read_h5(path: str, variable_name: str, has_time_row: bool) -> Recording:
    """Read a recording written by MatWriter or HDF5Writer.

    Shared between :mod:`gpype.backend.sources.mat_reader` and
    :mod:`gpype.backend.sources.hdf5_reader`, which agree on the data
    layout -- ``(1+n_channels, n_samples)`` with the time row first, if
    ``has_time_row`` -- because the two writers agree on it.

    Args:
        path: Path to the file.
        variable_name: Name of the dataset the recording was written
            under.
        has_time_row: Whether column 0 of the recovered array is a time
            axis rather than a signal channel. Explicit rather than
            sniffed: a heuristic guessing this would one day be wrong
            about a real signal channel, and the failure would be a
            silently dropped channel.

    Returns:
        The recording. ``sampling_rate`` is None when neither
        ``/gpype_meta``, native HDF5 attributes, nor a time row could
        supply one -- the caller's own ``sampling_rate`` argument is the
        last resort, exactly as it is for :class:`~gpype.CsvReader`.

    Raises:
        ImportError: If h5py is not installed.
        ValueError: If ``variable_name`` names no dataset in the file.
    """
    module = _h5py()
    with module.File(path, "r") as handle:
        meta = _read_gpype_meta(handle, module)

        if variable_name not in handle:
            available = sorted(handle.keys())
            raise ValueError(
                f"'{path}' has no dataset named '{variable_name}'; it "
                f"holds {available}."
            )
        ds = handle[variable_name]
        raw = np.asarray(ds[()])

        labels = None
        roles = None
        rate = None
        from_times = False
        extras: dict = {}

        if meta is not None:
            labels = meta.get("channel_labels")
            roles = meta.get("channel_roles")
            rate = meta.get("sampling_rate")
            if "device_serial" in meta:
                extras[Constants.Keys.DEVICE_SERIAL] = meta["device_serial"]
            if "channel_units" in meta:
                extras[Constants.Keys.CHANNEL_UNITS] = meta["channel_units"]
            for key, name in (
                (Constants.Keys.CHANNEL_GAINS, "channel_gains"),
                (Constants.Keys.CHANNEL_OFFSETS, "channel_offsets"),
                (Constants.Keys.CHANNEL_CLIPPING, "channel_clipping"),
                (Constants.Keys.CHANNEL_FILTERS, "channel_filters"),
            ):
                if name in meta:
                    extras[key] = meta[name]
            if "start_time" in meta:
                extras[Constants.Keys.START_TIME] = meta["start_time"]
            if "sampling_rate_exact" in meta:
                extras[Constants.Keys.SAMPLING_RATE_EXACT] = meta[
                    "sampling_rate_exact"
                ]
            for key in (
                Constants.Keys.MARKERS,
                Constants.Keys.GAPS,
                Constants.Keys.TRUST,
            ):
                if key in meta:
                    extras[key] = meta[key]
            if "mark" in meta:
                extras["mark"] = meta["mark"]
            # Not a context key: nested under this reader's own INPUT as
            # ``derived_from`` in ``__init__``, once the identity it
            # belongs to is known (LQ-P3). Popped back out before
            # ``extras`` reaches the port context.
            if "provenance" in meta:
                extras["_recorded_provenance"] = meta["provenance"]
        else:
            # No gpype_meta: this is either a file written by the
            # untracked HDF5Writer draft (native attrs may still carry
            # something) or a genuinely foreign file. Fall back to plain
            # HDF5 attributes on the data dataset before giving up.
            attrs = ds.attrs
            if "sampling_rate" in attrs:
                rate = float(attrs["sampling_rate"])
            if "channel_labels" in attrs:
                labels = [_as_str(x) for x in attrs["channel_labels"]]
            if "channel_roles" in attrs:
                roles = [_as_str(x) for x in attrs["channel_roles"]]
            if "device_serial" in attrs:
                extras[Constants.Keys.DEVICE_SERIAL] = _as_str(
                    attrs["device_serial"]
                )
            if "mark" in attrs:
                extras["mark"] = _as_str(attrs["mark"])

        # h5py/MATLAB store (cols, samples); this is the array a caller
        # actually indexes, samples first.
        data = raw.T
        if has_time_row:
            times = data[:, 0].astype(np.float64)
            samples = data[:, 1:]
            if rate is None:
                rate = rate_from_time_column(times)
                from_times = rate is not None
        else:
            samples = data

        samples = np.asarray(samples, dtype=Constants.DATA_TYPE)

    return Recording(
        samples=samples,
        labels=list(labels) if labels else None,
        roles=list(roles) if roles else None,
        sampling_rate=rate,
        extras=extras,
        rate_from_times=from_times,
    )


def _as_str(value) -> str:
    """Decode a value that may be h5py bytes or a numpy string scalar."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


class RecordingReader(FixedRateSource):
    """Internal node base replaying a saved recording as a live stream.

    Mirrors ``CsvReader`` exactly in its engine -- mode validation,
    the realtime/batch split, exhaustion, frame emission -- so that
    behaviour does not drift between formats. The one new thing is the
    subclass seam: :meth:`_load` returns a :class:`Recording`, and
    everything else here is format-agnostic.
    """

    #: Positions advance through the file, not with the host clock, so
    #: this stream cannot be merged with a live one.
    TIME_BASE = Constants.TimeBase.RECORDED

    def __init__(
        self,
        file_name: str,
        sampling_rate: Optional[float] = None,
        frame_size: int = 1,
        speed: float = 1.0,
        loop: bool = False,
        mode: str = Constants.ExecutionMode.REALTIME,
        **kwargs,
    ):
        """Initialize the reader core.

        Args:
            file_name: Path to the recording.
            sampling_rate: The recording's rate. Taken from the file
                when not given; required when the file records none. One
                given must agree with a rate the file records.
            frame_size: Samples emitted per cycle. In batch mode the
                frame is the whole recording.
            speed: Replay speed relative to real time. Zero runs as fast
                as the machine allows.
            loop: Start again from the beginning at the end of the file.
            mode: ``realtime`` to replay frame by frame, ``batch`` to
                emit the whole recording in one cycle.
            **kwargs: Additional arguments for the parent source.

        Raises:
            ValueError: If no file name is given, if the rate can neither
                be read from the file nor was supplied, if a
                realtime-only argument is combined with batch mode, or
                if a ``sampling_rate``, ``channel_count`` or batch
                ``frame_size`` given by hand disagrees with the file.
                A document's copy is warned about and replaced by the
                file's.
        """
        if file_name is None:
            raise ValueError("file_name must be provided.")
        if speed < 0:
            raise ValueError("speed must not be negative.")
        if mode not in (
            Constants.ExecutionMode.REALTIME,
            Constants.ExecutionMode.BATCH,
        ):
            raise ValueError(
                f"mode must be "
                f"'{Constants.ExecutionMode.REALTIME}' or "
                f"'{Constants.ExecutionMode.BATCH}'; got {mode!r}."
            )

        # **A server does not open the file, and nor does a replay.**
        # Under `chain-assembly` step 4 the public name is the core, so a
        # document naming a reader is *constructed* by every process that
        # reads it -- and the recording lives on the edge that made it,
        # or is stood in for by a raw run. Neither runs the node
        # (`raw.source_stage`), and nor does an edge the reader is not
        # assigned to, so what the author left to the file is stood in
        # for and written back as given: a server takes the shape from
        # the context the edge sends (D-CORE-70), a replay from the
        # recording. A 4.0.x document records no channel count for a
        # reader at all.
        #
        # A stored configuration returns per-port values as lists.
        if isinstance(frame_size, list):
            frame_size = frame_size[0]
        batch = mode == Constants.ExecutionMode.BATCH
        edge_id = kwargs.get(Constants.Keys.EDGE_ID)
        if not assembly.builds_core_for(edge_id) or raw.is_replaying():
            (sampling_rate,) = raw.stand_in(
                self, sampling_rate=(sampling_rate, raw.READER_STAND_IN_RATE)
            )
            raw.stand_in(
                self,
                channel_count=(
                    kwargs.get(Constants.Keys.CHANNEL_COUNT, ABSENT),
                    None,
                ),
            )
            self._samples = None
            self._labels = None
            self._roles = None
            self._mark = None
            self._extras = {}
        else:
            recording = self._load(file_name)
            # The file decides the shape where it is read: a stored
            # channel count used to win, and a re-pointed document
            # failed on the roles its new file gave (D-BATCH-75).
            sampling_rate, frame_size = file_shape.recording(
                self,
                file_name,
                kwargs,
                sampling_rate,
                frame_size,
                batch,
                recording.sampling_rate,
                recording.samples,
                rate_from_times=recording.rate_from_times,
            )
            if not sampling_rate:
                raise ValueError(
                    f"'{file_name}' carries no usable time column, so "
                    f"the sampling rate has to be given explicitly."
                )

            self._samples = recording.samples
            self._labels = recording.labels
            self._roles = recording.roles
            extras = dict(recording.extras or {})
            # Not a port-context key -- see the Recording docstring --
            # so it is exposed as `self.mark` instead of published
            # downstream.
            self._mark = extras.pop("mark", None)
            # Popped back out here too: this file's own recorded
            # provenance is not this run's PROVENANCE (that is built
            # fresh by Pipeline.start(), and would otherwise be
            # overwritten by it anyway) -- it is what *produced* the
            # file this run is reading, so it belongs on this reader's
            # own INPUT, not on the context's PROVENANCE key (LQ-P3).
            recorded_provenance = extras.pop("_recorded_provenance", None)
            identity = input_identity(file_name)
            if identity is not None:
                if recorded_provenance is not None:
                    identity = dict(identity)
                    identity["derived_from"] = recorded_provenance
                extras[Constants.Keys.INPUT] = identity
            # A file that stores no events of its own may still carry a
            # trigger channel; stored markers win, so nothing already
            # counted by the file is counted a second time (D-BATCH-87).
            if not extras.get(Constants.Keys.MARKERS) and self._roles:
                derived = events.derive_markers(recording.samples, self._roles)
                if derived:
                    extras[Constants.Keys.MARKERS] = derived
            self._extras = extras
        self._position = 0
        self._speed = float(speed)
        self._exhausted = False

        if batch:
            if float(speed) != 1.0:
                raise ValueError(
                    "speed describes pacing and a batch run has none: "
                    "the whole recording is processed in one cycle. "
                    "Drop speed, or use mode='realtime'."
                )
            if loop:
                raise ValueError(
                    "loop cannot be combined with mode='batch': a "
                    "looping reader never reaches the end, and a batch "
                    "run is defined by reaching it."
                )
            # The file's sample count where it was read, and the
            # document's frame size as given where it was not.
            kwargs["frame_size"] = int(frame_size or 1)
            self.EXECUTION_MODE = Constants.ExecutionMode.BATCH
        else:
            kwargs.setdefault("frame_size", int(frame_size))
            # FixedRateSource paces one cycle per *sample*, and step()
            # emits frame_size samples per cycle -- so without this a
            # recording replays frame_size times too fast, silently.
            # Generator carries the same compensation for the same
            # reason; see FixedRateSource._thread_function.
            #
            # Batch is excluded deliberately rather than by omission: it
            # runs no pacing thread, and a decimation factor there would
            # make the driver's single cycle() emit nothing at all.
            kwargs.setdefault("decimation_factor", int(frame_size))

        kwargs.setdefault("output_ports", [OPort.Configuration()])
        super().__init__(
            sampling_rate=float(sampling_rate),
            file_name=file_name,
            speed=float(speed),
            loop=bool(loop),
            mode=mode,
            **kwargs,
        )

    def _load(self, file_name: str) -> Recording:
        """Read the recording. Subclasses must implement this.

        Args:
            file_name: Path to the recording.

        Returns:
            The parsed recording.

        Raises:
            NotImplementedError: Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _load()."
        )

    def start(self):
        """Start the reader.

        A batch run has no pacing thread: the driver calls ``cycle()``
        once and the whole recording is emitted in that one call.
        Starting the thread as well would race the driver for the same
        file position.
        """
        if self.EXECUTION_MODE == Constants.ExecutionMode.BATCH:
            Source.start(self)
            return
        super().start()

    @property
    def sample_count(self) -> int:
        """Number of samples in the file."""
        return int(self._samples.shape[0])

    @property
    def is_exhausted(self) -> bool:
        """Whether the whole recording has been emitted.

        Returns:
            True once the last sample has been handed over. Never true
            for a looping reader.
        """
        return self._exhausted

    @property
    def mark(self) -> Optional[str]:
        """The entitlement mark recovered from the file, if it had one.

        None for a file written by an unmarked run, or one this format's
        writer never had the chance to mark at all.
        """
        return self._mark

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Describe the replayed channels.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts carrying the file's channel names,
            roles, and whatever extra provenance it recorded.
        """
        port_context_out = super().setup(data, port_context_in)
        self._position = 0
        self._exhausted = False

        if self._labels is not None:
            roles = self._roles
            if roles is None:
                roles = [Constants.ChannelRoles.SIGNAL] * len(self._labels)
            port_context_out[PORT_OUT].update(
                channels.describe(roles, list(self._labels), None)
            )
        elif self._roles is not None:
            port_context_out[PORT_OUT].update(
                channels.describe(list(self._roles))
            )

        port_context_out[PORT_OUT].update(self._extras)
        identity = self._extras.get(Constants.Keys.INPUT)
        if identity is not None:
            # A copy, as CsvReader and GtcReader publish: otherwise a
            # caller editing what setup() returned would change what the
            # next setup() publishes.
            port_context_out[PORT_OUT][Constants.Keys.INPUT] = dict(identity)
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Emit the next frame of the recording.

        Returns:
            The next frame, or None on a non-decimation step or once the
            file is exhausted.
        """
        # The pacing thread runs once per sample and this emits a whole
        # frame, so without this the recording advances frame_size times
        # too fast. In batch mode the factor is one and this is always
        # true, which is what a driver expecting one cycle needs.
        if not self.is_decimation_step():
            return None

        if self._exhausted:
            return None

        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
        if isinstance(frame_size, list):
            frame_size = frame_size[0]

        end = self._position + frame_size
        if end > self._samples.shape[0]:
            if self.config.get("loop"):
                self._position = 0
                end = frame_size
            else:
                # A partial frame would misrepresent the sampling grid,
                # so the file ends on the last whole frame.
                self._mark_exhausted()
                return None

        block = self._samples[self._position : end, :]
        self._position = end
        if end >= self._samples.shape[0] and not self.config.get("loop"):
            self._mark_exhausted()
        return {PORT_OUT: block}

    def _mark_exhausted(self) -> None:
        """Record that the recording has been fully emitted, once."""
        if self._exhausted:
            return
        self._exhausted = True
        self.log(
            f"Reached the end of "
            f"'{self.config['file_name']}' after "
            f"{self._position} sample(s).",
            type=Constants.LogTypes.INFO,
        )
