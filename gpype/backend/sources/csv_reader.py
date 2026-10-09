from __future__ import annotations

import csv
from typing import Optional

import numpy as np

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from ..core._private import assembly
from ..core.o_port import OPort
from .base import file_shape, raw
from .base.fixed_rate_source import FixedRateSource
from .base.recording_reader import (
    input_identity,
    rate_from_time_column,
)
from .base.source import ABSENT, Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Column name CsvWriter uses for the time axis.
TIME_COLUMN = "Time"


def _read_csv(path: str) -> tuple:
    """Read a recording written by CsvWriter.

    Args:
        path: Path to the file.

    Returns:
        A tuple of (labels, samples, sampling_rate, mark). The rate is
        None when it cannot be derived from the time column; the mark is
        the entitlement mark CsvWriter put on a marked run's first line,
        or None.

    Raises:
        ValueError: If the file has no header or no data rows.
    """
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    # Comment lines are dropped before the header is looked for, because
    # CsvWriter puts the entitlement mark on the first line of a marked
    # run -- so without this, g.Pype cannot read a file g.Pype wrote, and
    # the failure is a float conversion error naming the word "Time".
    # The mark itself is kept: dropped with the rest, a marked file
    # trained an unmarked artifact (D-BATCH-92).
    mark = None
    for row in rows:
        first = row[0].strip() if row else ""
        if not first:
            continue
        if not first.startswith("#"):
            break  # the header: the mark is a leading comment
        if first.lstrip("#").strip() == MARK:
            mark = MARK
            break
    rows = [
        row
        for row in rows
        if row
        and any(c.strip() for c in row)
        and not row[0].lstrip().startswith("#")
    ]
    if len(rows) < 2:
        raise ValueError(f"'{path}' has no header row and data rows to read.")

    header = [cell.strip() for cell in rows[0]]
    # Parsed as float64 and narrowed after the time column is taken out.
    # Parsed as float32, the time column lost the step to the stamps'
    # own resolution: 100000 samples at 250 Hz, written by CsvWriter,
    # read back as 250.137 Hz (measured).
    values = np.asarray(
        [[float(cell) for cell in row] for row in rows[1:]],
        dtype=np.float64,
    )

    rate = None
    if header and header[0] == TIME_COLUMN:
        labels = header[1:]
        samples = values[:, 1:]
        rate = rate_from_time_column(values[:, 0])
    else:
        labels = header
        samples = values

    samples = np.asarray(samples, dtype=Constants.DATA_TYPE)
    return labels, samples, rate, mark


class CsvReader(FixedRateSource):
    """Replays a recording as if it were a live source.

    This is what makes a session repeatable: a chain can be debugged
    without a subject present, yesterday's recording can be run through
    a changed pipeline, and a regression test can use real data instead
    of a synthetic approximation of it.

    Channel names are taken from the file's header, so a recording made
    with a montage keeps its electrode names on the way back in.

    ``speed`` decides what "replay" means. At one the file is paced to
    the wall clock, so the recording behaves like the amplifier that
    produced it and everything downstream sees the timing it expects. At
    zero it runs as fast as the machine allows, which is what offline
    reprocessing wants.
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
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the reader core.

        Args:
            file_name: Path to the recording.
            sampling_rate: The recording's rate. Taken from the file's
                time column when not given; required without one. One
                given must agree with the time column.
            frame_size: Samples emitted per cycle. In batch mode the
                frame is the whole recording.
            speed: Replay speed relative to real time. Zero runs as fast
                as the machine allows.
            loop: Start again from the beginning at the end of the file.
            mode: ``realtime`` to replay frame by frame, ``batch`` to
                emit the whole recording in one cycle. See
                Constants.ExecutionMode.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
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

        # **A server does not open the file, and nor does a replay.** See
        # `RecordingReader.__init__` for the whole argument: a document
        # naming a reader is constructed by every process that reads it,
        # neither of these runs the node -- nor does an edge it is not
        # assigned to -- and what the author left to the file is stood
        # in for and written back as given.
        #
        # A stored configuration returns per-port values as lists, so the
        # frame size can come back either way.
        if isinstance(frame_size, list):
            frame_size = frame_size[0]
        batch = mode == Constants.ExecutionMode.BATCH
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
            samples = None
            labels = None
            identity = None
            mark = None
        else:
            labels, samples, file_rate, mark = _read_csv(file_name)
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
                file_rate,
                samples,
                rate_from_times=True,
            )
            if not sampling_rate:
                raise ValueError(
                    f"'{file_name}' carries no usable time column, so "
                    f"the sampling rate has to be given explicitly."
                )
            identity = input_identity(file_name)

        self._samples = samples
        self._labels = labels
        #: What `Constants.Keys.INPUT` publishes, or None where the file
        #: was not read.
        self._input_identity = identity
        self._mark = mark
        self._position = 0
        self._speed = float(speed)
        self._exhausted = False

        if batch:
            # Both of these describe pacing, and a batch run has none.
            # Refused rather than ignored: a caller who asked for
            # half-speed playback and got a single block would have no
            # way to notice.
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
            # The frame *is* the recording: the file's sample count where
            # it was read, and the document's frame size as given where
            # it was not.
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
            edge_id=edge_id,
            **kwargs,
        )

    def start(self):
        """Start the reader.

        A batch run has no pacing thread: the driver calls ``cycle()``
        once and the whole recording is emitted in that one call.
        Starting the thread as well would race the driver for the same
        file position, and the run would end with part of the recording
        processed twice.
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
    def mark(self) -> Optional[str]:
        """The entitlement mark recovered from the file, if it had one.

        None for a file written by an unmarked run, or where the file
        was not read.
        """
        return self._mark

    @property
    def is_exhausted(self) -> bool:
        """Whether the whole recording has been emitted.

        Returns:
            True once the last sample has been handed over. Never true
            for a looping reader.
        """
        return self._exhausted

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Describe the replayed channels.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts carrying the file's channel names and,
            where the file was read, its identity.
        """
        port_context_out = super().setup(data, port_context_in)
        self._position = 0
        self._exhausted = False
        if self._labels:
            port_context_out[PORT_OUT].update(
                channels.describe(
                    [Constants.ChannelRoles.SIGNAL] * len(self._labels),
                    list(self._labels),
                    None,
                )
            )
        if self._input_identity is not None:
            port_context_out[PORT_OUT][Constants.Keys.INPUT] = dict(
                self._input_identity
            )
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
        # Exhausted as soon as the last sample has been handed over,
        # rather than one cycle later when the next read runs past the
        # end. A batch driver asks whether there is more, and answering
        # "perhaps" costs it a cycle that emits nothing.
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
