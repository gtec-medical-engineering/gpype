from __future__ import annotations

import csv
from typing import List, Optional

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.fixed_rate_source import FixedRateSource
from .base.source import Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Column name CsvWriter uses for the time axis.
TIME_COLUMN = "Time"


def _read_csv(path: str) -> tuple:
    """Read a recording written by CsvWriter.

    Args:
        path: Path to the file.

    Returns:
        A tuple of (labels, samples, sampling_rate). The rate is None
        when it cannot be derived from the time column.

    Raises:
        ValueError: If the file has no header or no data rows.
    """
    with open(path, "r", newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))

    # Comment lines are dropped before the header is looked for, because
    # CsvWriter puts the entitlement mark on the first line of a marked
    # run -- so without this, g.Pype cannot read a file g.Pype wrote, and
    # the failure is a float conversion error naming the word "Time".
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
    values = np.asarray(
        [[float(cell) for cell in row] for row in rows[1:]],
        dtype=Constants.DATA_TYPE,
    )

    rate = None
    if header and header[0] == TIME_COLUMN:
        times = values[:, 0]
        labels = header[1:]
        samples = values[:, 1:]
        if len(times) > 1:
            step = float(np.median(np.diff(times)))
            if step > 0:
                rate = 1.0 / step
                # The time column is written in a short format, so the
                # derived rate lands just beside the real one: 250 Hz
                # comes back as 249.99996. Sampling rates are whole
                # numbers in practice, so snap when the difference is
                # clearly rounding rather than a genuine fractional rate.
                nearest = round(rate)
                if nearest > 0 and abs(rate - nearest) / nearest < 1e-4:
                    rate = float(nearest)
    else:
        labels = header
        samples = values

    return labels, samples, rate


class _CsvReaderCore(FixedRateSource):
    """Internal node replaying a recording as a live stream."""

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
            sampling_rate: Rate to replay at. Taken from the file's time
                column when not given.
            frame_size: Samples emitted per cycle. Ignored in batch mode,
                where the frame is the whole recording.
            speed: Replay speed relative to real time. Zero runs as fast
                as the machine allows.
            loop: Start again from the beginning at the end of the file.
            mode: ``realtime`` to replay frame by frame, ``batch`` to
                emit the whole recording in one cycle. See
                Constants.ExecutionMode.
            **kwargs: Additional arguments for the parent source.

        Raises:
            ValueError: If no file name is given, if the rate can neither
                be read from the file nor was supplied, or if a
                realtime-only argument is combined with batch mode.
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

        labels, samples, file_rate = _read_csv(file_name)
        if sampling_rate is None:
            sampling_rate = file_rate
        if not sampling_rate:
            raise ValueError(
                f"'{file_name}' carries no usable time column, so the "
                f"sampling rate has to be given explicitly."
            )

        self._samples = samples
        self._labels = labels
        self._position = 0
        self._speed = float(speed)
        self._exhausted = False

        # A stored configuration returns per-port values as lists, so the
        # frame size can come back either way.
        if isinstance(frame_size, list):
            frame_size = frame_size[0]

        if mode == Constants.ExecutionMode.BATCH:
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
            # The frame *is* the recording. Assigned rather than
            # defaulted, so a stored configuration cannot reinstate a
            # realtime frame size on a batch reader.
            frame_size = int(samples.shape[0])
            kwargs["frame_size"] = frame_size
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
        # The channel count comes from the file, but a stored
        # configuration carries it too, so default rather than pass it.
        kwargs.setdefault("channel_count", int(samples.shape[1]))
        super().__init__(
            sampling_rate=float(sampling_rate),
            file_name=file_name,
            speed=float(speed),
            loop=bool(loop),
            mode=mode,
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
            Output port contexts carrying the file's channel names.
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


class CsvReader(ioc.OChain):
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
        """Initialize the reader chain.

        Args:
            file_name: Path to the recording.
            sampling_rate: Rate to replay at, if the file cannot say.
            frame_size: Samples emitted per cycle. One by default, which
                keeps every downstream node usable. Ignored in batch
                mode, where the frame is the whole recording.
            speed: Replay speed relative to real time; zero for as fast
                as possible.
            loop: Start again at the end of the file.
            mode: ``realtime`` (default) replays the file frame by frame,
                so everything downstream sees the timing the amplifier
                would have produced. ``batch`` emits the whole recording
                in one cycle, for offline processing driven by
                ``Pipeline.run()``.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If no file name is available.
        """
        self._link_stream_id = stream_id_for(kwargs)
        if file_name is None:
            file_name = kwargs.get("file_name")
        if file_name is None:
            raise ValueError("file_name must be provided.")

        self._core_params = {
            "file_name": file_name,
            "sampling_rate": sampling_rate,
            "frame_size": frame_size,
            "speed": speed,
            "loop": loop,
            "mode": mode,
        }
        self._core_params.update(strip_chain_keys(kwargs))

        kwargs.setdefault("file_name", file_name)
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration()],
        )
        # file_name already reaches kwargs via the setdefault above;
        # the rest of the chain's own parameters are forwarded here.
        ioc.OChain.__init__(
            self,
            sampling_rate=sampling_rate,
            frame_size=frame_size,
            speed=speed,
            loop=loop,
            mode=mode,
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
        # residency: the core lives on the edge, and so does
        # anything recording or replaying it.
        nodes.extend(
            raw.source_stage(self, lambda: _CsvReaderCore(**self._core_params))
        )
        if residency != Constants.Residency.STANDALONE:
            nodes.append(
                Link(
                    sender=Constants.Residency.EDGE,
                    receiver=Constants.Residency.SERVER,
                    stream_id=self._link_stream_id,
                )
            )
        nodes.append(Sync())
        return nodes
