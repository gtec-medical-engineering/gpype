from __future__ import annotations

from typing import List

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core.i_port import IPort
from .base.file_writer import FileWriter

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN


def _pyedflib():
    """Import pyedflib, or say what installs it.

    Deferred to construction time for the same reason
    ``mat_writer._h5py`` is: a module-level import would make
    ``scripts/generate_catalog.py`` raise on any machine without it.

    Returns:
        The pyedflib module.

    Raises:
        ImportError: If pyedflib is not installed.
    """
    try:
        import pyedflib
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "EDFWriter needs pyedflib, provided by the 'formats' extra: "
            "pip install 'gpype[formats]'"
        ) from error
    return pyedflib


class _EDFWriterCore(FileWriter):
    """Internal node writing a recording as an EDF+ (.edf) file.

    Rewritten from the untracked ``luna_edf_file_writer.py`` draft as a
    plain :class:`FileWriter` subclass: the draft's async ``record``
    port, ``set_file_prefix()`` and its own second queue and worker
    thread are dropped rather than ported, since ``FileWriter`` already
    owns the queue, the worker thread and the timestamped, collision-free
    output path.

    EDF stores samples in fixed-length *data records*, and pyedflib pads
    every ``writeSamples`` call out to a whole record -- silently, and
    not with zeros. Measured: three 100-sample ``writeSamples`` calls
    against a 250-sample record produced a 750-sample file of which 450
    samples were fabricated, interleaved *between* the real blocks (value
    counts ``{0.153: 450, 0.763: 100, 1.984: 100, 2.899: 100}``). So this
    writer buffers incoming blocks and writes only whole records; the
    buffering in :meth:`_write_block` is load-bearing, not tidiness.

    The true sample count is not recoverable from the header at all --
    measured, 1010 samples written at 250 Hz come back as
    ``datarecords=5``, ``getNSamples()[0]=1250`` -- and the 240 padding
    samples are not even zero (``0.15259022`` uV at the default range,
    i.e. between two digital codes), so a reader cannot detect them by
    looking for zeros either. The true count is therefore written as one
    EDF+ annotation at close, which :class:`~gpype.EDFReader` trims to.
    """

    def __init__(
        self,
        file_name: str,
        physical_min: float = -10000.0,
        physical_max: float = 10000.0,
        **kwargs,
    ):
        """Initialize the EDF writer core.

        Args:
            file_name: Base filename for the .edf output. A timestamp
                will be automatically appended.
            physical_min: Physical minimum value in uV. Kept wide
                (-10000) by default: EDF clips silently outside this
                range (measured -- see :meth:`_write_block`), and the
                header is written in ``setup()`` before any sample is
                seen, so a streaming writer cannot auto-range. A caller
                who knows their signal's true range can pass a tighter
                one for better resolution; clipping, if it then happens,
                is reported at close.
            physical_max: Physical maximum value in uV. See
                ``physical_min``.
            **kwargs: Additional arguments passed to parent FileWriter.

        Raises:
            ValueError: If physical_max is not greater than physical_min.
        """
        super().__init__(file_name=file_name, **kwargs)
        if float(physical_max) <= float(physical_min):
            raise ValueError("physical_max must be greater than physical_min.")
        self._physical_min = float(physical_min)
        self._physical_max = float(physical_max)
        self._writer = None
        self._buffer = None
        self._samples_per_record = None
        self._channel_count = None
        self._written = 0
        self._clipped = 0

    @property
    def file_extension(self) -> str:
        """Return the file extension for EDF files.

        Returns:
            The EDF file extension '.edf'.
        """
        return ".edf"

    def _open_file(
        self, file_path: str, port_context_in: dict[str, dict]
    ) -> None:
        """Open the EDF file and write its per-signal headers.

        Args:
            file_path: Full path to the output .edf file.
            port_context_in: Context information from input ports.
        """
        pyedflib = _pyedflib()
        context = port_context_in.get(PORT_IN) or {}
        rate = float(self._sampling_rate)
        n_channels = int(channels.channel_count(context))
        labels = channels.labels_of(context)
        roles = channels.roles_of(context)

        self._writer = pyedflib.EdfWriter(
            file_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
        )

        truncated = [label for label in labels if len(label) > 16]
        signal_headers = [
            {
                "label": labels[i],
                "dimension": "uV",
                # NOT "sample_rate": edfwriter.py raises FutureWarning
                # ("Use of `sample_rate` is deprecated, use
                # `sample_frequency` instead") on that older key.
                "sample_frequency": rate,
                "physical_min": self._physical_min,
                "physical_max": self._physical_max,
                "digital_min": -32768,
                "digital_max": 32767,
                # The 80-char transducer field round-trips exactly
                # (measured) and is the natural per-signal home for a
                # role EDF itself has no concept of.
                "transducer": f"gpype:role={roles[i]}",
                "prefilter": "",
            }
            for i in range(n_channels)
        ]
        self._writer.setSignalHeaders(signal_headers)

        if truncated:
            self.log(
                f"channel label(s) {truncated} are longer than EDF's "
                f"16-character label field and have been truncated on "
                f"disk (measured: 'a_very_long_channel_label_over_16' "
                f"-> 'a_very_long_chan'). The 80-character transducer "
                f"field is unaffected.",
                type=Constants.LogTypes.WARNING,
            )

        serial = context.get(Constants.Keys.DEVICE_SERIAL)
        if serial:
            self._writer.setEquipment(str(serial))

        if self._marked:
            # Measured: with equipment also set, this 18-character mark
            # survives intact (recording_additional truncates at 39
            # chars unmarked, 23 once equipment is set) -- nothing else
            # may share this field.
            self._writer.setRecordingAdditional(MARK)

        # NOT int(rate): pyedflib's own record duration is not always
        # one second. Measured: fs=500/3 gives record_duration=3.0 and
        # 500 samples per record, while int(500/3) is 166 -- reading the
        # rate directly would buffer the wrong size and reintroduce the
        # interleaved-padding corruption this class exists to avoid.
        self._samples_per_record = int(
            round(rate * self._writer.record_duration)
        )
        self._channel_count = n_channels
        self._buffer = np.empty((0, n_channels), dtype=np.float64)
        self._written = 0
        self._clipped = 0

    def _write_block(self, block: np.ndarray, timestamps: np.ndarray) -> None:
        """Buffer a block and flush whole records as they fill.

        Timestamps are discarded: EDF carries no per-sample time axis,
        only a start time and a rate, so there is nowhere to put them.

        Args:
            block: Data block to write, shape (samples, channels).
            timestamps: Ignored; see above.
        """
        del timestamps
        if self._writer is None:
            return

        # pyedflib clips silently -- measured, a +/-500 uV ramp written
        # into a +/-250 uV range came back as +/-250 with no exception
        # and no warning from the library -- so out-of-range samples are
        # counted here, before writing, and reported once at close.
        out_of_range = block < self._physical_min
        out_of_range |= block > self._physical_max
        self._clipped += int(np.count_nonzero(out_of_range))

        self._buffer = np.vstack((self._buffer, block.astype(np.float64)))
        n = self._samples_per_record
        while self._buffer.shape[0] >= n:
            record = np.ascontiguousarray(self._buffer[:n])
            self._buffer = self._buffer[n:]
            self._writer.writeSamples(
                [
                    np.ascontiguousarray(record[:, ch])
                    for ch in range(self._channel_count)
                ]
            )
            self._written += n

    def _close_file(self) -> None:
        """Flush the remaining partial record, annotate, warn, close.

        The trailing partial record is handed to pyedflib as-is rather
        than hand-padded with zeros: pyedflib pads to a whole record
        itself (measured: 999 samples in, 1000 out; 1010 in, 1250 out),
        so padding it here first would be redundant.
        """
        if self._writer is None:
            return

        if self._buffer is not None and self._buffer.shape[0] > 0:
            remainder = self._buffer.shape[0]
            record = np.ascontiguousarray(self._buffer)
            self._writer.writeSamples(
                [
                    np.ascontiguousarray(record[:, ch])
                    for ch in range(self._channel_count)
                ]
            )
            self._written += remainder

        # The only way the true count survives: the header expresses
        # only n_records x samples_per_record, so 1010 written reads
        # back as 1250 without this. duration=-1 marks it as an
        # unbounded-duration EDF+ annotation rather than an interval.
        self._writer.writeAnnotation(0.0, -1, f"gpype:samples={self._written}")

        if self._clipped > 0:
            self.log(
                f"{self._clipped} sample value(s) fell outside the "
                f"configured physical range "
                f"[{self._physical_min}, {self._physical_max}] uV and "
                f"were clipped by pyedflib with no exception raised. "
                f"Pass a wider physical_min/physical_max if this is "
                f"unexpected.",
                type=Constants.LogTypes.WARNING,
            )

        self._writer.close()
        self._writer = None
        self._buffer = None
        self._samples_per_record = None
        self._channel_count = None


class EDFWriter(ioc.IChain):
    """EDF+ (.edf) file writer chain for real-time data logging.

    Buffers incoming blocks to whole EDF data records -- writing a
    partial record per pipeline block was measured to interleave
    fabricated padding between the real samples -- and records the true
    sample count as an EDF+ annotation, since EDF's own header cannot
    express a count that is not a multiple of the record size.

    Args:
        file_name: Base filename for the .edf output.
        physical_min: Physical minimum value in uV (default -10000).
        physical_max: Physical maximum value in uV (default +10000).
        **kwargs: Additional arguments.
    """

    def __init__(
        self,
        file_name: str,
        physical_min: float = -10000.0,
        physical_max: float = 10000.0,
        **kwargs,
    ):
        """Initialize the EDF writer chain.

        Args:
            file_name: Base filename for the .edf output. A timestamp
                will be automatically appended. Optional only so that a
                stored configuration can supply it; one of the two must
                be given.
            physical_min: Physical minimum value in uV.
            physical_max: Physical maximum value in uV.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If no file name is available from either source.
        """
        self._link_stream_id = stream_id_for(kwargs)
        fn_key = _EDFWriterCore.Configuration.Keys.FILE_NAME
        if file_name is None:
            file_name = kwargs.get(fn_key)
        if file_name is None:
            raise ValueError("file_name must be provided.")
        self._core_params = {
            "file_name": file_name,
            "physical_min": physical_min,
            "physical_max": physical_max,
        }
        self._core_params.update(strip_chain_keys(kwargs))

        kwargs.setdefault(fn_key, file_name)
        kwargs.setdefault("physical_min", physical_min)
        kwargs.setdefault("physical_max", physical_max)
        kwargs.setdefault(
            self.Configuration.Keys.INPUT_PORTS,
            [IPort.Configuration()],
        )
        ioc.IChain.__init__(
            self,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create the internal node chain.

        Returns:
            List containing [Link, _EDFWriterCore].
        """
        return [
            Link(
                sender=Constants.Residency.SERVER,
                receiver=Constants.Residency.EDGE,
                stream_id=self._link_stream_id,
            ),
            _EDFWriterCore(**self._core_params),
        ]
