from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from .base import recording_meta
from .base.file_writer import FileWriter

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN

#: What EDF+ stores an annotation's onset and duration to, in seconds:
#: pyedflib rounds both to units of 100 microseconds.
ANNOTATION_RESOLUTION = 1e-4

#: EDF's 16-bit sample range. A trigger channel is written with this as
#: its physical range as well, so one physical unit is one digital step
#: and an integer code reads back as the same integer. Scaled into the
#: +/-10000 signal range, code 1 came back as 0.763 and the code 0
#: between markers as 0.153, and every one became a marker of its own
#: (D-BATCH-107).
DIGITAL_MIN = -32768
DIGITAL_MAX = 32767

#: How much of an annotation's text EDF+ keeps, in bytes of UTF-8.
#: Measured with pyedflib 0.1.42: a 41-byte label read back as its first
#: 40 bytes.
ANNOTATION_TEXT_BYTES = 40


def _annotation_text(label: str) -> str:
    """Cut *label* to what an annotation keeps, at a character boundary.

    pyedflib cuts at the byte. Cut inside a character, the text does not
    decode: 39 ASCII characters and an 'Ä' read back through pyedflib's
    latin1 fallback, with a UserWarning (measured).

    Args:
        label: The marker's label.

    Returns:
        The label, or its longest prefix that fits.
    """
    stored = label.encode("utf-8")[:ANNOTATION_TEXT_BYTES]
    return stored.decode("utf-8", "ignore")


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
    except ImportError as error:
        from ... import _missing_extra_hint

        hint = _missing_extra_hint(error) or str(error)
        raise ImportError(f"EDFWriter needs pyedflib. {hint}") from error
    return pyedflib


class EDFWriter(FileWriter):
    """EDF+ (.edf) file writer for real-time data logging.

    Buffers incoming blocks to whole EDF data records -- writing a
    partial record per pipeline block was measured to interleave
    fabricated padding between the real samples -- and records the true
    sample count as an EDF+ annotation, since EDF's own header cannot
    express a count that is not a multiple of the record size.
    :class:`~gpype.EDFReader` trims to it.

    The stream's markers are written as EDF+ annotations, onset and
    duration in seconds. The annotation channel holds one annotation per
    data record, and the sample count takes the first, so a file of *n*
    records stores *n* - 1 markers. EDF+ annotations name no channel. A
    marker that does not fit, or that names a channel, is not written,
    and the writer logs how many at close. An annotation keeps 40 bytes
    of UTF-8 text, so a longer label is cut at a character boundary, and
    the writer names it. A marker past the last sample written is left
    out.

    A trigger-role channel is written unscaled, with EDF's digital range
    as its physical range, so an integer code in that range survives
    exactly. ``physical_min`` and ``physical_max`` apply to the others.
    """

    def __init__(
        self,
        file_name: str,
        physical_min: float = -10000.0,
        physical_max: float = 10000.0,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the EDF writer core.

        Args:
            file_name: Base filename for the .edf output. A timestamp
                will be automatically appended.
            physical_min: Physical minimum value in uV, for every
                channel but a trigger (see above). Kept wide
                (-10000) by default: EDF clips silently outside this
                range (measured -- see :meth:`_write_block`), and the
                header is written in ``setup()`` before any sample is
                seen, so a streaming writer cannot auto-range. A caller
                who knows their signal's true range can pass a tighter
                one for better resolution; clipping, if it then happens,
                is reported at close.
            physical_max: Physical maximum value in uV. See
                ``physical_min``.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments passed to parent FileWriter.

        Raises:
            ValueError: If physical_max is not greater than physical_min.
        """
        if float(physical_max) <= float(physical_min):
            raise ValueError("physical_max must be greater than physical_min.")
        super().__init__(
            file_name=file_name,
            physical_min=float(physical_min),
            physical_max=float(physical_max),
            edge_id=edge_id,
            **kwargs,
        )
        self._physical_min = float(physical_min)
        self._physical_max = float(physical_max)
        self._writer = None
        self._buffer = None
        self._samples_per_record = None
        self._channel_count = None
        self._written = 0
        self._clipped = 0
        self._markers = []
        self._channel_min = None
        self._channel_max = None

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
        units = channels.units_of(context)

        self._writer = pyedflib.EdfWriter(
            file_path, n_channels, file_type=pyedflib.FILETYPE_EDFPLUS
        )

        truncated = [label for label in labels if len(label) > 16]
        trigger = [role == Constants.ChannelRoles.TRIGGER for role in roles]
        self._channel_min = np.array(
            [
                DIGITAL_MIN if trigger[i] else self._physical_min
                for i in range(n_channels)
            ],
            dtype=np.float64,
        )
        self._channel_max = np.array(
            [
                DIGITAL_MAX if trigger[i] else self._physical_max
                for i in range(n_channels)
            ],
            dtype=np.float64,
        )
        signal_headers = [
            {
                "label": labels[i],
                # From the context, per channel: empty where the
                # channel's unit is not recorded, e.g. a trigger, rather
                # than the 'uV' every channel used to get regardless
                # (D-BATCH-43).
                "dimension": (units[i] or "") if units is not None else "",
                # NOT "sample_rate": edfwriter.py raises FutureWarning
                # ("Use of `sample_rate` is deprecated, use
                # `sample_frequency` instead") on that older key.
                "sample_frequency": rate,
                "physical_min": float(self._channel_min[i]),
                "physical_max": float(self._channel_max[i]),
                "digital_min": DIGITAL_MIN,
                "digital_max": DIGITAL_MAX,
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
        # On the grid of the file's first sample, which is the stream's.
        self._markers = recording_meta.entries(
            context.get(Constants.Keys.MARKERS), 4
        )

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
        out_of_range = block < self._channel_min
        out_of_range |= block > self._channel_max
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
        # Written first: pyedflib keeps annotations in the order given
        # and drops those past the channel's capacity (measured).
        self._writer.writeAnnotation(0.0, -1, f"gpype:samples={self._written}")
        self._write_markers()

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
        self._markers = []

    def _write_markers(self) -> None:
        """Write the markers that fit as annotations, and log the rest.

        The annotation channel holds one annotation per data record
        (measured: 65 written into a 10-record file, the first 10 read
        back), and the sample count takes the first. A marker past the
        last sample written is about no sample of this file, and is left
        out, as ``recording_meta.finalize`` leaves it out of ``.mat`` and
        ``.h5``.
        """
        markers = [m for m in self._markers if int(m[0]) < self._written]
        if not markers:
            return
        rate = float(self._sampling_rate)
        n = self._samples_per_record
        records = -(-self._written // n) if n else 0
        room = max(records - 1, 0)

        scoped = [m for m in markers if m[2] is not None]
        whole = [m for m in markers if m[2] is None]
        kept = whole[:room]
        cut = {}
        for sample, duration, _, label in kept:
            text = str(label)
            stored = _annotation_text(text)
            if stored != text:
                cut.setdefault(text, stored)
            self._writer.writeAnnotation(
                int(sample) / rate,
                int(duration) / rate if int(duration) > 0 else -1,
                stored,
            )

        if cut:
            shown = ", ".join(
                f"{text!r} -> {stored!r}"
                for text, stored in list(cut.items())[:3]
            )
            more = f", and {len(cut) - 3} more" if len(cut) > 3 else ""
            self.log(
                f"{len(cut)} marker label(s) are longer than EDF+'s "
                f"{ANNOTATION_TEXT_BYTES}-byte annotation text and were "
                f"cut on disk: {shown}{more}.",
                type=Constants.LogTypes.WARNING,
            )

        lost = len(whole) - len(kept)
        if lost or scoped:
            reasons = []
            if lost:
                reasons.append(
                    f"{lost} did not fit: EDF+ keeps one annotation per "
                    f"data record, this file has {records}, and the "
                    f"sample count takes one"
                )
            if scoped:
                reasons.append(
                    f"{len(scoped)} name a channel, which an EDF+ "
                    f"annotation cannot"
                )
            self.log(
                f"{lost + len(scoped)} of {len(markers)} marker(s) "
                f"were not written to the file: {'; '.join(reasons)}.",
                type=Constants.LogTypes.WARNING,
            )
        # A marker comes back on its own sample while the rounding of
        # its onset stays under half a sample. At 10 kHz a sample is one
        # 100 us step, so nothing rounds; above it, a marker can move.
        if kept and rate * ANNOTATION_RESOLUTION > 1:
            self.log(
                f"EDF+ stores a marker's onset to "
                f"{ANNOTATION_RESOLUTION * 1e6:.0f} us, so at {rate:g} Hz "
                f"a marker can read back up to "
                f"{rate * ANNOTATION_RESOLUTION / 2:.1f} samples away.",
                type=Constants.LogTypes.WARNING,
            )
