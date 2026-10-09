from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from .base.recording_reader import Recording, RecordingReader

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: Prefix EDFWriter puts in front of a channel's role in its transducer
#: field. Shared with edf_writer.py only by convention -- the two modules
#: do not import each other -- since the whole point of writing it into
#: an ordinary EDF field is that any EDF tool, not only this reader,
#: can see it.
_ROLE_PREFIX = "gpype:role="

#: Prefix EDFWriter puts in front of the true sample count in its one
#: EDF+ annotation.
_SAMPLES_PREFIX = "gpype:samples="


def _pyedflib():
    """Import pyedflib, or say what installs it.

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
        raise ImportError(f"EDFReader needs pyedflib. {hint}") from error
    return pyedflib


def _read_edf(file_name: str) -> tuple:
    """Read an EDF/EDF+ file's signals and provenance.

    Args:
        file_name: Path to the recording.

    Returns:
        ``(recording, padding_undetected)``. The second element is True
        when the file carries no ``gpype:samples`` annotation, so the
        caller can log the warning once the node is far enough along in
        construction for ``self.log`` to work -- this function runs
        before that point.

    Raises:
        ImportError: If pyedflib is not installed.
        ValueError: If signals disagree on their sample frequency, or a
            transducer field names a role g.Pype does not recognise.
    """
    pyedflib = _pyedflib()
    reader = pyedflib.EdfReader(file_name)
    try:
        # signals_in_file already excludes the EDF+ annotation channel --
        # measured, a 2-channel file with 5 annotations reports 2
        # signals and 2 labels -- so no filtering is needed here.
        n = reader.signals_in_file
        rates = [float(reader.getSampleFrequency(i)) for i in range(n)]
        labels = list(reader.getSignalLabels())

        if n and any(r != rates[0] for r in rates):
            offending = [
                (labels[i], rates[i]) for i in range(n) if rates[i] != rates[0]
            ]
            raise ValueError(
                f"'{file_name}' has signals at different sample "
                f"frequencies ({offending} vs {rates[0]} Hz for the "
                f"rest); EDF permits this but a g.Pype port is "
                f"single-rate, and silently resampling or taking one "
                f"signal's rate would misplace every sample of the "
                f"others in time."
            )
        rate = rates[0] if rates else None

        roles = []
        for i in range(n):
            transducer = reader.getTransducer(i) or ""
            if transducer.startswith(_ROLE_PREFIX):
                candidate = transducer[len(_ROLE_PREFIX) :]
                if candidate not in channels.VALID_ROLES:
                    raise ValueError(
                        f"'{file_name}' channel {i} ({labels[i]!r}) "
                        f"declares transducer role {candidate!r}, which "
                        f"is not one g.Pype recognises. Valid roles are "
                        f"{sorted(channels.VALID_ROLES)}."
                    )
                roles.append(candidate)
            else:
                # A foreign file's transducer field means something
                # else entirely; read as signal, the format's own
                # default reading for a data channel.
                roles.append(Constants.ChannelRoles.SIGNAL)

        # The physical dimension EDFWriter writes from the context
        # (D-BATCH-43); empty where it wrote none, e.g. a trigger. A
        # foreign file with every dimension empty gives nothing to
        # publish -- an all-None list would claim it recorded units it
        # did not.
        dimensions = [reader.getPhysicalDimension(i) or None for i in range(n)]

        extras: dict = {}
        if any(dimensions):
            extras[Constants.Keys.CHANNEL_UNITS] = dimensions
        serial = reader.getEquipment()
        if serial:
            extras[Constants.Keys.DEVICE_SERIAL] = str(serial)
        additional = reader.getRecordingAdditional()
        if additional and MARK in additional:
            extras["mark"] = MARK

        raw = (
            np.stack([reader.readSignal(i) for i in range(n)], axis=1)
            if n
            else np.empty((0, 0))
        )

        true_count, markers = _annotations(reader, rate)
        if markers:
            extras[Constants.Keys.MARKERS] = markers

        padding_undetected = true_count is None
        if true_count is not None:
            raw = raw[:true_count]

        samples = np.asarray(raw, dtype=Constants.DATA_TYPE)
        return (
            Recording(
                samples=samples,
                labels=labels or None,
                roles=roles or None,
                sampling_rate=rate,
                extras=extras,
            ),
            padding_undetected,
        )
    finally:
        reader.close()


def _annotations(reader, rate: Optional[float]) -> tuple:
    """Split a file's EDF+ annotations into the sample count and markers.

    Args:
        reader: The open pyedflib reader.
        rate: The file's sampling rate, or None without signals.

    Returns:
        ``(true_count, markers)``. The count is None when no
        ``gpype:samples`` annotation is present. A marker is ``[sample,
        duration, None, label]``, with onset and duration moved from
        seconds onto the file's grid; an annotation without a duration
        is a point. EDF+ annotations name no channel. Without a rate
        there is no grid, and no markers.
    """
    true_count = None
    markers = []
    onsets, durations, descriptions = reader.readAnnotations()
    for onset, duration, description in zip(onsets, durations, descriptions):
        text = str(description)
        if text.startswith(_SAMPLES_PREFIX):
            if true_count is None:
                true_count = int(text[len(_SAMPLES_PREFIX) :])
            continue
        if not rate:
            continue
        # pyedflib reads a missing duration as -1.
        length = 0 if duration < 0 else int(round(float(duration) * rate))
        markers.append([int(round(float(onset) * rate)), length, None, text])
    return true_count, markers


class EDFReader(RecordingReader):
    """Replays an EDF+ (.edf) recording as if it were a live source.

    Reads back what :class:`~gpype.EDFWriter` writes: channel roles from
    the per-signal transducer field, each channel's unit from its
    physical dimension (None where that field is empty, e.g. a
    trigger), the device serial from the equipment field, the
    entitlement mark from the recording-additional field, and the true
    sample count -- not recoverable from the EDF header itself -- from a
    ``gpype:samples`` EDF+ annotation. A file without that annotation (a
    foreign EDF, or one from an earlier draft) is still read, in full,
    with a logged warning that its tail may carry up to one data record
    of undetectable padding.

    Every other EDF+ annotation is a marker: its onset and duration, in
    seconds, are moved onto the file's sample grid, and it names no
    channel.
    """

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
            file_name: Path to the .edf recording.
            sampling_rate: The recording's rate, if the file cannot say.
                EDF always can, so one given must agree with it.
            frame_size: Samples emitted per cycle. In batch mode the
                frame is the whole recording.
            speed: Replay speed relative to real time.
            loop: Start again at the end of the file.
            mode: ``realtime`` or ``batch``.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments for the parent reader.

        Raises:
            ImportError: If pyedflib is not installed.
            ValueError: If signals disagree on their sample frequency,
                or a ``sampling_rate``, ``channel_count`` or batch
                ``frame_size`` given by hand disagrees with the file.
        """
        self._padding_undetected = False
        super().__init__(
            file_name=file_name,
            sampling_rate=sampling_rate,
            frame_size=frame_size,
            speed=speed,
            loop=loop,
            mode=mode,
            edge_id=edge_id,
            **kwargs,
        )

    def _load(self, file_name: str) -> Recording:
        """Read the EDF file.

        Args:
            file_name: Path to the .edf recording.

        Returns:
            The parsed recording.
        """
        recording, undetected = _read_edf(file_name)
        # Recorded rather than logged here: `_load` runs from inside
        # `__init__`, before `Node.__init__` has created the
        # implementation `self.log` needs, so the warning is deferred to
        # `setup()`, which always runs afterwards.
        self._padding_undetected = undetected
        return recording

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Describe the replayed channels, and warn about undetectable
        padding when this file carries no ``gpype:samples`` annotation.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.
        """
        port_context_out = super().setup(data, port_context_in)
        if self._padding_undetected:
            self.log(
                f"'{self.config['file_name']}' carries no "
                f"'gpype:samples' annotation, so the "
                f"{self.sample_count} samples on disk may include up to "
                f"one EDF data record of padding this reader cannot "
                f"tell from real data -- measured, the padding is not "
                f"even zero (it reads back as 0.15259022 uV at the "
                f"+/-10000 uV default), so it cannot be detected by "
                f"looking for zeros. Trimming was skipped rather than "
                f"guessed.",
                type=Constants.LogTypes.WARNING,
            )
        return port_context_out
