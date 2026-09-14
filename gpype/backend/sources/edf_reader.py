from __future__ import annotations

from typing import List, Optional

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
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
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "EDFReader needs pyedflib, provided by the 'formats' extra: "
            "pip install 'gpype[formats]'"
        ) from error
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

        extras: dict = {}
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

        true_count = None
        onsets, durations, descriptions = reader.readAnnotations()
        del onsets, durations
        for description in descriptions:
            text = str(description)
            if text.startswith(_SAMPLES_PREFIX):
                true_count = int(text[len(_SAMPLES_PREFIX) :])
                break

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


class _EDFReaderCore(RecordingReader):
    """Internal node replaying an EDF+ (.edf) recording."""

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
            file_name: Path to the .edf recording.
            sampling_rate: Rate to replay at, if the file cannot say
                (EDF always can, so this is only a safety net).
            frame_size: Samples emitted per cycle. Ignored in batch mode.
            speed: Replay speed relative to real time.
            loop: Start again at the end of the file.
            mode: ``realtime`` or ``batch``.
            **kwargs: Additional arguments for the parent reader.

        Raises:
            ImportError: If pyedflib is not installed.
            ValueError: If signals disagree on their sample frequency.
        """
        self._padding_undetected = False
        super().__init__(
            file_name=file_name,
            sampling_rate=sampling_rate,
            frame_size=frame_size,
            speed=speed,
            loop=loop,
            mode=mode,
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


class EDFReader(ioc.OChain):
    """Replays an EDF+ (.edf) recording as if it were a live source.

    Reads back what :class:`~gpype.EDFWriter` writes: channel roles from
    the per-signal transducer field, the device serial from the
    equipment field, the entitlement mark from the recording-additional
    field, and the true sample count -- not recoverable from the EDF
    header itself -- from a ``gpype:samples`` EDF+ annotation. A file
    without that annotation (a foreign EDF, or one from an earlier
    draft) is still read, in full, with a logged warning that its tail
    may carry up to one data record of undetectable padding.
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
            file_name: Path to the .edf recording.
            sampling_rate: Rate to replay at, if the file cannot say.
            frame_size: Samples emitted per cycle. One by default.
                Ignored in batch mode.
            speed: Replay speed relative to real time; zero for as fast
                as possible.
            loop: Start again at the end of the file.
            mode: ``realtime`` (default) or ``batch``.
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
            raw.source_stage(self, lambda: _EDFReaderCore(**self._core_params))
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
