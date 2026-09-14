from __future__ import annotations

from typing import List, Optional

import ioiocore as ioc

from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.recording_reader import Recording, RecordingReader, read_h5

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class _MatReaderCore(RecordingReader):
    """Internal node replaying a MAT-file (.mat) recording."""

    def __init__(
        self,
        file_name: str,
        sampling_rate: Optional[float] = None,
        frame_size: int = 1,
        speed: float = 1.0,
        loop: bool = False,
        mode: str = Constants.ExecutionMode.REALTIME,
        variable_name: str = "data",
        has_time_row: bool = True,
        **kwargs,
    ):
        """Initialize the reader core.

        Args:
            file_name: Path to the .mat recording.
            sampling_rate: Rate to replay at, if the file cannot say.
            frame_size: Samples emitted per cycle. Ignored in batch mode.
            speed: Replay speed relative to real time.
            loop: Start again at the end of the file.
            mode: ``realtime`` or ``batch``.
            variable_name: Name of the dataset :class:`~gpype.MatWriter`
                wrote the recording under.
            has_time_row: Whether the dataset's first row is a time axis
                rather than a signal channel. Explicit rather than
                sniffed, so a real signal channel can never be mistaken
                for one -- ``/gpype_meta``, when present, overrides it
                with what the writer actually did.
            **kwargs: Additional arguments for the parent reader.

        Raises:
            ImportError: If h5py is not installed.
            ValueError: If no usable rate or dataset can be found.
        """
        self._variable_name = variable_name
        self._has_time_row = bool(has_time_row)
        super().__init__(
            file_name=file_name,
            sampling_rate=sampling_rate,
            frame_size=frame_size,
            speed=speed,
            loop=loop,
            mode=mode,
            variable_name=variable_name,
            has_time_row=has_time_row,
            **kwargs,
        )

    def _load(self, file_name: str) -> Recording:
        """Read the recording via the shared MAT/HDF5 body.

        Args:
            file_name: Path to the .mat recording.

        Returns:
            The parsed recording.
        """
        return read_h5(file_name, self._variable_name, self._has_time_row)


class MatReader(ioc.OChain):
    """Replays a MATLAB v7.3 (.mat) recording as if it were a live source.

    Reads back what :class:`~gpype.MatWriter` writes -- sampling rate,
    channel labels and roles from the ``gpype_meta`` document, falling
    back to the recorded time column when metadata is absent (a file
    written by an earlier draft, or a third-party MAT file), and to the
    caller's own ``sampling_rate`` when neither is available.
    """

    def __init__(
        self,
        file_name: str,
        sampling_rate: Optional[float] = None,
        frame_size: int = 1,
        speed: float = 1.0,
        loop: bool = False,
        mode: str = Constants.ExecutionMode.REALTIME,
        variable_name: str = "data",
        has_time_row: bool = True,
        **kwargs,
    ):
        """Initialize the reader chain.

        Args:
            file_name: Path to the .mat recording.
            sampling_rate: Rate to replay at, if the file cannot say.
            frame_size: Samples emitted per cycle. One by default.
                Ignored in batch mode.
            speed: Replay speed relative to real time; zero for as fast
                as possible.
            loop: Start again at the end of the file.
            mode: ``realtime`` (default) or ``batch``.
            variable_name: Name of the dataset the recording was written
                under.
            has_time_row: Whether the dataset's first row is a time axis.
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
            "variable_name": variable_name,
            "has_time_row": has_time_row,
        }
        self._core_params.update(strip_chain_keys(kwargs))

        kwargs.setdefault("file_name", file_name)
        kwargs.setdefault("variable_name", variable_name)
        kwargs.setdefault("has_time_row", has_time_row)
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
            raw.source_stage(self, lambda: _MatReaderCore(**self._core_params))
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
