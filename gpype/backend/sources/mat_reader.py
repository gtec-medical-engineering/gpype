from __future__ import annotations

from typing import Optional

from ...common.constants import Constants
from .base.recording_reader import Recording, RecordingReader, read_h5

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class MatReader(RecordingReader):
    """Replays a MATLAB v7.3 (.mat) recording as if it were a live source.

    Reads back what :class:`~gpype.MatWriter` writes -- sampling rate,
    channel labels and roles from the ``gpype_meta`` document, falling
    back to the recorded time column when metadata is absent (a file
    written by an earlier draft, or a third-party MAT file), and to the
    caller's own ``sampling_rate`` when neither is available. Markers,
    gaps and trust come back from ``gpype_meta`` too.
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
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the reader core.

        Args:
            file_name: Path to the .mat recording.
            sampling_rate: The recording's rate, if the file cannot say;
                one given must agree with a rate it records.
            frame_size: Samples emitted per cycle. In batch mode the
                frame is the whole recording.
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
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments for the parent reader.

        Raises:
            ImportError: If h5py is not installed.
            ValueError: If no usable rate or dataset can be found, or
                a ``sampling_rate``, ``channel_count`` or batch
                ``frame_size`` given by hand disagrees with the file.
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
            edge_id=edge_id,
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
