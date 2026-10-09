from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from .base.file_writer import FileWriter


class CsvWriter(FileWriter):
    """CSV file writer for real-time data logging.

    Writes multi-channel data to CSV files with timestamps in the first
    column. Automatically generates channel headers (Time, Ch01, Ch02, etc.).
    """

    def __init__(
        self, file_name: str, edge_id: Optional[str] = None, **kwargs
    ):
        """Initialize the CSV writer core.

        Args:
            file_name: Base filename for CSV output. Must have .csv extension.
                A timestamp will be automatically appended.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments passed to parent FileWriter class.
        """
        super().__init__(file_name=file_name, edge_id=edge_id, **kwargs)
        self._file_handle = None
        self._header_written = False

    @property
    def file_extension(self) -> str:
        """Return the file extension for CSV files.

        Returns:
            The CSV file extension '.csv'.
        """
        return ".csv"

    def _open_file(
        self, file_path: str, port_context_in: dict[str, dict]
    ) -> None:
        """Open the CSV file for writing.

        Args:
            file_path: Full path to the output CSV file.
            port_context_in: Context information from input ports.

        Raises:
            IOError: If the file cannot be created or opened.
        """
        self._file_handle = open(file_path, "w")
        self._header_written = False

        # Prefer the channel labels the source supplied. Fall back to
        # positional names so existing files keep their headers.
        self._channel_names = None
        context = port_context_in.get(Constants.Defaults.PORT_IN)
        if context is not None and channels.has_labels(context):
            self._channel_names = channels.labels_of(context)

        # Which physical amplifier produced this recording. Worth having
        # in the file rather than only in a log: a configuration records
        # which device was asked for, which for the usual serial=None is
        # nothing, and on analysis day the question is which device the
        # data actually came from.
        self._device_serial = ""
        if context is not None:
            self._device_serial = str(
                context.get(Constants.Keys.DEVICE_SERIAL, "") or ""
            )

    def _write_block(self, block: np.ndarray, timestamps: np.ndarray) -> None:
        """Write a data block to the CSV file.

        Generates CSV header on first write with Time and channel columns.
        Writes data with timestamps in the first column.

        Args:
            block: Data block to write, shape (samples, channels).
            timestamps: Timestamp array for each sample in the block.
        """
        if self._file_handle is None:
            return

        # Generate header only for first block
        if not self._header_written:
            # The mark precedes the column names, as a comment line, so a
            # reader that treats the first line as a header still finds
            # one, and a human opening the file sees the mark first.
            header = f"# {MARK}\n" if self._marked else ""
            serial = getattr(self, "_device_serial", "")
            if serial:
                header += f"# device: {serial}\n"
            header += "Time, "
            names = getattr(self, "_channel_names", None)
            if names is not None and len(names) == block.shape[1]:
                ch_names = list(names)
            else:
                ch_names = [f"Ch{d + 1:02d}" for d in range(block.shape[1])]
            header += ", ".join(ch_names)
            self._header_written = True
        else:
            header = ""

        # Combine timestamps with data (first column)
        full_block = np.column_stack((timestamps, block))

        # The time column needs as much care as the data. It was written
        # with bare "%g" -- six significant digits -- which is exact only
        # while a timestamp is short: past about 17 minutes at 250 Hz the
        # step between rows exceeds a whole sample period, and a 256 Hz
        # recording read back as 255.95 Hz. "%.15g" keeps every float64
        # timestamp a recording can reach, and unlike "%.17g" it does not
        # print the binary noise that turns 0.004 into
        # 0.0040000000000000001.
        np.savetxt(
            self._file_handle,
            full_block,
            fmt=["%.15g", *(["%.17g"] * block.shape[1])],
            delimiter=",",
            header=header,
            comments="",
        )

    def _close_file(self) -> None:
        """Close the CSV file.

        Properly closes the file handle and resets internal state.
        """
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
        self._header_written = False
