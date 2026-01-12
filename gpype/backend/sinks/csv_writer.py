from __future__ import annotations

import numpy as np

from .base.file_writer import FileWriter


class CsvWriter(FileWriter):
    """CSV file writer for real-time data logging.

    Writes multi-channel data to CSV files with timestamps in the first
    column. Automatically generates channel headers (Time, Ch01, Ch02, etc.).
    """

    def __init__(self, file_name: str, **kwargs):
        """Initialize the CSV writer.

        Args:
            file_name: Base filename for CSV output. Must have .csv extension.
                A timestamp will be automatically appended.
            **kwargs: Additional arguments passed to parent FileWriter class.
        """
        super().__init__(file_name=file_name, **kwargs)
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
            header = "Time, "
            ch_names = [f"Ch{d + 1:02d}" for d in range(block.shape[1])]
            header += ", ".join(ch_names)
            self._header_written = True
        else:
            header = ""

        # Combine timestamps with data (first column)
        full_block = np.column_stack((timestamps, block))

        # Write to CSV with reasonable precision formatting
        np.savetxt(
            self._file_handle,
            full_block,
            fmt=["%g", *(["%.17g"] * block.shape[1])],
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
