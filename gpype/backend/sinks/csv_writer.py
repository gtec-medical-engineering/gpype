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


class _CsvWriterCore(FileWriter):
    """Internal node implementing CSV file writing logic.

    This is the actual CSV writer node (pure INode inheritance via FileWriter).
    It is wrapped by the CsvWriter chain for distributed operation.

    Writes multi-channel data to CSV files with timestamps in the first
    column. Automatically generates channel headers (Time, Ch01, Ch02, etc.).
    """

    def __init__(self, file_name: str, **kwargs):
        """Initialize the CSV writer core.

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


class CsvWriter(ioc.IChain):
    """CSV file writer chain for real-time data logging.

    This is an IChain that contains:
    - Link: Bridge for distributed operation (passthrough in standalone)
    - _CsvWriterCore: The actual CSV writing node

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Writes multi-channel data to CSV files with timestamps in the first
    column. Automatically generates channel headers (Time, Ch01, Ch02, etc.).
    """

    def __init__(self, file_name: str, **kwargs):
        """Initialize the CSV writer chain.

        Args:
            file_name: Base filename for CSV output. Must have .csv extension.
                A timestamp will be automatically appended. Optional only so
                that a stored configuration can supply it; one of the two
                must be given.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If no file name is available from either source.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        fn_key = _CsvWriterCore.Configuration.Keys.FILE_NAME
        if file_name is None:
            file_name = kwargs.get(fn_key)
        if file_name is None:
            raise ValueError("file_name must be provided.")
        self._core_params = {"file_name": file_name}
        self._core_params.update(strip_chain_keys(kwargs))

        # Initialize IChain (calls create_internal_nodes). The file name
        # goes into the configuration as well: a chain that does not record
        # it cannot be rebuilt from what it stored.
        kwargs.setdefault(fn_key, file_name)
        kwargs.setdefault(
            self.Configuration.Keys.INPUT_PORTS,
            [IPort.Configuration()],
        )
        # file_name is already forwarded above via kwargs.setdefault;
        # only stream_id needs adding here. See Generator for the
        # rationale behind forwarding a chain's own parameters.
        ioc.IChain.__init__(
            self,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create the internal node chain.

        Returns:
            List containing [Link, _CsvWriterCore].
        """
        return [
            Link(
                sender=Constants.Residency.SERVER,
                receiver=Constants.Residency.EDGE,
                stream_id=self._link_stream_id,
            ),
            _CsvWriterCore(**self._core_params),
        ]
