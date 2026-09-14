from __future__ import annotations

import json
from typing import List

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core.i_port import IPort
from .base import recording_meta
from .base.file_writer import FileWriter


def _h5py():
    """Import h5py, or say what installs it.

    See :func:`gpype.backend.sinks.mat_writer._h5py` for why this is
    deferred to construction time rather than a module-level import.

    Returns:
        The h5py module.

    Raises:
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "HDF5Writer needs h5py, provided by the 'formats' extra: "
            "pip install 'gpype[formats]'"
        ) from error
    return h5py


class _HDF5WriterCore(FileWriter):
    """Internal node writing a recording as a plain HDF5 (.h5) file.

    Unlike :class:`~gpype.backend.sinks.mat_writer._MatWriterCore`, this
    is an ordinary HDF5 container with no userblock and no MATLAB
    conventions -- for a consumer that wants HDF5 without MATLAB's
    baggage, or a language whose HDF5 bindings do not speak MATLAB's
    dialect.

    The untracked draft this replaces ignored ``port_context_in``
    entirely, so a file it produced carried no rate, no labels and no
    roles -- a reader had nothing to go on but a bare array. This writer
    records the same ``gpype_meta`` JSON document
    :class:`~gpype.backend.sinks.mat_writer._MatWriterCore` writes (so
    one parser in :mod:`recording_meta` serves both readers) and, in
    addition, plain HDF5 attributes on the data dataset -- sampling
    rate, channel labels, channel roles, sample count -- so a tool with
    no g.Pype involvement at all can still make sense of the file with
    an ordinary HDF5 library.
    """

    def __init__(self, file_name: str, variable_name: str = "data", **kwargs):
        """Initialize the HDF5 writer core.

        Args:
            file_name: Base filename for the .h5 output. A timestamp
                will be automatically appended.
            variable_name: Name of the dataset the recording is written
                under.
            **kwargs: Additional arguments passed to parent FileWriter.
        """
        super().__init__(file_name=file_name, **kwargs)
        self._variable_name = variable_name
        self._file = None
        self._data_ds = None
        self._total_samples = 0
        self._meta = None
        self._h5py_module = None

    @property
    def file_extension(self) -> str:
        """Return the file extension for HDF5 files.

        A single extension only: ``FileWriter._check_extension`` compares
        against one string, and widening that base contract to accept
        both ``.h5`` and ``.hdf5`` is out of scope here.

        Returns:
            The HDF5 file extension '.h5'.
        """
        return ".h5"

    def _open_file(
        self, file_path: str, port_context_in: dict[str, dict]
    ) -> None:
        """Create the file and record what describes the recording.

        Args:
            file_path: Full path to the output .h5 file.
            port_context_in: Context information from input ports.
        """
        self._h5py_module = _h5py()
        self._file = self._h5py_module.File(file_path, "w")

        self._meta = recording_meta.collect(
            port_context_in, self._marked, self._sampling_rate
        )
        self._write_gpype_meta(self._meta)

        self._data_ds = None
        self._total_samples = 0

    def _write_gpype_meta(self, meta: dict) -> None:
        """(Re)write the ``/gpype_meta`` dataset.

        An ordinary HDF5 variable-length string, unlike the MAT writer's
        uint16 encoding -- this file carries no MATLAB userblock, so
        there is no MATLAB char-array convention to satisfy.

        Args:
            meta: The document to write.
        """
        if recording_meta.META_DATASET in self._file:
            del self._file[recording_meta.META_DATASET]
        self._file.create_dataset(
            recording_meta.META_DATASET,
            data=json.dumps(meta),
            dtype=self._h5py_module.string_dtype(),
        )

    def _apply_native_attrs(self, n_channels: int) -> None:
        """Set the plain-HDF5 attributes a non-g.Pype reader can use.

        Called once, when the data dataset is created -- attributes
        cannot be set before the dataset they describe exists.

        Args:
            n_channels: Channel count, used when the context carried
                none (so the file still says how many channels it has).
        """
        ds = self._data_ds
        ds.attrs["sampling_rate"] = float(self._sampling_rate)
        ds.attrs["channel_count"] = int(
            self._meta.get("channel_count", n_channels)
        )
        string_dtype = self._h5py_module.string_dtype()
        if "channel_labels" in self._meta:
            ds.attrs["channel_labels"] = np.array(
                self._meta["channel_labels"], dtype=string_dtype
            )
        if "channel_roles" in self._meta:
            ds.attrs["channel_roles"] = np.array(
                self._meta["channel_roles"], dtype=string_dtype
            )
        if "device_serial" in self._meta:
            ds.attrs["device_serial"] = self._meta["device_serial"]
        if "mark" in self._meta:
            ds.attrs["mark"] = self._meta["mark"]

    def _write_block(self, block: np.ndarray, timestamps: np.ndarray) -> None:
        """Append a data block, growing the dataset along the sample axis.

        Args:
            block: Data block to write, shape (samples, channels).
            timestamps: Timestamp array for each sample in the block.
        """
        if self._file is None:
            return

        n_samples, n_channels = block.shape
        # dtype is explicit float64 rather than combined.dtype, which is
        # what the untracked draft used: block arrives as
        # Constants.DATA_TYPE (float32), and letting the dataset dtype
        # follow the first block silently halved stored precision.
        combined = np.hstack(
            (
                timestamps.reshape(-1, 1).astype(np.float64),
                block.astype(np.float64),
            )
        ).T

        if self._data_ds is None:
            self._data_ds = self._file.create_dataset(
                self._variable_name,
                shape=(n_channels + 1, 0),
                maxshape=(n_channels + 1, None),
                dtype=np.float64,
                chunks=(n_channels + 1, max(1, n_samples)),
            )
            self._apply_native_attrs(n_channels)

        new_size = self._total_samples + n_samples
        self._data_ds.resize((n_channels + 1, new_size))
        self._data_ds[:, self._total_samples : new_size] = combined
        self._total_samples = new_size

        self._file.flush()

    def _close_file(self) -> None:
        """Record the final sample count in both places and close."""
        if self._file is not None:
            if self._meta is not None:
                final_meta = recording_meta.finalize(
                    self._meta, self._total_samples
                )
                self._write_gpype_meta(final_meta)
            if self._data_ds is not None:
                self._data_ds.attrs["sample_count"] = int(self._total_samples)
            self._file.flush()
            self._file.close()

        self._file = None
        self._data_ds = None
        self._total_samples = 0
        self._meta = None
        self._h5py_module = None


class HDF5Writer(ioc.IChain):
    """Plain HDF5 (.h5) file writer chain for real-time data logging.

    Writes multi-channel data with a time column plus one column per
    channel under ``variable_name``, alongside a ``gpype_meta`` JSON
    document and plain HDF5 attributes describing the rate, labels and
    roles -- so both a g.Pype :class:`~gpype.HDF5Reader` and a bare h5py
    script can make sense of the file.
    """

    def __init__(self, file_name: str, variable_name: str = "data", **kwargs):
        """Initialize the HDF5 writer chain.

        Args:
            file_name: Base filename for the .h5 output. A timestamp will
                be automatically appended. Optional only so that a stored
                configuration can supply it; one of the two must be
                given.
            variable_name: Name of the dataset the recording is written
                under.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If no file name is available from either source.
        """
        self._link_stream_id = stream_id_for(kwargs)
        fn_key = _HDF5WriterCore.Configuration.Keys.FILE_NAME
        if file_name is None:
            file_name = kwargs.get(fn_key)
        if file_name is None:
            raise ValueError("file_name must be provided.")
        self._core_params = {
            "file_name": file_name,
            "variable_name": variable_name,
        }
        self._core_params.update(strip_chain_keys(kwargs))

        kwargs.setdefault(fn_key, file_name)
        kwargs.setdefault("variable_name", variable_name)
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
            List containing [Link, _HDF5WriterCore].
        """
        return [
            Link(
                sender=Constants.Residency.SERVER,
                receiver=Constants.Residency.EDGE,
                stream_id=self._link_stream_id,
            ),
            _HDF5WriterCore(**self._core_params),
        ]
