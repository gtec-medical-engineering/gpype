from __future__ import annotations

import json
import struct
from typing import Optional

import numpy as np

from ...common.constants import Constants
from .base import recording_meta
from .base.file_writer import FileWriter


def _h5py():
    """Import h5py, or say what installs it.

    Imported here rather than at module scope: a module-level ``import
    h5py`` would make ``scripts/generate_catalog.py`` raise on any
    machine without it -- measured, it walks ``dir(gp)`` and treats a
    failed import as a hard error, not a skipped node. Deferring the
    import to construction time is the same pattern
    ``gtc_reader.py`` uses for the ``gtc`` package: ``gp.MatWriter``
    resolves as a class with h5py absent, and only building one raises.

    Returns:
        The h5py module.

    Raises:
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "MatWriter needs h5py, provided by the 'formats' extra: "
            'pip install "gpype[formats]"'
        ) from error
    return h5py


class MatWriter(FileWriter):
    """MATLAB v7.3 (.mat) file writer for real-time data logging.

    Writes multi-channel data as an HDF5-backed MAT-file MATLAB's own
    ``load()`` opens directly, with a time column plus one column per
    channel under ``variable_name``. Channel labels, roles and the
    sampling rate are recorded in a ``gpype_meta`` document that
    :class:`~gpype.MatReader` reads back exactly, and a plain
    ``sampling_rate`` variable is written alongside it for a MATLAB user
    who never asked about g.Pype's metadata format at all.

    MATLAB's v7.3 format *is* HDF5: a 128-byte header identifying the
    file as MAT-file version 2.0 little-endian, in a 512-byte userblock
    in front of an ordinary HDF5 container. The header is written when
    the file is created, not at close, so a recording killed mid-run
    still loads in MATLAB.
    """

    def __init__(
        self,
        file_name: str,
        variable_name: str = "data",
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the MAT writer core.

        Args:
            file_name: Base filename for the .mat output. A timestamp
                will be automatically appended.
            variable_name: Name of the HDF5 dataset MATLAB will load the
                recording as, e.g. ``load(file); data`` gives back the
                array under this name.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments passed to parent FileWriter.
        """
        super().__init__(
            file_name=file_name,
            variable_name=variable_name,
            edge_id=edge_id,
            **kwargs,
        )
        self._variable_name = variable_name
        self._file = None
        self._data_ds = None
        self._n_cols = None
        self._total_samples = 0
        self._meta = None

    @property
    def file_extension(self) -> str:
        """Return the file extension for MAT files.

        Returns:
            The MAT file extension '.mat'.
        """
        return ".mat"

    def _open_file(
        self, file_path: str, port_context_in: dict[str, dict]
    ) -> None:
        """Create the file, write its MATLAB header, and record metadata.

        Args:
            file_path: Full path to the output .mat file.
            port_context_in: Context information from input ports.
        """
        h5py = _h5py()

        # Created and closed first so the 128-byte header can be written
        # into the userblock while nothing else holds the file -- see the
        # class docstring for the measured PermissionError this avoids.
        created = h5py.File(file_path, "w", userblock_size=512)
        created.create_group("#refs#")  # MATLAB expects this to exist.
        created.close()
        self._write_matlab_header(file_path)
        self._file = h5py.File(file_path, "r+")

        self._meta = recording_meta.collect(
            port_context_in, self._marked, self._sampling_rate
        )

        # A plain MATLAB double as well as the JSON document, so a user
        # who never asked about metadata still gets `sampling_rate` as
        # an ordinary variable in the workspace.
        rate_ds = self._file.create_dataset(
            "sampling_rate",
            data=np.array([[float(self._sampling_rate)]], dtype=np.float64),
        )
        rate_ds.attrs["MATLAB_class"] = np.bytes_("double")

        self._write_gpype_meta(self._meta)

        self._data_ds = None
        self._n_cols = None
        self._total_samples = 0

    @staticmethod
    def _write_matlab_header(file_path: str) -> None:
        """Write the 128-byte MAT-file v7.3 header into the userblock.

        Layout verified against ``scipy.io.matlab._miobase.
        get_matfile_version``, an independent implementation of the
        spec: a 116-byte space-padded description, 8 zero bytes (the
        subsystem data offset, unused here), the version marker
        ``0x0200`` and the endian indicator ``b"IM"`` -- 128 bytes total,
        with the HDF5 superblock itself starting at the userblock
        boundary, offset 512.

        Args:
            file_path: Path to a file already created with
                ``userblock_size=512`` and currently closed. Measured:
                writing this while h5py holds the file open raises
                ``PermissionError`` on Windows.
        """
        desc = b"MATLAB 7.3 MAT-file, written by MatWriter"
        desc = (desc + b" " * 116)[:116]
        subsystem_offset = b"\x00" * 8
        version = struct.pack("<H", 0x0200)
        endian = b"IM"
        header = desc + subsystem_offset + version + endian
        with open(file_path, "r+b") as handle:
            handle.write(header)

    def _write_gpype_meta(self, meta: dict) -> None:
        """(Re)write the ``/gpype_meta`` dataset.

        MATLAB has no native JSON type, so the document is stored as a
        ``(N, 1)`` column of ``uint16`` character codes with
        ``MATLAB_int_decode=2`` -- MATLAB's own convention for "this
        integer array is really text" -- rather than as an HDF5 string,
        which MATLAB's HDF reader does not decode into a char array.
        Measured: this round-trips exactly, including non-ASCII labels.

        Args:
            meta: The document to write.
        """
        if recording_meta.META_DATASET in self._file:
            del self._file[recording_meta.META_DATASET]
        blob = json.dumps(meta)
        encoded = np.array([[ord(c)] for c in blob], dtype=np.uint16)
        ds = self._file.create_dataset(
            recording_meta.META_DATASET, data=encoded
        )
        ds.attrs["MATLAB_class"] = np.bytes_("char")
        ds.attrs["MATLAB_int_decode"] = np.int32(2)

    def _write_block(self, block: np.ndarray, timestamps: np.ndarray) -> None:
        """Append a data block, growing the dataset along the sample axis.

        Args:
            block: Data block to write, shape (samples, channels).
            timestamps: Timestamp array for each sample in the block.
        """
        if self._file is None:
            return

        # (1+n_channels, n_samples): MATLAB reverses row/col order on
        # load, so this becomes the (n_samples, 1+n_channels) array a
        # MATLAB user actually indexes, time column first.
        combined = np.column_stack(
            [timestamps.astype(np.float64), block.astype(np.float64)]
        ).T
        n_cols, n_samples = combined.shape

        if self._data_ds is None:
            self._n_cols = n_cols
            self._data_ds = self._file.create_dataset(
                self._variable_name,
                shape=(n_cols, 0),
                maxshape=(n_cols, None),
                dtype=np.float64,
                chunks=(n_cols, max(1, n_samples)),
            )
            self._data_ds.attrs["MATLAB_class"] = np.bytes_("double")

        new_size = self._total_samples + n_samples
        self._data_ds.resize((self._n_cols, new_size))
        self._data_ds[:, self._total_samples : new_size] = combined
        self._total_samples = new_size

        self._file.flush()

    def _close_file(self) -> None:
        """Record the final sample count and close the file.

        The MATLAB header was already written at open, so nothing about
        loadability depends on reaching this point -- a killed run is
        still a valid (if metadata-incomplete) MAT file.
        """
        if self._file is not None:
            if self._meta is not None:
                self._write_gpype_meta(
                    recording_meta.finalize(self._meta, self._total_samples)
                )
            self._file.flush()
            self._file.close()

        self._file = None
        self._data_ds = None
        self._n_cols = None
        self._total_samples = 0
        self._meta = None
