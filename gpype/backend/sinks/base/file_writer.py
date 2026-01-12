from __future__ import annotations

import os
import queue
import threading
import time
from abc import abstractmethod
from datetime import datetime
from typing import Optional

import ioiocore as ioc
import numpy as np

from ....common.constants import Constants
from ...core.i_node import INode
from ...core.i_port import IPort


class FileWriter(INode):
    """Abstract base class for threaded file writers.

    Implements a file writer that operates in a separate background thread
    to prevent blocking the main signal processing pipeline. Data is queued
    and written asynchronously to maintain real-time performance.

    Subclasses must implement format-specific file operations.
    """

    class Configuration(ioc.INode.Configuration):
        """Configuration class for FileWriter parameters."""

        class Keys(ioc.INode.Configuration.Keys):
            """Configuration keys for file writer settings."""

            #: File name configuration key
            FILE_NAME = "file_name"

    def __init__(
        self,
        file_name: str,
        **kwargs,
    ):
        """Initialize the file writer with specified filename.

        Args:
            file_name: Base filename for data output. A timestamp will be
                automatically appended.
            **kwargs: Additional arguments passed to parent INode class.
        """
        # Initialize parent INode with configuration
        INode.__init__(self, file_name=file_name, **kwargs)

        # Initialize threading components for background file operations
        self._file_queue = queue.Queue()  # Thread-safe data queue
        self._stop_event = threading.Event()  # Shutdown coordination
        self._worker_thread = None  # Background file writer thread
        self._sample_counter = 0  # Global sample index counter
        self._sampling_rate = None  # Sampling rate from port context
        self._file_path = None  # Full path to output file

    def _generate_file_path(self) -> str:
        """Generate the output file path with timestamp.

        Takes the configured file name, inserts a timestamp before the
        extension, and validates the extension matches the writer's format.

        Returns:
            Full file path with timestamp inserted.

        Raises:
            ValueError: If file extension doesn't match writer's format.
        """
        file_name = self.config[self.Configuration.Keys.FILE_NAME]
        name, ext = os.path.splitext(file_name)

        # Validate file extension
        expected_ext = self.file_extension
        if ext.lower() != expected_ext.lower():
            raise ValueError(
                f"Invalid file extension '{ext}'. "
                f"Expected '{expected_ext}' for {self.__class__.__name__}."
            )

        # Insert timestamp before extension
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{name}_{timestamp}{ext}"

    def start(self):
        """Start the file writer and initialize background thread.

        Generates the output file path with timestamp and starts the
        background worker thread for asynchronous writing. The actual
        file opening is deferred to setup() when port context is available.

        Raises:
            ValueError: If the file extension is invalid for this writer.
        """
        # Generate timestamped file path
        self._file_path = self._generate_file_path()

        # Reset sample counter
        self._sample_counter = 0

        # Initialize and start background worker thread
        self._worker_thread = threading.Thread(
            target=self._file_worker, daemon=True
        )
        self._stop_event.clear()  # Reset stop event
        self._worker_thread.start()

        # Call parent start method
        super().start()

    def stop(self):
        """Stop the file writer and clean up resources.

        Signals the background thread to stop, waits for it to finish
        processing remaining data, and properly closes the file.
        Ensures all queued data is written before stopping.
        """
        # Signal background thread to stop
        self._stop_event.set()

        # Wait for worker thread to finish processing remaining data
        if self._worker_thread is not None:
            self._worker_thread.join()
            self._worker_thread = None

        # Close file using format-specific implementation
        self._close_file()

        # Call parent stop method
        super().stop()

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup method called before processing begins.

        Extracts sampling rate from port context and opens the output file
        using the format-specific implementation.

        Args:
            data: Dictionary of input data arrays from connected ports.
            port_context_in: Context information from input ports.

        Returns:
            Empty dictionary as this is a sink node with no outputs.

        Raises:
            RuntimeError: If sampling rate is not provided in port context.
        """
        # Extract sampling rate from port context
        PORT_IN = Constants.Defaults.PORT_IN
        sr_key = Constants.Keys.SAMPLING_RATE
        self._sampling_rate = port_context_in[PORT_IN].get(sr_key, None)
        if self._sampling_rate is None:
            raise RuntimeError("Sampling rate not provided in port context.")

        # Open file using format-specific implementation
        self._open_file(self._file_path, port_context_in)

        # No output context for sink nodes
        return {}

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process incoming data by queuing it for background writing.

        Args:
            data: Dictionary containing input data arrays. Uses the default
                input port to retrieve data for writing.

        Returns:
            Empty dictionary as this is a sink node with no outputs.
        """
        # Get data from default input port
        d = data[Constants.Defaults.PORT_IN]

        # Copy data and queue for background writing (thread-safe)
        self._file_queue.put(d.copy())

        # No output data for sink nodes
        return {}

    def _file_worker(self):
        """Background worker thread for asynchronous file writing.

        Runs in a separate thread to handle file I/O operations without
        blocking the main signal processing pipeline. Continuously processes
        data blocks from the queue until stop is signaled.
        """
        # Continue processing until stop signaled AND queue is empty
        while not self._stop_event.is_set() or not self._file_queue.empty():
            if not self._file_queue.empty():
                try:
                    # Get next data block from queue
                    block = self._file_queue.get(timeout=1)

                    # Calculate timestamps for this block
                    start_idx = self._sample_counter
                    end_idx = self._sample_counter + block.shape[0]
                    indices = np.arange(start_idx, end_idx)
                    timestamps = indices / self._sampling_rate

                    # Write block using format-specific implementation
                    self._write_block(block, timestamps)

                    # Update sample counter
                    self._sample_counter += block.shape[0]

                except queue.Empty:
                    # Timeout occurred, continue loop to check stop condition
                    continue
            else:
                # Queue is empty, brief sleep to prevent busy waiting
                time.sleep(0.01)
                continue

    @property
    @abstractmethod
    def file_extension(self) -> str:
        """Return the file extension for this writer type.

        Returns:
            File extension including the dot (e.g., '.csv', '.hdf5').
        """
        pass  # pragma: no cover

    @abstractmethod
    def _open_file(
        self, file_path: str, port_context_in: dict[str, dict]
    ) -> None:
        """Open the output file for writing.

        Format-specific implementation to create and initialize the output
        file. Called during setup() when port context is available.

        Args:
            file_path: Full path to the output file.
            port_context_in: Context information from input ports, containing
                channel count, sampling rate, and other metadata.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _write_block(self, block: np.ndarray, timestamps: np.ndarray) -> None:
        """Write a data block to the file.

        Format-specific implementation to write data to the output file.
        Called from the background worker thread.

        Args:
            block: Data block to write, shape (samples, channels).
            timestamps: Timestamp array for each sample in the block.
        """
        pass  # pragma: no cover

    @abstractmethod
    def _close_file(self) -> None:
        """Close the output file and finalize.

        Format-specific implementation to properly close the file and
        perform any finalization steps. Called during stop().
        """
        pass  # pragma: no cover
