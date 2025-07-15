from __future__ import annotations
import queue
import threading
import time
from typing import Dict, Optional

import gtec_ble as ble
import numpy as np

from ...common.constants import Constants
from ..core.o_port import OPort
from .base.amplifier_source import AmplifierSource

# Convenience constants for default port names
PORT_OUT = Constants.Defaults.PORT_OUT
PORT_IN = Constants.Defaults.PORT_IN


class BCICore8(AmplifierSource):
    """g.tec BCI Core-8 amplifier source for wireless EEG acquisition.

    This class provides interface to the g.tec BCI Core-8 wireless EEG
    amplifier using Bluetooth Low Energy (BLE) communication. It handles
    device connection, data streaming, buffering, and real-time processing
    for BCI applications.

    The BCI Core-8 features:
    - 8-channel EEG acquisition at 250 Hz sampling rate
    - Wireless BLE connectivity with low latency
    - Real-time data streaming with configurable buffering
    - Optional buffer level monitoring for system diagnostics

    Features:
    - Automatic device discovery and connection by serial number
    - Frame-based data buffering with configurable sizes
    - Thread-safe data acquisition with precise timing
    - Buffer level monitoring for real-time system analysis
    - Automatic decimation and frame assembly

    Attributes:
        SAMPLING_RATE: Fixed 250 Hz sampling rate
        MAX_NUM_CHANNELS: Maximum 8 channels supported
        DEFAULT_BUFFER_SIZE_SAMPLES: Default buffer size (60ms at 250Hz)
        PORT_BUF_LEVEL: Port name for buffer level output

    Note:
        The amplifier uses a fixed 250 Hz sampling rate and supports up to
        8 channels. Buffer management is crucial for real-time performance
        and preventing data loss during intensive processing.
    """

    # Source code fingerprint
    FINGERPRINT = "b6370ce9b4f094bfafe29be37f975ea0"

    # Optional buffer level output port
    PORT_BUF_LEVEL = "buffer_level"

    # Hardware specifications
    SCANNING_TIMEOUT_S = 10
    SAMPLING_RATE = 250
    MAX_NUM_CHANNELS = 8
    DEFAULT_BUFFER_SIZE_SAMPLES = int(round(100e-3 * SAMPLING_RATE))

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for BCI Core-8 specific parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration keys for BCI Core-8 settings."""
            OUTPUT_BUFFER_LEVEL = 'output_buffer_level'

    # Type hints for amplifier-specific attributes
    _device: Optional[ble.Amplifier]
    _target_sn: Optional[str]

    def __init__(self,
                 serial: Optional[str] = None,
                 channel_count: Optional[int] = None,
                 frame_size: Optional[int] = None,
                 buffer_size_samples: Optional[int] = None,
                 output_buffer_level: Optional[bool] = None,
                 **kwargs):
        """Initialize BCI Core-8 amplifier source.

        Args:
            serial: Serial number of target BCI Core-8 device. If None,
                the first discovered device will be used.
            channel_count: Number of EEG channels to acquire (1-8).
                Defaults to maximum 8 channels if not specified.
            frame_size: Number of samples per processing frame. Affects
                latency and processing efficiency.
            buffer_size_samples: Size of internal buffer in samples.
                Defaults to 60ms worth of samples for low latency.
            output_buffer_level: If True, outputs buffer level monitoring
                data on a second port for system diagnostics.
            **kwargs: Additional arguments passed to parent AmplifierSource.

        Note:
            The amplifier automatically configures decimation and buffering
            based on frame size. Buffer level monitoring helps detect
            real-time performance issues.
        """
        # Validate and set channel count (1-8 channels supported)
        if channel_count is None:
            channel_count = self.MAX_NUM_CHANNELS
        channel_count = max(1, min(channel_count, self.MAX_NUM_CHANNELS))

        # Set default buffer level monitoring
        if output_buffer_level is None:
            output_buffer_level = False

        # Configure output ports based on buffer level monitoring
        output_ports = [OPort.Configuration()]
        if output_buffer_level:
            output_ports.append(OPort.Configuration(name=self.PORT_BUF_LEVEL))
            channel_count = [channel_count, 1]  # Main data + buffer level

        # Initialize parent amplifier source with BCI Core-8 specifications
        super().__init__(channel_count=channel_count,
                         sampling_rate=self.SAMPLING_RATE,
                         frame_size=frame_size,
                         decimation_factor=frame_size,
                         output_buffer_level=output_buffer_level,
                         output_ports=output_ports,
                         **kwargs)

        # Calculate buffer configuration
        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]
        if buffer_size_samples is None:
            buffer_size_samples = self.DEFAULT_BUFFER_SIZE_SAMPLES
        buffer_size_frames = int(np.ceil(buffer_size_samples / frame_size))

        # Store device configuration
        self._target_sn = serial

        # Initialize device connection (will be established in start())
        self._device = None

        # Initialize threading components for real-time processing
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._time_start: Optional[float] = None

        # Initialize data management components
        self._data_sample_counter: int = 0
        self._frame_buffer: Optional[queue.Queue] = None
        self._sample_buffer: Optional[np.ndarray] = None
        self._buffer_size_frames: int = buffer_size_frames

        # Calculate and set source delay for timing synchronization
        # Use 75% of buffer to allow for some safety margin
        self.source_delay = (buffer_size_frames * frame_size
                             / self.SAMPLING_RATE * 0.75)
        print(f"BCI Core-8 source delay is approximately "
              f"{self.source_delay * 1e3:.2f} ms.")

        # Initialize buffer level monitoring if enabled
        if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
            self._buffer_level_buffer: Optional[queue.Queue] = None

    def start(self) -> None:
        """Start the BCI Core-8 amplifier and begin data acquisition.

        Initializes data buffers, starts the background processing thread,
        establishes BLE connection to the amplifier, and begins real-time
        data streaming.

        Raises:
            ConnectionError: If amplifier connection fails.
            RuntimeError: If background thread creation fails.

        Note:
            The method first initializes internal buffers, then starts the
            background thread for timing control, connects to the amplifier
            hardware, and finally begins data acquisition.
        """
        # Get configuration parameters
        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
        channel_count = self.config[self.Configuration.Keys.CHANNEL_COUNT]

        # Initialize data buffers for frame-based processing
        self._frame_buffer = queue.Queue(maxsize=self._buffer_size_frames)
        self._sample_buffer = np.zeros((frame_size[0], channel_count[0]))

        # Initialize buffer level monitoring if enabled
        if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
            self._buffer_level_buffer = queue.Queue(
                maxsize=self._buffer_size_frames)

        # Start background thread for timing control
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._thread_function,
                                            daemon=True)
            self._thread.start()

        # Call parent start method
        super().start()

        # Initialize and connect to BCI Core-8 amplifier
        if self._device is None:
            self._device = ble.Amplifier(serial=self._target_sn)
            self._device.set_data_callback(self._data_callback)

        # Begin data acquisition from amplifier
        self._device.start()

    def setup(self,
              data: Dict[str, np.ndarray],
              port_context_in: Dict[str, dict]) -> Dict[str, dict]:
        """Setup output port contexts for BCI Core-8 data streams.

        Configures output port contexts by calling the parent setup method
        which handles sampling rate and channel configuration propagation.

        Args:
            data: Dictionary of input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with BCI Core-8 specific
            configuration including 250 Hz sampling rate.
        """
        return super().setup(data, port_context_in)

    def stop(self):
        """Stop the BCI Core-8 amplifier and clean up resources.

        Stops data acquisition, terminates the background thread, and
        properly disconnects from the amplifier hardware.

        Note:
            The method ensures proper shutdown sequence: stop amplifier,
            call parent stop, terminate background thread, and clean up
            device connection.
        """
        # Stop amplifier data acquisition
        if self._device is not None:
            self._device.stop()

        # Call parent stop method
        super().stop()

        # Stop background thread and wait for completion
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10)  # Wait up to 10 seconds

        # Clean up device connection
        self._device = None

    def step(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Retrieve processed data frames from the amplifier.

        Returns data frames when decimation step is active, otherwise
        returns None. Handles buffer underruns by providing zero-filled
        frames to maintain pipeline continuity.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing EEG data and optionally buffer level data,
            or None if not a decimation step.

        Note:
            Buffer underruns (empty queue) are handled by returning
            zero-filled frames to prevent pipeline stalls. This should
            be rare with proper buffer sizing.
        """
        if self.is_decimation_step():
            out_data = {}
            try:
                # Retrieve processed frame from buffer
                out_data = {PORT_OUT: self._frame_buffer.get_nowait()}

                # Add buffer level data if monitoring is enabled
                if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
                    buffer_level = self._buffer_level_buffer.get_nowait()
                    out_data[self.PORT_BUF_LEVEL] = buffer_level

            except queue.Empty:
                # Handle buffer underrun with zero-filled frames
                frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
                channel_count = self.config[
                    self.Configuration.Keys.CHANNEL_COUNT]

                self.log("Buffer underrun - returning zero-filled frame.",
                         type=Constants.LogTypes.WARNING)
                out_data = {PORT_OUT: np.zeros((frame_size[0],
                                                channel_count[0]))}

                if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
                    buf_shape = (frame_size[0], 1)
                    out_data[self.PORT_BUF_LEVEL] = np.zeros(buf_shape)

            return out_data
        else:
            # Not a decimation step, return None
            return None

    def _data_callback(self, data: np.ndarray):
        """Callback function for incoming amplifier data.

        Processes individual samples from the BCI Core-8 amplifier,
        assembles them into frames, and manages buffer queues for
        real-time processing.

        Args:
            data: Single sample data array from amplifier with shape
                (n_channels,). Only the configured number of channels
                are used.

        Note:
            This method runs in the amplifier's callback thread and must
            be efficient to prevent data loss. Frame assembly and buffer
            management are handled here to minimize processing overhead
            in the main pipeline.
        """
        # Safety check for buffer initialization
        if self._sample_buffer is None:
            return

        # Get configuration parameters
        cc_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT
        fsz_key = Constants.Keys.FRAME_SIZE
        frame_size = self.config[fsz_key][0]

        # Determine position within current frame
        idx_in_frame = self._data_sample_counter % frame_size

        # Store sample data (only configured number of channels)
        num_channels = self.config[cc_key][0]
        self._sample_buffer[idx_in_frame, :] = data[:num_channels]
        self._data_sample_counter += 1

        # Check if frame is complete
        if self._data_sample_counter % frame_size == 0:
            try:
                # Queue completed frame for processing
                self._frame_buffer.put_nowait(self._sample_buffer.copy())

                # Calculate and queue buffer level if monitoring enabled
                if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
                    # Buffer level: -100% (empty) to +100% (full)
                    # Centered around 50% fill (0% level)
                    fill_ratio = (self._frame_buffer.qsize()
                                  / self._frame_buffer.maxsize)
                    buf_lvl = (fill_ratio - 0.5) * 2 * 100

                    # Create buffer level array for entire frame
                    level_data = buf_lvl * np.ones((frame_size, 1))
                    self._buffer_level_buffer.put_nowait(level_data)

            except queue.Full:
                # Buffer overflow - drop oldest frame
                # This indicates the processing is too slow for real-time
                pass

    def _thread_function(self):
        """Background thread function for timing control and node cycles.

        This method implements the timing control for the BCI Core-8 data
        processing pipeline. It waits for the buffer to partially fill,
        then maintains precise timing for node cycles.

        The algorithm:
        1. Wait for buffer to reach 50% capacity (startup phase)
        2. Begin precise timing control using monotonic time
        3. Calculate expected sample times to prevent drift
        4. Sleep appropriately and trigger node cycles

        Note:
            The startup delay ensures stable data flow before beginning
            real-time processing. Precise timing prevents cumulative
            drift in the processing pipeline.
        """
        # Get sampling rate for timing calculations
        rate = self.config[self.Configuration.Keys.SAMPLING_RATE]

        # Startup phase: wait for buffer to partially fill
        # This ensures stable data flow before real-time processing begins
        target_fill = self._buffer_size_frames / 2
        while self._running and self._frame_buffer.qsize() < target_fill:
            time.sleep(1 / rate)

        # Initialize timing for precise pipeline control
        if self._time_start is None:
            self._time_start = time.monotonic()

        # Initialize timing variables for drift prevention
        thread_sample_count = 0
        expected_next_sample_time = self._time_start

        # Main timing loop
        while self._running:
            # Calculate expected time for next pipeline cycle
            thread_sample_count += 1
            expected_next_sample_time = (self._time_start
                                         + thread_sample_count / rate)

            # Determine sleep time to maintain precise timing
            current_time = time.monotonic()
            sleep_time = expected_next_sample_time - current_time

            # Sleep only if duration is significant (avoid timing jitter)
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            # Trigger pipeline processing cycle
            self.cycle()
