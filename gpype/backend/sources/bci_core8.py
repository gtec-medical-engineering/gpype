from __future__ import annotations

import queue
import threading
import time
from typing import Optional

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

    Interface to g.tec BCI Core-8 wireless EEG amplifier using BLE. Supports
    8-channel acquisition at 250 Hz.
    """

    # Source code fingerprint
    FINGERPRINT = "8b2d1645874477bb3fa5f76c27a7d10f"

    # Optional buffer level output port
    PORT_BUF_LEVEL = "buffer_level"

    # Hardware specifications
    SCANNING_TIMEOUT_S = 10
    SAMPLING_RATE = 250
    MAX_NUM_CHANNELS = 8
    DEFAULT_BUFFER_DELAY_MS = 40
    TARGET_FILL_RATIO = 0.5
    FILL_RATIO_ALPHA = 0.9995
    FILL_RATIO_CORRECTION_INTERVAL_S = 1.0
    NUM_UNDERRUNS_ALLOWED = 10
    DEVICE_DELAY_MS = 18

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for BCI Core-8 specific parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration keys for BCI Core-8 settings."""

            OUTPUT_BUFFER_LEVEL = "output_buffer_level"
            BUFFER_DELAY_MS = "buffer_delay_ms"

    # Type hints for amplifier-specific attributes
    _device: Optional[ble.Amplifier]
    _target_sn: Optional[str]

    def __init__(
        self,
        serial: Optional[str] = None,
        channel_count: Optional[int] = None,
        frame_size: Optional[int] = None,
        buffer_delay_ms: Optional[int] = None,
        output_buffer_level: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize BCI Core-8 amplifier source.

        Args:
            serial: Serial number of target device. Uses first discovered
                if None.
            channel_count: Number of EEG channels (1-8). Defaults to 8.
            frame_size: Samples per processing frame.
            buffer_delay_ms: Internal buffer delay in milliseconds.
            output_buffer_level: Enable buffer level monitoring output.
            **kwargs: Additional arguments for parent AmplifierSource.
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

        if buffer_delay_ms is None:
            buffer_delay_ms = self.DEFAULT_BUFFER_DELAY_MS

        # Initialize parent amplifier source with BCI Core-8 specifications
        super().__init__(
            channel_count=channel_count,
            sampling_rate=self.SAMPLING_RATE,
            frame_size=frame_size,
            decimation_factor=frame_size,
            output_buffer_level=output_buffer_level,
            buffer_delay_ms=buffer_delay_ms,
            output_ports=output_ports,
            **kwargs,
        )

        self._frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]

        # Calculate buffer configuration
        self._target_fill_samples = int(
            buffer_delay_ms / 1000 * self.SAMPLING_RATE
        )
        buf_size_samples = int(
            self._target_fill_samples / self.TARGET_FILL_RATIO
        )
        buf_size_frames = int(np.ceil(buf_size_samples / self._frame_size))

        # Store device configuration
        self._target_sn = serial

        # Initialize device connection (will be established in start())
        self._device = None

        # Initialize threading components for real-time processing
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._time_start: Optional[float] = None

        # Initialize data management components
        self._in_sample_counter: int = 0
        self._out_sample_counter: int = 0
        self._frame_buffer: Optional[queue.Queue] = None
        self._sample_buffer: Optional[np.ndarray] = None
        self._buffer_size_frames: int = buf_size_frames
        self._underrun_counter: int = None
        self._fill_ratio: float = None

        # Calculate and set source delay for timing synchronization
        self.source_delay = (buffer_delay_ms + self.DEVICE_DELAY_MS) / 1000
        print(
            f"BCI Core-8 buffer delay is set to " f"{buffer_delay_ms:.2f} ms."
        )

        # Initialize buffer level monitoring if enabled
        if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
            self._buffer_level_buffer: Optional[queue.Queue] = None

    def start(self) -> None:
        """Start BCI Core-8 amplifier and begin data acquisition.

        Initializes buffers, starts background thread, establishes BLE
        connection, and begins real-time data streaming.

        Raises:
            ConnectionError: If amplifier connection fails.
            RuntimeError: If background thread creation fails.
        """
        # Get configuration parameters
        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
        channel_count = self.config[self.Configuration.Keys.CHANNEL_COUNT]

        # Initialize data buffers for frame-based processing
        self._frame_buffer = queue.Queue(maxsize=self._buffer_size_frames)
        self._sample_buffer = np.zeros((frame_size[0], channel_count[0]))
        self._underrun_counter = 0
        self._fill_ratio = self.TARGET_FILL_RATIO

        # Start background thread for timing control
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._thread_function, daemon=True
            )
            self._thread.start()

        # Call parent start method
        super().start()

        # Initialize and connect to BCI Core-8 amplifier
        if self._device is None:
            self._device = ble.Amplifier(serial=self._target_sn)
            self._device.set_data_callback(self._data_callback)

        # Begin data acquisition from amplifier
        self._device.start()

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts for BCI Core-8 data streams.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with 250 Hz sampling rate.
        """
        return super().setup(data, port_context_in)

    def stop(self):
        """Stop BCI Core-8 amplifier and clean up resources.

        Stops data acquisition, terminates background thread, and disconnects
        from amplifier hardware.
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

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Retrieve processed data frames from the amplifier.

        Returns data frames when decimation step is active. Handles buffer
        underruns by providing zero-filled frames to maintain continuity.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing EEG data and optionally buffer level data,
            or None if not a decimation step.
        """
        if self.is_decimation_step():
            out_data = {}
            if self._frame_buffer.qsize() == 0:
                self._underrun_counter += 1
            else:
                self._underrun_counter = 0

            if self._underrun_counter > self.NUM_UNDERRUNS_ALLOWED:
                self.log(
                    "Buffer underrun - performance may lag. Consider "
                    "increasing buffer size.",
                    type=Constants.LogTypes.WARNING,
                )

            try:
                out_data = {PORT_OUT: self._frame_buffer.get(timeout=1)}

            except queue.Empty:
                # Provide zero-filled frame to maintain pipeline continuity
                frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]
                cc_key = self.Configuration.Keys.CHANNEL_COUNT
                channel_count = self.config[cc_key][0]
                zero_frame = np.zeros((frame_size, channel_count))
                out_data = {PORT_OUT: zero_frame}

            # Buffer level monitoring
            delta = self._in_sample_counter - self._out_sample_counter
            cur_fill_ratio = delta / self._frame_buffer.maxsize
            alpha = self.FILL_RATIO_ALPHA
            self._fill_ratio = (
                1 - alpha
            ) * cur_fill_ratio + alpha * self._fill_ratio

            if self.config[self.Configuration.Keys.OUTPUT_BUFFER_LEVEL]:
                buf_lvl = (cur_fill_ratio - 0.5) * 2 * 100
                level_data = buf_lvl * np.ones((self._frame_size, 1))
                out_data[self.PORT_BUF_LEVEL] = level_data

            return out_data
        else:
            # Not a decimation step, return None
            return None

    def _data_callback(self, data: np.ndarray):
        """Callback function for incoming amplifier data.

        Processes individual samples, assembles them into frames, and manages
        buffer queues for real-time processing.

        Args:
            data: Single sample data array from amplifier with shape
                (n_channels,). Only configured channels are used.
        """
        # Safety check for buffer initialization
        if self._sample_buffer is None:
            return

        # Determine position within current frame
        idx_in_frame = self._in_sample_counter % self._frame_size

        # Store sample data (only configured number of channels)
        cc_key = AmplifierSource.Configuration.Keys.CHANNEL_COUNT
        num_channels = self.config[cc_key][0]
        self._sample_buffer[idx_in_frame, :] = data[:num_channels]
        self._in_sample_counter += 1

        # Check if frame is complete
        if self._in_sample_counter % self._frame_size == 0:
            try:
                # Queue completed frame for processing
                self._frame_buffer.put_nowait(self._sample_buffer.copy())

            except queue.Full:
                self._in_sample_counter -= self._frame_size
                # self.log(
                #     "Buffer overflow - data lost. Consider "
                #     "increasing buffer size.",
                #     type=Constants.LogTypes.WARNING,
                # )
                pass

    def _thread_function(self):
        """Background thread function for timing control and node cycles.

        Implements timing control for data processing pipeline. Waits for
        buffer to partially fill, then maintains precise timing for cycles.
        """
        # Get sampling rate for timing calculations
        sampling_rate = self.config[self.Configuration.Keys.SAMPLING_RATE]

        # Startup phase: wait for buffer to partially fill
        # This ensures stable data flow before real-time processing begins
        limit = self._target_fill_samples
        while self._running and self._in_sample_counter < limit:
            time.sleep(1 / sampling_rate)

        # Initialize timing for precise pipeline control
        if self._time_start is None:
            self._time_start = time.monotonic()

        # Initialize variables
        self._out_sample_counter = 0
        next_wakeup_time = self._time_start
        time_correction = 0
        correction_interval_samples = int(
            self.FILL_RATIO_CORRECTION_INTERVAL_S * sampling_rate
        )
        buf_size = self._frame_buffer.maxsize
        fill_samples_target = self.TARGET_FILL_RATIO * buf_size

        # Main timing loop
        while self._running:
            # Calculate expected time for next pipeline cycle
            self._out_sample_counter += 1
            next_sample_time = self._out_sample_counter / sampling_rate
            next_wakeup_time = self._time_start + next_sample_time
            next_wakeup_time += time_correction

            # Determine sleep time to maintain precise timing
            current_time = time.monotonic()
            sleep_time = next_wakeup_time - current_time

            # Sleep only if duration is significant (avoid timing jitter)
            if sleep_time > 0.001:
                time.sleep(sleep_time)

            # Trigger pipeline processing cycle
            self.cycle()

            # Correction mechanism to adjust timing based on buffer fill ratio.
            # This is important to maintain the target buffer delay and
            # prevent drift.
            if self._out_sample_counter % correction_interval_samples == 0:
                fill_samples = self._fill_ratio * buf_size
                delta_samples = fill_samples - fill_samples_target
                delta_s = delta_samples / sampling_rate
                if abs(delta_s) > 1e-3:
                    # reset fill ratio to target to avoid overcorrection
                    self._fill_ratio = self.TARGET_FILL_RATIO
                    # Adjust time correction based on fill ratio deviation
                    time_correction -= delta_s
