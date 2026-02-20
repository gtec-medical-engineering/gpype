from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

import numpy as np

from ...common.constants import Constants
from ..core.o_port import OPort
from .base.amplifier_source import AmplifierSource

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN


def _get_unicorn_lib_path() -> str:
    """Get the path to the Unicorn Python library.

    The UnicornPy module requires the Unicorn.dll to be accessible.
    This function returns the path to the Lib folder in the user's
    Documents folder where the Unicorn Suite is typically installed.

    Returns:
        Path to the Unicorn Python Lib directory.
    """
    # Get user documents folder on Windows
    documents_path = os.path.join(os.environ["USERPROFILE"], "Documents")
    unicorn_lib_path = os.path.join(
        documents_path,
        "gtec",
        "Unicorn Suite",
        "Hybrid Black",
        "Unicorn Python",
        "Lib",
    )
    return unicorn_lib_path


def _ensure_unicorn_path():
    """Ensure the Unicorn Python library path is in sys.path and PATH.

    Adds the Unicorn Lib directory to sys.path for module import and
    to the PATH environment variable for DLL loading.
    """
    lib_path = _get_unicorn_lib_path()

    # Add to sys.path if not already present
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    # Add to PATH for DLL loading
    current_path = os.environ.get("PATH", "")
    if lib_path not in current_path:
        os.environ["PATH"] = lib_path + os.pathsep + current_path


class HybridBlack(AmplifierSource):
    """g.tec Unicorn Hybrid Black amplifier source for EEG acquisition.

    Interface to g.tec Unicorn Hybrid Black wireless EEG amplifier using
    Bluetooth. Supports 8-channel EEG acquisition at 250 Hz, plus optional
    accelerometer, gyroscope, battery, counter, and validation channels.
    """

    #: Source code fingerprint for licensing verification
    FINGERPRINT = "f83f85021ea989e78bf8dd50fbbdb796"

    #: Fixed sampling rate for Unicorn Hybrid Black amplifier in Hz
    SAMPLING_RATE = 250
    #: Number of EEG channels
    NUM_EEG_CHANNELS = 8
    #: Number of accelerometer channels (X, Y, Z)
    NUM_ACCEL_CHANNELS = 3
    #: Number of gyroscope channels (X, Y, Z)
    NUM_GYRO_CHANNELS = 3
    #: Total number of acquired channels (EEG + Accel + Gyro + Battery +
    #: Counter + Validation)
    TOTAL_ACQUIRED_CHANNELS = 17
    #: Hardware delay compensation in milliseconds (Bluetooth latency)
    DEVICE_DELAY_MS = 40
    #: Maximum allowed consecutive buffer underruns before warning
    NUM_UNDERRUNS_ALLOWED = 5
    #: Wait interval when behind (GetData < 1ms), in seconds
    WAIT_BEHIND_S = 0.003
    #: Wait interval when on time (GetData >= 1ms), in seconds
    WAIT_ON_TIME_S = 0.0039
    #: Threshold for determining if we're behind (GetData blocking time)
    BEHIND_THRESHOLD_S = 0.001

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for Unicorn Hybrid Black specific parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration keys for Unicorn Hybrid Black settings."""

            INCLUDE_ACCEL = "include_accel"
            INCLUDE_GYRO = "include_gyro"
            INCLUDE_AUX = "include_aux"
            TEST_SIGNAL = "test_signal"

    def __init__(
        self,
        serial: Optional[str] = None,
        channel_count: Optional[int] = None,
        frame_size: Optional[int] = None,
        include_accel: Optional[bool] = None,
        include_gyro: Optional[bool] = None,
        include_aux: Optional[bool] = None,
        test_signal: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize Unicorn Hybrid Black amplifier source.

        Args:
            serial: Serial number of target device. Uses first discovered
                if None.
            channel_count: Number of EEG channels (1-8). Defaults to 8.
            frame_size: Samples per processing frame.
            include_accel: Include accelerometer channels (3 channels).
            include_gyro: Include gyroscope channels (3 channels).
            include_aux: Include auxiliary channels (battery, counter,
                validation).
            test_signal: Enable test signal mode instead of live data.
            **kwargs: Additional arguments for parent AmplifierSource.

        Raises:
            NotImplementedError: If not running on Windows.
        """
        # Platform check - Hybrid Black is only supported on Windows
        if sys.platform != "win32":
            raise NotImplementedError(
                "HybridBlack amplifier is only supported on Windows."
            )

        # Ensure UnicornPy is importable
        _ensure_unicorn_path()

        # Validate and set channel count (1-8 EEG channels supported)
        if channel_count is None:
            channel_count = self.NUM_EEG_CHANNELS
        channel_count = max(1, min(channel_count, self.NUM_EEG_CHANNELS))

        # Set default values for optional parameters
        if include_accel is None:
            include_accel = False
        if include_gyro is None:
            include_gyro = False
        if include_aux is None:
            include_aux = False
        if test_signal is None:
            test_signal = False

        # Calculate total output channels based on configuration
        total_channels = channel_count
        if include_accel:
            total_channels += self.NUM_ACCEL_CHANNELS
        if include_gyro:
            total_channels += self.NUM_GYRO_CHANNELS
        if include_aux:
            total_channels += 3  # Battery, Counter, Validation

        # Configure output ports
        output_ports = [OPort.Configuration()]

        # Initialize parent amplifier source with Hybrid Black specifications
        super().__init__(
            channel_count=total_channels,
            sampling_rate=self.SAMPLING_RATE,
            frame_size=frame_size,
            decimation_factor=frame_size,
            include_accel=include_accel,
            include_gyro=include_gyro,
            include_aux=include_aux,
            test_signal=test_signal,
            output_ports=output_ports,
            **kwargs,
        )

        self._frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]

        # Store device configuration
        self._target_sn = serial
        self._eeg_channel_count = channel_count
        self._total_channels = total_channels
        self._include_accel = include_accel
        self._include_gyro = include_gyro
        self._include_aux = include_aux
        self._test_signal = test_signal

        # Initialize device connection (will be established in start())
        self._device = None

        # Initialize threading components
        self._running: bool = False
        self._acquisition_thread: Optional[threading.Thread] = None

        # Current frame for passing data from acquisition to step()
        self._current_frame: Optional[np.ndarray] = None

        # Underrun tracking
        self._underrun_counter: int = 0

        # Set source delay for timing synchronization (hardware delay only)
        self.source_delay = self.DEVICE_DELAY_MS / 1000

    def start(self) -> None:
        """Start Unicorn Hybrid Black amplifier and begin data acquisition.

        Establishes Bluetooth connection and starts background thread that
        acquires data and drives the pipeline via cycle().

        Raises:
            ConnectionError: If amplifier connection fails.
            RuntimeError: If background thread creation fails.
        """
        # Import UnicornPy when actually needed (lazy import)
        try:
            import UnicornPy
        except ImportError as e:
            raise RuntimeError(
                f"UnicornPy library not available: {e}. "
                "Please ensure the Unicorn Suite is installed and the "
                "library path is correct."
            ) from e

        # Initialize current frame holder
        self._current_frame = None
        self._underrun_counter = 0

        # Initialize and connect to Unicorn Hybrid Black amplifier
        if self._device is None:
            # Get available devices if no serial specified
            if self._target_sn is None:
                device_list = UnicornPy.GetAvailableDevices(True)
                if len(device_list) <= 0 or device_list is None:
                    raise ConnectionError(
                        "No Unicorn device available. Please pair with a "
                        "Unicorn first."
                    )
                self._target_sn = device_list[0]
                print(f"Using first available device: {self._target_sn}")

            # Connect to the device
            self._device = UnicornPy.Unicorn(self._target_sn)
            print(f"Connected to Unicorn Hybrid Black: {self._target_sn}")

        # Start single acquisition thread (handles both data and timing)
        if not self._running:
            self._running = True
            self._acquisition_thread = threading.Thread(
                target=self._acquisition_function, daemon=True
            )
            self._acquisition_thread.start()

        # Call parent start method
        super().start()

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts for Unicorn Hybrid Black data streams.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with 250 Hz sampling rate.
        """
        return super().setup(data, port_context_in)

    def stop(self):
        """Stop Unicorn Hybrid Black amplifier and clean up resources.

        Stops data acquisition, terminates background thread, and disconnects
        from amplifier hardware.
        """
        # Stop background thread and wait for completion
        if self._running:
            self._running = False
            acq_thread = self._acquisition_thread
            if acq_thread and acq_thread.is_alive():
                acq_thread.join(timeout=10)

        # Stop amplifier data acquisition
        if self._device is not None:
            try:
                self._device.StopAcquisition()
            except Exception:
                pass  # Device may already be stopped
            # Clean up device connection
            del self._device
            self._device = None

        # Call parent stop method
        super().stop()

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Retrieve processed data frames from the amplifier.

        Returns data frames when decimation step is active.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing EEG data, or None if not a decimation step.
        """
        if self.is_decimation_step():
            # Return current frame (set by acquisition thread before cycle())
            if self._current_frame is not None:
                out_data = {PORT_OUT: self._current_frame}
                self._current_frame = None
                return out_data
            else:
                # Provide zero-filled frame if no data available
                frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]
                cc_key = self.Configuration.Keys.CHANNEL_COUNT
                channel_count = self.config[cc_key][0]
                zero_frame = np.zeros((frame_size, channel_count))
                return {PORT_OUT: zero_frame}
        else:
            # Not a decimation step, return None
            return None

    def _acquisition_function(self):
        """Background thread function for data acquisition and pipeline timing.

        Uses adaptive timing: if GetData is fast (<1ms), we're behind and
        use shorter wait. If GetData blocks (>=1ms), we're on time and use
        normal wait interval.
        """
        try:
            import UnicornPy
        except ImportError:
            return

        # Get number of acquired channels from device
        num_acquired_channels = self._device.GetNumberOfAcquiredChannels()

        # Acquisition frame size matches output frame size
        acq_frame_length = self._frame_size
        receive_buffer_length = acq_frame_length * num_acquired_channels * 4
        receive_buffer = bytearray(receive_buffer_length)

        # Build column indices for channel selection
        col_indices = list(range(self._eeg_channel_count))
        if self._include_accel:
            start = self.NUM_EEG_CHANNELS
            col_indices.extend(range(start, start + 3))
        if self._include_gyro:
            start = self.NUM_EEG_CHANNELS + self.NUM_ACCEL_CHANNELS
            col_indices.extend(range(start, start + 3))
        if self._include_aux:
            start = self.NUM_EEG_CHANNELS + self.NUM_ACCEL_CHANNELS + 3
            col_indices.extend(range(start, num_acquired_channels))

        # Check if we can use fast contiguous slice (EEG only, all 8 channels)
        use_slice = (
            not self._include_accel
            and not self._include_gyro
            and not self._include_aux
            and self._eeg_channel_count == self.NUM_EEG_CHANNELS
        )

        # Start data acquisition from amplifier
        self._device.StartAcquisition(self._test_signal)

        # Initialize adaptive timing
        wait_interval_s = self.WAIT_ON_TIME_S
        t_last_call = time.perf_counter()

        while self._running:
            try:
                # Calculate expected next call time and sleep if needed
                t_expected_next = t_last_call + wait_interval_s
                t_now = time.perf_counter()
                t_remaining = t_expected_next - t_now
                if t_remaining > 0:
                    time.sleep(t_remaining)

                # GetData call with timing
                t_pre_get = time.perf_counter()
                self._device.GetData(
                    acq_frame_length, receive_buffer, receive_buffer_length
                )
                t_post_get = time.perf_counter()
                t_last_call = t_post_get

                # Adaptive timing: adjust wait based on GetData blocking time
                blocking_time_s = t_post_get - t_pre_get
                if blocking_time_s < self.BEHIND_THRESHOLD_S:
                    # Fast return = buffer had data = we're behind
                    wait_interval_s = self.WAIT_BEHIND_S
                    self._underrun_counter = 0
                else:
                    # Slow return = had to wait for data = we're on time
                    wait_interval_s = self.WAIT_ON_TIME_S
                    self._underrun_counter += 1

                if self._underrun_counter > self.NUM_UNDERRUNS_ALLOWED:
                    self.log(
                        "Buffer underrun - performance may lag.",
                        type=Constants.LogTypes.WARNING,
                    )
                    self._underrun_counter = 0

                # Convert to numpy array
                raw_data = np.frombuffer(
                    receive_buffer,
                    dtype=np.float32,
                    count=num_acquired_channels * acq_frame_length,
                ).reshape(acq_frame_length, num_acquired_channels)

                # Extract selected channels
                if use_slice:
                    frame_data = raw_data[:, :self.NUM_EEG_CHANNELS].copy()
                else:
                    frame_data = raw_data[:, col_indices].copy()

                # Set current frame and trigger pipeline cycle immediately
                self._current_frame = frame_data
                self.cycle()

            except Exception as e:
                if self._running:
                    self.log(
                        f"Data acquisition error: {e}",
                        type=Constants.LogTypes.WARNING,
                    )

        # Clean up receive buffer
        del receive_buffer

    @staticmethod
    def get_available_devices() -> list[str]:
        """Get list of available Unicorn Hybrid Black devices.

        Returns:
            List of device serial numbers that are available for connection.
            Returns empty list on non-Windows platforms.
        """
        if sys.platform != "win32":
            return []

        _ensure_unicorn_path()
        try:
            import UnicornPy

            device_list = UnicornPy.GetAvailableDevices(True)
            return device_list if device_list else []
        except ImportError:
            return []
        except Exception:
            return []
