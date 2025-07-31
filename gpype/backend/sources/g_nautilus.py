from __future__ import annotations

import sys
import threading

import numpy as np

from ...common.constants import Constants
from .base.amplifier_source import AmplifierSource

# Platform check - g.Nautilus is only supported on Windows
if sys.platform != "win32":
    raise NotImplementedError("This module is only supported on Windows.")

# Port identifiers for data flow
PORT_OUT = Constants.Defaults.PORT_OUT
PORT_IN = Constants.Defaults.PORT_IN


class GNautilus(AmplifierSource):
    """
    g.Nautilus EEG amplifier interface for real-time data acquisition.

    This class provides an interface to g.tec's g.Nautilus wireless EEG
    amplifier system. It handles device initialization, data streaming,
    and electrode impedance monitoring.

    Features:
        - Real-time EEG data acquisition
        - Configurable sampling rates and channel counts
        - Electrode impedance monitoring
        - Digital input channel support
        - Wireless connectivity

    Note:
        Requires g.tec GDS library and Windows operating system.
        The device must be paired and configured via g.tec software.
    """

    # Source code fingerprint
    FINGERPRINT = "80a8c8db4bb8a13cb4a71adeb50f310b"

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for g.Nautilus amplifier parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration key constants for the g.Nautilus amplifier."""

            SENSITIVITY = "sensitivity"

    def __init__(
        self,
        serial: str = None,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        sensitivity: float = None,
        enable_di: bool = False,
        **kwargs,
    ):
        """
        Initialize the g.Nautilus amplifier interface.

        Args:
            serial (str, optional): Device serial number for identification.
                If None, connects to first available device.
            sampling_rate (float, optional): Sampling frequency in Hz.
            channel_count (int, optional): Number of EEG channels to acquire.
            frame_size (int, optional): Number of samples per data frame.
                Affects latency and processing efficiency.
            sensitivity (float, optional): Amplifier sensitivity setting.
                Determines input voltage range.
            enable_di (bool): Enable digital input channel for triggers.
                Adds one additional channel for event markers.
            **kwargs: Additional configuration parameters passed to
                AmplifierSource.

        Raises:
            RuntimeError: If GDS library is not available or device
                initialization fails.
        """
        # Import gtec_gds only when actually needed (lazy import)
        try:
            import gtec_gds as gds
        except ImportError as e:
            raise RuntimeError(
                f"GDS library not available: {e}. "
                "This may be expected in CI environments where the GDS "
                "library is not installed."
            ) from e

        # Initialize impedance values to -10 kOhm (unknown state)
        self._z = np.ones(channel_count) * (-10)

        # Create and configure the g.Nautilus device
        self._device = gds.GNautilus(
            serial=serial,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            sensitivity=sensitivity,
            enable_di=enable_di,
        )

        # Update parameters with actual device configuration
        channel_count = self._device.channel_count
        sensitivity = self._device.sensitivity

        # Add digital input channel if enabled
        if enable_di:
            channel_count += 1

        # Set up data callback for real-time streaming
        self._device.set_data_callback(self._data_callback)

        # Initialize parent AmplifierSource with final configuration
        super().__init__(
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            sensitivity=sensitivity,
            enable_di=enable_di,
            **kwargs,
        )

        # Initialize impedance monitoring state
        self._impedance_check_running = False
        self._impedance_fresh = True

    def start(self) -> None:
        """
        Start the g.Nautilus data acquisition.

        Initiates the hardware data streaming and activates the amplifier.
        The device will begin acquiring EEG data and triggering data
        callbacks for real-time processing.
        """
        # Start hardware data acquisition
        self._device.start()
        # Start parent source processing
        super().start()

    def stop(self):
        """
        Stop the g.Nautilus data acquisition and cleanup resources.

        Stops the hardware streaming, cleans up device resources, and
        ensures proper shutdown of the amplifier connection.
        """
        # Stop hardware data acquisition
        self._device.stop()
        # Stop parent source processing
        super().stop()
        # Clean up device resources
        del self._device

    def start_impedance_check(self) -> None:
        """
        Start electrode impedance monitoring in a background thread.

        Initiates continuous impedance measurement for all electrodes.
        This provides real-time feedback on electrode contact quality,
        which is crucial for obtaining good EEG signals. Impedance values
        are updated periodically and can be retrieved with get_impedance().

        The impedance check runs in a separate daemon thread to avoid
        blocking the main data acquisition process.
        """
        # Start the impedance retrieval in a background thread
        self._impedance_check_running = True
        self._impedance_thread = threading.Thread(
            target=self._get_z_thread, daemon=True
        )
        self._impedance_thread.daemon = True
        self._impedance_thread.start()

    def stop_impedance_check(self):
        """
        Stop electrode impedance monitoring and cleanup the thread.

        Stops the background impedance measurement thread and waits for
        it to complete. Call this before stopping the main data acquisition
        to ensure clean shutdown.
        """
        # Signal thread to stop
        self._impedance_check_running = False
        # Wait for thread completion if it exists
        if self._impedance_thread:
            self._impedance_thread.join()

    def get_impedance(self):
        """
        Get current electrode impedance values and freshness status.

        Returns the most recent impedance measurements for all electrodes.
        The freshness flag indicates whether new impedance data is available
        since the last call to this method.

        Returns:
            tuple: (impedance_array, is_fresh)
                - impedance_array (np.ndarray): Impedance values in kOhms
                    for each electrode. -10 indicates unknown/disconnected.
                - is_fresh (bool): True if impedance data has been updated
                    since last call, False otherwise.

        Note:
            Impedance values < 5 kOhm are typically considered good for EEG.
            Values > 25 kOhm may indicate poor electrode contact.
        """
        # Get current freshness status and mark as read
        imp_fresh = self._impedance_fresh
        self._impedance_fresh = False
        return self._z, imp_fresh

    def _data_callback(self, data: np.ndarray):
        """
        Handle incoming data from the g.Nautilus device.

        This callback is invoked by the GDS library whenever new EEG data
        is available from the amplifier. It forwards the data through the
        g.Pype pipeline using the cycle mechanism.

        Args:
            data (np.ndarray): Raw EEG data from the amplifier.
                Shape: (frame_size, channel_count)
                Data type typically float32 or float64.
        """
        # Forward data through the pipeline using the input port
        self.cycle(data={PORT_IN: data})

    def _get_z_thread(self):
        """
        Background thread function for continuous impedance monitoring.

        This function runs in a separate thread to periodically retrieve
        electrode impedance values from the g.Nautilus device. The first
        measurement may take longer as it initializes the impedance
        measurement circuit.

        The thread continues until _impedance_check_running is set to False.
        Each measurement updates the _z array and sets the freshness flag.
        """
        # First impedance measurement requires initialization
        first = True
        while self._impedance_check_running:
            # Get impedance values from device
            self._z = self._device.get_impedance(first)
            first = False
            # Mark impedance data as fresh/updated
            self._impedance_fresh = True

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Process one step of data through the g.Nautilus source.

        This method implements the core data processing step for the
        AmplifierSource interface. It simply passes through the input
        data from PORT_IN to PORT_OUT without modification.

        Args:
            data (dict): Input data dictionary with PORT_IN key containing
                EEG data array from the amplifier.

        Returns:
            dict: Output data dictionary with PORT_OUT key containing
                the same EEG data array for downstream processing.
        """
        # Pass through data from input to output port
        return {PORT_OUT: data[PORT_IN]}
