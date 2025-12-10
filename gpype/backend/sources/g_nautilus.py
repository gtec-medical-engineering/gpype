from __future__ import annotations

import sys
import threading

import numpy as np

from ...common.constants import Constants
from .base.amplifier_source import AmplifierSource

# Platform check - g.Nautilus is only supported on Windows
if sys.platform != "win32":
    raise NotImplementedError("This module is only supported on Windows.")

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN


class GNautilus(AmplifierSource):
    """g.Nautilus EEG amplifier interface for real-time data acquisition.

    Interface to g.tec's g.Nautilus wireless EEG amplifier system. Handles
    device initialization, data streaming, and electrode impedance monitoring.
    Requires g.tec GDS library and Windows operating system.
    """

    #: Source code fingerprint for licensing verification
    FINGERPRINT = "4012ac0540d2ff22f6d87cda7cf3c167"

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for g.Nautilus amplifier parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration key constants for the g.Nautilus amplifier."""

            #: Configuration key for amplifier sensitivity setting
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
        """Initialize g.Nautilus amplifier interface.

        Args:
            serial: Device serial number. Uses first available if None.
            sampling_rate: Sampling frequency in Hz.
            channel_count: Number of EEG channels to acquire.
            frame_size: Samples per data frame.
            sensitivity: Amplifier sensitivity setting.
            enable_di: Enable digital input channel for triggers.
            **kwargs: Additional parameters for AmplifierSource.

        Raises:
            RuntimeError: If GDS library unavailable or device init fails.
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

        #: Electrode impedance values in kOhms (-10 indicates unknown)
        self._z = np.ones(channel_count) * (-10)

        #: g.Nautilus device interface instance
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

        #: Flag indicating if impedance monitoring is active
        self._impedance_check_running = False
        #: Flag indicating if impedance data has been updated
        self._impedance_fresh = True

    def start(self) -> None:
        """Start g.Nautilus data acquisition.

        Initiates hardware data streaming and activates the amplifier for
        real-time EEG data processing.
        """
        # Start hardware data acquisition
        self._device.start()
        # Start parent source processing
        super().start()

    def stop(self):
        """Stop g.Nautilus data acquisition and cleanup resources.

        Stops hardware streaming and ensures proper shutdown of amplifier
        connection.
        """
        # Stop hardware data acquisition
        self._device.stop()
        # Stop parent source processing
        super().stop()
        # Clean up device resources
        del self._device

    def start_impedance_check(self) -> None:
        """Start electrode impedance monitoring in background thread.

        Initiates continuous impedance measurement for all electrodes.
        Provides real-time feedback on electrode contact quality.
        """
        # Start the impedance retrieval in a background thread
        self._impedance_check_running = True
        self._impedance_thread = threading.Thread(
            target=self._get_z_thread, daemon=True
        )
        self._impedance_thread.daemon = True
        self._impedance_thread.start()

    def stop_impedance_check(self):
        """Stop electrode impedance monitoring and cleanup thread.

        Stops background impedance measurement thread and waits for completion.
        """
        # Signal thread to stop
        self._impedance_check_running = False
        # Wait for thread completion if it exists
        if self._impedance_thread:
            self._impedance_thread.join()

    def get_impedance(self):
        """Get current electrode impedance values and freshness status.

        Returns:
            tuple: (impedance_array, is_fresh)
                - impedance_array: Impedance values in kOhms per electrode.
                    -10 indicates unknown/disconnected.
                - is_fresh: True if data updated since last call.
        """
        # Get current freshness status and mark as read
        imp_fresh = self._impedance_fresh
        self._impedance_fresh = False
        return self._z, imp_fresh

    def _data_callback(self, data: np.ndarray):
        """Handle incoming data from g.Nautilus device.

        Callback invoked by GDS library when new EEG data is available.
        Forwards data through g.Pype pipeline using cycle mechanism.

        Args:
            data: Raw EEG data with shape (frame_size, channel_count).
        """
        # Forward data through the pipeline using the input port
        self.cycle(data={PORT_IN: data})

    def _get_z_thread(self):
        """Background thread function for continuous impedance monitoring.

        Periodically retrieves electrode impedance values from device. Runs
        until _impedance_check_running is set to False.
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
        """Process one step of data through g.Nautilus source.

        Args:
            data: Input data dictionary with PORT_IN key containing EEG data.

        Returns:
            Output data dictionary with PORT_OUT key containing EEG data.
        """
        # Pass through data from input to output port
        return {PORT_OUT: data[PORT_IN]}
