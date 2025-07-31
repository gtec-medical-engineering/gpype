from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ....common.constants import Constants
from ...core.o_port import OPort
from .source import Source

# Convenience constant for default output port name
OUT_PORT = Constants.Defaults.PORT_OUT


class FixedRateSource(Source):
    """Fixed-rate source for continuous data generation at sampling rate.

    This class implements a source that generates data at a fixed sampling rate
    using a background thread. It maintains precise timing by tracking expected
    sample times and adjusting sleep periods to compensate for delays.

    Features:
    - Precise timing control with drift compensation
    - Background threading for non-blocking operation
    - Configurable sampling rate validation
    - Automatic context propagation with sampling rate information

    Attributes:
        _running: Flag indicating if the background thread is active
        _thread: Background thread for data generation
        _time_start: Timestamp when data generation started

    Note:
        The source uses absolute timing to prevent cumulative drift and
        handles timing lag gracefully by skipping excessive sleep periods.
    """

    class Configuration(Source.Configuration):
        """Configuration class for FixedRateSource parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for fixed-rate source settings."""

            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE  # 'sampling_rate'

        def __init__(self, sampling_rate: float, **kwargs):
            """Initialize configuration with sampling rate validation.

            Args:
                sampling_rate: Sampling rate in Hz. Must be positive.
                **kwargs: Additional configuration parameters.

            Raises:
                ValueError: If sampling_rate is not positive.
            """
            if sampling_rate <= 0:
                raise ValueError("sampling_rate must be greater than zero.")
            super().__init__(sampling_rate=sampling_rate, **kwargs)

    def __init__(
        self,
        sampling_rate: float,
        output_ports: Optional[list[OPort.Configuration]] = None,
        **kwargs,
    ):
        """Initialize the fixed-rate source with sampling configuration.

        Args:
            sampling_rate: Sampling rate in Hz for data generation.
                Must be positive. Higher rates require more precise timing.
            output_ports: List of output port configurations. If None,
                default port configuration will be used.
            **kwargs: Additional arguments passed to parent Source class.
        """
        # Initialize parent source with configuration
        Source.__init__(
            self,
            sampling_rate=sampling_rate,
            output_ports=output_ports,
            **kwargs,
        )

        # Initialize threading components
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._time_start: Optional[float] = None

    def start(self):
        """Start the fixed-rate source and begin data generation.

        Initializes the parent source and starts the background thread
        for continuous data generation at the specified sampling rate.

        Note:
            The background thread is created as a daemon thread to ensure
            clean shutdown when the main program exits. Data generation
            begins immediately after the thread starts.
        """
        # Start parent source first
        Source.start(self)

        # Start background thread if not already running
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._thread_function, daemon=True
            )
            self._thread.start()

    def stop(self):
        """Stop the fixed-rate source and terminate data generation.

        Signals the background thread to stop and waits for it to complete.
        The parent stop method handles the main cleanup.

        Note:
            The method waits up to 500ms for the thread to join. After this
            timeout, the thread will be abandoned but should terminate
            shortly due to daemon status.
        """
        # Stop parent source first
        Source.stop(self)

        # Signal thread to stop and wait for completion
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)  # Wait up to 500ms

    def _thread_function(self):
        """Background thread function for fixed-rate data generation.

        This method runs continuously in a background thread, generating
        data at precise intervals determined by the sampling rate. It uses
        absolute timing to prevent cumulative drift and handles timing
        delays gracefully.

        The timing algorithm:
        1. Calculate expected time for next sample based on sample count
        2. Compare with current time to determine sleep duration
        3. Sleep only if duration is > 1ms (avoid inaccurate short sleeps)
        4. Handle timing lag by proceeding immediately when behind schedule

        Note:
            Uses time.time() for wall-clock timing and maintains sample
            count to calculate absolute expected times, preventing drift
            accumulation that would occur with relative timing.
        """
        # Get configured sampling rate
        rate = self.config[FixedRateSource.Configuration.Keys.SAMPLING_RATE]

        # Initialize start time if not set
        if self._time_start is None:
            self._time_start = time.time()

        # Initialize timing variables
        sample_count = 0
        expected_next_sample_time = self._time_start

        while self._running:
            # Calculate the absolute time for the next sample
            sample_count += 1
            expected_next_sample_time = self._time_start + sample_count / rate

            # Calculate how long to sleep until next sample
            current_time = time.time()
            sleep_time = expected_next_sample_time - current_time

            # Handle timing control
            if sleep_time > 0.001:
                # Sleep period is significant, wait for proper timing
                time.sleep(sleep_time)
            elif sleep_time < 0:
                # We are lagging behind schedule - proceed immediately
                # This handles cases where processing takes longer than
                # the sample period, preventing infinite catch-up loops
                pass
            # For 0 <= sleep_time <= 0.001, proceed immediately to avoid
            # inaccurate short sleeps that can cause timing jitter

            # Generate next sample
            self.cycle()

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with sampling rate information.

        Configures output port contexts by adding sampling rate information
        to each configured output port. This ensures downstream nodes
        receive accurate timing information for their processing.

        Args:
            data: Dictionary of input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with sampling_rate information
            added to each configured output port.

        Note:
            The sampling rate is propagated to all output ports, enabling
            downstream nodes to perform sample-rate dependent processing
            such as filtering, windowing, and frequency analysis.
        """
        # Call parent setup to initialize base contexts
        port_context_out = super().setup(data, port_context_in)

        # Get configuration parameters
        sampling_rate = self.config[self.Configuration.Keys.SAMPLING_RATE]
        frame_size = port_context_out[Constants.Defaults.PORT_OUT][
            Constants.Keys.FRAME_SIZE
        ]
        frame_rate = sampling_rate / frame_size
        out_ports = self.config[self.Configuration.Keys.OUTPUT_PORTS]

        # Add sampling rate context to each output port
        for i in range(len(out_ports)):
            # Create context with sampling rate information
            context = {
                Constants.Keys.SAMPLING_RATE: sampling_rate,
                Constants.Keys.FRAME_RATE: frame_rate,
            }

            # Get port name and update its context
            port_name = out_ports[i][OPort.Configuration.Keys.NAME]
            port_context_out[port_name].update(context)

        return port_context_out
