from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ....common.constants import Constants
from ...core._private.timeline import MasterPriority
from ...core.o_port import OPort
from .source import Source

#: Default output port identifier
OUT_PORT = Constants.Defaults.PORT_OUT


class FixedRateSource(Source):
    """Fixed-rate source for continuous data generation at sampling rate.

    Generates data at fixed sampling rate using background thread with
    precise timing control and drift compensation.

    **The pacing thread runs one cycle per sample, not per frame.**
    A subclass whose ``step()`` returns ``frame_size`` samples per call
    must therefore set ``decimation_factor=frame_size``, so that
    ``step()`` runs on every ``frame_size``-th cycle and the stream
    leaves at its declared rate. A subclass that omits this emits
    ``frame_size`` times too much data, and nothing reports it: the
    frames are individually well-formed, the declared sampling rate is
    unchanged, and only wall-clock measurement reveals it. Measured on
    a reader before this was applied: 1.01x at frame_size 1, 5.02x at
    5, 24.98x at 25, 50.03x at 50.

    Every subclass here does this -- ``_GeneratorCore``,
    ``_CsvReaderCore`` and ``RecordingReader`` -- and a fourth must too.
    Two exemptions, both real:

    * **Batch mode.** A batch reader starts no pacing thread; the driver
      calls ``cycle()`` once for the whole recording. A decimation
      factor there would make that single cycle emit nothing.
    * **Externally driven sources.** An amplifier is clocked by its own
      device callback rather than by this thread, so it emits one frame
      per callback and needs a factor of one. Those extend
      :class:`~gpype.backend.sources.base.source.Source` directly.
    """

    class Configuration(Source.Configuration):
        """Configuration class for FixedRateSource parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for fixed-rate source settings."""

            #: Configuration key for sampling rate in Hz
            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE

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
        """Initialize fixed-rate source with sampling configuration.

        Args:
            sampling_rate: Sampling rate in Hz for data generation.
            output_ports: List of output port configurations. Defaults to
                default port configuration if None.
            **kwargs: Additional arguments for parent Source class.
        """
        # Initialize parent source with configuration
        Source.__init__(
            self,
            sampling_rate=sampling_rate,
            output_ports=output_ports,
            **kwargs,
        )

        # Initialize threading components
        #: Flag indicating if the source is currently running
        self._running: bool = False
        #: Background thread for continuous data generation
        self._thread: Optional[threading.Thread] = None
        #: Start time for precise timing calculations
        self._time_start: Optional[float] = None

    def start(self):
        """Start fixed-rate source and begin data generation.

        Initializes parent source and starts background thread for continuous
        data generation at specified sampling rate.
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
        """Stop fixed-rate source and terminate data generation.

        Signals background thread to stop and waits for completion.
        """
        # Stop parent source first
        Source.stop(self)

        # Forget the pacing anchor. It is set only when None, so a
        # second run would keep run 1's anchor, see a negative sleep
        # time, and free-run to catch up on every second that has
        # passed since the FIRST start. Measured after a 6 s gap:
        # 1750 samples, seven seconds of nominal signal, emitted in
        # 0.55 s of wall time -- and once a restart re-runs setup()
        # that burst is no longer discarded but recorded.
        self._time_start = None

        # Signal thread to stop and wait for completion
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=0.5)  # Wait up to 500ms

    def _thread_function(self):
        """Background thread function for fixed-rate data generation.

        Runs continuously generating data at precise intervals using absolute
        timing to prevent cumulative drift. Handles timing delays gracefully.

        One cycle per **sample**. A subclass emitting whole frames per
        ``step()`` compensates with ``decimation_factor=frame_size`` --
        see the class docstring, which explains what happens when it
        does not.
        """
        # Get configured sampling rate
        rate = self.config[FixedRateSource.Configuration.Keys.SAMPLING_RATE]

        # A source that is replaying stored data may want to run faster
        # than the clock, or as fast as the machine allows. Live sources
        # leave this at one and are paced exactly as before.
        speed = getattr(self, "_speed", 1.0)
        if not speed:
            while self._running:
                self.cycle()
            return
        rate = rate * speed

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

    def master_candidacy(self):
        """Claim as a counted stream.

        Contiguous by construction and with a declared rate, so it is a
        valid timeline -- but not a loss-aware one, since a source that
        reports no counter cannot say what it dropped. A numbered device
        therefore outranks it.

        Returns:
            ``(rate, COUNTED)``, or None if no usable rate is set.
        """
        rate = self.scalar(
            self.config.get(self.Configuration.Keys.SAMPLING_RATE)
        )
        if not rate or rate <= 0:
            return None
        return float(rate), MasterPriority.COUNTED

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with sampling rate information.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with sampling_rate information.
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
