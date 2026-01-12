from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.o_port import OPort
from .base.fixed_rate_source import FixedRateSource

#: Port identifier for signal output
OUT_PORT = ioc.Constants.Defaults.PORT_OUT


class Generator(FixedRateSource):
    """Signal generator source for creating synthetic test signals.

    Generates configurable test signals with optional noise for testing
    pipelines. Supports multiple waveforms (sine, rectangular, pulse) with
    multi-channel output.
    """

    #: Sinusoidal waveform signal shape
    SHAPE_SINUSOID = "sine"
    #: Square wave signal shape
    SHAPE_RECTANGULAR = "rect"
    #: Brief pulses signal shape
    SHAPE_PULSE = "pulse"

    #: Source code fingerprint for integrity verification
    FINGERPRINT = "468a05f0a39ca1b4b51d26604cfbd992"

    #: Default sampling rate in Hz
    DEFAULT_SAMPLING_RATE = 250.0
    #: Default number of channels
    DEFAULT_CHANNEL_COUNT = 8
    #: Default signal frequency in Hz
    DEFAULT_SIGNAL_FREQUENCY = 10.0
    #: Default signal shape
    DEFAULT_SIGNAL_SHAPE = SHAPE_SINUSOID
    #: Default signal amplitude
    DEFAULT_SIGNAL_AMPLITUDE = 0.0
    #: Default noise amplitude
    DEFAULT_NOISE_AMPLITUDE = 0.0

    class Configuration(FixedRateSource.Configuration):
        """Configuration class for Generator signal parameters."""

        class Keys(FixedRateSource.Configuration.Keys):
            """Configuration key constants for the Generator."""

            #: Signal frequency configuration key
            SIGNAL_FREQUENCY = "signal_frequency"
            #: Signal shape configuration key
            SIGNAL_SHAPE = "signal_shape"
            #: Signal amplitude configuration key
            SIGNAL_AMPLITUDE = "signal_amplitude"
            #: Noise amplitude configuration key
            NOISE_AMPLITUDE = "noise_amplitude"

    def __init__(
        self,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        signal_frequency: float = None,
        signal_shape: str = None,
        signal_amplitude: float = 0.0,
        noise_amplitude: float = 0.0,
        **kwargs,
    ):
        """Initialize signal generator.

        Args:
            sampling_rate: Sampling frequency in Hz.
            channel_count: Number of output channels. All get same signals.
            frame_size: Samples per output frame.
            signal_frequency: Signal frequency in Hz. Defaults to 10.0.
            signal_shape: Waveform shape (sine, rect, pulse). Defaults to sine.
            signal_amplitude: Peak amplitude of signal component.
            noise_amplitude: Standard deviation of Gaussian noise.
            **kwargs: Additional parameters for FixedRateSource.

        Raises:
            ValueError: If signal_frequency or noise_amplitude is negative,
                or signal_shape is unsupported.
        """
        # Set default values and validate parameters
        if sampling_rate is None:
            sampling_rate = self.DEFAULT_SAMPLING_RATE
        if channel_count is None:
            channel_count = self.DEFAULT_CHANNEL_COUNT
        if signal_frequency is None:
            signal_frequency = self.DEFAULT_SIGNAL_FREQUENCY
        if signal_frequency < 0:
            raise ValueError("signal_frequency must be positive.")
        if signal_shape is None:
            signal_shape = self.DEFAULT_SIGNAL_SHAPE
        if signal_amplitude is None:
            signal_amplitude = self.DEFAULT_SIGNAL_AMPLITUDE
        if noise_amplitude is None:
            noise_amplitude = self.DEFAULT_NOISE_AMPLITUDE
        if noise_amplitude < 0:
            raise ValueError("noise_amplitude must be positive.")

        # Configure output ports
        output_ports = kwargs.pop(
            Generator.Configuration.Keys.OUTPUT_PORTS, [OPort.Configuration()]
        )

        # Initialize parent FixedRateSource with all parameters
        FixedRateSource.__init__(
            self,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            decimation_factor=frame_size,
            signal_frequency=signal_frequency,
            signal_amplitude=signal_amplitude,
            signal_shape=signal_shape,
            noise_amplitude=noise_amplitude,
            output_ports=output_ports,
            **kwargs,
        )

        # Initialize time tracking for continuous signal generation
        self._time = 0.0
        # Initialize random number generator for noise generation
        self._rng = np.random.default_rng()

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Generate one frame of synthetic signal data.

        Creates a frame containing the configured waveform plus optional noise.
        Maintains phase continuity across frames through time tracking.

        Args:
            data: Input data dictionary (unused for signal generation).

        Returns:
            Output data dictionary with generated signal frame of shape
            (frame_size, channel_count), or None if not a decimation step.
        """
        # Check if this is a decimation step (frame generation timing)
        if not self.is_decimation_step():
            return None

        # Get configuration parameters
        config = self.config
        frame_size = config[self.Configuration.Keys.FRAME_SIZE][0]
        ch_count = config[self.Configuration.Keys.CHANNEL_COUNT][0]

        # Initialize output frame with zeros
        output = np.zeros((frame_size, ch_count), dtype=Constants.DATA_TYPE)

        # Create time vector for this frame
        dt = 1.0 / config[self.Configuration.Keys.SAMPLING_RATE]
        t = np.linspace(
            self._time, self._time + (frame_size - 1) * dt, frame_size
        )

        # Generate signal component if amplitude > 0
        freq = config[self.Configuration.Keys.SIGNAL_FREQUENCY]
        amp = config[self.Configuration.Keys.SIGNAL_AMPLITUDE]
        shape = config[self.Configuration.Keys.SIGNAL_SHAPE]

        if freq and amp > 0.0:
            # Generate waveform based on selected shape
            if shape == self.SHAPE_SINUSOID:
                # Smooth sinusoidal waveform
                wave = amp * np.sin(2 * np.pi * freq * t)
            elif shape == self.SHAPE_RECTANGULAR:
                # Square wave (sign of sine function)
                wave = amp * np.sign(np.sin(2 * np.pi * freq * t))
            elif shape == self.SHAPE_PULSE:
                # Brief pulses at specified frequency
                period = 1.0 / freq
                wave = np.zeros_like(t)
                for i, ti in enumerate(t):
                    # Generate pulse at start of each period
                    if (ti % period) < dt:
                        wave[i] = amp
            else:
                raise ValueError(f"Unsupported signal shape: {shape}")

            # Broadcast signal to all channels
            output += wave[:, np.newaxis]

        # Update internal time for next frame (maintains phase continuity)
        self._time += frame_size * dt

        # Add noise component if amplitude > 0
        noise_amp = config[self.Configuration.Keys.NOISE_AMPLITUDE]
        if noise_amp > 0.0:
            # Generate Gaussian noise for all channels
            noise = self._rng.standard_normal(size=output.shape) * noise_amp
            output += noise.astype(Constants.DATA_TYPE)

        return {OUT_PORT: output}
