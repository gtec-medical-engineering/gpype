from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.o_port import OPort
from .base.fixed_rate_source import FixedRateSource

# Port identifier for signal output
OUT_PORT = ioc.Constants.Defaults.PORT_OUT


class Generator(FixedRateSource):
    """
    Signal generator source for creating synthetic test signals.

    This class generates configurable test signals with optional noise for
    testing and development of BCI pipelines. It supports multiple waveform
    types including sinusoidal, rectangular, and pulse signals.

    The generator can produce multi-channel signals with identical waveforms
    across all channels, making it useful for testing filters, analyzing
    pipeline performance, and creating controlled test scenarios.

    Features:
        - Multiple signal shapes (sine, rectangular, pulse)
        - Configurable frequency, amplitude, and noise levels
        - Multi-channel output support
        - Frame-based output compatible with g.Pype architecture
    """

    # Signal shape constants
    SHAPE_SINUSOID = "sine"  # Sinusoidal waveform
    SHAPE_RECTANGULAR = "rect"  # Square wave
    SHAPE_PULSE = "pulse"  # Brief pulses at specified frequency

    # Source code fingerprint
    FINGERPRINT = "da7c08f1537b99baea5a47cfe8c1b47a"

    class Configuration(FixedRateSource.Configuration):
        """Configuration class for Generator signal parameters."""

        class Keys(FixedRateSource.Configuration.Keys):
            """Configuration key constants for the Generator."""

            SIGNAL_FREQUENCY = "signal_frequency"
            SIGNAL_SHAPE = "signal_shape"
            SIGNAL_AMPLITUDE = "signal_amplitude"
            NOISE_AMPLITUDE = "noise_amplitude"

    def __init__(
        self,
        sampling_rate: float,
        channel_count: int,
        frame_size: int = None,
        signal_frequency: float = None,
        signal_shape: str = None,
        signal_amplitude: float = 0.0,
        noise_amplitude: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the signal generator.

        Args:
            sampling_rate (float): Sampling frequency in Hz for signal
                generation and output timing.
            channel_count (int): Number of output channels. All channels
                will receive identical generated signals.
            frame_size (int, optional): Number of samples per output frame.
                Affects processing latency and computational efficiency.
            signal_frequency (float, optional): Frequency of the generated
                signal in Hz. Defaults to 10.0 Hz if None.
            signal_shape (str, optional): Shape of the generated waveform.
                Must be one of SHAPE_SINUSOID, SHAPE_RECTANGULAR, or
                SHAPE_PULSE. Defaults to SHAPE_SINUSOID if None.
            signal_amplitude (float): Peak amplitude of the signal component.
                Set to 0.0 for noise-only output.
            noise_amplitude (float): Standard deviation of Gaussian noise
                added to the signal. Set to 0.0 for noise-free output.
            **kwargs: Additional configuration parameters passed to
                FixedRateSource.

        Raises:
            ValueError: If signal_frequency or noise_amplitude is negative,
                or if signal_shape is not a supported waveform type.
        """
        # Set default values and validate parameters
        if signal_frequency is None:
            signal_frequency = 10.0
        if signal_frequency < 0:
            raise ValueError("signal_frequency must be positive.")

        if signal_shape is None:
            signal_shape = self.SHAPE_SINUSOID

        if signal_amplitude is None:
            signal_amplitude = 0.0

        if noise_amplitude is None:
            noise_amplitude = 0.0
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
        """
        Generate one frame of synthetic signal data.

        This method implements the core signal generation logic, creating
        a frame of data containing the configured waveform plus optional
        noise. The signal maintains phase continuity across frames through
        internal time tracking.

        Args:
            data (dict): Input data dictionary (unused for signal generation
                but required by the FixedRateSource interface).

        Returns:
            dict: Output data dictionary with OUT_PORT key containing the
                generated signal frame of shape (frame_size, channel_count).
                Returns None if not a decimation step.

        Note:
            The generated signal is identical across all channels. Time
            continuity is maintained between frames for seamless waveforms.
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
