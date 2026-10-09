from __future__ import annotations

import time
from typing import Optional

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common._private.naming import public_name
from ...common.constants import Constants
from ..core.o_port import OPort
from .base.fixed_rate_source import FixedRateSource

#: Port identifier for signal output
OUT_PORT = ioc.Constants.Defaults.PORT_OUT


def _check_signal_shape(signal_shape: str) -> None:
    """Refuse a waveform this node cannot produce.

    The shape used to be checked in step(), and only on the branch taken
    when the signal is audible at all -- ``if freq and amp > 0.0``. Since
    the default amplitude is 0.0, a document naming a nonsense shape ran
    silently and indefinitely, reporting nothing. The constructor is
    where an author can still act on it.

    None is left alone: it means "not supplied".

    Args:
        signal_shape: The requested shape, or None if not supplied.

    Raises:
        ValueError: If the shape is given and is not one of
            SIGNAL_SHAPES.
    """
    if signal_shape is None:
        return
    if signal_shape not in Generator.SIGNAL_SHAPES:
        accepted = ", ".join(repr(s) for s in Generator.SIGNAL_SHAPES)
        raise ValueError(
            f"signal_shape must be one of {accepted}; got "
            f"{signal_shape!r}."
        )


class Generator(FixedRateSource):
    """Signal generator for creating synthetic test signals.

    Generates configurable test signals with optional noise for testing
    pipelines. Supports multiple waveforms (sine, rectangular, pulse) with
    multi-channel output; every channel carries the same signal.
    """

    #: Sinusoidal waveform signal shape
    SHAPE_SINUSOID = "sine"
    #: Square wave signal shape
    SHAPE_RECTANGULAR = "rect"
    #: Brief pulses signal shape
    SHAPE_PULSE = "pulse"

    #: Every shape this node can produce. Named, and named with the
    #: parameter it constrains, because the set had no representation at
    #: all before: the shapes existed as three separate constants and the
    #: only thing that knew which were valid was an if/elif chain in
    #: step(). A form offering a dropdown, or a generated catalog, has
    #: nowhere else to read this from.
    SIGNAL_SHAPES = (SHAPE_SINUSOID, SHAPE_RECTANGULAR, SHAPE_PULSE)

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
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the signal generator.

        Args:
            sampling_rate: Sampling frequency in Hz.
            channel_count: Number of output channels. All get same signals.
            frame_size: Samples per output frame.
            signal_frequency: Signal frequency in Hz. Defaults to 10.0.
            signal_shape: Waveform shape (sine, rect, pulse). Defaults to sine.
            signal_amplitude: Peak amplitude of signal component.
            noise_amplitude: Standard deviation of Gaussian noise.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
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
        _check_signal_shape(signal_shape)
        if signal_amplitude is None:
            signal_amplitude = self.DEFAULT_SIGNAL_AMPLITUDE
        if noise_amplitude is None:
            noise_amplitude = self.DEFAULT_NOISE_AMPLITUDE
        if noise_amplitude < 0:
            raise ValueError("noise_amplitude must be positive.")
        frame_size = kwargs.pop(
            Generator.Configuration.Keys.FRAME_SIZE, frame_size
        )
        # Resolved before it is reused below. Left as None, the frame size
        # still defaulted further down while decimation_factor kept the
        # None and defaulted to 1 -- so the source emitted one sample per
        # cycle while declaring frames of Defaults.FRAME_SIZE. That was
        # invisible while the default was 1 and the two happened to
        # agree; at 10 it put every placed observation out of reach and
        # nothing was drawn.
        if frame_size is None:
            # Derived from the rate, so one frame spans roughly the same
            # stretch of time whatever the rate: 250 Hz gives 4, 500
            # gives 8, 2400 gives 32. Resolved here rather than left to
            # Source, because decimation_factor below has to be the same
            # number -- FixedRateSource paces this source per sample, so
            # it emits one frame every frame_size cycles.
            frame_size = Generator.frame_size_for(sampling_rate)
        decimation_factor = frame_size
        decimation_factor = kwargs.pop(
            Generator.Configuration.Keys.DECIMATION_FACTOR,
            decimation_factor,
        )

        # Configure output ports
        # None means "the default port", as it did while Generator was a
        # chain deriving its port from the node inside.
        output_ports = kwargs.pop(
            Generator.Configuration.Keys.OUTPUT_PORTS, None
        ) or [OPort.Configuration()]

        # Initialize parent FixedRateSource with all parameters
        FixedRateSource.__init__(
            self,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            decimation_factor=decimation_factor,
            signal_frequency=signal_frequency,
            signal_amplitude=signal_amplitude,
            signal_shape=signal_shape,
            noise_amplitude=noise_amplitude,
            output_ports=output_ports,
            edge_id=edge_id,
            **kwargs,
        )

        # Initialize time tracking for continuous signal generation
        self._time = 0.0
        # Initialize random number generator for noise generation
        self._rng = np.random.default_rng()

    #: One channel beyond the configured count, carrying the instant
    #: each frame was produced.
    #:
    #: A generator numbers nothing, so Sync numbers it by counting
    #: arrivals -- and under distributed residency that Sync runs in the
    #: server process, counting arrivals off a socket. Pairing those
    #: positions with a locally read clock fits the master timeline to
    #: the transport rather than to the signal. The stamp is what makes
    #: the two residencies the same measurement.
    NUM_TIMELINE_CHANNELS = 1

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Declare the generated channels and the timestamp beside them.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Output port contexts, describing one timestamp channel after
            the signal channels.
        """
        port_context_out = super().setup(data, port_context_in)
        n_signal = self.config[self.Configuration.Keys.CHANNEL_COUNT][0]
        roles = [Constants.ChannelRoles.SIGNAL] * n_signal
        roles += [Constants.ChannelRoles.TIMESTAMP]
        for context in port_context_out.values():
            # Declared on the port, not in the configuration: the
            # frame step() emits is this wide, while the number the
            # author asked for stays what it was and survives a
            # round trip unchanged.
            context[Constants.Keys.CHANNEL_COUNT] = (
                n_signal + self.NUM_TIMELINE_CHANNELS
            )
            # Roles only. Labelling the signal channels here would
            # make every generator pipeline a labelled one, and an
            # unlabelled pipeline is meant to stay unlabelled; the
            # stamp is found by role, never by name.
            context.update(channels.describe(roles))
        return port_context_out

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
        n_signal = config[self.Configuration.Keys.CHANNEL_COUNT][0]

        # Initialize output frame with zeros
        output = np.zeros((frame_size, n_signal), dtype=Constants.DATA_TYPE)

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
                # Unreachable through the constructor, which validates
                # signal_shape (see _check_signal_shape). It stays for a
                # configuration mutated after construction, and says
                # what that check says.
                accepted = ", ".join(repr(s) for s in self.SIGNAL_SHAPES)
                raise ValueError(
                    f"signal_shape must be one of {accepted}; "
                    f"{public_name(type(self).__name__)} got {shape!r} "
                    f"while generating a frame."
                )

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

        # Stamped as the frame leaves, which for a generator is
        # when it was produced: there is no acquisition to lag
        # behind. One value for the whole frame, read back off the
        # last row, matching how Sync observes once per frame.
        stamp = np.full(
            (frame_size, 1),
            channels.wrap_time(channels.stamp_clock()),
            dtype=Constants.DATA_TYPE,
        )
        return {OUT_PORT: np.hstack((output, stamp))}
