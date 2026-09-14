from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ...common._private import channels
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Decimator(IONode):
    """Decimator node for downsampling data streams.

    Reduces data rate by outputting only every Nth sample based on decimation
    factor. Adjusts sampling rate and frame size in output context accordingly.

    Everything above the new Nyquist frequency is removed first. Dropping
    samples without that step folds the discarded band back into the
    retained one, where it is indistinguishable from real signal: at 250 Hz
    decimated by 50, everything from 2.5 Hz to 125 Hz lands on top of the
    band of interest. The filter runs on the measured-signal channels only,
    so an in-band master index or a trigger channel survives intact.
    """

    #: Cutoff as a fraction of the output Nyquist frequency. Below one so
    #: that the transition band fits below Nyquist rather than straddling
    #: it, which is where aliasing would reappear.
    CUTOFF_RATIO = 0.8
    #: Order of the anti-alias low-pass.
    FILTER_ORDER = 8

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            pass

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Whether to low-pass before dropping samples.
            ANTI_ALIAS = "anti_alias"

    def __init__(
        self,
        decimation_factor: int = 1,
        anti_alias: bool = True,
        **kwargs,
    ):
        """Initialize decimator with decimation factor.

        Args:
            decimation_factor: Factor by which to reduce data rate. Must be
                positive integer. Value of 1 means no decimation.
            anti_alias: Remove everything above the new Nyquist frequency
                before dropping samples. Only turn this off if the input is
                already band-limited below the output Nyquist frequency.
            **kwargs: Additional arguments for parent IONode.
        """
        super().__init__(
            decimation_factor=decimation_factor,
            anti_alias=anti_alias,
            **kwargs,
        )
        self._sos = None  # Anti-alias coefficients, built in setup
        self._zi = None  # Streaming filter state, one set per channel
        self._split = None  # Which channels the filter may touch
        self._n_seen = 0  # Input samples seen; carries the retained phase
        self._decimation_factor = decimation_factor

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output context with adjusted sampling rate and frame size.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts containing frame size and
                sampling rate information.

        Returns:
            Output port contexts with decimated sampling rate.

        Raises:
            ValueError: If frame_size is not provided, or is not exactly 1.
        """

        port_context_out = super().setup(data, port_context_in)
        frame_size = port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE]
        if frame_size is None:
            raise ValueError("frame_size must be provided in context.")

        M = self.config[self.Configuration.Keys.DECIMATION_FACTOR]

        # Decimation is a property of the SAMPLE stream, so it is counted
        # in samples here rather than in cycles.
        #
        # This node used to refuse every frame size but 1, because it
        # leaned on ioiocore's is_decimation_step(), which fires once
        # every decimation_factor *cycles* -- and a cycle carries
        # frame_size samples, so the two only coincide at frame_size 1.
        # An earlier round tried to reconcile them by rewriting
        # self.config's decimation_factor to decimation_factor //
        # frame_size; that is not idempotent, because setup() reruns per
        # run, so the second run divided what the first had already
        # divided (measured: M=5, frame_size=5 worked on run 1 and raised
        # on run 2) and the corruption reached serialize().
        #
        # Keeping an own sample phase costs one integer and removes the
        # restriction. decimation_factor stays exactly what the caller
        # passed, for any number of runs, and is still what the output
        # rate is derived from. ioiocore no longer decimates a second
        # time on top of what step() returns, which is what made this
        # possible.
        #
        # One constraint is real and survives: a cycle can emit at most
        # one output frame, so these are the only two shapes that give a
        # regular output cadence.
        if M >= frame_size:
            # At most one retained sample can fall inside a frame.
            frame_size_out = 1
        elif frame_size % M == 0:
            # Every frame contributes the same number of samples.
            frame_size_out = frame_size // M
        else:
            raise ValueError(
                f"Decimator cannot emit a regular frame from "
                f"frame_size {frame_size} at decimation_factor {M}: a "
                f"frame would yield {frame_size / M} samples. Use a "
                f"decimation_factor of at least {frame_size}, or one "
                f"that divides {frame_size} exactly."
            )

        #: Input samples seen, so the retained phase survives frames of
        #: any size. Reset here because setup() reruns per run.
        self._n_seen = 0
        self._decimation_factor = M

        port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE] = frame_size_out
        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rate_in = port_context_in[PORT_IN][sr_key]
        sampling_rate_out = sampling_rate_in / M
        port_context_out[PORT_OUT][sr_key] = sampling_rate_out

        self._split = channels.SignalSplit(port_context_in[PORT_IN])
        self._sos = None
        self._zi = None
        if M > 1 and self.config.get(
            self.Configuration.OptionalKeys.ANTI_ALIAS, True
        ):
            n_signal = len(self._split.signal)
            if self._split.all_signal:
                n_signal = port_context_out[PORT_OUT][
                    Constants.Keys.CHANNEL_COUNT
                ]
            if n_signal:
                cutoff = self.CUTOFF_RATIO * sampling_rate_out / 2.0
                self._sos = butter(
                    self.FILTER_ORDER,
                    cutoff,
                    btype="lowpass",
                    output="sos",
                    fs=sampling_rate_in,
                )
                # (sections, 2, channels) for filtering along axis 0. The
                # state is left unscaled until the first sample arrives,
                # so the filter starts from the signal's own level rather
                # than from an assumed unit step.
                self._zi = np.repeat(
                    sosfilt_zi(self._sos)[:, :, None], n_signal, axis=2
                )
                self._primed = False
        return port_context_out

    def step(self, data: dict):
        """Process one step of decimation.

        The anti-alias filter runs on every incoming frame, not only on
        the frames that are emitted: a streaming filter carries state, and
        skipping the discarded samples would leave that state describing a
        signal that was never presented.

        Args:
            data: Input data dictionary containing data to be decimated.

        Returns:
            Dictionary with last sample of input data if decimation step,
            None otherwise.
        """
        block = data[PORT_IN]
        if self._sos is not None:
            operated = self._split.take(block)
            if not self._primed:
                # Anchor the steady-state template on the first sample, so
                # a stream that starts at a constant offset does not decay
                # from an assumed 1.0 towards it.
                self._zi = self._zi * operated[0, :]
                self._primed = True
            filtered, self._zi = sosfilt(
                self._sos, operated, axis=0, zi=self._zi
            )
            block = self._split.merge(block, filtered)

        # Which rows of this block are the retained ones. A sample at
        # absolute index i is kept when (i + 1) % M == 0 -- the same
        # phase is_decimation_step() applied when a cycle was a sample,
        # so frame_size 1 behaves exactly as it always did.
        M = self._decimation_factor
        rows = block.shape[0]
        first = (M - 1 - self._n_seen) % M
        self._n_seen += rows
        if first >= rows:
            return None
        return {PORT_OUT: block[first::M, :]}
