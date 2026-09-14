from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos

from ....common._private import channels
from ....common.constants import Constants
from ...core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Butterworth(IONode):
    """Butterworth filter implementation for real-time signal processing.

    Implements a Butterworth digital filter using second-order sections for
    numerical stability. Supports lowpass, highpass, bandpass, and bandstop
    filtering with configurable order and maintains state for streaming data.
    """

    #: Default filter order for Butterworth filters
    DEFAULT_ORDER = 2

    #: Filtered forwards only, carrying the filter's group delay. The
    #: only thing a realtime pipeline can do, because the samples after
    #: the current one have not been acquired yet.
    PHASE_CAUSAL = "causal"
    #: Filtered forwards and backwards, so the result has no group delay
    #: at any frequency. Needs every sample, and therefore a batch run.
    PHASE_ZERO = "zero"
    #: Accepted values of ``phase``, in the order the catalog lists them.
    PHASE_CHOICES = (PHASE_CAUSAL, PHASE_ZERO)
    #: What a filter does when nothing is said, which has to stay the
    #: realtime answer: a default that changed with the execution mode
    #: would give two different results for one document.
    DEFAULT_PHASE = PHASE_CAUSAL

    class Configuration(IONode.Configuration):
        """Configuration class for Butterworth filter parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for Butterworth filter settings."""

            #: Cutoff frequencies configuration key
            FN = "fn"
            #: Filter type configuration key
            BTYPE = "btype"
            #: Filter order configuration key
            ORDER = "order"
            #: Causal or zero-phase filtering
            PHASE = "phase"

    def __init__(
        self,
        fn: list,
        btype: str,
        order: int = None,
        phase: str = None,
        **kwargs,
    ):
        """Initialize the Butterworth filter with specified parameters.

        Args:
            fn: List of cutoff frequencies in Hz. Single value for lowpass/
                highpass, two values [low, high] for bandpass/bandstop.
            btype: Filter type ('lowpass', 'highpass', 'bandpass', 'bandstop').
            order: Filter order. Defaults to 2 if not specified.
            phase: ``causal`` (default) filters forwards only and carries
                the filter's group delay, which is the only thing a
                realtime pipeline can do. ``zero`` filters forwards and
                backwards, leaving no group delay at any frequency --
                which needs every sample, so it runs in a batch pipeline
                and is refused in a realtime one.
            **kwargs: Additional arguments passed to parent IONode class.

        Raises:
            ValueError: If fn is not a list, btype is invalid, order <= 0,
                or phase is not one of PHASE_CHOICES.
        """
        # Validate cutoff frequencies
        if type(fn) is not list:
            raise ValueError("fn must be a list.")

        # Validate filter type
        if btype not in ["lowpass", "highpass", "bandpass", "bandstop"]:
            raise ValueError(
                "btype must be 'lowpass', 'highpass', 'bandpass' "
                "or 'bandstop'."
            )

        # Set default order if not provided
        if order is None:
            order = self.DEFAULT_ORDER
        if order <= 0:
            raise ValueError("Filter order must be greater than 0.")

        if phase is None:
            phase = self.DEFAULT_PHASE
        if phase not in self.PHASE_CHOICES:
            raise ValueError(
                f"phase must be one of "
                f"{', '.join(repr(p) for p in self.PHASE_CHOICES)}; "
                f"got {phase!r}."
            )

        # Initialize parent class with filter configuration
        super().__init__(
            fn=fn, btype=btype, order=order, phase=phase, **kwargs
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the Butterworth filter before processing begins.

        Initializes filter coefficients and state based on sampling rate
        and channel configuration from input context.

        Args:
            data: Initial data dictionary (not used in setup).
            port_context_in: Input port context with sampling_rate and
                channel_count metadata.

        Returns:
            Output port context dictionary with updated metadata.

        Raises:
            ValueError: If required context keys are missing.
        """
        # Extract required context information
        ct = port_context_in[PORT_IN]
        channel_count = ct.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in context.")
        sampling_rate = ct.get(Constants.Keys.SAMPLING_RATE)
        if sampling_rate is None:
            raise ValueError("Sampling rate must be provided in context.")

        # Get filter configuration
        btype = self.config[self.Configuration.Keys.BTYPE]
        fn = self.config[self.Configuration.Keys.FN]
        N = self.config[self.Configuration.Keys.ORDER]
        self._phase = self.config[self.Configuration.Keys.PHASE]

        # Zero-phase filtering runs the filter backwards as well, so it
        # needs samples that do not exist yet in a realtime run. Refused
        # here, at setup, rather than emitting nothing and leaving the
        # user to work out why: this is the earliest point at which the
        # node learns how the pipeline is being driven.
        if self._phase == self.PHASE_ZERO:
            mode = ct.get(
                Constants.Keys.EXECUTION_MODE,
                Constants.ExecutionMode.REALTIME,
            )
            if mode != Constants.ExecutionMode.BATCH:
                raise ValueError(
                    f"phase='{self.PHASE_ZERO}' filters forwards and "
                    f"backwards, so it needs the whole recording and "
                    f"only runs in a batch pipeline; this one is "
                    f"'{mode}'. Use a source in batch mode -- "
                    f"CsvReader(..., mode='batch') -- and "
                    f"Pipeline.run(), or leave phase at "
                    f"'{self.PHASE_CAUSAL}'."
                )

        # A cutoff at or above Nyquist cannot be filtered, and scipy
        # refuses it -- but with a message about "Wn", a normalised
        # variable the author never wrote, naming neither the parameter
        # nor the sampling rate that makes it invalid.
        #
        # Here is the earliest this can be checked at all: the rate
        # arrives with the port context, so a constructor cannot know it.
        # A document naming Bandpass(f_hi=500) is therefore only refused
        # once it is connected to a 250 Hz source and started -- roughly
        # two seconds in, when the monitoring thread reports the error.
        # Catching it before a run needs the sampling rate propagated
        # from the sources through the graph, which is a whole-document
        # question and belongs to a graph validator, not to a node.
        nyquist = sampling_rate / 2
        names = (
            ("f_lo", "f_hi") if btype in ("bandpass", "bandstop") else ("f_c",)
        )
        for name, f in zip(names, fn):
            if f >= nyquist:
                raise ValueError(
                    f"{name} must be below the Nyquist frequency "
                    f"({nyquist:g} Hz, half the {sampling_rate:g} Hz "
                    f"sampling rate); got {f:g} Hz."
                )

        # Convert cutoff frequencies to normalized frequencies (0-1)
        # Nyquist frequency is sampling_rate/2, so normalize by 2*fn/fs
        Wn = [f / sampling_rate * 2 for f in fn]

        # For lowpass and highpass, scipy expects a scalar, not a list
        if btype in ["lowpass", "highpass"]:
            Wn = Wn[0]  # Extract single element for scalar conversion

        # Design Butterworth filter using scipy
        b, a = butter(N=N, Wn=Wn, btype=btype)

        # Convert to second-order sections for better numerical stability
        self._sos = tf2sos(b, a)

        # A filter must not touch channels that are not measured signal:
        # filtering a trigger smears its edges and filtering the in-band
        # master index destroys the timeline. With no roles declared every
        # channel is signal and this is a no-op.
        self._split = channels.SignalSplit(port_context_in[PORT_IN])
        if not self._split.all_signal:
            channel_count = len(self._split.signal)

        # Initialize filter state for each channel
        # Shape: (n_sections, n_states, n_channels)
        self._z = np.tile(
            sosfilt_zi(self._sos), (channel_count, 1, 1)
        ).transpose(1, 2, 0)

        # The state above is the steady state for a unit step. It is scaled
        # by the first sample when it arrives, so the filter starts from the
        # level the signal actually has rather than behaving as though the
        # input had been 1.0 forever.
        self._primed = False

        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply Butterworth filter to input data.

        Processes input data through the filter while maintaining filter
        state for continuous operation.

        Args:
            data: Dictionary containing input data with key PORT_IN.
                Input should be 2D array (samples x channels).

        Returns:
            Dictionary with filtered data under key PORT_OUT.

        Raises:
            ValueError: If input data is not 2D array.
        """
        data_in = data[PORT_IN]

        # Validate input data format
        if data_in.ndim != 2:
            raise ValueError(
                "Input data must be a 2D array (samples x channels)."
            )

        operated = self._split.take(data_in)

        if not self._primed:
            self._z = self._z * operated[0, :]
            self._primed = True

        # Apply filter with state preservation
        data_out, self._z = sosfilt(
            sos=self._sos,
            x=operated,
            axis=0,
            zi=self._z,  # Filter along time axis
        )

        return {PORT_OUT: self._split.merge(data_in, data_out)}

    def process(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Filter a whole recording in one call.

        Called instead of :meth:`step` in a batch run. With
        ``phase='causal'`` this hands the block straight to ``step`` --
        ``sosfilt`` over N rows is the same state machine as over N
        frames, so a batch run and a realtime run give identical
        samples, which is the property that makes validating offline
        worth anything.

        With ``phase='zero'`` it filters forwards and backwards, leaving
        no group delay at any frequency. That is what a realtime
        pipeline cannot do, and the reason this mode exists.

        Args:
            data: The complete recording, under key PORT_IN.

        Returns:
            The filtered recording under key PORT_OUT.

        Raises:
            ValueError: If the recording is too short for the filter's
                edge padding.
        """
        if self._phase == self.PHASE_CAUSAL:
            return self.step(data)

        data_in = data[PORT_IN]
        if data_in.ndim < 2:
            raise ValueError(
                "Input data must have at least two dimensions "
                "(samples x channels)."
            )

        operated = self._split.take(data_in)

        # sosfiltfilt extends the signal at both ends before filtering,
        # and refuses a signal shorter than that extension. Its own
        # message names padlen, a variable the user never wrote, so say
        # it in terms of the recording and the filter order instead.
        padlen = 3 * (2 * len(self._sos) + 1)
        if operated.shape[0] <= padlen:
            raise ValueError(
                f"phase='{self.PHASE_ZERO}' needs more than {padlen} "
                f"samples to filter in both directions at order "
                f"{self.config[self.Configuration.Keys.ORDER]}; this "
                f"recording has {operated.shape[0]}."
            )

        data_out = sosfiltfilt(sos=self._sos, x=operated, axis=0)
        return {PORT_OUT: self._split.merge(data_in, data_out)}
