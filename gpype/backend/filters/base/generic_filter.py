from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, lfilter_zi, sosfilt, sosfilt_zi, tf2sos

from ....common._private import channels
from ....common.constants import Constants
from ...core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class GenericFilter(IONode):
    """Generic Linear Time-Invariant (LTI) digital filter for real-time use.

    Implements a flexible LTI filter using transfer function coefficients
    (numerator 'b' and denominator 'a' polynomials). For IIR filters, converts
    to second-order sections for numerical stability. For FIR filters, uses
    direct form implementation to avoid unnecessary decomposition overhead.
    Maintains state for streaming data.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for GenericFilter parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for filter coefficients."""

            #: Numerator coefficients configuration key
            B = "b"
            #: Denominator coefficients configuration key
            A = "a"

    def __init__(self, b: np.ndarray, a: np.ndarray, **kwargs):
        """Initialize the generic filter with transfer function coefficients.

        Args:
            b: Numerator coefficients of the transfer function.
            a: Denominator coefficients of the transfer function.
            **kwargs: Additional arguments passed to parent IONode class.

        Raises:
            ValueError: If coefficients are empty or invalid.
        """
        # None reaches here from a document rather than from a caller:
        # serialize() writes a null for a parameter the author omitted,
        # and deserialize() hands it straight back. Without this the
        # failure was "object of type 'NoneType' has no len()", which
        # names neither the parameter nor the node.
        missing = [
            name for name, value in (("b", b), ("a", a)) if value is None
        ]
        if missing:
            raise ValueError(
                f"Filter coefficients {missing} must be given; they are "
                f"the numerator and denominator of the transfer function."
            )

        # Validate coefficient arrays are not empty
        if len(b) == 0 or len(a) == 0:
            raise ValueError(
                "Filter coefficients 'b' and 'a' must not be empty."
            )

        # Initialize parent class with filter configuration
        super().__init__(b=b, a=a, **kwargs)

        # Initialize filter state (will be set up in setup() method)
        self._sos = None  # For IIR filters (second-order sections)
        self._b = None  # For FIR filters (numerator coefficients)
        self._a = None  # For FIR filters (denominator coefficients)
        self._z = None  # Filter state
        self._is_fir = None  # Flag to distinguish FIR/IIR

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the generic filter before processing begins.

        Converts transfer function to second-order sections for numerical
        stability and initializes filter state based on channel configuration.

        Args:
            data: Initial data dictionary (not used in setup).
            port_context_in: Input port context containing channel_count
                and other metadata.

        Returns:
            Output port context dictionary with updated metadata.

        Raises:
            ValueError: If required context keys are missing or coefficients
                are invalid.
        """
        # Extract required context information
        md = port_context_in[PORT_IN]
        channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
        if channel_count is None:
            raise ValueError("Channel count must be provided in context.")

        # Get filter coefficients from configuration
        b = self.config[self.Configuration.Keys.B]
        a = self.config[self.Configuration.Keys.A]

        # Validate denominator coefficients
        if len(a) < 1 or a[0] == 0:
            raise ValueError(
                "Invalid 'a' coefficients: first element must be non-zero."
            )

        # Determine if this is an FIR filter (a = [1.0] or equivalent)
        # FIR filters have no feedback, so a should only contain a[0] = 1
        self._is_fir = len(a) == 1 and np.isclose(a[0], 1.0)

        # A filter must not touch channels that are not measured signal:
        # filtering a trigger smears its edges and filtering the in-band
        # master index destroys the timeline. With no roles declared every
        # channel is signal and this is a no-op.
        self._split = channels.SignalSplit(md)
        if not self._split.all_signal:
            channel_count = len(self._split.signal)

        if self._is_fir:
            # FIR filter: use direct form implementation (no biquad overhead)
            self._b = np.asarray(b)
            self._a = np.asarray(a)
            if len(self._b) > 1:
                state = lfilter_zi(self._b, self._a)
                self._z = np.tile(state, (channel_count, 1)).T
            else:
                # A single coefficient is a pure gain: there is no history
                # to carry, and lfilter_zi cannot describe one.
                self._z = None
        else:
            # IIR filter: convert to second-order sections for stability
            # This avoids numerical issues with direct form implementation
            self._sos = tf2sos(b, a)
            state = sosfilt_zi(self._sos)
            self._z = np.tile(state, (channel_count, 1, 1)).transpose(1, 2, 0)

        # Both branches hold the steady state for a unit step. It is scaled
        # by the first sample when it arrives, so the filter starts from the
        # level the signal actually has. Leaving it unscaled makes a filter
        # behave as though the input had been 1.0 forever: a 10 Hz lowpass
        # fed silence would emit 0.99, 0.94, 0.85 ... instead of zero.
        self._primed = False

        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply the generic filter to input data.

        Processes input data through the filter while maintaining filter
        state for continuous operation across processing steps.

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
                "Input data must be a 2D array (samples x " "channels)."
            )

        operated = self._split.take(data_in)

        if not self._primed:
            if self._z is not None:
                self._z = self._z * operated[0, :]
            self._primed = True

        # Apply filter with state preservation
        if self._is_fir:
            if self._z is None:
                # Pure gain: nothing to carry between frames.
                data_out = lfilter(b=self._b, a=self._a, x=operated, axis=0)
            else:
                # FIR filter: use direct form lfilter
                data_out, self._z = lfilter(
                    b=self._b,
                    a=self._a,
                    x=operated,
                    axis=0,
                    zi=self._z,
                )
        else:
            # IIR filter: use second-order sections for stability
            data_out, self._z = sosfilt(
                sos=self._sos,
                x=operated,
                axis=0,
                zi=self._z,
            )

        return {PORT_OUT: self._split.merge(data_in, data_out)}
