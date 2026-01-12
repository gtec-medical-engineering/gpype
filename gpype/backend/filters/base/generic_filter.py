from __future__ import annotations

import numpy as np
from scipy.signal import lfilter, lfilter_zi, sosfilt, sosfilt_zi, tf2sos

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

    def __init__(self, b: np.ndarray = None, a: np.ndarray = None, **kwargs):
        """Initialize the generic filter with transfer function coefficients.

        Args:
            b: Numerator coefficients of the transfer function.
            a: Denominator coefficients of the transfer function.
            **kwargs: Additional arguments passed to parent IONode class.

        Raises:
            ValueError: If coefficients are empty or invalid.
        """
        # Validate coefficient arrays are not empty
        if len(b) == 0 or len(a) == 0:
            raise ValueError(
                "Filter coefficients 'b' and 'a' must not be " "empty."
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

        if self._is_fir:
            # FIR filter: use direct form implementation (no biquad overhead)
            self._b = np.asarray(b)
            self._a = np.asarray(a)

            # Initialize filter state for each channel
            # State size is len(b) - 1 for FIR filters
            state_len = len(self._b) - 1
            self._z = np.zeros((state_len, channel_count))
        else:
            # IIR filter: convert to second-order sections for stability
            # This avoids numerical issues with direct form implementation
            self._sos = tf2sos(b, a)

            # Initialize filter state for each channel
            # sosfilt_zi provides initial conditions for zero-phase start
            initial_state = sosfilt_zi(self._sos)
            self._z = np.tile(initial_state, (channel_count, 1, 1)).transpose(
                1, 2, 0
            )

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

        # Apply filter with state preservation
        if self._is_fir:
            # FIR filter: use direct form lfilter
            data_out, self._z = lfilter(
                b=self._b,
                a=self._a,
                x=data_in,
                axis=0,
                zi=self._z,
            )
        else:
            # IIR filter: use second-order sections for stability
            data_out, self._z = sosfilt(
                sos=self._sos,
                x=data_in,
                axis=0,
                zi=self._z,
            )

        return {PORT_OUT: data_out}
