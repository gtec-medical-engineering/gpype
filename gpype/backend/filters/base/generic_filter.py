from __future__ import annotations

import numpy as np
from scipy.signal import sosfilt, sosfilt_zi, tf2sos

from ....common.constants import Constants
from ...core.io_node import IONode

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class GenericFilter(IONode):
    """Generic Linear Time-Invariant (LTI) digital filter for real-time use.

    This class implements a flexible LTI filter using transfer function
    coefficients (numerator 'b' and denominator 'a' polynomials). It
    automatically converts to second-order sections (SOS) for improved
    numerical stability and maintains internal state for continuous
    processing of streaming data.

    The filter supports both FIR (Finite Impulse Response) and IIR
    (Infinite Impulse Response) implementations, making it suitable for
    a wide range of signal processing applications in BCI and real-time
    systems.

    The transfer function is defined as:
        H(z) = (b[0] + b[1]*z^-1 + ... + b[M]*z^-M) /
               (a[0] + a[1]*z^-1 + ... + a[N]*z^-N)

    This class uses second-order sections (SOS) internally instead of
    direct form implementation to avoid numerical issues with high-order
    filters or filters with coefficients spanning many orders of magnitude.

    Attributes:
        DEFAULT_ORDER: Default filter order (2) when not specified.
        _sos: Second-order sections representation of the filter for
            numerical stability.
        _z: Filter state array maintaining continuity between processing
            steps. Shape: (n_sections, n_states, n_channels).
    """

    DEFAULT_ORDER = 2

    class Configuration(IONode.Configuration):
        """Configuration class for GenericFilter parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for filter coefficients."""

            B = "b"  # Numerator coefficients
            A = "a"  # Denominator coefficients

    def __init__(self, b: np.ndarray = None, a: np.ndarray = None, **kwargs):
        """Initialize the generic filter with transfer function coefficients.

        Args:
            b: Numerator coefficients of the transfer function. Must be a
                non-empty numpy array.
            a: Denominator coefficients of the transfer function. Must be a
                non-empty numpy array with non-zero first element.
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
        self._sos = None
        self._z = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the generic filter before processing begins.

        This method converts the transfer function to second-order sections
        for numerical stability and initializes the filter state based on
        the channel configuration.

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

        # Convert transfer function to second-order sections for stability
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

        Processes the input data through the filter while maintaining
        filter state for continuous operation across processing steps.

        Args:
            data: Dictionary containing input data with key PORT_IN.
                Input data should be 2D array (samples x channels).

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

        # Apply filter using second-order sections with state preservation
        data_out, self._z = sosfilt(
            sos=self._sos,
            x=data_in,
            axis=0,
            zi=self._z,  # Filter along time axis
        )

        return {PORT_OUT: data_out}
