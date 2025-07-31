from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, tf2sos

from ....common.constants import Constants
from ...core.io_node import IONode

PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class Butterworth(IONode):
    """Butterworth filter implementation for real-time signal processing.

    This class implements a Butterworth digital filter using second-order
    sections (SOS) for improved numerical stability. It supports lowpass,
    highpass, bandpass, and bandstop filtering with configurable order.

    The filter maintains internal state for continuous processing of streaming
    data, making it suitable for real-time applications in BCI and signal
    processing pipelines.

    Attributes:
        DEFAULT_ORDER: Default filter order (2) when not specified.
        _sos: Second-order sections representation of the filter.
        _z: Filter state for each channel to maintain continuity between steps.
    """

    DEFAULT_ORDER = 2

    class Configuration(IONode.Configuration):
        """Configuration class for Butterworth filter parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for Butterworth filter settings."""

            FN = "fn"  # Cutoff frequencies
            BTYPE = "btype"  # Filter type
            ORDER = "order"  # Filter order

    def __init__(self, fn: list, btype: str, order: int = None, **kwargs):
        """Initialize the Butterworth filter with specified parameters.

        Args:
            fn: List of cutoff frequencies in Hz. For lowpass/highpass: single
                value in list. For bandpass/bandstop: two values [low, high].
            btype: Filter type. Must be one of 'lowpass', 'highpass',
                'bandpass', or 'bandstop'.
            order: Filter order. Higher orders provide steeper rolloff but
                may introduce more phase distortion. Defaults to 2.
            **kwargs: Additional arguments passed to parent IONode class.

        Raises:
            ValueError: If fn is not a list, btype is invalid, or order <= 0.
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

        # Initialize parent class with filter configuration
        super().__init__(fn=fn, btype=btype, order=order, **kwargs)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the Butterworth filter before processing begins.

        This method initializes the filter coefficients and state based on
        the sampling rate and channel configuration from the input context.

        Args:
            data: Initial data dictionary (not used in setup).
            port_context_in: Input port context containing sampling_rate,
                channel_count, and other metadata.

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

        # Convert cutoff frequencies to normalized frequencies (0-1)
        # Nyquist frequency is sampling_rate/2, so normalize by 2*fn/fs
        Wn = [f / sampling_rate * 2 for f in fn]

        # Design Butterworth filter using scipy
        b, a = butter(N=N, Wn=Wn, btype=btype)

        # Convert to second-order sections for better numerical stability
        self._sos = tf2sos(b, a)

        # Initialize filter state for each channel
        # Shape: (n_sections, n_states, n_channels)
        self._z = np.tile(
            sosfilt_zi(self._sos), (channel_count, 1, 1)
        ).transpose(1, 2, 0)

        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Apply Butterworth filter to input data.

        Processes the input data through the Butterworth filter while
        maintaining filter state for continuous operation.

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
                "Input data must be a 2D array (samples x channels)."
            )

        # Apply filter with state preservation
        data_out, self._z = sosfilt(
            sos=self._sos,
            x=data_in,
            axis=0,
            zi=self._z,  # Filter along time axis
        )

        return {PORT_OUT: data_out}
