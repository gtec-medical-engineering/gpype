from __future__ import annotations

import numpy as np
from scipy.signal import get_window

from ...common.constants import Constants
from ..core.io_node import IONode

# Port name constants for convenience
PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT


class FFT(IONode):
    """
    Fast Fourier Transform node for frequency domain analysis.

    The FFT node performs windowed Fast Fourier Transform on input data
    to convert time-domain signals to frequency-domain representations.
    It supports overlapping windows, various window functions, and proper
    amplitude scaling for spectral analysis.

    This node is essential for BCI applications that analyze frequency
    content of neural signals, such as detecting alpha rhythms, motor
    imagery patterns, or steady-state visual evoked potentials (SSVEP).

    Features:
        - Configurable window size and overlap
        - Multiple window functions (Hanning, Hamming, Blackman, etc.)
        - Proper amplitude scaling and normalization
        - Rolling buffer for continuous processing
        - Decimation for overlapping windows
        - FFT optimization for real-valued signals

    Args:
        window_size (int): Size of the FFT window in samples. Must be > 1.
        window_function (str): Window function name (default: 'boxcar').
            Supports all scipy.signal window functions.
        overlap (float): Overlap ratio between windows (0.0 to 1.0).
            Default is 0.5 (50% overlap).

    Note:
        Input data must have frame_size=1 for proper windowing.
        Output frame rate is reduced by the step size (decimation).
    """

    # Type annotation for the rolling buffer
    _buf: np.ndarray

    class Configuration(IONode.Configuration):
        """Configuration class for FFT parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for the FFT."""

            WINDOW_SIZE = "window_size"
            WINDOW_FUNCTION = "window_function"
            OVERLAP = "overlap"

    def __init__(
        self,
        window_size: int = None,
        window_function: str = None,
        overlap: float = None,
        **kwargs,
    ):
        """
        Initialize the FFT node with windowing and overlap parameters.

        Sets up the FFT node with specified window size, window function,
        and overlap ratio. Calculates the step size for overlapping windows
        and configures the node for decimated output.

        Args:
            window_size (int): Size of the FFT window in samples. Must be
                an integer greater than 1.
            window_function (str, optional): Name of the window function
                to apply. Defaults to 'boxcar' (rectangular window).
                Supports all scipy.signal window functions like 'hanning',
                'hamming', 'blackman', etc.
            overlap (float, optional): Overlap ratio between consecutive
                windows, ranging from 0.0 (no overlap) to 1.0 (full overlap).
                Defaults to 0.5 (50% overlap).
            **kwargs: Additional configuration parameters passed to IONode.

        Raises:
            ValueError: If window_size is None, not an integer, or <= 1.
            ValueError: If overlap is not a float or outside [0, 1] range.
        """
        # Validate window size parameter
        if window_size is None:
            raise ValueError("window_size must not be None.")
        if type(window_size) is not int:
            raise ValueError("window_size must be integer.")
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1.")

        # Set default window function if not provided
        if window_function is None:
            window_function = "boxcar"  # Rectangular window (no windowing)

        # Set default overlap if not provided
        if overlap is None:
            overlap = 0.5  # 50% overlap is common for spectral analysis

        # Validate overlap parameter
        if type(overlap) is not float:
            raise ValueError("overlap must be float.")
        if overlap < 0 or overlap > 1:
            raise ValueError("overlap must be between 0 and 1.")

        # Calculate step size for overlapping windows
        # Step size determines how many samples to advance between windows
        self._step_size = int(np.round(window_size * (1 - overlap)))

        # Recalculate actual overlap based on integer step size
        overlap = self._step_size / window_size

        # Initialize parent IONode with windowing configuration
        super().__init__(
            window_size=window_size,
            frame_size=window_size,
            decimation_factor=self._step_size,
            window_function=window_function,
            overlap=overlap,
            **kwargs,
        )

        # Initialize internal state variables
        self._buf = None  # Rolling input buffer for windowing
        self._w = None  # Precomputed window function weights

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Set up the FFT node and initialize windowing components.

        Validates input requirements, initializes the rolling buffer for
        windowing, and precomputes the window function weights for
        efficient real-time processing.

        Args:
            data (dict): Initial data dictionary for port configuration.
            port_context_in (dict): Input port context information containing
                sampling rates, frame sizes, and channel counts.

        Returns:
            dict: Output port context with updated frame size for FFT output.

        Raises:
            ValueError: If input frame size is not 1.
        """
        # Call parent setup to get base output context
        port_context_out = super().setup(data, port_context_in)

        # Validate input frame size requirement for windowing
        frame_size_in = port_context_in[PORT_IN][Constants.Keys.FRAME_SIZE]
        if frame_size_in != 1:
            raise ValueError("Input frame size must be 1.")

        # Get configuration parameters
        frame_size_out = self.config[Constants.Keys.FRAME_SIZE]
        channel_count = port_context_out[PORT_OUT][
            Constants.Keys.CHANNEL_COUNT
        ]

        # Initialize rolling buffer for windowing
        # Shape: (window_size, channel_count)
        self._buf = np.zeros(shape=(frame_size_out, channel_count))
        self._frame_size = frame_size_out

        sampling_rate = port_context_in[PORT_IN][Constants.Keys.SAMPLING_RATE]
        frame_rate = sampling_rate / self._step_size

        # Update output context with FFT frame size
        port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE] = frame_size_out
        port_context_out[PORT_OUT][Constants.Keys.FRAME_RATE] = frame_rate

        # Get window function configuration
        window_function = self.config[self.Configuration.Keys.WINDOW_FUNCTION]
        window_size = self.config[self.Configuration.Keys.WINDOW_SIZE]

        # Create and normalize window function
        w = get_window(window_function, window_size)
        w = w / np.sum(w)  # Normalize to preserve signal power

        # Reshape for broadcasting with multi-channel data
        self._w = w[:, np.newaxis]  # Shape: (window_size, 1)

        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Process one frame of data and compute FFT when ready.

        Updates the rolling buffer with new input data and performs FFT
        computation when the decimation step is reached (based on overlap
        configuration). Returns frequency domain amplitude spectrum.

        Args:
            data (dict): Dictionary containing input data arrays. The input
                data should have shape (1, channel_count) for each frame.

        Returns:
            dict or None: Dictionary containing FFT amplitude spectrum when
                decimation step is reached, None otherwise. The output has
                shape (frequency_bins, channel_count) where frequency_bins
                is (window_size // 2 + 1) for real FFT.

        Note:
            The output amplitude is properly scaled and normalized for
            spectral analysis. DC and Nyquist components are scaled
            appropriately, while other frequencies are scaled by factor 2.
        """
        # Update rolling buffer with new data sample
        # Efficiently shift all samples: move rows up by one position
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = data[PORT_IN][-1]  # Add newest sample at end

        # Check if this is a decimation step (based on overlap configuration)
        if not self.is_decimation_step():
            return None  # Skip FFT computation for this frame

        # Apply window function to the buffered data
        # Broadcasting: (window_size, channels) * (window_size, 1)
        windowed = self._buf * self._w

        # Compute real FFT (optimized for real-valued signals)
        # Transpose for FFT: (channels, window_size) -> (channels, freq_bins)
        fft = np.fft.rfft(windowed.T, axis=1)

        # Get window size for scaling calculations
        N = self._frame_size

        # Create scaling array for proper amplitude normalization
        scale_arr = np.ones((fft.shape[1],))
        scale_arr[1:-1] = 2.0  # Scale non-DC/Nyquist frequencies by 2
        scale_arr[0] = 1.0  # DC component (0 Hz) scaled by 1

        # Handle Nyquist frequency (only exists for even N)
        if N % 2 == 0:
            scale_arr[-1] = 1.0  # Nyquist bin scaled by 1

        # Calculate magnitude spectrum
        # Normalize by window sum to preserve signal power
        magnitude = np.abs(fft) / np.sum(self._w)

        # Apply proper amplitude scaling for single-sided spectrum
        # Factor of 2 accounts for negative frequencies in two-sided spectrum
        amplitude = magnitude * scale_arr[np.newaxis, :] * 2

        # Return transposed result: (frequency_bins, channels)
        return {PORT_OUT: amplitude.T}
