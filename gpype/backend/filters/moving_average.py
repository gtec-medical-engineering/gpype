from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.signal import get_window

from .base.generic_filter import GenericFilter


class MovingAverage(GenericFilter):
    """Moving average filter for signal smoothing and noise reduction.

    Implements a moving average filter using various window functions. Moving
    average filters are linear phase FIR filters that smooth signals by
    averaging values within a sliding window. Supports window functions
    including rectangular, Hamming, Hanning, and others.
    """

    class Configuration(GenericFilter.Configuration):
        """Configuration class for MovingAverage filter parameters."""

        class Keys(GenericFilter.Configuration.Keys):
            """Configuration keys for moving average parameters."""

            WINDOW_SIZE = "window_size"
            WINDOW_FUNCTION = "window_function"

    def __init__(
        self,
        window_size: Optional[int] = None,
        window_function: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the moving average filter.

        Args:
            window_size: Number of samples in the averaging window.
                Must be greater than 1.
            window_function: Type of window function to apply. Supported
                functions include 'boxcar', 'hamming', 'hanning', etc.
                Defaults to 'boxcar'.
            **kwargs: Additional arguments passed to parent GenericFilter.

        Raises:
            ValueError: If window_size is None, <= 1, or if window_function
                is not supported by scipy.signal.get_window.
        """

        # Validate window size parameter
        if window_size is None:
            raise ValueError("window_size must not be None.")
        if window_size <= 1:
            raise ValueError("window_size must be greater than 1.")

        # Set default window function
        if window_function is None:
            window_function = "boxcar"  # Rectangular window

        # Generate window coefficients using scipy
        try:
            b = get_window(window_function, window_size, fftbins=False)
        except ValueError as e:
            msg = f"Invalid window function '{window_function}': {e}"
            raise ValueError(msg)

        # Normalize coefficients to maintain unity gain
        b = b / np.sum(b)

        # FIR filter (no feedback, denominator = 1)
        a = np.array([1.0])

        # Initialize parent generic filter with computed coefficients
        super().__init__(
            b=b,
            a=a,
            window_size=window_size,
            window_function=window_function,
            **kwargs,
        )
