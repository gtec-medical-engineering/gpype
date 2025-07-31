from __future__ import annotations

from scipy import signal

from .base.butterworth import Butterworth


class Bandpass(Butterworth):
    """Bandpass filter implementation using Butterworth design.

    This class provides a convenient interface for creating bandpass filters
    by extending the Butterworth filter implementation. It automatically
    configures the filter for bandpass operation with specified lower and
    upper cutoff frequencies.

    Bandpass filters allow frequencies within a specific range to pass through
    while attenuating frequencies outside this range. This is commonly used
    in BCI applications to isolate specific frequency bands of interest
    (e.g., alpha, beta, or gamma rhythms).

    The filter uses Butterworth design characteristics, providing maximally
    flat response in the passband with no ripples, making it suitable for
    applications requiring smooth frequency response.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Bandpass filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for bandpass-specific parameters."""

            F_HI = "f_hi"  # Upper cutoff frequency
            F_LO = "f_lo"  # Lower cutoff frequency

    def __init__(self, f_lo: float, f_hi: float, order: int = None, **kwargs):
        """Initialize the bandpass filter with cutoff frequencies.

        Args:
            f_lo: Lower cutoff frequency in Hz. Frequencies below this
                value will be attenuated.
            f_hi: Upper cutoff frequency in Hz. Frequencies above this
                value will be attenuated.
            order: Filter order. Higher orders provide steeper rolloff
                but may introduce more phase distortion. Defaults to
                DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_lo >= f_hi or if frequencies are invalid.

        Note:
            The passband is defined as [f_lo, f_hi]. Frequencies within
            this range will pass through with minimal attenuation, while
            frequencies outside will be attenuated according to the filter
            order and Butterworth characteristics.
        """
        # Validate cutoff frequency relationship
        if f_lo >= f_hi:
            raise ValueError(
                "Lower cutoff frequency must be less than upper "
                "cutoff frequency."
            )
        if f_lo <= 0 or f_hi <= 0:
            raise ValueError("Cutoff frequencies must be positive.")

        # Configure frequency list for bandpass operation
        fn = [f_lo, f_hi]
        btype = "bandpass"

        # Use default order if not specified
        if order is None:
            order = self.DEFAULT_ORDER

        # Initialize parent Butterworth filter with bandpass configuration
        super().__init__(
            fn=fn, f_lo=f_lo, f_hi=f_hi, btype=btype, order=order, **kwargs
        )
