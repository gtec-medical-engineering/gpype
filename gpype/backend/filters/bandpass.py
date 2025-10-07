from __future__ import annotations

from scipy import signal

from .base.butterworth import Butterworth


class Bandpass(Butterworth):
    """Bandpass filter implementation using Butterworth design.

    Provides a convenient interface for creating bandpass filters that allow
    frequencies within a specific range to pass while attenuating frequencies
    outside this range. Uses Butterworth design for maximally flat response.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Bandpass filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for bandpass-specific parameters."""

            #: Upper cutoff frequency configuration key
            F_HI = "f_hi"

            #: Lower cutoff frequency configuration key
            F_LO = "f_lo"

    def __init__(self, f_lo: float, f_hi: float, order: int = None, **kwargs):
        """Initialize the bandpass filter with cutoff frequencies.

        Args:
            f_lo: Lower cutoff frequency in Hz.
            f_hi: Upper cutoff frequency in Hz.
            order: Filter order. Defaults to DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_lo >= f_hi or if frequencies are invalid.
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
