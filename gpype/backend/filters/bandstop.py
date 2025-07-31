from __future__ import annotations

from scipy import signal

from .base.butterworth import Butterworth


class Bandstop(Butterworth):
    """Bandstop filter implementation using Butterworth design.

    This class provides a convenient interface for creating bandstop (notch)
    filters by extending the Butterworth filter implementation. It
    automatically configures the filter for bandstop operation with specified
    lower and upper cutoff frequencies.

    Bandstop filters attenuate frequencies within a specific range while
    allowing frequencies outside this range to pass through. This is commonly
    used in BCI applications to remove specific interference frequencies
    (e.g., 50/60 Hz power line noise, or specific artifact frequencies).

    The filter uses Butterworth design characteristics, providing maximally
    flat response in the passband with no ripples, making it suitable for
    applications requiring smooth frequency response outside the stopband.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Bandstop filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for bandstop-specific parameters."""

            F_HI = "f_hi"  # Upper cutoff frequency
            F_LO = "f_lo"  # Lower cutoff frequency

    def __init__(self, f_lo: float, f_hi: float, order: int = None, **kwargs):
        """Initialize the bandstop filter with cutoff frequencies.

        Args:
            f_lo: Lower cutoff frequency in Hz. This defines the lower
                boundary of the stopband.
            f_hi: Upper cutoff frequency in Hz. This defines the upper
                boundary of the stopband.
            order: Filter order. Higher orders provide steeper rolloff
                but may introduce more phase distortion. Defaults to
                DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_lo >= f_hi or if frequencies are invalid.

        Note:
            The stopband is defined as [f_lo, f_hi]. Frequencies within
            this range will be attenuated, while frequencies outside will
            pass through with minimal attenuation according to the filter
            characteristics.
        """
        # Validate cutoff frequency relationship
        if f_lo >= f_hi:
            raise ValueError(
                "Lower cutoff frequency must be less than "
                "upper cutoff frequency."
            )
        if f_lo <= 0 or f_hi <= 0:
            raise ValueError("Cutoff frequencies must be positive.")

        # Configure frequency list for bandstop operation
        fn = [f_lo, f_hi]
        btype = "bandstop"

        # Use default order if not specified
        if order is None:
            order = self.DEFAULT_ORDER

        # Initialize parent Butterworth filter with bandstop configuration
        super().__init__(
            fn=fn, f_lo=f_lo, f_hi=f_hi, btype=btype, order=order, **kwargs
        )
