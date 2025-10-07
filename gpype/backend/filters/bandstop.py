from __future__ import annotations

from scipy import signal

from .base.butterworth import Butterworth


class Bandstop(Butterworth):
    """Bandstop filter implementation using Butterworth design.

    Provides a convenient interface for creating bandstop (notch) filters that
    attenuate frequencies within a specific range while allowing frequencies
    outside this range to pass. Uses Butterworth design for smooth response.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Bandstop filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for bandstop-specific parameters."""

            #: Upper cutoff frequency configuration key
            F_HI = "f_hi"
            #: Lower cutoff frequency configuration key
            F_LO = "f_lo"

    def __init__(self, f_lo: float, f_hi: float, order: int = None, **kwargs):
        """Initialize the bandstop filter with cutoff frequencies.

        Args:
            f_lo: Lower cutoff frequency in Hz (lower boundary of stopband).
            f_hi: Upper cutoff frequency in Hz (upper boundary of stopband).
            order: Filter order. Defaults to DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_lo >= f_hi or if frequencies are invalid.
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
