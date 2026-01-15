from __future__ import annotations

from .base.butterworth import Butterworth


class Lowpass(Butterworth):
    """Lowpass filter implementation using Butterworth design.

    Provides a convenient interface for creating lowpass filters that allow
    frequencies below the cutoff to pass while attenuating frequencies above
    the cutoff. Uses Butterworth design for maximally flat response.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Lowpass filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for lowpass-specific parameters."""

            #: Cutoff frequency configuration key
            F_C = "f_c"

    def __init__(self, f_c: float, order: int = None, **kwargs):
        """Initialize the lowpass filter with cutoff frequency.

        Args:
            f_c: Cutoff frequency in Hz. Must be positive.
            order: Filter order. Defaults to DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_c is not positive.
        """
        # Validate cutoff frequency
        if f_c <= 0:
            raise ValueError("Cutoff frequency must be positive.")

        # Configure frequency list for lowpass operation
        fn = [f_c]
        btype = "lowpass"

        # Use default order if not specified
        if order is None:
            order = self.DEFAULT_ORDER

        fn = kwargs.pop(Butterworth.Configuration.Keys.FN, fn)
        btype = kwargs.pop(Butterworth.Configuration.Keys.BTYPE, btype)

        # Initialize parent Butterworth filter with lowpass configuration
        super().__init__(fn=fn, f_c=f_c, btype=btype, order=order, **kwargs)
