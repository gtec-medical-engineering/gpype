from __future__ import annotations

from .base.butterworth import Butterworth


class Lowpass(Butterworth):
    """Lowpass filter implementation using Butterworth design.

    This class provides a convenient interface for creating lowpass filters
    by extending the Butterworth filter implementation. It automatically
    configures the filter for lowpass operation with a specified cutoff
    frequency.

    Lowpass filters allow frequencies below the cutoff frequency to pass
    through while attenuating frequencies above the cutoff. This is commonly
    used in BCI applications for anti-aliasing, noise reduction, or isolating
    slow cortical potentials and low-frequency brain rhythms.

    The filter uses Butterworth design characteristics, providing maximally
    flat response in the passband with no ripples, making it suitable for
    applications requiring smooth frequency response.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Lowpass filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for lowpass-specific parameters."""

            F_C = "f_c"  # Cutoff frequency

    def __init__(self, f_c: float, order: int = None, **kwargs):
        """Initialize the lowpass filter with cutoff frequency.

        Args:
            f_c: Cutoff frequency in Hz. Frequencies below this value will
                pass through with minimal attenuation, while frequencies
                above will be attenuated according to the filter order.
            order: Filter order. Higher orders provide steeper rolloff
                but may introduce more phase distortion. Defaults to
                DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_c is not positive.

        Note:
            The -3dB point (half-power frequency) occurs at the cutoff
            frequency f_c. The actual attenuation at f_c depends on the filter
            order, with higher orders providing steeper rolloff
            characteristics.
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

        # Initialize parent Butterworth filter with lowpass configuration
        super().__init__(fn=fn, f_c=f_c, btype=btype, order=order, **kwargs)
