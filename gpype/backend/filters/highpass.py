from __future__ import annotations

from scipy import signal

from .base.butterworth import Butterworth


class Highpass(Butterworth):
    """Highpass filter implementation using Butterworth design.

    This class provides a convenient interface for creating highpass filters
    by extending the Butterworth filter implementation. It automatically
    configures the filter for highpass operation with a specified cutoff
    frequency.

    Highpass filters allow frequencies above the cutoff frequency to pass
    through while attenuating frequencies below the cutoff. This is commonly
    used in BCI applications to remove low-frequency drift, DC offset, or
    slow artifacts while preserving higher-frequency neural activity.

    The filter uses Butterworth design characteristics, providing maximally
    flat response in the passband with no ripples, making it suitable for
    applications requiring smooth frequency response above the cutoff.
    """

    class Configuration(Butterworth.Configuration):
        """Configuration class for Highpass filter parameters."""

        class Keys(Butterworth.Configuration.Keys):
            """Configuration keys for highpass-specific parameters."""

            F_C = "f_c"  # Cutoff frequency

    def __init__(self, f_c: float, order: int = None, **kwargs):
        """Initialize the highpass filter with cutoff frequency.

        Args:
            f_c: Cutoff frequency in Hz. Frequencies below this value will
                be attenuated, while frequencies above will pass through.
                Must be positive and less than Nyquist frequency.
            order: Filter order. Higher orders provide steeper rolloff
                but may introduce more phase distortion. Defaults to
                DEFAULT_ORDER from parent class.
            **kwargs: Additional arguments passed to parent Butterworth class.

        Raises:
            ValueError: If f_c is not positive.

        Note:
            The cutoff frequency is defined as the -3dB point where the
            filter response is 0.707 (-3dB) of the passband gain. Higher
            order filters will have steeper transitions around this point.
        """
        # Validate cutoff frequency
        if f_c <= 0:
            raise ValueError("Cutoff frequency must be positive.")

        # Configure frequency list for highpass operation
        fn = [f_c]
        btype = "highpass"

        # Use default order if not specified
        if order is None:
            order = self.DEFAULT_ORDER

        # Initialize parent Butterworth filter with highpass configuration
        super().__init__(fn=fn, f_c=f_c, btype=btype, order=order, **kwargs)
