import ioiocore as ioc
import numpy as np


class Constants(ioc.Constants):
    """Application-wide constants for g.Pype BCI framework.

    Extends ioiocore Constants with g.Pype-specific constants,
    data types, and configuration keys for signal processing.
    """

    #: Default data type for numerical operations in the pipeline
    DATA_TYPE = np.float32

    #: Special value indicating inherited timing or configuration
    INHERITED = -1

    class Keys(ioc.Constants.Keys):
        """Configuration key constants for pipeline components.

        Standard key names for configuration dictionaries used
        throughout the g.Pype framework.
        """

        #: Sampling rate in Hz (samples per second)
        SAMPLING_RATE: str = "sampling_rate"

        #: Number of data channels in the signal
        CHANNEL_COUNT: str = "channel_count"

        #: Number of samples processed per frame
        FRAME_SIZE: str = "frame_size"

        #: Frame rate in Hz (frames per second, optional)
        FRAME_RATE: str = "frame_rate"

        #: Factor by which to reduce the sampling rate
        DECIMATION_FACTOR: str = "decimation_factor"

    class Defaults(ioc.Constants.Defaults):

        #: Default frame size in samples
        FRAME_SIZE: int = 1
