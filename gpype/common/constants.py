import ioiocore as ioc
import numpy as np


class Constants(ioc.Constants):
    """
    Application-wide constants for g.Pype BCI framework.

    Extends the ioiocore Constants class to provide g.Pype-specific
    constants, data types, and configuration keys used throughout
    the BCI processing pipeline.

    This class centralizes all constant values to ensure consistency
    across the framework and provides a single point of maintenance
    for shared configuration parameters.
    """

    # Default data type for numerical operations in the pipeline
    # Using float32 for memory efficiency while maintaining precision
    DATA_TYPE = np.float32

    # Special value indicating inherited timing or configuration
    # Used when a node should inherit timing from its parent/input
    INHERITED = -1

    class Keys(ioc.Constants.Keys):
        """
        Configuration key constants for pipeline components.

        Defines standard key names used in configuration dictionaries
        throughout the g.Pype framework. These keys ensure consistent
        naming across different nodes and components.
        """

        # Sampling rate in Hz (samples per second)
        SAMPLING_RATE: str = "sampling_rate"

        # Number of data channels in the signal
        CHANNEL_COUNT: str = "channel_count"

        # Number of samples processed per frame
        FRAME_SIZE: str = "frame_size"

        # Frame rate in Hz (frames per second, optional)
        FRAME_RATE: str = "frame_rate"

        # Factor by which to reduce the sampling rate
        DECIMATION_FACTOR: str = "decimation_factor"
