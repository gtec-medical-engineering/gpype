from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants


class IPort(ioc.IPort):
    """Input port class for g.Pype signal processing nodes.

    IPort extends the ioiocore.IPort to provide g.Pype-specific functionality
    for input ports. It handles data input with configurable timing modes and
    type validation for signal processing pipelines.

    The port supports different timing modes (synchronous/asynchronous) and
    automatic type inference for numpy arrays, making it suitable for
    real-time signal processing applications.
    """

    class Configuration(ioc.IPort.Configuration):
        """Configuration class for IPort with g.Pype-specific extensions."""

        class Keys(ioc.IPort.Configuration.Keys):
            """Configuration keys inherited from ioiocore with extensions."""

            pass

    def __init__(
        self,
        name: str = Constants.Defaults.PORT_IN,
        timing: Constants.Timing = Constants.Timing.SYNC,
        **kwargs,
    ):
        """Initialize the input port with g.Pype-specific defaults.

        Args:
            name: Name of the input port. Defaults to the standard input port
                name defined in Constants.Defaults.PORT_IN.
            timing: Timing mode for the port (SYNC or ASYNC). Defaults to
                synchronous timing for real-time processing.
            **kwargs: Additional configuration parameters passed to the parent
                class, including 'type' for data type specification.

        Note:
            If no 'type' is specified in kwargs, it defaults to np.ndarray
            which is the standard data type for signal processing in g.Pype.
        """
        # Extract and set default type for signal processing
        type_key = self.Configuration.Keys.TYPE
        type: str = kwargs.pop(
            type_key, np.ndarray.__name__
        )  # Default to numpy arrays

        # Initialize parent class with g.Pype-specific configuration
        super().__init__(name=name, type=type, timing=timing, **kwargs)
