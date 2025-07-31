from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants


class OPort(ioc.OPort):
    """Output port class for g.Pype signal processing nodes.

    OPort extends the ioiocore.OPort to provide g.Pype-specific functionality
    for output ports. It handles data output with configurable timing modes and
    type validation for signal processing pipelines.

    The port supports different timing modes (synchronous/asynchronous) and
    automatic type inference for numpy arrays, making it suitable for
    real-time signal processing applications where processed data needs to be
    passed to subsequent nodes in the pipeline.
    """

    class Configuration(ioc.OPort.Configuration):
        """Configuration class for OPort with g.Pype-specific extensions."""

        class Keys(ioc.OPort.Configuration.Keys):
            """Configuration keys inherited from ioiocore with extensions."""

            pass

    def __init__(
        self,
        name: str = Constants.Defaults.PORT_OUT,
        timing: Constants.Timing = Constants.Timing.SYNC,
        **kwargs,
    ):
        """Initialize the output port with g.Pype-specific defaults.

        Args:
            name: Name of the output port. Defaults to the standard output port
                name defined in Constants.Defaults.PORT_OUT.
            timing: Timing mode for the port (SYNC or ASYNC). Defaults to
                synchronous timing for real-time processing.
            **kwargs: Additional configuration parameters passed to the parent
                class, including 'type' for data type specification.

        Note:
            If no 'type' is specified in kwargs, it defaults to np.ndarray
            which is the standard data type for signal processing in g.Pype.
            This ensures consistent data flow between nodes in the pipeline.
        """
        # Extract and set default type for signal processing
        type_key = self.Configuration.Keys.TYPE
        type: str = kwargs.pop(
            type_key, np.ndarray.__name__
        )  # Default to numpy arrays

        # Initialize parent class with g.Pype-specific configuration
        super().__init__(name=name, type=type, timing=timing, **kwargs)
