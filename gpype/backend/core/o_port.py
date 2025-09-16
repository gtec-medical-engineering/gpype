from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants


class OPort(ioc.OPort):
    """Output port class for g.Pype signal processing nodes.

    Extends ioiocore.OPort with g.Pype-specific functionality for handling
    data output with configurable timing modes and type validation.
    """

    class Configuration(ioc.OPort.Configuration):
        """Configuration class for OPort with g.Pype-specific extensions."""

        class Keys(ioc.OPort.Configuration.Keys):
            """Configuration keys inherited from ioiocore."""

            pass

    def __init__(
        self,
        name: str = Constants.Defaults.PORT_OUT,
        timing: Constants.Timing = Constants.Timing.SYNC,
        **kwargs,
    ):
        """Initialize the output port with g.Pype-specific defaults.

        Args:
            name: Name of the output port.
            timing: Timing mode (SYNC or ASYNC). Defaults to SYNC.
            **kwargs: Additional configuration parameters including 'type'.
                Defaults to np.ndarray if not specified.
        """
        # Extract and set default type for signal processing
        type_key = self.Configuration.Keys.TYPE
        type: str = kwargs.pop(
            type_key, np.ndarray.__name__
        )  # Default to numpy arrays

        # Initialize parent class with g.Pype-specific configuration
        super().__init__(name=name, type=type, timing=timing, **kwargs)
