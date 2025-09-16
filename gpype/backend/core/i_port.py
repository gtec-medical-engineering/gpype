from __future__ import annotations

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants


class IPort(ioc.IPort):
    """Input port class for g.Pype signal processing nodes.

    Extends ioiocore.IPort with g.Pype-specific functionality for handling
    data input with configurable timing modes and type validation.
    """

    class Configuration(ioc.IPort.Configuration):
        """Configuration class for IPort with g.Pype-specific extensions."""

        class Keys(ioc.IPort.Configuration.Keys):
            """Configuration keys inherited from ioiocore."""

            pass

    def __init__(
        self,
        name: str = Constants.Defaults.PORT_IN,
        timing: Constants.Timing = Constants.Timing.SYNC,
        **kwargs,
    ):
        """Initialize the input port with g.Pype-specific defaults.

        Args:
            name: Name of the input port.
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
