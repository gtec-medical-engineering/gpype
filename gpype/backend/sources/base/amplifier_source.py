from __future__ import annotations

from typing import Any

import numpy as np

from ....common.constants import Constants
from ...core.o_port import OPort
from .source import Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class AmplifierSource(Source):
    """Base class for amplifier-based data acquisition sources.

    Provides hardware device management, sampling rate configuration,
    and multi-channel data acquisition setup for BCI applications.
    """

    class Configuration(Source.Configuration):
        """Configuration class for AmplifierSource parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for amplifier source settings."""

            #: Sampling rate configuration key
            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE

        def __init__(self, sampling_rate: float, **kwargs):
            """Initialize configuration with sampling rate validation.

            Args:
                sampling_rate: Sampling rate in Hz. Must be positive or
                    Constants.INHERITED for runtime determination.
                **kwargs: Additional configuration parameters.

            Raises:
                ValueError: If sampling_rate is not positive and not INHERITED.
            """
            # Validate sampling rate (allow INHERITED for runtime config)
            if sampling_rate != Constants.INHERITED and sampling_rate <= 0:
                raise ValueError("sampling_rate must be greater than zero.")
            super().__init__(sampling_rate=sampling_rate, **kwargs)

    # Class attributes for device management
    _devices: list[Any]
    _device: Any

    def __init__(
        self,
        sampling_rate: float,
        channel_count: int,
        frame_size: int,
        **kwargs,
    ):
        """Initialize amplifier source with acquisition parameters.

        Args:
            sampling_rate: Sampling rate in Hz. Must be positive or
                Constants.INHERITED for runtime determination.
            channel_count: Number of data channels to acquire.
            frame_size: Number of samples per data frame.
            **kwargs: Additional arguments for parent Source class.
        """
        # Extract output_ports from kwargs with default configuration
        op_key = AmplifierSource.Configuration.Keys.OUTPUT_PORTS
        output_ports: list[OPort.Configuration] = kwargs.pop(
            op_key, [OPort.Configuration()]
        )

        # Initialize device management
        self._devices = []  # List of available devices

        # Initialize parent Source with amplifier configuration
        Source.__init__(
            self,
            output_ports=output_ports,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            **kwargs,
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with sampling rate information.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with sampling_rate information.
        """
        # Call parent setup to initialize base contexts
        port_context_out = super().setup(data, port_context_in)

        # Get actual sampling rate (may have been resolved from INHERITED)
        sampling_rate = self.config[self.Configuration.Keys.SAMPLING_RATE]
        frame_size = port_context_out[Constants.Defaults.PORT_OUT][
            Constants.Keys.FRAME_SIZE
        ]
        frame_rate = sampling_rate / frame_size
        out_ports = self.config[self.Configuration.Keys.OUTPUT_PORTS]

        # Add sampling rate context to each output port
        for i in range(len(out_ports)):
            # Create context with sampling rate information
            context = {
                Constants.Keys.SAMPLING_RATE: sampling_rate,
                Constants.Keys.FRAME_RATE: frame_rate,
            }

            # Get port name and update its context
            port_name = out_ports[i][OPort.Configuration.Keys.NAME]
            port_context_out[port_name].update(context)

        return port_context_out
