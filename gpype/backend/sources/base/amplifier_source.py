from __future__ import annotations

from typing import Any

import numpy as np

from ....common.constants import Constants
from ...core.o_port import OPort
from .source import Source

# Convenience constant for default output port name
PORT_OUT = Constants.Defaults.PORT_OUT


class AmplifierSource(Source):
    """Base class for amplifier-based data acquisition sources.

    This abstract class provides the foundation for interfacing with hardware
    amplifiers and data acquisition devices in BCI applications. It extends
    the basic Source functionality with amplifier-specific configuration
    and device management capabilities.

    The class handles:
    - Hardware device enumeration and management
    - Sampling rate configuration with inheritance support
    - Multi-channel data acquisition setup
    - Frame-based data buffering configuration

    Features:
    - Support for INHERITED sampling rate (determined at runtime)
    - Device list management for hardware abstraction
    - Automatic port configuration for amplifier data streams
    - Integration with g.Pype timing and context framework

    Attributes:
        _devices: List of available amplifier devices
        _device: Currently selected/connected amplifier device

    Note:
        This is an abstract base class. Concrete implementations must
        provide device-specific connection and data acquisition logic.
        The sampling rate can be set to Constants.INHERITED (-1) to
        indicate it should be determined during device setup.
    """

    class Configuration(Source.Configuration):
        """Configuration class for AmplifierSource parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for amplifier source settings."""

            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE  # 'sampling_rate'

        def __init__(self, sampling_rate: float, **kwargs):
            """Initialize configuration with sampling rate validation.

            Args:
                sampling_rate: Sampling rate in Hz. Must be positive, or
                    Constants.INHERITED (-1) to indicate the rate should
                    be determined during device setup.
                **kwargs: Additional configuration parameters.

            Raises:
                ValueError: If sampling_rate is not positive and not INHERITED.

            Note:
                INHERITED sampling rate allows the amplifier hardware to
                determine the optimal sampling rate during initialization.
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
        """Initialize the amplifier source with acquisition parameters.

        Args:
            sampling_rate: Sampling rate in Hz. Must be positive or
                Constants.INHERITED for runtime determination.
            channel_count: Number of data channels to acquire from amplifier.
                Must be positive integer.
            frame_size: Number of samples per data frame. Must be positive
                integer. Affects processing latency and throughput.
            **kwargs: Additional arguments passed to parent Source class.

        Example:
            Standard EEG amplifier configuration::

                amp = AmplifierSource(sampling_rate=1000.0,
                                    channel_count=64,
                                    frame_size=256)

            Inherited sampling rate (determined by hardware)::

                amp = AmplifierSource(sampling_rate=Constants.INHERITED,
                                    channel_count=32,
                                    frame_size=128)

        Note:
            Output ports are automatically configured if not provided.
            Device list is initialized empty and should be populated
            by concrete implementations during device discovery.
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

        Configures output port contexts by adding sampling rate information
        to each configured output port. This method ensures downstream nodes
        receive accurate timing information for amplifier data processing.

        Args:
            data: Dictionary of input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with sampling_rate information
            added to each configured output port.

        Note:
            The sampling rate may have been determined during device
            initialization if it was set to Constants.INHERITED. All
            downstream nodes will receive the actual sampling rate value
            for proper signal processing configuration.
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
