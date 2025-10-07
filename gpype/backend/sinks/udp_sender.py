from __future__ import annotations

import socket
from typing import Optional

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.i_node import INode


class UDPSender(INode):
    """UDP sink node for real-time data transmission.

    Transmits data as float64 numpy arrays via UDP packets to a configurable
    target address. Each step() sends one packet with direct transmission.
    """

    #: Default target IP address (localhost)
    DEFAULT_IP = "127.0.0.1"
    #: Default target UDP port number
    DEFAULT_PORT = 56000

    class Configuration(ioc.INode.Configuration):
        """Configuration class for UDPSender parameters."""

        class Keys(ioc.INode.Configuration.Keys):
            """Configuration keys for UDP sender settings."""

            #: IP address configuration key
            IP = "ip"
            #: Port number configuration key
            PORT = "port"

    def __init__(
        self, ip: Optional[str] = None, port: Optional[int] = None, **kwargs
    ):
        """Initialize UDP sender with target address and port.

        Args:
            ip: Target IP address. Defaults to localhost if None.
            port: Target port number. Defaults to DEFAULT_PORT if None.
            **kwargs: Additional arguments for parent INode.
        """
        # Use default values if not specified
        if ip is None:
            ip = UDPSender.DEFAULT_IP
        if port is None:
            port = UDPSender.DEFAULT_PORT

        # Initialize parent INode with configuration
        INode.__init__(self, ip=ip, port=port, **kwargs)

        # Initialize networking components
        self._socket = None  # UDP socket (created on start)
        self._target = (ip, port)  # Target address tuple

    def start(self):
        """Start UDP sender and initialize socket connection.

        Creates UDP socket and configures target address from configuration.

        Raises:
            OSError: If socket creation fails.
        """
        # Create UDP socket for data transmission
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Update target address from current configuration
        self._target = (
            self.config[self.Configuration.Keys.IP],
            self.config[self.Configuration.Keys.PORT],
        )

        # Call parent start method
        super().start()

    def stop(self):
        """Stop UDP sender and clean up socket resources."""
        # Close socket and clean up resources
        if self._socket:
            self._socket.close()
            self._socket = None

        # Call parent stop method
        super().stop()

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup method for pipeline initialization.

        Args:
            data: Input data arrays from connected ports.
            port_context_in: Context information from input ports.

        Returns:
            Empty dictionary (sink node has no output context).
        """
        # No setup required for UDP transmission
        return {}

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and transmit data via UDP.

        Converts data to float64, serializes to bytes, and sends via UDP.

        Args:
            data: Input data arrays. Uses default input port.

        Returns:
            Empty dictionary (sink node has no output).
        """
        # Get data from default input port
        d = data[Constants.Defaults.PORT_IN]

        # Transmit data if socket is available
        if self._socket:
            # Convert to float64 and serialize to bytes for transmission
            payload = d.astype(np.float64).tobytes()

            # Send UDP packet to target address
            self._socket.sendto(payload, self._target)

        # No output data for sink nodes
        return {}
