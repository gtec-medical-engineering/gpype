from __future__ import annotations

import socket
from typing import Optional

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.i_node import INode


class UDPSender(INode):
    """UDP sender for real-time data transmission over network.

    This class implements a UDP-based data sender that transmits incoming
    data blocks immediately during step() execution. The sender uses direct
    UDP transmission without additional threading to maintain real-time
    performance and minimize latency.

    The data is automatically serialized as float64 numpy arrays before
    transmission, ensuring consistent data format across the network. Each
    step() call results in one UDP packet being sent.

    Features:
    - Direct UDP transmission without threading overhead
    - Automatic float64 serialization for network compatibility
    - Configurable IP address and port settings
    - Real-time safe operation with minimal latency

    Attributes:
        DEFAULT_IP: Default target IP address (localhost)
        DEFAULT_PORT: Default target port number
        _socket: UDP socket for data transmission
        _target: Target address tuple (ip, port)

    Note:
        Data is serialized as float64 numpy array before sending. One UDP
        packet is sent per step() call. The receiving end is responsible
        for proper deserialization of the binary data.
    """

    DEFAULT_IP = "127.0.0.1"
    DEFAULT_PORT = 56000

    class Configuration(ioc.INode.Configuration):
        """Configuration class for UDPSender parameters."""

        class Keys(ioc.INode.Configuration.Keys):
            """Configuration keys for UDP sender settings."""

            IP = "ip"
            PORT = "port"

    def __init__(
        self, ip: Optional[str] = None, port: Optional[int] = None, **kwargs
    ):
        """Initialize the UDP sender with target address and port.

        Args:
            ip: Target IP address for UDP transmission. If None, uses
                DEFAULT_IP (localhost). Can be any valid IPv4 address.
            port: Target port number for UDP transmission. If None, uses
                DEFAULT_PORT. Must be a valid port number (1-65535).
            **kwargs: Additional arguments passed to parent INode class.
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
        """Start the UDP sender and initialize socket connection.

        Creates a UDP socket and configures the target address from
        the current configuration. The socket is ready for immediate
        data transmission after this method completes.

        Raises:
            OSError: If socket creation fails or address is invalid.

        Note:
            The target address is read from configuration to support
            dynamic address changes between start/stop cycles.
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
        """Stop the UDP sender and clean up socket resources.

        Properly closes the UDP socket if it exists and resets the
        socket reference to None. This ensures clean resource cleanup
        and prevents potential network resource leaks.

        Note:
            Socket closure is handled gracefully - if the socket is
            already closed or None, no error is raised.
        """
        # Close socket and clean up resources
        if self._socket:
            self._socket.close()
            self._socket = None

        # Call parent stop method
        super().stop()

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup method called before processing begins.

        This method is called during pipeline initialization but requires
        no specific setup for UDP transmission since all configuration
        is handled during start().

        Args:
            data: Dictionary of input data arrays from connected ports.
            port_context_in: Context information from input ports.

        Returns:
            Empty dictionary as this is a sink node with no output context.

        Note:
            UDP transmission requires no data-dependent configuration,
            so this method simply returns an empty context dictionary.
        """
        # No setup required for UDP transmission
        return {}

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and transmit data via UDP.

        Retrieves data from the default input port, converts it to float64
        format, serializes it to bytes, and transmits it via UDP to the
        configured target address.

        Args:
            data: Dictionary containing input data arrays. Uses the default
                input port to retrieve data for transmission.

        Returns:
            Empty dictionary as this is a sink node with no output data.

        Note:
            Data is automatically converted to float64 format before
            serialization to ensure consistent network representation.
            Each call results in exactly one UDP packet transmission.
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
