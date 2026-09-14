from __future__ import annotations

import socket
from typing import List, Optional

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core.i_node import INode
from ..core.i_port import IPort


class _UDPSenderCore(INode):
    """Internal node implementing UDP data transmission.

    This is the actual UDP sender node (pure INode inheritance).
    It is wrapped by the UDPSender chain for distributed operation.

    Transmits data as float64 numpy arrays via UDP packets to a configurable
    target address. Supports optional frame size conversion via buffering.
    """

    #: Default target IP address (localhost)
    DEFAULT_IP = "127.0.0.1"
    #: Default target UDP port number
    DEFAULT_PORT = 56000

    class Configuration(INode.Configuration):
        """Configuration class for UDPSender parameters."""

        class Keys(INode.Configuration.Keys):
            """Configuration keys for UDP sender settings."""

            #: IP address configuration key
            IP = "ip"
            #: Port number configuration key
            PORT = "port"

        class OptionalKeys(INode.Configuration.OptionalKeys):
            """Optional configuration keys.

            The sender may forward data at its incoming frame size, so
            the output frame size must not be declared as required.
            """

            #: Output frame size configuration key
            FRAME_SIZE_OUT = "frame_size_out"

    def __init__(
        self,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        frame_size_out: Optional[int] = None,
        **kwargs,
    ):
        """Initialize UDP sender with target address and port.

        Args:
            ip: Target IP address. Defaults to localhost if None.
            port: Target port number. Defaults to DEFAULT_PORT if None.
            frame_size_out: Output frame size. If None, data is sent with
                its original frame size. If specified, data is buffered
                and sent when frame_size_out samples are accumulated.
            **kwargs: Additional arguments for parent INode.
        """
        # Use default values if not specified
        if ip is None:
            ip = UDPSender.DEFAULT_IP
        if port is None:
            port = UDPSender.DEFAULT_PORT

        # Initialize parent INode with configuration
        INode.__init__(
            self, ip=ip, port=port, frame_size_out=frame_size_out, **kwargs
        )

        # Initialize networking components
        self._socket = None  # UDP socket (created on start)
        self._target = (ip, port)  # Target address tuple

        # Initialize buffering components
        self._buffer = None  # Buffer for frame size conversion
        self._buffer_idx = 0  # Current write position in buffer
        self._frame_size_out = None  # Output frame size (set in setup)

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

        Initializes the output buffer if frame_size_out is configured.

        Args:
            data: Input data arrays from connected ports.
            port_context_in: Context information from input ports.

        Returns:
            Empty dictionary (sink node has no output context).
        """
        # Get frame_size_out from configuration
        self._frame_size_out = self.config.get(
            self.Configuration.OptionalKeys.FRAME_SIZE_OUT
        )

        # Initialize buffer if frame size conversion is needed
        if self._frame_size_out is not None:
            # Get channel count from input context
            channel_count = port_context_in[Constants.Defaults.PORT_IN][
                Constants.Keys.CHANNEL_COUNT
            ]
            # Allocate buffer for frame assembly
            self._buffer = np.zeros(
                shape=(self._frame_size_out, channel_count), dtype=np.float64
            )
            self._buffer_idx = 0

        return {}

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and transmit data via UDP.

        If frame_size_out is configured, accumulates data in buffer and sends
        when the output frame size is reached. Otherwise, sends immediately.

        Args:
            data: Input data arrays. Uses default input port.
                Expected shape: (frame_size_in, channels)

        Returns:
            Empty dictionary (sink node has no output).
        """
        # Get data from default input port
        d = data[Constants.Defaults.PORT_IN]

        # Transmit data if socket is available
        if self._socket:
            if self._frame_size_out is not None:
                # Buffered mode: accumulate until output frame size is reached
                frame_size_in = d.shape[0]
                samples_remaining = frame_size_in
                src_idx = 0

                while samples_remaining > 0:
                    # Calculate how many samples we can copy to the buffer
                    space_in_buffer = self._frame_size_out - self._buffer_idx
                    samples_to_copy = min(samples_remaining, space_in_buffer)

                    # Copy samples to buffer
                    self._buffer[
                        self._buffer_idx : self._buffer_idx + samples_to_copy,
                        :,
                    ] = d[src_idx : src_idx + samples_to_copy, :]

                    self._buffer_idx += samples_to_copy
                    src_idx += samples_to_copy
                    samples_remaining -= samples_to_copy

                    # Send when buffer is full
                    if self._buffer_idx >= self._frame_size_out:
                        payload = self._buffer.tobytes()
                        self._socket.sendto(payload, self._target)
                        self._buffer_idx = 0
            else:
                # Direct mode: send immediately
                # Convert to float64 and serialize to bytes for transmission
                # Using copy=False to avoid unnecessary allocation if already
                # float64
                payload = d.astype(np.float64, copy=False).tobytes()

                # Send UDP packet to target address
                self._socket.sendto(payload, self._target)

        # No output data for sink nodes
        return {}


class UDPSender(ioc.IChain):
    """UDP sender chain for real-time data transmission.

    This is an IChain that contains:
    - Link: Bridge for distributed operation (passthrough in standalone)
    - _UDPSenderCore: The actual UDP transmission node

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Transmits data as float64 numpy arrays via UDP packets to a configurable
    target address.
    """

    #: Default target IP address (localhost)
    DEFAULT_IP = _UDPSenderCore.DEFAULT_IP
    #: Default target UDP port number
    DEFAULT_PORT = _UDPSenderCore.DEFAULT_PORT

    def __init__(
        self,
        ip: Optional[str] = None,
        port: Optional[int] = None,
        frame_size_out: Optional[int] = None,
        **kwargs,
    ):
        """Initialize UDP sender chain.

        Args:
            ip: Target IP address. Defaults to localhost if None.
            port: Target port number. Defaults to DEFAULT_PORT if None.
            frame_size_out: Output frame size. If None, data is sent with
                its original frame size. If specified, data is buffered
                and sent when frame_size_out samples are accumulated.
            **kwargs: Additional arguments.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "ip": ip,
            "port": port,
            "frame_size_out": frame_size_out,
        }
        self._core_params.update(strip_chain_keys(kwargs))

        # Initialize IChain (calls create_internal_nodes)
        kwargs.setdefault(
            self.Configuration.Keys.INPUT_PORTS,
            [IPort.Configuration()],
        )
        # The chain's own parameters go into the chain's own
        # configuration; see Generator for the full rationale.
        ioc.IChain.__init__(
            self,
            ip=ip,
            port=port,
            frame_size_out=frame_size_out,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create the internal node chain.

        Returns:
            List containing [Link, _UDPSenderCore].
        """
        return [
            Link(
                sender=Constants.Residency.SERVER,
                receiver=Constants.Residency.EDGE,
                stream_id=self._link_stream_id,
            ),
            _UDPSenderCore(**self._core_params),
        ]
