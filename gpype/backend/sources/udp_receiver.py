import select
import socket
import threading
import time

from ...common.constants import Constants
from .base.event_source import EventSource

# Output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class UDPReceiver(EventSource):
    """UDP network receiver for capturing remote trigger events.

    Provides network-based event capture. Listens on specified IP/port
    for UDP packets containing numeric trigger values. Each trigger
    outputs the received value, followed immediately by a zero.
    """

    # Source code fingerprint
    FINGERPRINT = "e4558a93e67f779b9a8fd010962ebc93"

    # Default network configuration
    DEFAULT_IP: str = "127.0.0.1"  # Localhost
    DEFAULT_PORT: int = 1000  # Default UDP port

    class Configuration(EventSource.Configuration):
        """Configuration class for UDP receiver network parameters."""

        class Keys(EventSource.Configuration.Keys):
            """Configuration key constants for the UDP receiver."""

            IP: str = "ip"  # IP address to bind to
            PORT: str = "port"  # UDP port number to listen on

    def __init__(
        self, ip: str = DEFAULT_IP, port: int = DEFAULT_PORT, **kwargs
    ):
        """Initialize UDP receiver.

        Args:
            ip: IP address to bind socket to. Use "0.0.0.0" for all interfaces
                or "127.0.0.1" for localhost. Defaults to localhost.
            port: UDP port number to listen on. Defaults to 1000.
            **kwargs: Additional parameters for EventSource base class.
        """
        # Initialize parent EventSource with network configuration
        super().__init__(ip=ip, port=port, **kwargs)

        # Initialize UDP receiver state
        self._udp_thread_running = False
        self._socket = None
        self._udp_thread = None
        self._t_start = None  # Start time tracking

    def _udp_listener(self):
        """Background thread function for UDP message reception.

        Creates UDP socket and continuously listens for incoming messages.
        Parses numeric string data and triggers events. Uses select() for
        non-blocking operation to allow clean shutdown.
        """
        # Create UDP socket for receiving messages
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setblocking(False)  # Non-blocking for select()

        # Get network configuration
        ip_key = self.Configuration.Keys.IP
        port_key = self.Configuration.Keys.PORT
        ip = self.config[ip_key]
        port = self.config[port_key]

        # Bind socket to specified address and port
        self._socket.bind((ip, port))

        # Main reception loop
        while self._udp_thread_running:
            # Use select with timeout for non-blocking check
            ready, _, _ = select.select([self._socket], [], [], 0.01)
            if not ready:
                continue
            if not self._udp_thread_running:
                break

            try:
                # Receive UDP packet (max 1024 bytes)
                data, _ = self._socket.recvfrom(1024)
                message = data.decode().strip()

                # Parse message - only numeric strings are supported
                if message.isdigit():
                    # Valid integer format: "123"
                    value = int(message)
                else:
                    # Non-numeric message, skip silently
                    raise ValueError("Unsupported message format")

                # Trigger events: first the value, then 0 for completion
                self.trigger(value)
                self.trigger(0)

            except Exception:
                # Handle any decoding or parsing errors silently
                continue

    def start(self):
        """Start UDP receiver and begin listening for messages.

        Initializes background UDP listener thread and starts monitoring
        for incoming trigger messages.
        """
        # Start parent EventSource
        super().start()

        # Start UDP listener thread if not already running
        if not self._udp_thread_running:
            self._udp_thread_running = True
            self._udp_thread = threading.Thread(
                target=self._udp_listener, daemon=True
            )
            self._udp_thread.start()

        # Record start time for potential timing analysis
        self._t_start = time.perf_counter()

    def stop(self):
        """Stop UDP receiver and cleanup network resources.

        Stops background listener thread, closes UDP socket, and waits for
        clean thread termination.
        """
        # Stop parent EventSource first
        super().stop()

        # Stop UDP listener thread and cleanup resources
        if self._udp_thread_running:
            self._udp_thread_running = False

            # Close socket to release network resources
            if self._socket:
                self._socket.close()
                self._socket = None

            # Wait for thread to complete
            if self._udp_thread:
                self._udp_thread.join()
                self._udp_thread = None
