import select
import socket
import threading
import time

from ...common.constants import Constants
from .base.event_source import EventSource

# Output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class UDPReceiver(EventSource):
    """
    UDP network receiver for capturing remote trigger events.

    This class provides network-based event capture for BCI applications,
    enabling remote systems to send trigger signals via UDP packets. It
    listens on a specified IP address and port for incoming messages,
    parsing them to extract trigger values.

    The receiver supports plain integer message format where the entire
    UDP packet content should be a numeric string (e.g., "123").

    Each received trigger generates two events: first the trigger value,
    then immediately a value of 0 to indicate trigger completion.

    Features:
        - UDP network communication
        - Non-blocking socket operation
        - Plain integer message format
        - Background thread operation
        - Configurable IP address and port
        - Automatic trigger completion signaling

    Note:
        Requires network permissions and available UDP port.
        Firewall settings may need adjustment for remote connections.
        Only numeric string messages are supported (e.g., "123", "42").
    """

    # Source code fingerprint
    FINGERPRINT = "16094605bf234cec0f98f6edf2392b8d"

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
        """
        Initialize the UDP receiver.

        Args:
            ip (str): IP address to bind the UDP socket to. Use "0.0.0.0"
                to listen on all available interfaces, or "127.0.0.1" for
                localhost only. Defaults to localhost.
            port (int): UDP port number to listen on. Must be available
                and not blocked by firewall. Defaults to 1000.
            **kwargs: Additional configuration parameters passed to
                EventSource base class.
        """
        # Initialize parent EventSource with network configuration
        super().__init__(ip=ip, port=port, **kwargs)

        # Initialize UDP receiver state
        self._udp_thread_running = False
        self._socket = None
        self._udp_thread = None
        self._t_start = None  # Start time tracking

    def _udp_listener(self):
        """
        Background thread function for UDP message reception.

        Creates and manages the UDP socket, continuously listening for
        incoming messages. Parses received data and triggers appropriate
        events in the BCI pipeline. Runs until _udp_thread_running is False.

        Message format:
        - Plain integers only: "123" -> triggers 123, then 0

        The method uses select() for non-blocking operation to allow
        clean shutdown when requested. Only numeric string messages
        are accepted; non-numeric messages are silently ignored.
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
        """
        Start the UDP receiver and begin listening for messages.

        Initializes the background UDP listener thread and starts monitoring
        for incoming trigger messages. The receiver will continue operating
        until stop() is called.
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
        """
        Stop the UDP receiver and cleanup network resources.

        Stops the background listener thread, closes the UDP socket, and
        waits for clean thread termination. This ensures proper resource
        cleanup and network port release.
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
