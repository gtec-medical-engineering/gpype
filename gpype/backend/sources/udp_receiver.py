import socket
import threading
import numpy as np
from ...common.constants import Constants
from .base.event_source import EventSource

PORT_OUT = Constants.Defaults.PORT_OUT


class UDPReceiver(EventSource):

    FINGERPRINT = "9f9be5078cbbe3aa921df3b740a304b1"
    DEFAULT_IP: str = "127.0.0.1"
    DEFAULT_PORT: str = 1000

    class Configuration(EventSource.Configuration):
        class Keys(EventSource.Configuration.Keys):
            IP: str = "ip"
            PORT: str = "port"

    def __init__(self,
                 ip: str = DEFAULT_IP,
                 port: int = DEFAULT_PORT,
                 **kwargs):
        super().__init__(ip=ip, port=port, **kwargs)
        self._running = False
        self._socket = None
        self._thread = None

    def _udp_listener(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ip_key = self.Configuration.Keys.IP
        port_key = self.Configuration.Keys.PORT
        ip = self.config[ip_key]
        port = self.config[port_key]
        self._socket.bind((ip, port))
        import re

        while self._running:
            try:
                data, _ = self._socket.recvfrom(1024)  # Buffer size
                message = data.decode().strip()
                if message.isdigit():  # Ensure it's an integer
                    value = int(message)
                else:
                    m = re.search(r'name="(\d+)"', message)
                    if m:
                        value = int(m.group(1))
                    else:
                        continue
                self.trigger({PORT_OUT: np.array([[value]])})
                self.trigger({PORT_OUT: np.array([[0]])})
            except Exception:
                continue

    def start(self):
        """ Starts the UDP receiver. """
        super().start()
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._udp_listener,
                                            daemon=True)
            self._thread.start()

    def stop(self):
        """ Stops the UDP receiver. """
        super().stop()
        if self._running:
            self._running = False
            if self._socket:
                self._socket.close()
            if self._thread:
                self._thread.join()
