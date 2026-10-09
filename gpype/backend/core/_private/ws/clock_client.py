"""The edge's connection for clock probes.

A second WebSocket to the same broker endpoint as the data, opened with
``compression=None``: permessage-deflate adds variable latency to a
message whose only purpose is to be timed, so it is off here and only
here (D-TIME-19). The data connections keep it. The endpoint gets
:data:`~gpype._wire.CLOCK_QUERY`, so the broker does not take this
connection for an edge that holds a device (D-TIME-66).

Blocking, on the websockets sync client, because the one caller is a
start that waits for the answer anyway: an event loop would add a
thread hop to every round trip it measures.
"""

from __future__ import annotations

import inspect
import json
import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import websockets.sync.client

from ....._transport import endpoint_is_secure
from ....._wire import CLOCK_QUERY, decode_clock, encode_clock
from .client import CLOSE_TIMEOUT_S

#: ``websockets.sync.client.connect``, bound at import time, on the
#: thread that imports this module: see ``_ws_connect`` in ``client.py``
#: for the bus error that a first touch from another thread caused.
_ws_sync_connect = websockets.sync.client.connect

#: The keyword the sync client takes a TLS context under: ``ssl`` from
#: websockets 14, ``ssl_context`` before it (the floor is 12).
_SSL_KEYWORD = (
    "ssl"
    if "ssl" in inspect.signature(_ws_sync_connect).parameters
    else "ssl_context"
)

#: How long one probe waits for its reply before it counts as lost.
PROBE_TIMEOUT_S = 0.5


def clock_endpoint(endpoint: str) -> str:
    """Return *endpoint* with the clock role added to its query.

    Args:
        endpoint: The broker endpoint the data connections dial.

    Returns:
        The same URL, its query extended by :data:`CLOCK_QUERY`.
    """
    parts = urlsplit(endpoint)
    query = f"{parts.query}&{CLOCK_QUERY}" if parts.query else CLOCK_QUERY
    path = parts.path or "/"
    return urlunsplit((parts.scheme, parts.netloc, path, query, ""))


class ClockConnection:
    """One edge's probe connection to its broker.

    The transport ``clock_sync.synchronise`` drives: :meth:`connect`,
    then :meth:`probe` as ``ClockSync``'s probe, then :meth:`close`.

    Args:
        endpoint: The broker endpoint, as ``LaunchConfig.endpoint``.
        probe_timeout: Seconds a probe waits for its reply.
    """

    def __init__(self, endpoint: str, probe_timeout: float = PROBE_TIMEOUT_S):
        self.endpoint = endpoint
        self._probe_timeout = float(probe_timeout)
        self._ws = None
        #: Why the last connection attempt or probe failed, for a message.
        self.last_error: Optional[str] = None

    @property
    def connected(self) -> bool:
        """Whether a connection is open."""
        return self._ws is not None

    def connect(self, timeout: float) -> bool:
        """Open the connection and authenticate, within *timeout* seconds.

        Args:
            timeout: Seconds the whole attempt may take.

        Returns:
            True if the connection is open and, when this process has a
            data token, the broker accepted it.
        """
        self.close()
        deadline = time.monotonic() + max(0.0, float(timeout))
        options = {
            "compression": None,
            "open_timeout": max(0.1, timeout),
            "close_timeout": CLOSE_TIMEOUT_S,
        }
        tls = self._tls_context()
        if tls is not None:
            options[_SSL_KEYWORD] = tls
        try:
            self._ws = _ws_sync_connect(
                clock_endpoint(self.endpoint), **options
            )
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            return False
        try:
            if not self._authenticate(deadline):
                self.close()
                return False
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.close()
            return False
        self.last_error = None
        return True

    def _tls_context(self):
        """The TLS context to dial with, or None for a plain socket."""
        if not endpoint_is_secure(self.endpoint):
            return None
        from .....common.launch_config import LaunchConfig

        tls = LaunchConfig.get().client_tls()
        return tls.context() if tls is not None else None

    def _authenticate(self, deadline: float) -> bool:
        """Present this process's data token, and wait for the verdict.

        Unlike ``WsClient._authenticate`` this waits: the round trip is
        paid once, at start, and a refused token is then named as such
        rather than as probes that went unanswered.

        Args:
            deadline: ``time.monotonic()`` reading to give up at.

        Returns:
            True when there is no token or the broker accepted it.
        """
        from .....common.launch_config import LaunchConfig

        config = LaunchConfig.get()
        token = getattr(config, "data_token", None) if config else None
        if not token:
            return True
        self._ws.send(json.dumps({"command": "auth", "token": token}))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                self.last_error = "the broker did not answer the data token"
                return False
            message = self._ws.recv(timeout=remaining)
            try:
                reply = json.loads(message)
            except Exception:
                continue
            if isinstance(reply, dict) and reply.get("event") == "auth":
                if reply.get("status") == "ok":
                    return True
                self.last_error = "the broker refused this edge's data token"
                return False

    def probe(self, t0: float) -> Optional[tuple]:
        """Send one probe and wait for its reply.

        Args:
            t0: The edge's clock at sending, echoed by the broker.

        Returns:
            ``(t1, t2, epoch)``, or None for a reply that did not come.
        """
        if self._ws is None:
            self.last_error = "not connected"
            return None
        deadline = time.monotonic() + self._probe_timeout
        try:
            self._ws.send(encode_clock(t0))
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("a clock probe got no reply")
                reply = decode_clock(self._ws.recv(timeout=remaining))
                # A reply to an earlier, lost probe arrives late and
                # carries its own t0: skip it rather than time it.
                if reply is not None and "t1" in reply and reply["t0"] == t0:
                    return (reply["t1"], reply["t2"], reply["epoch"])
        except TimeoutError as error:
            self.last_error = str(error)
            return None
        except Exception as error:
            self.last_error = f"{type(error).__name__}: {error}"
            self.close()
            return None

    def extensions(self) -> Optional[str]:
        """The extensions the handshake negotiated, or None for none."""
        response = getattr(self._ws, "response", None)
        headers = getattr(response, "headers", None)
        return (
            None
            if headers is None
            else headers.get("Sec-WebSocket-Extensions")
        )

    def close(self) -> None:
        """Close the connection, if one is open. Safe to call twice."""
        ws, self._ws = self._ws, None
        if ws is None:
            return
        try:
            ws.close()
        except Exception:
            pass
