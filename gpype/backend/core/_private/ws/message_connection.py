"""A broker connection whose messages the host carries, not a socket.

Where no thread can be started there is no server to bind either: a
browser page talks to the pipeline in its Web Worker by ``postMessage``.
:class:`WsBroker` serves such a client through the same connection
handler it runs for a WebSocket, which needs only this much of one:
asynchronous iteration over incoming messages, ``send``, ``close``,
``ping``, ``transport.abort()`` and ``remote_address``. Everything the
handler does with them -- tokens, subscriptions, one sender per stream,
contexts, the frame codec -- is unchanged, so a page and an edge speak
one protocol.

The host owns both ends of the carrier: it calls :meth:`feed` with each
message the client sent and :meth:`disconnect` when the client goes; the
broker's messages reach ``on_send``. All of it runs on the event loop's
thread -- under Pyodide there is no other.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from typing import Callable, Optional, Union

log = logging.getLogger(__name__)

#: Wakes the reader once the connection is closed.
_CLOSED = object()

_serials = itertools.count(1)


class _Transport:
    """What ``WsBroker._ping_sender`` aborts: the connection itself."""

    def __init__(self, connection: "MessageConnection"):
        self._connection = connection

    def abort(self) -> None:
        self._connection.end(1006, "aborted")


class MessageConnection:
    """One client of the broker, carried by the host's messages.

    Args:
        on_send: Called with each message for the client, a ``str`` for
            a text frame and ``bytes`` for a binary one. Under Pyodide a
            JS function receives ``bytes`` as a proxy; ``toJs()`` makes
            it a ``Uint8Array``.
        on_close: Called once, with a close code and reason, when the
            broker closes the connection or the client disconnects.
        name: How the connection is named in the broker's log lines.
    """

    def __init__(
        self,
        on_send: Callable[[Union[str, bytes]], None],
        on_close: Optional[Callable[[int, str], None]] = None,
        name: str = "page",
    ):
        self._on_send = on_send
        self._on_close = on_close
        self._incoming: asyncio.Queue = asyncio.Queue()
        self._closed = False
        #: Read by the broker's log lines and its sender bookkeeping.
        self.remote_address = (name, next(_serials))
        #: No handshake, so no path: never a clock-only connection.
        self.request = None
        self.transport = _Transport(self)

    @property
    def closed(self) -> bool:
        """Whether either side has closed the connection."""
        return self._closed

    # -- the host's side -------------------------------------------------

    def feed(self, message) -> None:
        """Deliver one message the client sent.

        Args:
            message: ``str`` for text; ``bytes``, a buffer, or a JS
                typed array (anything with ``to_bytes()``) for binary.
                Ignored once the connection is closed.
        """
        if self._closed:
            return
        if not isinstance(message, (str, bytes)):
            to_bytes = getattr(message, "to_bytes", None)
            message = to_bytes() if callable(to_bytes) else bytes(message)
        self._incoming.put_nowait(message)

    def disconnect(self) -> None:
        """The client went away; the broker releases what it held."""
        self.end(1001, "client disconnected")

    def end(self, code: int, reason: str) -> None:
        """Close once, wake the reader, and tell the host.

        Args:
            code: WebSocket close code, for the host.
            reason: Why, for the host.
        """
        if self._closed:
            return
        self._closed = True
        # What the client sent and the broker has not read goes with the
        # connection. Read after a stop it would rebuild what the stop
        # has just cleared, and a run started at once would receive it.
        while not self._incoming.empty():
            self._incoming.get_nowait()
        self._incoming.put_nowait(_CLOSED)
        if self._on_close is not None:
            try:
                self._on_close(code, reason)
            except Exception:  # a host's callback must not break a stop
                log.debug("on_close raised", exc_info=True)

    # -- the broker's side ---------------------------------------------

    def __aiter__(self) -> "MessageConnection":
        return self

    async def __anext__(self):
        if self._closed:
            raise StopAsyncIteration
        message = await self._incoming.get()
        if message is _CLOSED or self._closed:
            raise StopAsyncIteration
        return message

    async def send(self, message: Union[str, bytes]) -> None:
        """Hand one message to the client.

        Raises:
            ConnectionError: Once the connection is closed, as a closed
                WebSocket raises.
        """
        if self._closed:
            raise ConnectionError("the connection is closed")
        self._on_send(message)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        """Close from the broker's side."""
        self.end(code, reason)

    async def ping(self) -> asyncio.Future:
        """Answer at once: the client is in this process.

        Returns:
            An already-resolved waiter, as ``websockets`` returns one.

        Raises:
            ConnectionError: Once the connection is closed.
        """
        if self._closed:
            raise ConnectionError("the connection is closed")
        pong = asyncio.get_running_loop().create_future()
        pong.set_result(None)
        return pong
