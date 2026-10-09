"""Singleton WebSocket broker for SERVER-side Link data routing.

One broker serves all Link nodes in a pipeline.  It binds on a single
port (derived from LaunchConfig.endpoint) and multiplexes data by
stream_id:

  PUT      – a binary frame from an EDGE sender is routed to the
             registered receiver callback for that stream_id. A stream
             has one sender: the first connection to write it, by a
             frame or a context, holds it until it disconnects, and
             another connection's writes to it are refused.

  SUBSCRIBE – a text-JSON message ``{"command": "subscribe",
             "stream_id": "..."}`` from an EDGE receiver registers the
             WS connection as a subscriber for a stream_id.  The broker
             pushes binary frames to it when push() is called.

  CLOCK    – a text-JSON probe ``{"command": "clock", "t0": ...}`` is
             answered on the same connection with this process's
             ``channels.stamp_clock()`` at receive and at send, and its
             epoch: the server's reference an edge syncs to (D-TIME-66).

The broker is a process-wide singleton.  Link nodes call acquire() when
they start and release() when they stop; the underlying WS server starts
on the first acquire() and stops after the last release().
"""

from __future__ import annotations

import asyncio
import errno
import hmac
import json
import logging
import queue
import socket
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Set
from urllib.parse import urlparse

import numpy as np
import websockets

# The codec and the transport policy live at the package root, five
# levels up: both are contracts with the runtime, not part of the node
# layer.
from ....._transport import ServerTls, endpoint_is_secure
from ....._wire import (
    CLOCK_QUERY,
    decode,
    decode_attest,
    decode_clock,
    decode_context,
    encode,
    encode_attest,
    encode_clock_reply,
    encode_context,
)
from .....common._private import channels
from .. import threads

# Both ends of the link run a background asyncio loop and both have to
# close it, so the teardown lives once, next to the other one, rather
# than as two copies that can drift. The dependency already runs this
# way -- attest_channels() imports WsAttestChannel from the same module.
from .client import close_loop

log = logging.getLogger(__name__)

#: ``websockets.serve``, bound at import time.
#:
#: `websockets` defers its submodules to a module-level ``__getattr__``,
#: so the first touch of ``websockets.serve`` runs a real import. That
#: first touch used to be the ``await`` in ``_serve``, inside
#: the broker's own event-loop thread -- which ran the import
#: machinery, and a bytecode compile, on a background thread while the
#: main thread was importing and collecting.
#:
#: Measured in CI on macos-14 / CPython 3.12, intermittently, taking the
#: whole suite down with exit -10:
#:
#:     Fatal Python error: Bus error
#:     Current thread (most recent call first):
#:       Garbage-collecting
#:       <frozen importlib._bootstrap_external>:757 in _compile_bytecode
#:       websockets/asyncio/server.py:13 in <module>
#:       websockets/imports.py:78 in __getattr__
#:
#: SIGBUS rather than SIGSEGV is the tell: it is a fault reading a
#: mapped file, which is what a bytecode cache written by one thread and
#: read by another looks like. Binding the name here moves all of it
#: onto whichever thread imports this module, which is the thread that
#: builds the node. Never reach for ``websockets.serve`` from the
#: event-loop thread.
_ws_serve = websockets.serve

#: Port used when an endpoint names none.
DEFAULT_PORT = 8765

#: Reading a stream: ``subscribe``, and receiving its context.
CAP_SUBSCRIBE = "subscribe"

#: Writing to a stream: binary PUT frames, and describing them with a
#: ``context``. Separate from reading because fan-in is a *single
#: receiver slot* -- a connection that may PUT can take a stream before
#: the amplifier's edge connects, and it was once able to interleave
#: frames into one that edge was already feeding. Neither is a privilege
#: a viewer should get by being able to reach the port. A stream's
#: sender is now the first connection to write it (``_claim_sender``),
#: which closes the second and not the first.
CAP_PUT = "put"

#: What a connection may do when the deployment configured no tokens at
#: all, which is every STANDALONE run and every LAN deployment to date.
_ALL_CAPABILITIES = frozenset({CAP_SUBSCRIBE, CAP_PUT})

#: How long a connection has to present a token before the broker
#: closes it, when this deployment requires one. Generous, because a
#: client authenticates as its first act and a slow one is a slow
#: network rather than an attack -- the point is that an idle
#: unauthenticated socket cannot be held open indefinitely, not to
#: race a legitimate client.
AUTH_TIMEOUT_S = 10.0

#: How long acquire() waits for the server thread to reach its listen
#: call before declaring the bind stuck.
#:
#: Was 10 s, and that was measured too tight. Under a loaded process --
#: a full test suite, or a container sharing a busy host -- the thread
#: can be starved long enough to miss the window while being perfectly
#: healthy, and the caller then sees a bind failure for a port nothing
#: was ever wrong with. Observed at 94% of a 2582-test run, on a bind
#: that was not the first in the process, alongside a backlog of pending
#: asyncio tasks from earlier work.
#:
#: The window exists to catch a bind that is *stuck*, and a stuck bind
#: stays stuck: a longer wait costs nothing diagnostically and removes a
#: false positive that is real. It is not a test-only concern -- the same
#: starvation on a busy server is a session that will not start.
BIND_TIMEOUT_S = 30.0

#: How long the shutdown waits for the WS server to finish closing
#: before it stops the loop regardless.
#:
#: ``Server.close()`` closes every open connection with code 1001 and
#: waits for each handler to return, and closing a connection means
#: waiting for the peer's half of the closing handshake. A peer that has
#: gone away without closing -- a crashed edge, a pulled cable -- never
#: sends it, so this wait has to be bounded or a stop never finishes.
#:
#: It is a bound, not the mechanism: the reason a *cooperating* peer used
#: to hit it was a defect in WsClient.disconnect(), which abandoned its
#: socket instead of closing it. With that fixed, a loopback shutdown
#: completes in single-digit milliseconds and never reaches this number.
SHUTDOWN_TIMEOUT_S = 2.0

#: How long a stop waits for the loop thread to finish before giving up
#: on it and saying so.
STOP_TIMEOUT_S = 5.0

#: How long the connection holding a stream has to answer a ping once
#: another connection tries to write that stream, before the broker
#: closes it and the stream passes on. See WsBroker._probe_sender.
#:
#: The case it exists for is a sender that reconnected: a phone that
#: changed network dials again at once, while its old socket is still
#: open here until websockets' own keepalive gives up on it, 20 to 40 s
#: later. If the old socket also subscribed, what the broker pushes to
#: it piles up until its writer pauses, and the keepalive never gives
#: up: its ping waits for that writer. Only TCP then closes it. Without
#: a probe the new connection's frames are refused for that long.
#:
#: The budget covers getting the ping out as well as its answer, so a
#: live subscriber more than this far behind is taken for gone. Five
#: seconds is generous for a pong on a live link and short next to the
#: keepalive.
SENDER_PROBE_TIMEOUT_S = 5.0

#: How often one holder is probed at most, however many refused frames
#: arrive. Two live senders then cost one ping every few seconds.
SENDER_PROBE_INTERVAL_S = 5.0

#: The option that keeps the broker's port to itself: Windows only, None
#: elsewhere, where the bind is unchanged (D-CORE-133). Without it,
#: Windows lets another program bind the port on ``127.0.0.1`` or
#: ``::1``, before the broker or after it, and hands that address's
#: connections to the other program: a local edge dialling ``localhost``
#: reaches it instead of the broker. With it, either bind fails.
_EXCLUSIVE_OPTION = (
    getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
    if sys.platform == "win32"
    else None
)


def _exclusive_sockets(
    hosts: Optional[List[str]], port: int
) -> List[socket.socket]:
    """Bind *port* on *hosts* with :data:`_EXCLUSIVE_OPTION` set.

    Does what ``loop.create_server`` does with a host and a port, which
    takes no socket option: every address *hosts* resolve to, the
    wildcard of each family for None, IPv6 sockets IPv6-only, and an
    address this host does not have skipped.

    Args:
        hosts: The hosts to bind, or None for the wildcard.
        port: The port.

    Returns:
        The bound sockets, not yet listening.

    Raises:
        OSError: If an address cannot be bound, naming the port, with
            the bind's errno: 10048 when the port is held; 13
            (``EACCES``, WinError 10013) when a holder has it exclusively
            on the wildcard and this is a loopback address, or Windows
            reserves it.
    """
    infos = []
    for host in hosts or [None]:
        infos.extend(
            socket.getaddrinfo(
                host,
                port,
                socket.AF_UNSPEC,
                socket.SOCK_STREAM,
                0,
                socket.AI_PASSIVE,
            )
        )
    sockets: List[socket.socket] = []
    try:
        for family, kind, proto, _, address in dict.fromkeys(infos):
            sock = socket.socket(family, kind, proto)
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, _EXCLUSIVE_OPTION, 1)
            if family == socket.AF_INET6:
                sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
            try:
                sock.bind(address)
            except OSError as error:
                if error.errno == errno.EADDRNOTAVAIL:
                    sockets.pop().close()
                    continue
                message = (
                    f"the broker cannot bind port {port} on {address[0]}: "
                    f"{str(error.strerror or error).rstrip('.')}"
                )
                if error.errno == errno.EADDRINUSE:
                    message += (
                        f". Another program holds port {port}, on this "
                        f"address or on a loopback one."
                    )
                elif error.errno == errno.EACCES:
                    # Windows answers EACCES, not 10048, when a holder has
                    # the port exclusively on the wildcard and this is a
                    # loopback address -- a broker bound ANY, then one
                    # bound LOOPBACK -- and for a port it reserves.
                    message += (
                        f". Another program holds port {port} "
                        f"exclusively on every address, as a g.Pype "
                        f"broker bound to all of them does, or Windows "
                        f"reserves it."
                    )
                raise OSError(error.errno, message) from None
        if not sockets:
            raise OSError(
                errno.EADDRNOTAVAIL,
                f"the broker found no address to bind port {port} on",
            )
    except BaseException:
        for sock in sockets:
            sock.close()
        raise
    return sockets


class _Servers:
    """Several ``websockets`` servers that report and close as one.

    ``loop.create_server`` takes one socket, so each of
    :func:`_exclusive_sockets` gets a server of its own.
    """

    def __init__(self, servers: List[Any]) -> None:
        self._servers = servers

    @property
    def sockets(self) -> List[socket.socket]:
        """Every server's listening sockets."""
        return [sock for server in self._servers for sock in server.sockets]

    def close(self) -> None:
        """Close every server; see ``websockets``' ``Server.close``."""
        for server in self._servers:
            server.close()

    async def wait_closed(self) -> None:
        """Wait until every server has closed."""
        await asyncio.gather(
            *(server.wait_closed() for server in self._servers)
        )


def _port_of(endpoint: Optional[str]) -> int:
    """Return the port a broker endpoint names.

    Args:
        endpoint: A ``ws://host:port`` or ``wss://host:port`` endpoint,
            or None.

    Returns:
        The port, or :data:`DEFAULT_PORT` when the endpoint names none.
    """
    if not endpoint:
        return DEFAULT_PORT
    return urlparse(endpoint).port or DEFAULT_PORT


def _is_clock_connection(ws) -> bool:
    """Whether *ws* was opened only to probe the clock.

    An edge marks that connection with :data:`CLOCK_QUERY` on the
    endpoint it dials. The request path is ``ws.request.path`` on the
    asyncio implementation of ``websockets`` and ``ws.path`` on the
    legacy one; either may be absent on a stand-in.

    Args:
        ws: A server-side connection.

    Returns:
        True if the handshake named the clock role.
    """
    request = getattr(ws, "request", None)
    path = getattr(request, "path", None) or getattr(ws, "path", None)
    if not isinstance(path, str):
        return False
    return CLOCK_QUERY in urlparse(path).query.split("&")


def _peer_of(ws) -> str:
    """Name a connection by the address it came from, for a log line.

    Args:
        ws: A connection, or anything standing in for one.

    Returns:
        ``host:port``, with an IPv6 host in brackets, or a description
        when the address is unknown.
    """
    address = getattr(ws, "remote_address", None)
    if isinstance(address, tuple) and len(address) >= 2:
        host, port = address[0], address[1]
        if ":" in str(host):
            host = f"[{host}]"
        return f"{host}:{port}"
    return "an unknown address"


class WsBroker:
    """Process-wide singleton WebSocket broker for SERVER-side Links."""

    _instance: Optional["WsBroker"] = None
    _class_lock: threading.Lock = threading.Lock()
    _counter: int = 0
    _counter_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Singleton
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> "WsBroker":
        """Return the process-wide singleton, creating it if necessary."""
        with cls._class_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    @classmethod
    def allocate_stream_id(cls) -> str:
        """Return the next deterministic stream_id (``link_0``, ``link_1``, …).

        Called by :class:`Link` when no explicit *stream_id* is supplied.
        Both EDGE and SERVER processes must create Links in the same order
        for the auto-generated IDs to match -- which is a strong
        assumption, and one nothing checks. Every ``Link`` built from a
        document passes an explicit id derived from the node's own id, so
        this path is reached only by direct construction.

        Kept rather than removed because that path is legitimate in
        tests, but warned about: an id that depends on construction order
        is an id that two processes can disagree about, and disagreement
        here is silent at the wire level.
        """
        with cls._counter_lock:
            idx = cls._counter
            cls._counter += 1
        stream_id = f"link_{idx}"
        log.warning(
            "Link built with no stream_id; using %r, which matches the "
            "other side only if it constructs its Links in the same "
            "order. Pass stream_id explicitly, or canonicalize the "
            "document so the id comes from the node.",
            stream_id,
        )
        return stream_id

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton and reset the ID counter (for testing)."""
        with cls._class_lock:
            if cls._instance is not None:
                try:
                    cls._instance._force_stop()
                except Exception:
                    pass
            cls._instance = None
        with cls._counter_lock:
            cls._counter = 0

    # ------------------------------------------------------------------
    # Instance initialisation
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._ref_count: int = 0
        self._ref_lock: threading.Lock = threading.Lock()
        self._endpoint: Optional[str] = None
        #: The certificate this broker serves with, if the endpoint
        #: asked for one. Set in _start, beside the endpoint it follows.
        self._tls: Optional[ServerTls] = None

        # stream_id -> PUT callback (SERVER-side receiver Links)
        self._receivers: Dict[str, Callable[[np.ndarray], None]] = {}
        self._receivers_lock: threading.Lock = threading.Lock()

        #: stream_id -> the per-connection state of the one connection
        #: writing it. See _claim_sender.
        self._senders: Dict[str, dict] = {}
        self._senders_lock: threading.Lock = threading.Lock()

        # stream_id -> set of live WS connections (EDGE subscriber Links)
        self._subscribers: Dict[str, Set] = {}
        self._subscribers_lock: threading.Lock = threading.Lock()

        # stream_id -> last known port context (persists for late subscribers)
        self._contexts: Dict[str, dict] = {}
        self._contexts_lock: threading.Lock = threading.Lock()

        # stream_id -> SERVER-side context-received callbacks
        self._context_receivers: Dict[str, Callable[[dict], None]] = {}
        self._context_receivers_lock: threading.Lock = threading.Lock()

        #: stream_id -> where a warning about that stream goes besides
        #: the console: the receiving Link's own log. See _warn_stream.
        self._warning_receivers: Dict[str, Callable[[str], None]] = {}
        self._warning_receivers_lock: threading.Lock = threading.Lock()

        #: Causes already reported, so a per-frame failure is said once
        #: rather than at the sampling rate. See _warn_once.
        self._reported: set = set()
        self._reported_lock: threading.Lock = threading.Lock()

        # Background asyncio event loop / thread
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server = None

        #: Connections handed in by the host (accept), where there is no
        #: socket: closed by a stop, as the server closes its own.
        self._connections: Set = set()

        #: Which hosts to bind, resolved from LaunchConfig at _start().
        #: None is the wildcard; a list names the loopback literals.
        self._bind_hosts: Optional[List[str]] = None

        # Bind completion signalling
        self._bind_done: threading.Event = threading.Event()
        self._bind_error: Optional[Exception] = None

        # connection -> mailbox of attestation-protocol messages.
        # Keyed per connection rather than shared: a verdict is about one
        # device, and a shared mailbox would let one edge answer for
        # another -- which is precisely the substitution the signature is
        # there to prevent.
        self._attest_inbox: Dict[Any, "queue.Queue"] = {}
        self._attest_lock: threading.Lock = threading.Lock()

        # Set once any edge has connected. The server cannot challenge
        # before there is somebody to challenge.
        #
        # Latches for the life of the broker rather than tracking whether
        # anyone is *currently* connected. A restarted pipeline therefore
        # skips the wait -- and that is safe rather than lucky, because
        # attest_channels() is what actually decides: an edge that left
        # has no mailbox, so the caller gets an empty list and reports
        # "no edge to challenge" instead of a false pass.
        self._client_joined: threading.Event = threading.Event()

    # ------------------------------------------------------------------
    # Ref-counting: acquire / release
    # ------------------------------------------------------------------

    def acquire(self, endpoint: str) -> None:
        """Increment the ref count; start the server on the first one.

        Args:
            endpoint: Endpoint to serve. Callers sharing a broker have to
                agree on the port, because one broker binds once.

        Raises:
            ValueError: If the broker is already serving a different
                port.
        """
        with self._ref_lock:
            if self._ref_count == 0:
                self._start(endpoint)
            else:
                self._require_same_port(endpoint)
            self._ref_count += 1

    def _require_same_port(self, endpoint: str) -> None:
        """Refuse an endpoint this broker cannot also be serving.

        ``acquire()`` used to take the endpoint and ignore it whenever
        the ref count was already non-zero, because only the 0 -> 1 edge
        starts the server. A second caller naming a *different* port was
        therefore told it had succeeded while its port stayed closed, and
        the failure surfaced later as a client that could not connect --
        with nothing pointing back here.

        Only the port is compared. An endpoint's host is the address a
        *client* should dial -- a container binds a wildcard and
        advertises its own address -- so two endpoints differing only in
        host are the same instruction to the broker, and rejecting those
        would be a false alarm. Which interfaces it binds is a separate
        decision, read from ``LaunchConfig.bind``, and process-global
        rather than per-caller.

        Args:
            endpoint: The endpoint the caller asked for.

        Raises:
            ValueError: If the port differs from the one being served.
        """
        wanted = _port_of(endpoint)
        current = _port_of(self._endpoint)
        if wanted == current:
            return
        raise ValueError(
            f"WsBroker is already serving port {current} (endpoint "
            f"{self._endpoint!r}) and cannot also serve port {wanted} "
            f"for endpoint {endpoint!r}. One broker binds one port: use "
            f"a single endpoint per process, or release the broker "
            f"before acquiring a different one."
        )

    def release(self) -> None:
        """Decrement ref count; stop the WS server on the last release."""
        with self._ref_lock:
            self._ref_count = max(0, self._ref_count - 1)
            if self._ref_count == 0:
                self._stop()

    # ------------------------------------------------------------------
    # Remote attestation (C10): the SERVER is the verifier
    # ------------------------------------------------------------------

    def wait_for_client(self, timeout: float) -> bool:
        """Block until at least one edge has connected.

        The ordering here is the substance of C10 and it is awkward by
        nature: the server must not start its nodes before the verdict
        exists, because a sink writes its header at ``start()`` -- but
        the verdict needs an edge, and the edge can only connect once the
        server is listening. So the server binds, waits here, and only
        then challenges.

        Bounded on purpose. An unbounded wait would turn a missing edge
        into a pipeline that hangs with no message, which is worse than
        one that fails and says why.

        Without threads it answers at once. A page connects through the
        loop this call would be blocking, so nothing could arrive while
        it waited, and the worker would stand still for the whole
        timeout (D-CORE-129).

        Args:
            timeout: Seconds to wait.

        Returns:
            True if an edge connected in time.
        """
        if not threads.available():
            return self._client_joined.is_set()
        return self._client_joined.wait(timeout=timeout)

    def attest_channels(self) -> List[Any]:
        """Return one attestation channel per connected edge.

        Returns:
            A list of channels satisfying
            ``gtec_attest.protocol.Channel``, one per live connection.
        """
        from .client import WsAttestChannel

        with self._attest_lock:
            items = list(self._attest_inbox.items())
        return [
            WsAttestChannel(self._attest_sender(ws), inbox)
            for ws, inbox in items
        ]

    def _attest_sender(self, ws) -> Callable[[bytes], None]:
        """Return a callable that sends attest bytes to one connection.

        Failures are *not* swallowed here, unlike the data path. The
        caller is the attestation protocol, which reads an exception as
        "no attestation" -- and needs to, rather than waiting out a
        timeout for a reply that was never sent.

        Args:
            ws: The connection to send on.

        Returns:
            A one-argument callable.
        """

        def send(data: bytes) -> None:
            loop = self._loop
            if loop is None:
                raise ConnectionError("WsBroker is not running")
            if not threads.available():
                # Waiting for the send would block the loop that sends.
                raise ConnectionError(
                    "remote attestation needs threads: without them a "
                    "send cannot be waited for"
                )
            future = asyncio.run_coroutine_threadsafe(
                ws.send(encode_attest(data)), loop
            )
            future.result(timeout=5)

        return send

    # ------------------------------------------------------------------
    # Receiver registration (SERVER-side receiver Links)
    # ------------------------------------------------------------------

    def register_receiver(
        self, stream_id: str, callback: Callable[[np.ndarray], None]
    ) -> None:
        """Register the PUT handler for *stream_id*.

        One handler per stream, and a second claim is refused rather than
        accepted. ``_receivers`` holds a single callback per stream by
        design -- fan-in is one writer, unlike ``_subscribers``, which is
        a set -- so an assignment over an existing entry does not fail,
        it silently redirects every frame of that stream to the newcomer
        and leaves the original receiving nothing.

        That is undiagnosable from either end: no exception, no discard
        counter, no log. It is the failure mode two edges hit first,
        because both build Links from one document and any disagreement
        about which stream is whose lands here. Refusing turns it into a
        message naming the stream.

        Args:
            stream_id: The stream this handler receives.
            callback: Invoked with each array PUT to that stream.

        Raises:
            ValueError: If a handler is already registered for
                *stream_id*.
        """
        with self._receivers_lock:
            if stream_id in self._receivers:
                raise ValueError(
                    f"stream {stream_id!r} already has a receiver. Two "
                    f"Links are claiming one stream: every frame would "
                    f"go to whichever registered last and the other "
                    f"would receive nothing. Give them distinct ids -- "
                    f"gp.canonicalize() assigns them from the document."
                )
            self._receivers[stream_id] = callback

    def unregister_receiver(self, stream_id: str) -> None:
        """Remove the PUT handler for *stream_id*."""
        with self._receivers_lock:
            self._receivers.pop(stream_id, None)

    def register_warning(
        self, stream_id: str, callback: Callable[[str], None]
    ) -> None:
        """Send warnings about *stream_id* to *callback* as well.

        The broker is below the node layer and shared by the process, so
        its warnings reached the console alone -- not the session log and
        not the condition, where a refused stream's discarded frames
        belong. The receiving Link registers its own log here; a warning
        about no particular stream stays on the console (D-CORE-125).

        Args:
            stream_id: The stream the warnings concern.
            callback: Takes the formatted warning.
        """
        with self._warning_receivers_lock:
            self._warning_receivers[stream_id] = callback

    def unregister_warning(self, stream_id: str) -> None:
        """Stop sending warnings about *stream_id* anywhere but the console."""
        with self._warning_receivers_lock:
            self._warning_receivers.pop(stream_id, None)

    def _warn_stream(self, stream_ids, text: str) -> None:
        """Warn on the console once, and to the Link of each stream.

        Args:
            stream_ids: The stream, or the streams, the warning concerns.
            text: The warning, formatted.
        """
        log.warning(text)
        if isinstance(stream_ids, str):
            stream_ids = [stream_ids]
        for stream_id in stream_ids:
            with self._warning_receivers_lock:
                callback = self._warning_receivers.get(stream_id)
            if callback is None:
                continue
            try:
                callback(text)
            except Exception:  # pragma: no cover - must not break I/O
                log.debug("warning callback for %r raised", stream_id)

    # ------------------------------------------------------------------
    # Context: store / deliver / callbacks
    # ------------------------------------------------------------------

    def store_context(self, stream_id: str, context: dict) -> None:
        """Store *context* for *stream_id* and deliver it immediately.

        - Calls the registered SERVER-side context-receiver callback.
        - Pushes a context text-frame to all current EDGE subscribers.
        - Stored context is re-delivered to future subscribers on subscribe.
        """
        with self._contexts_lock:
            self._contexts[stream_id] = context

        with self._context_receivers_lock:
            cb = self._context_receivers.get(stream_id)
        if cb is not None:
            cb(context)

        with self._subscribers_lock:
            conns = list(self._subscribers.get(stream_id, set()))
        if conns and self._loop is not None:
            payload = encode_context(stream_id, context)
            for ws in conns:
                asyncio.run_coroutine_threadsafe(
                    self._safe_send_text(ws, payload), self._loop
                )

    def register_context_receiver(
        self, stream_id: str, callback: Callable[[dict], None]
    ) -> None:
        """Register *callback* for context on *stream_id*.

        If context is already stored, *callback* is invoked immediately.
        """
        with self._context_receivers_lock:
            self._context_receivers[stream_id] = callback
        with self._contexts_lock:
            ctx = self._contexts.get(stream_id)
        if ctx is not None:
            callback(ctx)

    def unregister_context_receiver(self, stream_id: str) -> None:
        """Remove the context callback for *stream_id*."""
        with self._context_receivers_lock:
            self._context_receivers.pop(stream_id, None)

    # ------------------------------------------------------------------
    # Push (SERVER-side sender Links)
    # ------------------------------------------------------------------

    def push(self, stream_id: str, data: np.ndarray) -> None:
        """Encode *data* and send it to all EDGE subscribers of *stream_id*."""
        with self._subscribers_lock:
            conns = list(self._subscribers.get(stream_id, set()))
        if not conns or self._loop is None:
            return
        payload = encode(data, stream_id)
        for ws in conns:
            asyncio.run_coroutine_threadsafe(
                self._safe_send(ws, payload), self._loop
            )

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def _start(self, endpoint: str) -> None:
        """Bind the WS server (called under _ref_lock).

        Reads the certificate out of LaunchConfig rather than taking it
        as an argument, for the same reason the endpoint used to be read
        from there: a Link calls acquire() during its own start, and it
        has no business knowing about certificates.
        """
        self._endpoint = endpoint
        if not threads.available():
            self._start_in_process()
            return
        self._tls = self._configured_tls(endpoint)
        # Resolved here, on the caller's thread, so a bad value is a
        # start-time error rather than a bind that silently never
        # happens on the broker thread.
        self._bind_hosts = self._configured_bind()
        self._bind_done.clear()
        self._bind_error = None

        loop = asyncio.new_event_loop()
        self._loop = loop
        # Handed to the thread as an argument rather than read back off
        # self, which a stop clears as soon as it has asked for one: the
        # thread still has to run this loop to its end and close it.
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(loop,),
            name="WsBroker._run_loop",
            daemon=True,
        )
        self._thread.start()

        # Three outcomes, and the third used to be silent: the bind
        # succeeded, it failed with an OSError, or it never reported at
        # all. The return value of wait() was discarded, so a timeout left
        # _bind_error as None and this returned as though the server were
        # listening. acquire() then succeeded, the caller connected to a
        # port nothing was bound to, and the symptom surfaced somewhere
        # else entirely -- a client that "never connected", ten seconds
        # later and with no mention of the broker.
        #
        # Note _serve() imports websockets inside the coroutine, so the
        # first bind in a process pays that import before it can report.
        bound = self._bind_done.wait(timeout=BIND_TIMEOUT_S)
        if self._bind_error is not None or not bound:
            # Tear the loop thread down before reporting. acquire() has
            # not taken a reference yet, so nothing will ever release
            # this broker, and the thread would otherwise sit in
            # run_forever() for the life of the process -- and a later
            # acquire() on a working port would find that stale loop
            # still installed.
            error = self._bind_error
            self._force_stop()
            if isinstance(error, Exception):
                raise error
            if error is not None:
                # A BaseException that is not an Exception -- in
                # practice a CancelledError out of the bind -- must not
                # travel out of a plain synchronous call. Callers guard
                # acquire() with `except Exception`, so a bare
                # CancelledError would walk past every one of them, and
                # to anything that did catch it, it would read as "your
                # task was cancelled" rather than "the bind was".
                raise RuntimeError(
                    f"binding {endpoint} was interrupted by "
                    f"{type(error).__name__}: {error}"
                ) from error
            raise TimeoutError(
                f"the broker did not finish binding {endpoint} within "
                f"{BIND_TIMEOUT_S:g} s and reported no error. The server "
                f"thread never reached its listen call, so nothing is "
                f"accepting on that port."
            )

        log.debug("WsBroker listening on port %s", _port_of(endpoint))

    def _start_in_process(self) -> None:
        """Serve without a thread or a socket (called under _ref_lock).

        Where no thread can be started, as under Pyodide, there is no
        server to run either: a browser page reaches the pipeline in its
        Web Worker by ``postMessage``, and the host hands each client in
        through :meth:`accept`. The broker serves on the loop already
        running in this thread, and borrows it: a stop closes this
        broker's connections, never the loop.

        Raises:
            RuntimeError: If there is no event loop to serve on.
        """
        loop = threads.event_loop()
        if loop is None:
            raise RuntimeError(
                "WsBroker cannot serve without threads unless an event "
                "loop is running: start the pipeline from a coroutine, as "
                "a Pyodide worker does."
            )
        self._loop = loop
        self._bind_done.set()
        log.debug("WsBroker serving in process, without a socket")

    def accept(self, connection) -> None:
        """Serve one connection the host carries itself.

        The in-process counterpart of a client dialling the port: the
        connection runs through the same handler, with the same tokens,
        subscriptions and one-sender rule. See
        :class:`~.message_connection.MessageConnection`.

        Args:
            connection: A MessageConnection, or anything with its
                surface.

        Raises:
            RuntimeError: If the broker is not serving yet; a page must
                connect after the pipeline has started.
        """
        loop = self._loop
        if loop is None:
            raise RuntimeError(
                "WsBroker is not serving. Start the pipeline before "
                "connecting to it."
            )
        self._connections.add(connection)

        async def serve() -> None:
            try:
                # Closed before its turn came -- a stop in the same call
                # that accepted it -- so serving it would rebuild state
                # the stop has cleared.
                if not getattr(connection, "closed", False):
                    await self._handle_client(connection)
            finally:
                self._connections.discard(connection)

        asyncio.run_coroutine_threadsafe(serve(), loop)

    def _stop(self) -> None:
        """Close the WS server (called under _ref_lock)."""
        self._force_stop()

    def _force_stop(self) -> None:
        """Stop without acquiring _ref_lock (used by reset()).

        The whole stop is now one request handed to the loop -- close
        the server, then stop yourself -- followed by a join. Two things
        were wrong with asking for it from out here.

        It was conditional on ``is_running()``, and that guard races the
        window between ``run_until_complete()`` returning and
        ``run_forever()`` starting. A stop landing inside it did
        nothing: no shutdown, no stop, a join that timed out, and then
        ``_loop`` and ``_thread`` cleared -- leaving a live thread
        holding the port for the life of the process with nothing left
        that could reach it. Measured at 1 of 8 fast acquire/release
        cycles, on both 3.10/websockets 16.1.1 and 3.11/websockets 17.1.
        ``call_soon_threadsafe`` needs the loop *open*, not *running*: a
        callback queued before it starts is the first thing it runs.

        And the shutdown was awaited from this thread with
        ``.result(timeout=3)``, so a shutdown that could not finish cost
        three seconds and then had its loop stopped out from under it
        mid-close. Letting the loop stop *itself* when the close is done
        makes the join the only wait, and the join is the one that knows
        the difference between finished and abandoned.
        """
        # Handed over before anything is asked of it: from here the
        # thread owns the loop, including closing it. Clearing _loop
        # first also stops push() and store_context() from scheduling
        # onto a loop that is on its way out.
        loop, thread = self._loop, self._thread
        self._loop = None
        self._thread = None

        # Served in process, the loop is borrowed (_start_in_process):
        # close this broker's connections and leave the loop running.
        for connection in list(self._connections):
            end = getattr(connection, "end", None)
            if callable(end):
                try:
                    end(1001, "server stopped")
                except Exception:  # a stop that raises never finishes
                    log.debug("ending a connection raised", exc_info=True)
        self._connections.clear()

        if loop is not None and thread is not None:
            coro = self._shutdown()
            try:
                loop.call_soon_threadsafe(loop.create_task, coro)
            except RuntimeError:
                # Closed already, so its thread has finished and there
                # is nothing left to stop. Closing the coroutine keeps
                # it from being reported as never awaited.
                coro.close()

        if thread is not None:
            thread.join(timeout=STOP_TIMEOUT_S)
            if thread.is_alive():
                log.warning(
                    "WsBroker: the server on port %s did not shut down "
                    "within %g s; abandoning its thread and event loop. "
                    "That port stays held until the process exits.",
                    _port_of(self._endpoint),
                    STOP_TIMEOUT_S,
                )

        self._server = None
        self._bind_done.clear()
        self._bind_error = None
        self._client_joined.clear()
        with self._attest_lock:
            self._attest_inbox.clear()
        with self._subscribers_lock:
            self._subscribers.clear()
        with self._receivers_lock:
            self._receivers.clear()
        with self._senders_lock:
            self._senders.clear()
        with self._contexts_lock:
            self._contexts.clear()
        with self._context_receivers_lock:
            self._context_receivers.clear()

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            try:
                loop.run_until_complete(self._serve())
            except BaseException as exc:
                # _serve now reports its own failures, so arriving here
                # means something outside it failed -- the loop refusing
                # to run, set_event_loop on a policy that will not have
                # it, or the bind being cancelled. Recorded rather than
                # left to end this thread in silence: the waiter in
                # _start cannot tell a thread that died from one that is
                # slow, and would wait the full timeout either way.
                #
                # BaseException, not Exception, and that difference is a
                # red CI cell on its own. CancelledError does not derive
                # from Exception, so a cancelled bind walked straight
                # past the handler put here to report failures, ended
                # this thread silently with _bind_done never set and
                # _bind_error still None, and _start() then spent the
                # full thirty seconds before reporting the one outcome
                # it cannot explain -- "did not finish binding and
                # reported no error". Measured on cycle 15 of 20 fast
                # acquire/release cycles, cancelled inside
                # loop.getaddrinfo while the loops abandoned by earlier
                # cycles were being collected. Closing every loop
                # removes that cause; catching this makes the next
                # cancellation cost a millisecond and name itself.
                if self._bind_error is None:
                    self._bind_error = exc
                self._bind_done.set()
                return
            loop.run_forever()
        finally:
            close_loop(loop)

    @staticmethod
    def _configured_tls(endpoint: str) -> Optional[ServerTls]:
        """The certificate to serve with, or None for a plain socket.

        Raises:
            ValueError: If the endpoint asks for TLS and no certificate
                is configured. Refused here rather than served in the
                clear: an endpoint that says ``wss`` and a socket that
                is not encrypted is the one outcome nobody could
                detect from either side.
        """
        if not endpoint_is_secure(endpoint):
            return None
        # Imported here: launch_config imports the transport policy, and
        # this module is reached from the node layer during a start.
        from .....common.launch_config import LaunchConfig

        tls = LaunchConfig.get().server_tls()
        if tls is None:
            raise ValueError(
                f"endpoint {endpoint!r} asks for TLS, so the broker "
                "needs a certificate: configure LaunchConfig with "
                "certfile (and keyfile, if the key is separate), or "
                "use a ws:// endpoint."
            )
        return tls

    @staticmethod
    def _configured_bind() -> Optional[List[str]]:
        """Which hosts to bind, out of LaunchConfig.

        Read from there rather than taken as an argument, for the same
        reason the certificate is: a Link calls acquire() during its own
        start and has no business knowing how the deployment is exposed.

        Returns:
            None for the wildcard, or the loopback literals.
        """
        # Imported here for the same reason _configured_tls does it:
        # launch_config imports the transport policy, and this module is
        # reached from the node layer during a start.
        from .....common.launch_config import LaunchConfig

        return LaunchConfig.get().bind_hosts()

    async def _serve(self) -> None:

        port = _port_of(self._endpoint)
        try:
            # Whichever mode is configured, the bind names every
            # address family this host has and never just one.
            # "0.0.0.0" is IPv4-only, and `localhost` resolves to ::1
            # first here, so every client paid ~2.0 s for a refused
            # IPv6 attempt before falling back to 127.0.0.1 (measured:
            # 2.021 s, against 0.002 s dual-stack). That cost was
            # charged per connection, and under load it was long enough
            # to surface as a client that "never connected".
            #
            # ANY is host=None, the wildcard on both families. LOOPBACK
            # is the loopback addresses as *literals*, which keeps that
            # property while refusing everything off this machine.
            ssl = self._tls.context() if self._tls else None
            if _EXCLUSIVE_OPTION is None:
                self._server = await _ws_serve(
                    self._handle_client, self._bind_hosts, port, ssl=ssl
                )
            else:
                self._server = await self._serve_exclusive(port, ssl)
        except Exception as exc:
            # Every failure, not only OSError. A bind can fail for
            # reasons that carry no errno: a TypeError from an argument
            # this version of websockets does not accept, or a
            # ValueError out of the TLS context. Those used to
            # propagate out of run_until_complete
            # and kill the loop thread with _bind_done never set and
            # _bind_error still None -- which _start then reported,
            # thirty seconds later, as "did not finish binding and
            # reported no error", having discarded the one thing that
            # said what went wrong.
            self._bind_error = exc
            self._bind_done.set()
            return
        self._bind_done.set()

    async def _serve_exclusive(self, port: int, ssl: Any) -> _Servers:
        """Serve on sockets that hold *port* alone (Windows, D-CORE-133).

        Args:
            port: The port.
            ssl: The TLS context, or None.

        Returns:
            The servers, one per bound socket.
        """
        sockets = _exclusive_sockets(self._bind_hosts, port)
        servers = []
        try:
            for sock in sockets:
                servers.append(
                    await _ws_serve(self._handle_client, sock=sock, ssl=ssl)
                )
        except BaseException:
            for server in servers:
                server.close()
            for sock in sockets:
                sock.close()
            raise
        return _Servers(servers)

    async def _shutdown(self) -> None:
        """Close the server, then stop this broker's loop.

        Runs on the loop thread, so the close completes before the loop
        that is performing it goes away -- see :meth:`_force_stop` for
        what the old arrangement did instead.
        """
        server, self._server = self._server, None
        if server is not None:
            # close() stops accepting, closes every open connection with
            # code 1001, and waits for each handler to return.
            server.close()
            try:
                await asyncio.wait_for(
                    server.wait_closed(), timeout=SHUTDOWN_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                log.warning(
                    "WsBroker: a connection on port %s did not complete "
                    "its closing handshake within %g s; dropping it. The "
                    "peer stopped answering rather than closing.",
                    _port_of(self._endpoint),
                    SHUTDOWN_TIMEOUT_S,
                )
            except Exception:
                # A server already closed, or one whose close failed.
                # Either way the loop below still has to stop, and a
                # stop that raises is a thread that never ends.
                log.debug("closing the WS server raised", exc_info=True)
        asyncio.get_running_loop().stop()

    # ------------------------------------------------------------------
    # Client connection handler
    # ------------------------------------------------------------------

    def _required_tokens(self) -> Optional[dict]:
        """The tokens this broker accepts, or None to accept anything.

        Read out of LaunchConfig rather than taken as an argument, for
        the same reason the certificate is: a Link calls ``acquire()``
        during its own start and has no business knowing about
        credentials.

        Returns:
            A ``{capability: token}`` mapping, or None when the
            deployment configured none -- which is what a STANDALONE or
            trusted-LAN run has always done.
        """
        from .....common.launch_config import LaunchConfig

        config = LaunchConfig.get()
        tokens = getattr(config, "data_tokens", None) if config else None
        if not tokens:
            return None
        return {
            str(cap): str(token) for cap, token in tokens.items() if token
        } or None

    def _granted(self, message: str, required: dict) -> Optional[frozenset]:
        """The capabilities an ``auth`` message earns, if any.

        Args:
            message: The raw text message.
            required: The configured ``{capability: token}`` mapping.

        Returns:
            The capabilities granted, or None if the token matched
            nothing. Compared with ``compare_digest`` so a wrong token
            costs the same time as a right one.
        """
        try:
            presented = json.loads(message).get("token")
        except Exception:
            return None
        if not isinstance(presented, str) or not presented:
            return None

        granted = {
            cap
            for cap, token in required.items()
            if hmac.compare_digest(presented, token)
        }
        # Writing implies reading. An edge is one connection that does
        # both -- its source chains PUT and its sink and scope chains
        # subscribe -- so a token that only wrote would make the common
        # case need two tokens on one socket. A reader is still only a
        # reader, which is the direction that matters: the privilege
        # being withheld is the ability to replace somebody's stream.
        if CAP_PUT in granted:
            granted.add(CAP_SUBSCRIBE)
        return frozenset(granted) or None

    async def _handle_client(self, ws) -> None:
        """Handle one connected EDGE client."""
        subscribed: List[str] = []

        # What this connection is allowed to do. A deployment that
        # configured tokens starts every connection at *nothing* and
        # makes it earn a capability; one that configured none keeps the
        # historical behaviour, because a STANDALONE run has no control
        # plane to mint a token and nothing to protect it from.
        required = self._required_tokens()
        caps = _ALL_CAPABILITIES if required is None else frozenset()
        # The state is also this connection's identity as a sender: a
        # stream is held by the state of the connection writing it, and
        # released with it below.
        state = {
            "caps": caps,
            "peer": _peer_of(ws),
            "refused": set(),
            "ws": ws,
            # Claimed by the client and not proven, so it may only
            # narrow what this connection can do, never widen it.
            "clock_only": _is_clock_connection(ws),
        }

        log.debug("WsBroker: client connected from %s", ws.remote_address)
        # Register the mailbox before reading anything: a challenge sent
        # the instant this connection appears must not fall on the floor.
        # Not for a connection opened only to probe the clock. An edge
        # opens that one before its Links connect, so counting it would
        # let the server challenge a socket that carries no device, and
        # find no edge at all once it has closed.
        if not state["clock_only"]:
            with self._attest_lock:
                self._attest_inbox[ws] = queue.Queue()
            self._client_joined.set()

        # An unauthenticated connection can do nothing, but it can sit
        # there holding a slot -- and on a socket with no other
        # protection, "does nothing" is not the same as "costs nothing".
        # Only armed when this broker requires a token: without one there
        # is nothing to wait for and every connection would be closed.
        reaper = None
        if required is not None:
            reaper = asyncio.create_task(self._reap_unauthenticated(ws, state))

        try:
            async for message in ws:
                if isinstance(message, str):
                    await self._handle_text_message(
                        message, ws, subscribed, state, required
                    )
                elif isinstance(message, bytes):
                    # A clock connection is out of attestation's sight,
                    # so it may carry no frame (D-TIME-71).
                    if state["clock_only"]:
                        self._warn_once(
                            "clock:put",
                            "Ignored a frame on a clock connection.",
                        )
                    else:
                        self._handle_put_message(message, state)
        except Exception:
            pass
        finally:
            if reaper is not None:
                reaper.cancel()
            with self._subscribers_lock:
                for sid in subscribed:
                    conns = self._subscribers.get(sid)
                    if conns:
                        conns.discard(ws)
            self._release_sender(state)
            with self._attest_lock:
                self._attest_inbox.pop(ws, None)
            log.debug("WsBroker: client disconnected")

    async def _reap_unauthenticated(self, ws, state: dict) -> None:
        """Close *ws* if it has not authenticated in time.

        Args:
            ws: The connection to watch.
            state: Its per-connection state.
        """
        try:
            await asyncio.sleep(AUTH_TIMEOUT_S)
        except asyncio.CancelledError:
            return
        if state.get("caps"):
            return
        log.warning(
            "WsBroker: closing a connection from %s that did not "
            "authenticate within %g s.",
            ws.remote_address,
            AUTH_TIMEOUT_S,
        )
        try:
            await ws.close(code=1008, reason="unauthorized")
        except Exception:
            pass

    async def _handle_text_message(
        self,
        message: str,
        ws,
        subscribed: List[str],
        state: dict,
        required: Optional[dict],
    ) -> None:
        """Dispatch text messages by command, enforcing capabilities.

        Args:
            message: The raw text message.
            ws: The connection it arrived on.
            subscribed: Stream ids this connection has registered for.
            state: Per-connection state; ``state["caps"]`` is what it may
                currently do.
            required: The configured tokens, or None when this broker
                requires none.
        """
        # First, before parsing: a clock probe's receive instant. Taken for
        # every text message because the read costs less than deciding.
        received = channels.stamp_clock()
        try:
            msg = json.loads(message)
            cmd = msg.get("command", "").lower()

            if cmd == "auth":
                await self._handle_auth(message, ws, state, required)
                return

            # A clock connection is never challenged, so it may only
            # probe: anything else would reach the data plane unattested.
            # One key, not one per command: the name is the client's, and
            # every new one would be kept, and logged, for good.
            if state.get("clock_only") and cmd != "clock":
                self._warn_once(
                    "clock:command",
                    "Ignored a command other than auth and clock on a "
                    "clock connection.",
                )
                return

            if cmd == "subscribe":
                if self._permits(state, CAP_SUBSCRIBE, "subscribe"):
                    self._handle_subscribe(msg, ws, subscribed)
            elif cmd == "unsubscribe":
                # The same capability as subscribing, because it is the
                # same relationship being ended. A connection that could
                # not have subscribed has nothing to stop.
                if self._permits(state, CAP_SUBSCRIBE, "unsubscribe"):
                    self._handle_unsubscribe(msg, ws, subscribed)
            elif cmd == "context":
                # A context describes a stream its sender is feeding, so
                # it is a write, not a read: accepting one from a
                # read-only connection would let a viewer relabel
                # somebody else's channels.
                if self._permits(state, CAP_PUT, "context"):
                    self._handle_context_from_edge(msg, state)
            elif cmd == "attest":
                # Attestation is a challenge/response *about a device*,
                # carried on whichever connection holds it, and it grants
                # nothing on its own -- so it needs no capability.
                self._handle_attest_message(message, ws)
            elif cmd == "clock":
                # No capability either, but an authenticated connection
                # when tokens are configured: an answer before auth
                # would make this socket a timing oracle for anybody who
                # can reach the port (D-TIME-66).
                if state.get("caps"):
                    await self._handle_clock(message, ws, received)
                else:
                    self._warn_once(
                        "clock:unauthenticated",
                        "Ignored a clock probe from a connection that has "
                        "not authenticated. This deployment configures "
                        "data tokens, so an edge presents its token "
                        "before it probes.",
                    )
        except Exception:
            pass

    async def _handle_clock(self, message: str, ws, received: float) -> None:
        """Answer one clock probe on the connection it came from.

        Args:
            message: The raw text message.
            ws: The connection it arrived on.
            received: ``channels.stamp_clock()`` when it arrived.
        """
        probe = decode_clock(message)
        if probe is None or "t1" in probe:
            return
        epoch = channels.stamp_epoch()
        sent = channels.stamp_clock()
        await self._safe_send_text(
            ws, encode_clock_reply(probe["t0"], received, sent, epoch)
        )

    def _permits(self, state: dict, capability: str, what: str) -> bool:
        """Whether this connection may do *what*, and say so if not.

        Args:
            state: Per-connection state.
            capability: The capability required.
            what: What was attempted, for the log line.

        Returns:
            bool: True if permitted.
        """
        if capability in state.get("caps", frozenset()):
            return True
        self._warn_once(
            f"cap:{capability}",
            f"Refused {what} from a connection without the "
            f"{capability!r} capability. A deployment that configures "
            f"data tokens requires every connection to present one; the "
            f"control plane hands the right token to the right party.",
        )
        return False

    async def _handle_auth(
        self, message: str, ws, state: dict, required: Optional[dict]
    ) -> None:
        """Grant this connection whatever its token earns.

        Args:
            message: The raw text message, carrying ``token``.
            ws: The connection.
            state: Per-connection state to update.
            required: The configured tokens, or None.
        """
        if required is None:
            # Nothing to prove. Answer anyway, so a client that always
            # authenticates works against a broker that does not care.
            state["caps"] = _ALL_CAPABILITIES
        else:
            granted = self._granted(message, required)
            if granted is None:
                log.warning(
                    "WsBroker: refused a connection from %s -- its token "
                    "matched no configured capability.",
                    ws.remote_address,
                )
                await self._safe_send_text(
                    ws, json.dumps({"event": "auth", "status": "refused"})
                )
                await ws.close(code=1008, reason="unauthorized")
                return
            state["caps"] = granted

        await self._safe_send_text(
            ws,
            json.dumps(
                {
                    "event": "auth",
                    "status": "ok",
                    "capabilities": sorted(state["caps"]),
                }
            ),
        )

    def _handle_attest_message(self, message: str, ws) -> None:
        """Deliver an attestation message to its connection's mailbox.

        Args:
            message: The raw text message.
            ws: The connection it arrived on.
        """
        payload = decode_attest(message)
        if payload is None:
            return
        with self._attest_lock:
            inbox = self._attest_inbox.get(ws)
        if inbox is not None:
            inbox.put(payload)

    def _handle_subscribe(self, msg: dict, ws, subscribed: List[str]) -> None:
        """Register *ws* as a subscriber for the requested stream_id."""
        stream_id = msg.get("stream_id")
        if stream_id:
            with self._subscribers_lock:
                if stream_id not in self._subscribers:
                    self._subscribers[stream_id] = set()
                self._subscribers[stream_id].add(ws)
            subscribed.append(stream_id)
            log.debug("WsBroker: client subscribed to %s", stream_id)
            # Deliver stored context immediately if available
            with self._contexts_lock:
                ctx = self._contexts.get(stream_id)
            if ctx is not None and self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    self._safe_send_text(ws, encode_context(stream_id, ctx)),
                    self._loop,
                )

    def _handle_unsubscribe(
        self, msg: dict, ws, subscribed: List[str]
    ) -> None:
        """Stop sending *stream_id* to this connection.

        Closing the socket was the only way to stop a stream before
        this, which is tolerable for the streams a document declares --
        they last as long as the run -- and not for a probe. A probe is
        published only while somebody is listening, so a subscription
        that cannot be ended is a probe that cannot be turned off: the
        producer keeps copying, encoding and sending for a viewer that
        stopped looking.

        Silent when the connection was not subscribed. There is nothing
        to report: the caller wanted the stream to stop arriving, and it
        is not arriving.

        Args:
            msg: The decoded message.
            ws: The connection it arrived on.
            subscribed: Stream ids this connection registered for,
                modified in place so the disconnect sweep does not try
                to remove it a second time.
        """
        stream_id = msg.get("stream_id")
        if not stream_id:
            return
        with self._subscribers_lock:
            conns = self._subscribers.get(stream_id)
            if conns:
                conns.discard(ws)
                if not conns:
                    # Drop the empty set rather than keep it: `push`
                    # tests this dict, and an id that nobody watches
                    # should read as absent.
                    self._subscribers.pop(stream_id, None)
        while stream_id in subscribed:
            subscribed.remove(stream_id)
        log.debug("WsBroker: client unsubscribed from %s", stream_id)

    def subscriber_count(self, stream_id: str) -> int:
        """How many connections are watching *stream_id*.

        Lets a producer skip work nobody will receive. ``push`` already
        returns early with no subscribers, but by then the frame has
        been copied and queued; a probe asks this first and does not
        copy at all.

        Args:
            stream_id: The stream to count.

        Returns:
            Number of subscribed connections.
        """
        with self._subscribers_lock:
            return len(self._subscribers.get(stream_id) or ())

    @property
    def is_running(self) -> bool:
        """Whether this broker is serving.

        Returns:
            True once the event loop is up, which is the same test
            ``push`` and ``store_context`` make before handing anything
            to it.
        """
        return self._loop is not None

    def _handle_context_from_edge(self, msg: dict, state: dict) -> None:
        """Store context received from an EDGE sender and route to
        SERVER callback.

        Args:
            msg: The decoded ``context`` message.
            state: Per-connection state of the connection it arrived on.
                A context is refused, like a frame, from a connection
                that does not hold the stream (:meth:`_claim_sender`):
                accepted, it would relabel the channels of the stream
                another sender is feeding.
        """
        stream_id = msg.get("stream_id")
        context = msg.get("context")
        if stream_id and context is not None:
            if not self._claim_sender(stream_id, state):
                # Kept, not dropped: a sender that reconnected onto its
                # own half-open socket sends its context first and never
                # again, so once this connection takes the stream it is
                # the context the server has to hold (_take_refused).
                state.setdefault("refused_contexts", {})[stream_id] = context
                return
            state.get("refused_contexts", {}).pop(stream_id, None)
            self._apply_context(stream_id, context)

    def _apply_context(self, stream_id: str, context: dict) -> None:
        """Store a stream's context and hand it to the server's receiver.

        A receiver that raises is reported once, as a frame receiver is.
        This runs inside frame routing too (:meth:`_take_refused`), where
        an exception would lose the frame that took the stream and end
        the connection that had just taken it.

        Args:
            stream_id: The stream it describes.
            context: The context its sender sent.
        """
        with self._contexts_lock:
            self._contexts[stream_id] = context
        with self._context_receivers_lock:
            cb = self._context_receivers.get(stream_id)
        if cb is None:
            return
        try:
            cb(context)
        except Exception as e:
            self._warn_once(
                f"context_receiver:{stream_id}",
                f"The context receiver for stream '{stream_id}' raised: "
                f"{e}. The context is stored, and frames for it continue "
                f"to be delivered.",
            )

    def _take_refused(self, stream_id: str, state: dict) -> None:
        """Apply the context a connection sent while it was refused.

        Called once that connection holds the stream. 4.0 applied a
        reconnected sender's context at once; since the stream binding,
        the context arrived while the old socket still held the stream
        and the server kept the previous connection's description.

        Args:
            stream_id: The stream it now holds.
            state: Per-connection state of that connection.
        """
        pending = state.get("refused_contexts")
        if pending:
            context = pending.pop(stream_id, None)
            if context is not None:
                self._apply_context(stream_id, context)

    def _handle_put_message(self, message: bytes, state: dict) -> None:
        """Decode *message* and route to the registered receiver callback.

        Every way this can fail used to be silent -- a frame that would
        not decode, one carrying no stream id, and one naming a stream
        nobody is listening to were all discarded without a word, so a
        data path that had stopped working looked exactly like one with
        nothing to carry.

        Each cause is now reported once. Once, not per frame: this runs
        per frame at the sampling rate, and a warning per frame would
        bury the first one and cost more than the routing it guards.

        Args:
            message: The raw binary frame.
            state: Per-connection state; a frame is routed only if this
                connection earned :data:`CAP_PUT`. Checked here rather
                than at the socket because routing reads the frame's own
                header and never looked at the connection at all -- which
                is exactly why an unauthenticated PUT could replace the
                stream an amplifier was feeding. And only if this
                connection holds the stream (:meth:`_claim_sender`).
        """
        if not self._permits(state, CAP_PUT, "a binary frame"):
            return
        try:
            array, stream_id = decode(message)
        except Exception as e:
            self._warn_once(
                "undecodable",
                f"Discarding a frame that could not be decoded: {e}. "
                f"Further undecodable frames on this broker will not be "
                f"reported.",
            )
            return

        if stream_id is None:
            self._warn_once(
                "no_stream_id",
                "Discarding a frame that carries no stream id. The frame "
                "decoded, so it shares this format's prefix -- most "
                "likely it is another dialect using the same magic bytes "
                "rather than a corrupt frame.",
            )
            return

        if not self._claim_sender(stream_id, state):
            return
        if state.get("refused_contexts"):
            self._take_refused(stream_id, state)

        with self._receivers_lock:
            cb = self._receivers.get(stream_id)
            known = sorted(self._receivers)
        if cb is None:
            self._warn_once(
                f"unknown:{stream_id}",
                f"Discarding a frame for stream '{stream_id}', which "
                f"nothing is receiving. Registered streams: "
                f"{known or 'none'}. A stream id is derived from the "
                f"node id in the document, so the two sides are reading "
                f"different documents or one has not set up yet.",
            )
            return

        try:
            cb(array)
        except Exception as e:
            self._warn_once(
                f"receiver:{stream_id}",
                f"The receiver for stream '{stream_id}' raised: {e}. "
                f"Frames for it continue to be delivered.",
            )

    def _claim_sender(self, stream_id: str, state: dict) -> bool:
        """Whether this connection may write *stream_id*, taking it if free.

        A stream has one sender. The first connection to write it -- a
        context or a frame -- holds it until that connection closes, and
        every write from another connection is refused, with one warning
        per stream and connection.

        Before this the broker routed a frame by its header alone, and a
        second sender did not fail: its frames were interleaved with the
        first one's in the single receiver slot. Measured with a
        Generator that names no edge, in a document two edges ran: both
        edges opened it and each sent about 112 frames, the server's
        receiving Link counted 226, and the file recording it held twice
        the rows its sampling rate allows. Nothing said so.

        First rather than last: a later sender taking over would move
        the stream to whichever of two live senders wrote most recently,
        which is the interleaving again. A sender that reconnects is a
        new connection, and resumes once its old one is seen to close --
        which a refused write hastens: the holder is pinged, and closed
        if it does not answer (:meth:`_probe_sender`).

        Args:
            stream_id: The stream being written.
            state: Per-connection state of the writing connection, which
                is also its identity as a sender.

        Returns:
            True if the write should be routed.
        """
        with self._senders_lock:
            holder = self._senders.get(stream_id)
            if holder is state:
                return True
            if holder is None:
                self._senders[stream_id] = state
        peer = state.get("peer", "an unknown address")
        refused = state.setdefault("refused", set())
        if holder is None:
            if stream_id in refused:
                refused.discard(stream_id)
                self._warn_stream(
                    stream_id,
                    f"WsBroker: stream '{stream_id}' is now sent from "
                    f"{peer}, which was refused it while another "
                    f"connection held it; that connection has closed. If "
                    f"it was another sender rather than this one "
                    f"reconnecting, the stream now carries a different "
                    f"device's samples.",
                )
            return True
        if stream_id not in refused:
            refused.add(stream_id)
            other = holder.get("peer", "an unknown address")
            self._warn_stream(
                stream_id,
                f"WsBroker: refusing stream '{stream_id}' from {peer}: the "
                f"connection from {other} is already sending it, and a "
                f"stream has one sender. Frames and context for it from "
                f"{peer} are discarded while {other} stays connected. Most "
                f"often this is a source that names no edge, in a document "
                f"two edges run: every edge opens such a source, so give "
                f"it an edge_id. Otherwise two edges were started with one "
                f"edge id, or this sender reconnected while its previous "
                f"connection still looks open; that one is being pinged, "
                f"and is closed if it does not answer.",
            )
        self._probe_sender(holder)
        return False

    def _probe_sender(self, holder: dict) -> None:
        """Ping the connection holding a stream another one tried to write.

        A holder that answers is a live sender, and the refusal stands.
        One that does not is a socket whose peer has gone -- most often
        the same sender, reconnected -- and is closed and released, so
        the next write takes the stream. Without this the new connection
        waits out websockets' keepalive on the old one, 20 to 40 s, or,
        when the old one also subscribed, until TCP gives up on it.

        At most one ping in flight per holder, and one every
        :data:`SENDER_PROBE_INTERVAL_S`, since this is reached on every
        refused frame. Nothing is probed without a running loop or a
        connection to ping, which is the routing tested on its own.

        Args:
            holder: Per-connection state of the connection holding it.
        """
        ws, loop = holder.get("ws"), self._loop
        if ws is None or loop is None:
            return
        now = time.monotonic()
        with self._senders_lock:
            if holder.get("probing"):
                return
            if now - holder.get("probed_at", -1e9) < SENDER_PROBE_INTERVAL_S:
                return
            holder["probing"] = True
            holder["probed_at"] = now
        try:
            asyncio.run_coroutine_threadsafe(self._ping_sender(holder), loop)
        except RuntimeError:
            # The loop is closing; there is nobody left to hand over to.
            holder["probing"] = False

    async def _ping_sender(self, holder: dict) -> None:
        """Close and release *holder* if it does not answer a ping.

        The timeout covers sending the ping, not only its answer:
        websockets' ``ping()`` waits for the writer to drain before it
        returns, and the writer to a subscriber whose peer has gone
        stays paused, so the wait would never end. The close is an
        abort for the same reason; a close frame waits behind
        everything that peer has not read.

        Args:
            holder: Per-connection state of the connection to probe.
        """
        ws = holder["ws"]

        async def answered() -> None:
            await (await ws.ping())

        try:
            # wait_for, not asyncio.timeout, which 3.10 does not have.
            await asyncio.wait_for(answered(), timeout=SENDER_PROBE_TIMEOUT_S)
            return
        except Exception:
            pass
        finally:
            holder["probing"] = False
        with self._senders_lock:
            held = sorted(
                stream_id
                for stream_id, state in self._senders.items()
                if state is holder
            )
        if not held:
            # It closed while the ping was out, and gave everything up.
            return
        # Released before the abort, so the stream is free the moment
        # the next write arrives rather than when the handler notices.
        self._release_sender(holder)
        self._warn_stream(
            held,
            f"WsBroker: closing the connection from "
            f"{holder.get('peer', 'an unknown address')}, which was sending "
            f"{', '.join(repr(s) for s in held)} but did not answer a ping "
            f"within {SENDER_PROBE_TIMEOUT_S:g} s after another connection "
            f"wrote to it. The next connection to write "
            f"{'it' if len(held) == 1 else 'one of them'} takes it.",
        )
        try:
            ws.transport.abort()
        except Exception:
            pass

    def _release_sender(self, state: dict) -> None:
        """Give up every stream the connection with *state* was sending.

        Args:
            state: Per-connection state of a connection that has closed.
        """
        with self._senders_lock:
            held = [
                stream_id
                for stream_id, holder in self._senders.items()
                if holder is state
            ]
            for stream_id in held:
                del self._senders[stream_id]

    def _warn_once(self, cause: str, message: str) -> None:
        """Report *cause* the first time it happens and never again.

        Args:
            cause: Key identifying the cause, so distinct problems are
                each reported once rather than the first one masking the
                rest.
            message: What to say.
        """
        with self._reported_lock:
            if cause in self._reported:
                return
            self._reported.add(cause)
        log.warning(message)

    async def _safe_send(self, ws, payload: bytes) -> None:
        try:
            await ws.send(payload)
        except Exception:
            pass

    async def _safe_send_text(self, ws, payload: str) -> None:
        try:
            await ws.send(payload)
        except Exception:
            pass
