"""WebSocket client for EDGE-side Link data I/O.

Connects to the WsBroker running on the SERVER side.

Two usage modes
---------------
Sender (EDGE → SERVER)::

    client = WsClient(endpoint)
    client.start_sending(stream_id)
    client.put(data)          # non-blocking fire-and-forget

Receiver (EDGE ← SERVER)::

    client = WsClient(endpoint)
    client.start_receiving(stream_id, callback)
    # callback(np.ndarray) is invoked from the background thread

Auto-reconnect is active in both modes.  put() silently drops frames
while disconnected and emits a rate-limited WARNING log.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np
import websockets

# The codec lives at the package root, five levels up: it is a
# contract with the runtime, not part of the node layer.
from ....._transport import endpoint_is_secure
from ....._wire import (
    decode,
    decode_attest,
    decode_context,
    encode,
    encode_attest,
    encode_context,
)

log = logging.getLogger(__name__)

#: ``websockets.connect``, bound at import time.
#:
#: `websockets` defers its submodules to a module-level ``__getattr__``,
#: so the first touch of ``websockets.connect`` runs a real import. That
#: first touch used to be the ``async with`` in ``_connect_loop``, inside
#: the client's own event-loop thread -- which ran the import
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
#: builds the node. Never reach for ``websockets.connect`` from the
#: event-loop thread.
_ws_connect = websockets.connect

#: How long the closing handshake may take before the socket is simply
#: dropped.
#:
#: Short on purpose: on a live connection the handshake is one round
#: trip, and this bound exists only for the peer that has stopped
#: answering. ``websockets`` applies its own ``close_timeout`` of 10 s,
#: which is the right budget for a connection that is merely slow and
#: the wrong one for a shutdown that a person is waiting on.
CLOSE_TIMEOUT_S = 2.0

#: How long ``disconnect()`` waits for the background thread to close
#: the socket, stop its loop and close it.
DISCONNECT_TIMEOUT_S = 5.0


def close_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Cancel what is left on *loop*, then close it.

    Called from the thread that ran the loop, once ``run_forever()`` has
    returned. That is the only place it can be called safely: ``close()``
    refuses a loop that is still running, and the thread asking for the
    stop cannot know the loop has stopped until it has joined -- so it
    would have to guess. The thread that owns the loop does not.

    Nothing here used to happen at all. Both ends stopped their loop,
    joined its thread and dropped the loop object on the floor, and
    ``close()`` is the only thing that shuts down a loop's default
    ``ThreadPoolExecutor`` and closes its selector or IOCP. Every bind
    goes through ``loop.getaddrinfo``, which runs in exactly that
    executor, so the first thing a leak of loops breaks is the *next
    bind* -- which is why this surfaced as a broker that would not start
    rather than as a process that grew.

    Measured over 20 fast acquire/release cycles before the fix: worker
    threads accumulated to 24, and cycle 15 failed with the CI symptom,
    "the broker did not finish binding within 30 s and reported no
    error". After it: 20 cycles, no growth, no failure.

    Pending tasks are cancelled and given a bounded moment to unwind
    rather than having the loop shut underneath them. A loop closed on
    top of a live task logs "Task was destroyed but it is pending!" and
    leaves the socket that task was holding for the garbage collector to
    find; cancelling lets ``websockets`` close its own transports. The
    moment is bounded because a task that declines to be cancelled must
    not be able to wedge the thread tidying up after it.

    Args:
        loop: The loop to close. Must not be running.
    """
    try:
        pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(
                asyncio.wait(pending, timeout=CLOSE_TIMEOUT_S)
            )
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception:
        # Best-effort by nature: this runs while everything around it is
        # already being torn down, and an exception escaping here would
        # trade a tidy shutdown for a dead thread and an unclosed loop --
        # precisely the leak the function exists to prevent. The close
        # below still happens, because it is in the finally.
        log.debug("tidying up an event loop raised", exc_info=True)
    finally:
        try:
            loop.close()
        except Exception:
            log.debug("closing an event loop raised", exc_info=True)


class WsClient:
    """EDGE-side WebSocket client for a single Link."""

    def __init__(
        self,
        endpoint: str,
        retry_interval: float = 1.0,
        drop_log_interval: float = 5.0,
    ) -> None:
        self._endpoint = endpoint
        self._retry_interval = retry_interval
        self._drop_log_interval = drop_log_interval
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ws = None
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._last_drop_log: float = 0.0

        # Why the last connection attempt failed. _connect_loop retries
        # forever and used to discard the exception, so a wrong endpoint
        # produced no log line at all and no way to ask.
        self._last_error: Optional[BaseException] = None
        self._last_error_log: float = 0.0
        self._reported_error: bool = False

        self._mode: Optional[str] = None  # "send" | "receive"
        self._stream_id: Optional[str] = None
        self._receive_callback: Optional[Callable[[np.ndarray], None]] = None
        self._context_callback: Optional[Callable[[dict], None]] = None
        self._pending_context: Optional[dict] = None

        # Attestation-protocol messages from the broker. A sender reads
        # them too -- see _control_loop -- because the edge holds the
        # amplifier and therefore has to answer challenges regardless of
        # which direction its data flows.
        self._attest_inbox: "queue.Queue" = queue.Queue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True while a live WebSocket connection is open."""
        return self._ready.is_set()

    @property
    def last_error(self) -> Optional[BaseException]:
        """Why the most recent connection attempt failed, or None.

        Readable because this client retries forever by design, so a
        wrong endpoint or an unverifiable certificate is otherwise only
        a log line -- and a caller that wants to give up, or a test that
        wants to know *why* nothing connected, has nothing to look at.
        Cleared on a successful connection.
        """
        return self._last_error

    def start_sending(self, stream_id: str) -> None:
        """Connect to the broker and prepare to PUT data for *stream_id*."""
        self._mode = "send"
        self._stream_id = stream_id
        self._launch()

    def start_receiving(
        self,
        stream_id: str,
        callback: Callable[[np.ndarray], None],
        context_callback: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """Connect to the broker and SUBSCRIBE to *stream_id*."""
        self._mode = "receive"
        self._stream_id = stream_id
        self._receive_callback = callback
        self._context_callback = context_callback
        self._launch()

    def send_context(self, context: dict) -> None:
        """Store *context* and send it now (if connected); re-sent on
        every reconnect."""
        self._pending_context = context
        if (
            self._ws is not None
            and self._ready.is_set()
            and self._loop is not None
        ):
            asyncio.run_coroutine_threadsafe(
                self._safe_send_text(encode_context(self._stream_id, context)),
                self._loop,
            )

    def put(self, data: np.ndarray) -> None:
        """Send a data frame (non-blocking fire-and-forget).

        Drops silently when not connected; emits a rate-limited WARNING.
        """
        if self._ws is None or not self._ready.is_set():
            now = time.monotonic()
            if now - self._last_drop_log >= self._drop_log_interval:
                log.warning(
                    "WsClient: not connected to %s — dropping data",
                    self._endpoint,
                )
                self._last_drop_log = now
            return
        payload = encode(data, self._stream_id)
        asyncio.run_coroutine_threadsafe(self._safe_send(payload), self._loop)

    def disconnect(self) -> None:
        """Close the connection and stop the background thread.

        The close is now *completed on the loop* rather than fired at
        it. It used to be scheduled with ``run_coroutine_threadsafe``
        and then followed immediately by ``loop.stop()``, which put both
        callbacks in the same pass of the loop: the close task was
        created, the stop ran, and the loop was gone before the task
        ever got a turn. The socket was therefore never closed -- the
        process just abandoned it -- and the broker at the far end was
        left waiting for a closing handshake from a peer whose event
        loop no longer existed. That is what made ``WsBroker.release()``
        cost exactly 3.000 s on every one of the 15 cycles measured: the
        three seconds were the broker's own shutdown timeout expiring.

        Nor is the stop conditional on ``is_running()`` any more. That
        guard raced the window between the loop being created and
        ``run_forever()`` starting, and a disconnect landing inside it
        left the thread running for the life of the process with
        ``_loop`` and ``_thread`` cleared, so nothing could ever stop it
        -- measured at 1 of 8 fast cycles. ``call_soon_threadsafe``
        needs the loop *open*, not *running*: a callback queued before
        it starts is the first thing it does.
        """
        self._stop.set()

        # Handed over rather than read back later: from here the thread
        # owns the loop, including closing it, and disconnect() must not
        # be able to act on a loop that is already being torn down.
        loop, thread = self._loop, self._thread
        self._loop = None
        self._thread = None

        if loop is not None:
            coro = self._shutdown()
            try:
                loop.call_soon_threadsafe(loop.create_task, coro)
            except RuntimeError:
                # Closed already, so its thread has finished and there
                # is nothing left to close. Closing the coroutine keeps
                # it from being reported as never awaited.
                coro.close()

        if thread is not None:
            thread.join(timeout=DISCONNECT_TIMEOUT_S)
            if thread.is_alive():
                log.warning(
                    "WsClient: the connection to %s did not shut down "
                    "within %g s; abandoning its thread and event loop. "
                    "The socket and one worker thread stay held until "
                    "the process exits.",
                    self._endpoint,
                    DISCONNECT_TIMEOUT_S,
                )

        self._ready.clear()
        self._ws = None

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _launch(self) -> None:
        self._stop.clear()
        loop = asyncio.new_event_loop()
        self._loop = loop
        # The loop is handed to the thread as an argument rather than
        # read back off self: disconnect() clears the attribute the
        # moment it has asked for a shutdown, and the thread still has
        # to run that loop to its end and close it.
        self._thread = threading.Thread(
            target=self._run_loop,
            args=(loop,),
            name="WsClient._run_loop",
            daemon=True,
        )
        self._thread.start()

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        try:
            loop.call_soon(loop.create_task, self._connect_loop())
            loop.run_forever()
        finally:
            close_loop(loop)

    async def _shutdown(self) -> None:
        """Close the socket, then stop this client's loop.

        Runs on the loop thread, which is what makes the close actually
        happen before the loop stops -- see :meth:`disconnect` for what
        the process did when it did not. Bounded, because the closing
        handshake needs the peer's half and a peer that has stopped
        answering must not be able to hold this client open.
        """
        ws = self._ws
        if ws is not None:
            try:
                await asyncio.wait_for(ws.close(), timeout=CLOSE_TIMEOUT_S)
            except Exception:
                # A socket that is already gone, or a peer that never
                # answered. Either way there is nothing more to close,
                # and the loop below still has to stop.
                pass
        asyncio.get_running_loop().stop()

    def _tls_context(self):
        """The TLS context to dial with, or None for a plain socket.

        Read once per connection attempt rather than cached: a
        reconnect after a certificate rotation should pick up the new
        one, and building a context is cheap next to a TCP handshake.
        """
        if not endpoint_is_secure(self._endpoint):
            return None
        from .....common.launch_config import LaunchConfig

        tls = LaunchConfig.get().client_tls()
        return tls.context() if tls is not None else None

    async def _authenticate(self, ws) -> None:
        """Present this process's data-plane token, if it has one.

        Sent unconditionally when a token is configured, because an edge
        cannot tell from the endpoint whether the broker requires one --
        and a broker that does not require one answers anyway.

        The reply is not awaited. A refusal closes the socket from the
        far end, which the reconnect loop already reports, and blocking
        here would add a round trip to every reconnection on the path
        that carries acquisition. What a refusal must not do is look like
        silence, so the broker logs it as a refusal and so does this.

        Args:
            ws: The freshly connected socket.
        """
        from .....common.launch_config import LaunchConfig

        config = LaunchConfig.get()
        token = getattr(config, "data_token", None) if config else None
        if not token:
            return
        await ws.send(json.dumps({"command": "auth", "token": token}))

    async def _connect_loop(self) -> None:
        """Reconnect indefinitely until disconnect() is called."""

        was_connected = False
        while not self._stop.is_set():
            try:
                async with _ws_connect(
                    self._endpoint, ssl=self._tls_context()
                ) as ws:
                    self._ws = ws
                    # Before anything else: a broker that requires a
                    # token refuses subscribe and every PUT until this
                    # has been sent, so it cannot wait until after the
                    # first frame.
                    await self._authenticate(ws)
                    if self._mode == "receive":
                        await ws.send(
                            json.dumps(
                                {
                                    "command": "subscribe",
                                    "stream_id": self._stream_id,
                                }
                            )
                        )
                    self._ready.set()
                    self._last_error = None
                    self._reported_error = False
                    log.debug(
                        "WsClient %s to %s (stream=%s)",
                        "reconnected" if was_connected else "connected",
                        self._endpoint,
                        self._stream_id,
                    )
                    was_connected = True
                    # Re-send stored context on every (re)connection
                    if (
                        self._mode == "send"
                        and self._pending_context is not None
                    ):
                        await self._safe_send_text(
                            encode_context(
                                self._stream_id, self._pending_context
                            )
                        )
                    if self._mode == "receive":
                        await self._receive_loop(ws)
                    else:
                        await self._control_loop(ws)
            except Exception as exc:
                self._last_error = exc
            self._ws = None
            self._ready.clear()
            if not self._stop.is_set():
                self._log_attempt_failed(was_connected)
                await asyncio.sleep(self._retry_interval)

    def _log_attempt_failed(self, was_connected: bool) -> None:
        """Report a failed connection attempt, rate-limited.

        The first failure is always reported. Previously the only
        warning here was gated on ``was_connected``, so a client that
        never reached the broker at all -- a wrong endpoint, a broker
        that never bound -- retried in complete silence.

        Args:
            was_connected: Whether this client had been connected
                before, which distinguishes a dropped connection from
                one that was never established.
        """
        exc = self._last_error
        # No exception means the socket closed cleanly rather than
        # failing, which only happens once a connection was established.
        reason = f"{type(exc).__name__}: {exc}" if exc else "closed by peer"
        now = time.monotonic()
        if (
            self._reported_error
            and now - self._last_error_log < self._drop_log_interval
        ):
            return
        self._last_error_log = now
        self._reported_error = True
        if was_connected:
            log.warning(
                "WsClient: connection to %s lost (%s), retrying every "
                "%.1fs",
                self._endpoint,
                reason,
                self._retry_interval,
            )
        else:
            log.warning(
                "WsClient: cannot reach %s (%s), retrying every %.1fs",
                self._endpoint,
                reason,
                self._retry_interval,
            )

    async def _receive_loop(self, ws) -> None:
        """Read frames from the broker and invoke data/context callbacks."""
        try:
            async for message in ws:
                if isinstance(message, bytes) and self._receive_callback:
                    try:
                        array, _ = decode(message)
                        self._receive_callback(array)
                    except Exception:
                        pass
                elif isinstance(message, str):
                    self._handle_text(message)
        except Exception:
            pass

    async def _control_loop(self, ws) -> None:
        """Read control messages while in sending mode.

        A sender used to wait for the socket to close and ignore
        everything arriving on it. It cannot: an attestation challenge
        arrives as a text message on this same connection, and the edge
        that holds the amplifier is usually the sender. Iterating ends
        when the socket closes, exactly as wait_closed() did.
        """
        try:
            async for message in ws:
                if isinstance(message, str):
                    self._handle_text(message)
        except Exception:
            pass

    def _handle_text(self, message: str) -> None:
        """Route one text control message.

        Args:
            message: The raw JSON text.
        """
        payload = decode_attest(message)
        if payload is not None:
            self._attest_inbox.put(payload)
            return
        if self._context_callback:
            try:
                _, context = decode_context(message)
                if context is not None:
                    self._context_callback(context)
            except Exception:
                pass

    async def _safe_send(self, payload: bytes) -> None:
        try:
            if self._ws is not None:
                await self._ws.send(payload)
        except Exception:
            pass

    async def _safe_send_text(self, payload: str) -> None:
        try:
            if self._ws is not None:
                await self._ws.send(payload)
        except Exception:
            pass

    def send_attest(self, payload: bytes) -> None:
        """Send one attestation-protocol message to the broker.

        Unlike put(), this does not swallow failures: the caller is
        ``gtec_attest.protocol``, which treats an exception as "no
        attestation" and needs to see one rather than wait for a reply
        that was never sent.

        Args:
            payload: Bytes from the attestation protocol.

        Raises:
            ConnectionError: If this client is not connected.
        """
        if self._ws is None or self._loop is None or not self._ready.is_set():
            raise ConnectionError(
                f"WsClient is not connected to {self._endpoint}"
            )
        future = asyncio.run_coroutine_threadsafe(
            self._ws.send(encode_attest(payload)), self._loop
        )
        future.result(timeout=5)

    def attest_channel(self) -> "WsAttestChannel":
        """Return this connection as an attestation channel.

        Returns:
            A channel satisfying ``gtec_attest.protocol.Channel``.
        """
        return WsAttestChannel(self.send_attest, self._attest_inbox)


class WsAttestChannel:
    """A ``gtec_attest`` Channel over one WebSocket connection.

    Deliberately holds a send *callable* and a queue rather than the
    client or broker itself: both sides of the link produce one of these,
    and the attestation protocol should not be able to tell which end it
    is talking to.
    """

    def __init__(self, send, inbox) -> None:
        """Initialise the channel.

        Args:
            send: Callable taking the bytes to deliver.
            inbox: Queue of received message bytes.
        """
        self._send = send
        self._inbox = inbox

    def send(self, data: bytes) -> None:
        """Deliver one message.

        Args:
            data: The message bytes.
        """
        self._send(data)

    def recv(self, timeout: float) -> Optional[bytes]:
        """Return one message, or None on timeout.

        Args:
            timeout: Seconds to wait.

        Returns:
            The message bytes, or None.
        """
        try:
            return self._inbox.get(timeout=timeout)
        except queue.Empty:
            return None
