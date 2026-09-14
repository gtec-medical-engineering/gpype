import re
import select
import socket
import threading
import time
from typing import List

import ioiocore as ioc

from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.event_source import EventSource

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: A numeric payload, with the tolerance a sender on the other side of
#: a C API actually needs: optional sign, decimal point, exponent.
#: ``str.isdigit`` accepted none of those, and the bare ``except`` it
#: sat behind discarded b"+1", b"1.0" and b"-1" without a word.
_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


class _UDPReceiverCore(EventSource):
    """Internal node implementing UDP network event reception.

    This is the actual UDP receiver node (pure ONode inheritance).
    It is wrapped by the UDPReceiver chain for distributed operation.

    Listens on specified IP/port for UDP packets containing numeric
    trigger values. Each trigger outputs the received value, and then a
    zero once ``HOLD_TIME_MS`` has passed -- not immediately. Emitted
    back to back, the value and its reset are observed microseconds
    apart, which at any real sampling rate is the *same sample*: they
    then place onto one row of the grid and the reset overwrites the
    value it was meant to follow. Measured through a Router at 250 Hz,
    six datagrams with frames of 25: one trigger reached the consumer
    with the immediate reset, five with it held.
    """

    #: Default IP address for localhost binding
    DEFAULT_IP: str = "127.0.0.1"
    #: Default UDP port number for listening
    DEFAULT_PORT: int = 1000
    #: Hold time in milliseconds before the reset trigger follows a
    #: value. Declared with the protocol from the start and then never
    #: read: the reset went out on the line after the value, so the
    #: two shared a sample and the trigger was erased before any
    #: consumer saw it. Ten milliseconds is 2-3 samples at 250 Hz --
    #: enough to land on a row of its own at every rate g.Pype
    #: supports, and short enough that a paradigm asserting a code
    #: for 50 ms still reads as one pulse.
    HOLD_TIME_MS: int = 10

    #: How long stop() waits for the listener thread before giving up on
    #: it. The loop wakes on a 10 ms select timeout, so this is about a
    #: hundred chances to see the cleared flag -- generous enough that
    #: reaching it means something is wrong, and short enough that it is
    #: not mistaken for a hang.
    STOP_TIMEOUT_S: float = 1.0

    class Configuration(EventSource.Configuration):
        """Configuration class for UDP receiver network parameters."""

        class Keys(EventSource.Configuration.Keys):
            """Configuration key constants for the UDP receiver."""

            #: Configuration key for IP address binding
            IP: str = "ip"
            #: Configuration key for UDP port number
            PORT: str = "port"

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

        #: Flag indicating if UDP listener thread is running
        self._udp_thread_running = False
        #: UDP socket instance for message reception
        self._socket = None
        #: Guards the hand-off of the socket between the listener thread
        #: and stop(). The thread creates and binds the socket itself, so
        #: without this stop() can run to completion in the window before
        #: the thread has published it -- and then the thread binds a
        #: port that nothing will ever close. Measured: 50 of 50 cycles
        #: with the thread delayed 50 ms left the port held after stop()
        #: returned, reported by a later bind as WinError 10048.
        self._socket_lock = threading.Lock()
        #: Background thread for UDP message listening
        self._udp_thread = None
        #: Start time for timing analysis
        self._t_start = None
        #: When the pending reset is due, or None. Kept on the loop
        #: rather than handed to a timer thread: the loop already
        #: wakes on a select timeout, so it can do this itself.
        self._reset_due = None
        #: Payloads rejected as non-numeric, and the count at which
        #: the next report is due.
        self._rejected = 0
        self._reject_report_due = 1
        #: Reads that failed at the socket, and the count at which the
        #: next report is due. The same doubling scheme as the
        #: rejections above, and for the same reason: one failure is
        #: usually a previous send being refused by ICMP and clears
        #: itself, while one line per datagram would bury the run the
        #: report is about.
        self._failed_reads = 0
        self._read_report_due = 1

    def _udp_listener(self):
        """Background thread function for UDP message reception.

        Creates UDP socket and continuously listens for incoming messages.
        Parses numeric string data and triggers events. Uses select() for
        non-blocking operation to allow clean shutdown.
        """
        # Get network configuration
        ip_key = self.Configuration.Keys.IP
        port_key = self.Configuration.Keys.PORT
        ip = self.config[ip_key]
        port = self.config[port_key]

        # Held locally until it is bound and this thread is sure it is
        # still wanted. Publishing to self._socket first is what let a
        # stop() that ran during startup return while this thread went
        # on to bind the port anyway.
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)  # Non-blocking for select()

        # Bind socket to specified address and port.
        #
        # Reported rather than raised: this runs on a background thread,
        # so an exception here reaches nobody. Measured with the port
        # already held -- a second UDPReceiver on the same port, or a
        # stale process -- start() returned normally, this thread died
        # with WinError 10048, and a windowed application showed a
        # running pipeline that would never receive a trigger.
        try:
            sock.bind((ip, port))
        except OSError as error:
            self._udp_thread_running = False
            self.log(
                f"Could not listen on {ip}:{port} -- {error}. No trigger "
                f"will be received on this node. Another receiver or a "
                f"leftover process is holding the port; note that "
                f"ParadigmPresenter is itself a UDPReceiver and defaults "
                f"to the same address, so wiring both binds it twice.",
                type=Constants.LogTypes.ERROR,
            )
            try:
                sock.close()
            except OSError:  # pragma: no cover
                pass
            return

        # The hand-off. stop() takes the same lock to clear the flag and
        # close whatever is published, so exactly one of the two runs
        # first: either stop() sees this socket and closes it, or this
        # thread sees the cleared flag and closes it here. Neither can
        # leave it open.
        with self._socket_lock:
            if not self._udp_thread_running:
                sock.close()
                return
            self._socket = sock

        # Bound once, on entry, rather than looked up on the module every
        # iteration. Two things follow, and the second is why it matters:
        #
        # A thread must not have its behaviour changed by whatever else
        # the process does to `select` while it is running -- and holding
        # the reference means the object cannot be collected out from
        # under a loop that is about to call it. That is the documented
        # crash: an orphaned listener, from a test whose assertion failed
        # between start() and stop(), picking up a *later* test's
        # `unittest.mock` patch and racing its teardown, which surfaces
        # as an access violation during garbage collection rather than as
        # anything resembling its cause.
        select_ready = select.select

        try:
            # Main reception loop
            while self._udp_thread_running:
                # Wake early enough to send a due reset, rather than sitting
                # in select for the full timeout while one is outstanding.
                timeout = 0.01
                if self._reset_due is not None:
                    timeout = max(
                        0.0, min(timeout, self._reset_due - time.monotonic())
                    )
                ready, _, _ = select_ready([sock], [], [], timeout)
                if not self._udp_thread_running:
                    break

                self._release_reset()

                if not ready:
                    continue

                try:
                    # Receive UDP packet (max 1024 bytes)
                    data, _ = sock.recvfrom(1024)
                    value = self._parse(data)
                    if value is None:
                        continue

                    # A value supersedes a reset that has not gone out yet.
                    # Emitting the old reset first would put it in this
                    # value's sample and erase it -- the failure this hold
                    # exists to prevent -- and a paradigm re-asserting a code
                    # inside 10 ms means the code is still on, not that it
                    # blinked off in between.
                    self.trigger(value)
                    self._reset_due = time.monotonic() + (
                        self.HOLD_TIME_MS / 1000.0
                    )

                except Exception as error:  # noqa: BLE001
                    # A socket error, not a payload problem: the payload path
                    # reports for itself. Never break the loop over one
                    # packet -- Windows reports a previous send's ICMP
                    # rejection as WSAECONNRESET on the *next* recvfrom,
                    # which is spurious and clears itself.
                    #
                    # "Never swallow it without trace either" is what this
                    # claimed while doing the opposite. It went to the
                    # module logger, and nothing in g.Pype configures
                    # stdlib logging, so log.debug was dropped outright:
                    # the session log, the monitor thread and the GUI's
                    # failure box never learned a trigger had been lost.
                    self._report_failed_read(error, ip, port)
                    continue

        finally:
            # A reset still owed would strand the trigger asserted for
            # the rest of the run. In a finally, because the loop can
            # also leave by exception -- and a listener that dies is
            # exactly when a stuck trigger would be hardest to spot.
            self._reset_due = 0.0
            self._release_reset()

    def _release_reset(self) -> None:
        """Emit the pending reset, if it has come due."""
        due = self._reset_due
        if due is None or time.monotonic() < due:
            return
        self._reset_due = None
        self.trigger(0)

    def _report_failed_read(self, error, ip, port) -> None:
        """Report a read that failed at the socket, but not too often.

        Args:
            error: What ``recvfrom`` -- or the trigger it drove --
                raised.
            ip: The bound address, for the message.
            port: The bound port, for the message.
        """
        self._failed_reads += 1
        if self._failed_reads < self._read_report_due:
            return
        self._read_report_due = self._failed_reads * 10
        self.log(
            f"{node_label(self)}: {self._failed_reads} read(s) on "
            f"{ip}:{port} failed; the most recent said: {error}. Any "
            f"trigger sent while a read was failing is lost. One "
            f"failure is usually a previous send refused by ICMP and "
            f"needs nothing; a rising count is a socket that is not "
            f"working -- check the address and port the sender uses.",
            type=Constants.LogTypes.WARNING,
        )

    def _parse(self, data: bytes):
        """Return the numeric value in one datagram, or None.

        Args:
            data: The datagram as received.

        Returns:
            The value, or None if the payload does not carry one -- in
            which case the rejection is counted and reported. It used to
            be dropped in silence, which made a sender whose format did
            not match indistinguishable from a sender that was not
            running: b"1\x00" -- what a C caller passing strlen + 1
            produces -- was discarded without a word.
        """
        text = data.decode("utf-8", errors="replace")
        # NULs first: a C sender's terminator is not whitespace, so
        # strip() leaves it in place and every numeric test then fails.
        text = text.replace("\x00", "").strip().lstrip("\ufeff")
        if _NUMBER.match(text):
            try:
                return int(text)
            except ValueError:
                return float(text)

        self._rejected += 1
        if self._rejected >= self._reject_report_due:
            self._reject_report_due = self._rejected * 10
            # self.log and not the module logger: nothing in g.Pype
            # configures stdlib logging, so a report whose whole purpose
            # is to tell an operator their sender's format is wrong
            # reached the session log exactly never.
            self.log(
                f"{node_label(self)}: {self._rejected} UDP payload(s) "
                f"were not numeric and were ignored; the most recent "
                f"was {data[:32]!r}. A trigger source whose format does "
                f"not match is otherwise indistinguishable from one "
                f"that is not sending at all.",
                type=Constants.LogTypes.WARNING,
            )
        return None

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

        # Stop UDP listener thread and cleanup resources.
        #
        # Under the lock, so that a thread still starting up cannot
        # publish its socket after this has already looked for one.
        # The join is deliberately outside it: the listener takes the
        # same lock to publish, so joining while holding it deadlocks.
        #
        # The socket is taken here and closed *after* the join, not
        # here. The listener blocks in select() on this very socket, and
        # closing it underneath a thread sitting in select is an access
        # violation on Windows -- a hard process crash, not an
        # exception. Measured: it killed the suite at
        # test_start_creates_thread with the fault inside
        # `_udp_listener`. Clearing the flag is enough to end the loop,
        # which wakes every 10 ms to check it.
        with self._socket_lock:
            was_running = self._udp_thread_running
            self._udp_thread_running = False
            socket_to_close = self._socket
            self._socket = None

        if was_running and self._udp_thread:
            # Bounded, because an unbounded join here is an unbounded
            # stop(): a listener wedged for any reason would hang the
            # pipeline's shutdown, and a windowed application's exit,
            # with nothing said. The loop wakes on a 10 ms select
            # timeout, so a second is roughly a hundred chances to
            # notice the cleared flag.
            self._udp_thread.join(timeout=self.STOP_TIMEOUT_S)
            if self._udp_thread.is_alive():
                self.log(
                    f"the UDP listener for {node_label(self)} did not stop "
                    f"within {self.STOP_TIMEOUT_S} s and was abandoned. "
                    f"It is a daemon thread, so it cannot hold the "
                    f"process open, but the port may stay bound until "
                    f"the process exits.",
                    type=Constants.LogTypes.WARNING,
                )
            self._udp_thread = None

        # Now that nothing is selecting on it. Closing it before the
        # join is what crashed the process; see the note above.
        if socket_to_close is not None:
            socket_to_close.close()


class UDPReceiver(ioc.OChain):
    """UDP network receiver chain for capturing remote trigger events.

    This is an OChain that contains:
    - _UDPReceiverCore: The actual UDP reception node
    - Link: Bridge for distributed operation (passthrough in standalone)

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Listens on specified IP/port for UDP packets containing numeric
    trigger values. Each trigger outputs the received value, and then a
    zero once ``HOLD_TIME_MS`` has passed -- not immediately. Emitted
    back to back, the value and its reset are observed microseconds
    apart, which at any real sampling rate is the *same sample*: they
    then place onto one row of the grid and the reset overwrites the
    value it was meant to follow. Measured through a Router at 250 Hz,
    six datagrams with frames of 25: one trigger reached the consumer
    with the immediate reset, five with it held.
    """

    #: Default IP address for localhost binding
    DEFAULT_IP = _UDPReceiverCore.DEFAULT_IP
    #: Default UDP port number for listening
    DEFAULT_PORT = _UDPReceiverCore.DEFAULT_PORT

    def __init__(
        self,
        ip: str = _UDPReceiverCore.DEFAULT_IP,
        port: int = _UDPReceiverCore.DEFAULT_PORT,
        **kwargs,
    ):
        """Initialize UDP receiver chain.

        Args:
            ip: IP address to bind socket to. Use "0.0.0.0" for all interfaces
                or "127.0.0.1" for localhost. Defaults to localhost.
            port: UDP port number to listen on. Defaults to 1000.
            **kwargs: Additional parameters.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {"ip": ip, "port": port}
        self._core_params.update(strip_chain_keys(kwargs))

        # Initialize OChain (calls create_internal_nodes)
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration()],
        )
        ioc.OChain.__init__(
            self,
            ip=ip,
            port=port,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create the internal node chain.

        Returns:
            The head of the list is whatever stands in for this chain's
            core: the core itself, the core followed by a raw tap
            under ``save_as``, a replay core under ``load_from``,
            or nothing at all under SERVER residency. Then a Link
            where the pipeline is distributed, and Sync last.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        # Recorded, replaced, or simply built, as the launch
        # configuration says. Contributes nothing under server
        # residency: the core lives on the edge, and so does
        # anything recording or replaying it.
        nodes.extend(
            raw.source_stage(
                self,
                lambda: _UDPReceiverCore(**self._core_params),
                timing=Constants.Timing.ASYNC,
            )
        )
        if residency != Constants.Residency.STANDALONE:
            nodes.append(
                Link(
                    sender=Constants.Residency.EDGE,
                    receiver=Constants.Residency.SERVER,
                    stream_id=self._link_stream_id,
                    # An event stream is sparse on both sides of the
                    # link, so the ports have to say so. Without this the
                    # ASYNC output of the event core meets a SYNC input
                    # and the chain cannot be built at all.
                    timing=Constants.Timing.ASYNC,
                )
            )
        # Sync places each event on the master timeline. ASYNC
        # because a sparse stream must stay sparse: announced as
        # continuous, every downstream node would wait for data on
        # every cycle.
        nodes.append(Sync(timing=Constants.Timing.ASYNC))
        return nodes
