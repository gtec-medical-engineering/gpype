from __future__ import annotations

import atexit
import queue
import threading
import time
from collections import deque
from typing import List, Optional

import ioiocore as ioc
import numpy as np

from ...common._private import channels, driver_usage
from ...common.constants import Constants
from ...common.montage import Montage
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.oscar import Oscar
from ..core._private.sync import Sync
from ..core._private.timeline import MasterPriority
from ..core.o_port import OPort
from .base import raw
from .base.amplifier_source import AmplifierSource, as_whole_number

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN

#: Device-name prefixes advertised by g.tec BLE amplifiers. A BLE scan
#: reports every peripheral in range, so the discovery list has to be
#: filtered before a device is opened.
GTEC_NAME_PREFIXES = ("UN-", "U8-", "U4-")

#: Guards the process-wide BLE runtime state.
_ble_lock = threading.Lock()
#: True once an amplifier has been opened in this process.
_ble_used = False


#: The gtec_ble module, or None until something needs it. Not imported
#: at module scope: a document naming BCICore (or the deprecated
#: BCICore8 alias) is deserialised in every residency, but the
#: acquisition core is built only where the device is.
#: Importing eagerly dragged gtec_ble and its native wrapper into server
#: processes that never touch a device -- and where that wrapper is not
#: packaged, the import fails and takes the whole document with it.
ble = None


def _load() -> None:
    """Import gtec_ble, once, on first use.

    Only fills the name if it is still None, so a test that has replaced
    ``ble`` with a stand-in keeps it.
    """
    global ble
    if ble is None:
        import gtec_ble

        ble = gtec_ble


def _ble_acquire() -> None:
    """Note that the process-wide BLE runtime is now in use."""
    global _ble_used
    with _ble_lock:
        _ble_used = True


@atexit.register
def _ble_shutdown_at_exit() -> None:  # pragma: no cover - exit hook
    """Release the BLE runtime once, at process exit.

    gtec-ble requires an explicit shutdown before the process ends;
    skipping it leaves the Bluetooth adapter claimed, so the next run
    fails to discover or open the device. It must not be called between
    acquisitions: the native runtime cannot be re-initialised afterwards,
    and every later scan would silently return no devices.
    """
    global _ble_used
    with _ble_lock:
        if not _ble_used:
            return
        _ble_used = False
        try:
            ble.Amplifier.shutdown()
        except Exception:
            pass


class _BCICoreCore(AmplifierSource):
    """Internal node implementing BCI Core amplifier acquisition logic.

    This is the actual BCI Core node (pure ONode inheritance).
    It is wrapped by the BCICore chain for distributed operation.

    Interface to g.tec BCI Core-8 wireless EEG amplifier using BLE. By
    default it acquires every EEG channel the amplifier reports at
    250 Hz; ``channel_count``, ``channels`` and ``sampling_rate`` narrow,
    reorder or redeclare that.
    """

    #: Bluetooth scanning timeout in seconds
    SCANNING_TIMEOUT_S = 6
    #: Default sampling rate in Hz, for an amplifier not told otherwise.
    #: Not a hardware limit: the BLE amplifier configuration is external
    #: to this node (see ``sampling_rate`` on __init__), and this is only
    #: what a caller gets by not asking.
    SAMPLING_RATE = 250
    #: Maximum number of supported EEG channels
    MAX_NUM_CHANNELS = 8
    #: Capacity of the hand-off queue between the BLE callback thread and
    #: the worker that drives the pipeline, in samples.
    #:
    #: This is not a jitter buffer and it is not held at any fill level:
    #: the worker drains it as data arrives, so it normally sits near
    #: empty and adds no latency. It only needs to absorb scheduling
    #: slack between the two threads, and one second of room makes
    #: overflow a symptom of something genuinely wrong rather than a
    #: tuning question.
    QUEUE_CAPACITY_SAMPLES = 250
    #: Number of discovery/open attempts before giving up.
    #: One: a second scan of a radio that just answered
    #: nothing costs the whole timeout again, and start()
    #: blocks the calling thread for all of it.
    CONNECT_ATTEMPTS = 1
    #: Pause between connection attempts in seconds
    CONNECT_RETRY_DELAY_S = 1.0
    #: Minimum seconds between repeated packet-loss warnings
    LOSS_LOG_INTERVAL_S = 10.0
    #: Samples lost in one dropout above which it is reported as a
    #: warning rather than a note. A lost BLE payload is four samples,
    #: so 25 -- 100 ms at this rate -- is well clear of the
    #: single-payload dropouts an ordinary link produces, and short
    #: enough that anything above it is worth an operator's attention.
    LOSS_WARN_SAMPLES = 25
    #: Dropouts within LOSS_WARN_WINDOW_S above which the link is
    #: reported as degrading, however small each one is on its own.
    #: Repetition is the signal there: one four-sample dropout says
    #: nothing, five of them in a minute say the radio is struggling.
    LOSS_WARN_BURSTS = 5
    #: Window over which repeated dropouts are counted, in seconds.
    LOSS_WARN_WINDOW_S = 60.0
    #: Highest device counter value that can still be a restart. The
    #: driver resumes numbering at 1 after a reconnect, so only the first
    #: few counts of a fresh connection look like this; a mid-stream
    #: counter never does.
    RECONNECT_COUNTER_MAX = 16
    #: Shortest arrival gap that can be an outage, in seconds. A reconnect
    #: takes far longer than this; a notification overtaking its
    #: predecessor takes essentially no time at all.
    RECONNECT_MIN_GAP_S = 0.5
    #: How long a stream may deliver nothing before it is reported, in
    #: seconds. Twice the driver's one-second back-fill horizon, so an
    #: ordinary dropout the driver repairs itself never trips this.
    STALL_TIMEOUT_S = 2.0
    #: Channels carrying the timeline: master index, sample
    #: validity, and the host time the sample arrived at, in that
    #: order. Appended to the EEG channels in band, and stripped
    #: again by the Sync node at the end of the chain.
    #:
    #: The arrival time is in band because it has to survive a
    #: Link. Under distributed residency the Sync that builds the
    #: master-timeline relation runs in the server process, where
    #: reading a clock measures when the frame finished crossing
    #: the network. The instant that matters is this one, known
    #: only in the acquisition callback -- and not even at the
    #: moment the frame is emitted, because frames wait in
    #: _frame_buffer first.
    NUM_TIMELINE_CHANNELS = 3
    #: Labels of the timeline channels
    TIMELINE_LABELS = ("master_index", "valid", "host_time")

    #: The electrode montage the 8-channel device ships with, in
    #: acquisition order.
    #:
    #: Applied only when eight EEG channels are acquired and the author
    #: named none, because it describes *that* headset. A four-channel
    #: device wears a different cap, and guessing its electrodes from
    #: this list -- by truncation or otherwise -- would put a confident
    #: wrong name on every column.
    #:
    #: This does change what an unlabelled recording looks like: the
    #: same session that produced Ch01..Ch08 now produces Fz..PO8. That
    #: is the point. A positional name is not neutral, it is merely
    #: silent about something the device does know, and the electrode a
    #: column came from is the one fact a recording cannot recover
    #: afterwards. An author who wants different names still passes a
    #: montage, and one who wants none cannot -- which is the right way
    #: round for a device whose electrode positions are fixed.
    DEFAULT_MONTAGE_8 = (
        "Fz",
        "C3",
        "Cz",
        "C4",
        "Pz",
        "PO7",
        "POz",
        "PO8",
    )

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for BCI Core specific parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration keys for BCI Core settings."""

            #: Number of EEG channels, as requested. Recorded separately
            #: because the port's channel_count is the total including the
            #: timeline channels: reading the EEG count back out of that
            #: total would add the timeline channels a second time on
            #: every round trip.
            EEG_CHANNEL_COUNT = "eeg_channel_count"

        class OptionalKeys(AmplifierSource.Configuration.OptionalKeys):
            """Optional configuration keys.

            A key listed in Keys must be present and non-None, and a
            montage is genuinely optional, so it belongs here.
            """

            #: Electrode labels, as a plain list. A Montage object is not
            #: JSON-serialisable, so the labels travel and the object is
            #: rebuilt from them.
            MONTAGE_LABELS = "montage_labels"
            #: Serial number of the target device. Recorded because it
            #: selects which physical amplifier is opened: a pipeline
            #: rebuilt without it would silently connect to whichever
            #: device happened to answer first.
            SERIAL = "serial"

    #: BLE amplifier device connection instance
    _device: Optional[ble.Amplifier]
    #: Target device serial number for connection
    _target_sn: Optional[str]

    def __init__(
        self,
        serial: Optional[str] = None,
        channel_count: Optional[int] = None,
        frame_size: Optional[int] = None,
        montage: Optional[Montage] = None,
        sampling_rate: Optional[int] = None,
        **kwargs,
    ):
        """Initialize BCI Core-8 amplifier source.

        Samples are numbered on the master timeline as they arrive and the
        pipeline advances when data is available. There is no regenerated
        sampling clock, so nothing has to be kept in step with the device
        and no placeholder samples are ever produced.

        Args:
            serial: Serial number of target device. Uses first discovered
                if None.
            channel_count: Number of EEG channels (1-8). Defaults to 8,
                or to the size of the montage if one is given.
            frame_size: Samples per processing frame.
            montage: Electrode labels for the EEG channels. Names the
                channels for downstream nodes, files and streams.
            sampling_rate: Rate the device runs at, in Hz. Leave it
                unset and the device is asked: the node connects during
                the master-timeline election and adopts whatever rate the
                amplifier reports. That is the only way a gCore device can
                work unattended, because it is runtime-configurable --
                16 channels with 256 Hz, 8 with 512, 4 with 1024, 1 with
                4096 -- and nothing outside the device knows which.

                Give a value only to assert one. It is then checked
                against the device at connect and a disagreement is an
                error rather than a silent re-interpretation.
            **kwargs: Additional arguments for parent AmplifierSource.

        Raises:
            ValueError: If montage size and channel_count disagree.
        """
        # The library is imported here, not at module scope: this core is
        # built only where the hardware is, while the chain that names it
        # is built in every residency.
        _load()
        keys = self.Configuration.Keys

        # Restoring from a stored configuration: the EEG count and the
        # montage labels were recorded in their own keys precisely so
        # they can be read back without re-deriving them from the port's
        # total, which already includes the timeline channels.
        # A restored configuration binds to the named parameters, in the
        # per-port list form it was stored as, so normalise them here.
        channel_count = self.scalar(channel_count)
        frame_size = self.scalar(frame_size)
        restored = self.scalar(kwargs.pop(keys.EEG_CHANNEL_COUNT, None))
        if restored is not None:
            channel_count = restored
        opt = self.Configuration.OptionalKeys
        labels = kwargs.pop(opt.MONTAGE_LABELS, None)
        if montage is None and labels:
            montage = Montage(list(labels))
        if serial is None:
            serial = kwargs.pop(opt.SERIAL, None)
        else:
            kwargs.pop(opt.SERIAL, None)
        # An empty serial means "no specific device"; normalise it here so
        # the stored configuration, _target_sn and _discover() agree. The
        # key is validated once declared optional, and an empty value is
        # rejected, so it must not travel as "".
        serial = serial or None

        # A montage describes exactly the EEG channels, so it can supply
        # the channel count when one was not requested explicitly.
        if montage is not None:
            if channel_count is None:
                channel_count = len(montage)
            elif len(montage) != channel_count:
                raise ValueError(
                    f"montage has {len(montage)} labels but "
                    f"channel_count is {channel_count}."
                )

        # Validate and set channel count (1-8 channels supported)
        if channel_count is None:
            channel_count = self.MAX_NUM_CHANNELS
        channel_count = max(1, min(channel_count, self.MAX_NUM_CHANNELS))
        num_eeg_channels = channel_count

        if montage is not None and len(montage) != num_eeg_channels:
            raise ValueError(
                f"montage has {len(montage)} labels but this amplifier "
                f"supports at most {self.MAX_NUM_CHANNELS} EEG channels."
            )
        self._montage = montage
        #: Frame taken off the queue by the worker, awaiting emission by
        #: the next step().
        self._pending: Optional[tuple] = None
        #: Serialises notification handling. The driver calls back from a
        #: thread pool, and frame assembly mutates shared buffers.
        self._callback_lock = threading.Lock()
        #: Notifications that arrived after a later one.
        self._reordered_blocks: int = 0
        #: Rate limit for the out-of-order warning.
        self._last_reorder_log: float = 0.0

        # The timeline rides in-band, on the same port as the EEG, so it
        # survives the Link without a second stream to re-pair. The Sync
        # node at the end of the chain strips it again.
        channel_count += self.NUM_TIMELINE_CHANNELS

        # Everything with a fixed value goes in through kwargs rather than
        # as an explicit keyword: on a round trip the stored configuration
        # arrives in kwargs too, and passing both is a duplicate keyword.
        kwargs.setdefault(keys.OUTPUT_PORTS, [OPort.Configuration()])
        # The rate is a property of the device, but not one this node can
        # discover before it has to be declared: the pipeline elects its
        # master timeline from the *configured* rate, and that election
        # runs before any device handle is opened. So it is declared here
        # and verified against the device at connect, where a disagreement
        # is an error rather than a warning.
        #
        # It is also read back from a stored configuration, unlike before.
        # Dropping it made a document authoritative about the channel count
        # and silent about the rate, so a session saved at one rate
        # reloaded at the class default without anything reporting it.
        restored_rate = self.scalar(kwargs.pop(keys.SAMPLING_RATE, None))
        if sampling_rate is None:
            sampling_rate = restored_rate
        #: Whether a rate was asked for, as opposed to defaulted to. It
        #: decides what a disagreement with the device means: a declared
        #: rate that does not match is an error, while a defaulted one is
        #: simply replaced by what the device reports.
        self._rate_declared = sampling_rate is not None
        if sampling_rate is None:
            sampling_rate = self.SAMPLING_RATE
        sampling_rate = int(sampling_rate)
        #: The rate this node runs at. Every timeline computation reads
        #: this rather than the class constant, which is only the value
        #: used until the device says otherwise.
        self._sampling_rate = sampling_rate
        # A property of this device, so a stored copy carries no
        # information; dropping it keeps it from arriving twice.
        kwargs.pop(keys.DECIMATION_FACTOR, None)

        if montage is not None:
            kwargs[opt.MONTAGE_LABELS] = list(montage.labels)
        if serial:
            kwargs[opt.SERIAL] = serial

        super().__init__(
            sampling_rate=sampling_rate,
            frame_size=frame_size,
            # The worker cycles once per frame, so no decimation applies.
            decimation_factor=1,
            channel_count=channel_count,
            # Recorded so a round trip can rebuild both exactly.
            eeg_channel_count=num_eeg_channels,
            **kwargs,
        )

        self._frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]
        buf_size_frames = int(
            np.ceil(self.QUEUE_CAPACITY_SAMPLES / self._frame_size)
        )

        # Store device configuration
        self._target_sn = serial
        self._num_eeg_channels = num_eeg_channels

        # Initialize device connection (will be established in start())
        self._device = None

        # Sample layout, resolved from the device once it is opened. The
        # amplifier streams EEG together with accelerometer, battery and
        # counter channels, and their order is device-dependent.
        self._eeg_idx: Optional[np.ndarray] = None
        self._sample_width: Optional[int] = None

        # Columns of the device counter and validity channels, resolved
        # once the device reports its channel map.
        self._cnt_idx: Optional[int] = None
        self._valid_idx: Optional[int] = None

        #: Pipeline timeline, bound before start if one exists.
        self._timeline = None
        #: True when this source won the master-timeline claim.
        self._is_master_source: bool = False

        # Timeline state must exist from construction: step() may be
        # driven before start() has run.
        self._reset_timeline()

        # Initialize threading components for real-time processing
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None

        # Initialize data management components
        self._in_sample_counter: int = 0
        self._frame_buffer: Optional[queue.Queue] = None
        self._sample_buffer: Optional[np.ndarray] = None
        self._buffer_size_frames: int = buf_size_frames
        #: Frames discarded because the queue was full.
        self._overflow_count: int = 0
        self._last_overflow_log: float = 0.0

    def start(self) -> None:
        """Start BCI Core-8 amplifier and begin data acquisition.

        Initializes buffers, starts background thread, establishes BLE
        connection, and begins real-time data streaming.

        Raises:
            ConnectionError: If amplifier connection fails.
            RuntimeError: If background thread creation fails.
        """
        # Get configuration parameters
        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
        # Initialize data buffers for frame-based processing. The sample
        # buffer holds EEG only; timeline position and validity travel
        # alongside it so they can never drift apart from their samples.
        self._frame_buffer = queue.Queue(maxsize=self._buffer_size_frames)
        self._sample_buffer = np.zeros((frame_size[0], self._num_eeg_channels))
        self._meta_buffer = np.zeros(
            (frame_size[0], self.NUM_TIMELINE_CHANNELS)
        )
        self._overflow_count = 0
        # Start the interval now so a single transient drop stays quiet.
        self._last_overflow_log = time.monotonic()

        # Reset sample accounting so a restarted pipeline does not inherit
        # the previous run's epoch and counters.
        self._in_sample_counter = 0
        self._reset_timeline()

        # Connect before the worker starts consuming, so that a failed
        # connection does not leave a thread spinning.
        if self._device is None:
            self._connect()

        # Bring the base class up before the worker starts cycling, so
        # the worker never drives a node that is not fully started. The
        # device already streams by this point; its frames simply wait in
        # the queue.
        super().start()

        # Start the worker that drives pipeline cycles from arrival.
        if not self._running:
            self._running = True
            self._thread = threading.Thread(
                target=self._worker_function, daemon=True
            )
            self._thread.start()

        # Begin data acquisition from amplifier
        self._device.start()
        # Anchors the silence watchdog, so a stream that never delivers a
        # first sample is reported rather than waited on forever.
        self._stream_since = channels.stamp_clock()
        self._stalled = False
        self._last_stall_log = 0.0

    def open_for_attestation(self) -> None:
        """Connect before the entitlement gate runs. See the base class.

        Safe to call twice: `_connect` is already guarded by
        ``if self._device is None`` in `start()`, and the same guard is
        applied here, so the handle opened for the challenge is the one
        `start()` goes on to use. Opening a second BLE handle would be
        refused anyway -- the device is exclusive.

        Never raises. A Core-8 that is out of range or off is not an
        entitlement question: the run stays unattested and `start()`
        reports the connection failure with its own error.
        """
        if self._device is not None:
            return
        try:
            self._connect()
        except Exception as error:  # noqa: BLE001
            self.log(
                f"could not open the amplifier before the entitlement "
                f"gate, so this run is unattested: {error}",
                type=Constants.LogTypes.WARNING,
            )

    def _discover(self) -> str:
        """Scan for a BCI Core-8 and return the device name to open.

        Returns:
            Name of the amplifier to connect to.

        Raises:
            ConnectionError: If no suitable device is discovered.
        """
        seen: List[str] = []
        for attempt in range(self.CONNECT_ATTEMPTS):
            try:
                if attempt == 0:
                    # Fast path: reuse the process-wide cache if a previous
                    # scan already found the device.
                    devices = list(ble.Amplifier.get_connected_devices())
                else:
                    # gtec-ble caches empty results too, so one unlucky
                    # scan would otherwise make every later attempt fail
                    # without ever touching the radio again.
                    ble.Amplifier.clear_device_cache()
                    devices = list(
                        ble.Amplifier.get_connected_devices(rescan=True)
                    )
            except Exception as exc:
                self.log(
                    f"BLE scan failed: {exc}",
                    type=Constants.LogTypes.WARNING,
                )
                devices = []
            seen = devices

            if self._target_sn is not None:
                if self._target_sn in devices:
                    return self._target_sn
            else:
                # The scan returns as soon as any peripheral answers, so
                # the list can contain unrelated BLE devices. Prefer names
                # that look like a g.tec amplifier.
                gtec = [d for d in devices if d.startswith(GTEC_NAME_PREFIXES)]
                candidates = gtec or devices
                if candidates:
                    if len(candidates) > 1:
                        self.log(
                            f"Multiple BLE devices discovered {candidates}; "
                            f"using {candidates[0]}. Pass serial=... to "
                            f"select a specific amplifier."
                        )
                    return candidates[0]

            # No pause after the cache lookup: go straight to a real scan.
            if attempt > 0 and attempt + 1 < self.CONNECT_ATTEMPTS:
                time.sleep(self.CONNECT_RETRY_DELAY_S)

        target = self._target_sn or "BCI Core-8"
        raise ConnectionError(
            f"Could not discover {target}. Devices seen: "
            f"{seen if seen else 'none'}. Check that the amplifier is "
            f"powered on, charged and not connected elsewhere."
        )

    def _connect(self) -> None:
        """Discover, open and configure the amplifier.

        Raises:
            ConnectionError: If the device cannot be opened.
        """
        # gtec_ble refuses to construct an Amplifier until the local usage
        # key is registered, so this precedes discovery as well as the
        # handle itself.
        driver_usage.register()

        serial = self._discover()
        last_error: Optional[Exception] = None

        for attempt in range(self.CONNECT_ATTEMPTS):
            try:
                device = ble.Amplifier(serial=serial)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.CONNECT_ATTEMPTS:
                    # The name may have come from a stale cache entry;
                    # force the next discovery to hit the radio.
                    try:
                        ble.Amplifier.clear_device_cache()
                    except Exception:
                        pass
                    time.sleep(self.CONNECT_RETRY_DELAY_S)
                continue

            try:
                self._resolve_channels(device)
                self._register_link_state(device)
                device.set_data_callback(self._data_callback)
            except Exception:
                try:
                    device.close()
                except Exception:
                    pass
                raise

            self._device = device
            _ble_acquire()
            return

        raise ConnectionError(
            f"Could not open BCI Core-8 '{serial}': {last_error}"
        )

    def _register_link_state(self, device) -> None:
        """Use the driver's link-state notification as the reconnect signal.

        A generation that numbers its samples reveals a reconnect by
        restarting its counter, which is what :meth:`_detect_reconnect`
        reads. The gCore generation reports no counter channel at all, so
        that signal does not exist -- and without a replacement the
        timeline would silently fail to advance across an outage, leaving
        every later sample misplaced with nothing reporting an error.

        The driver's own notification is the better signal in any case: it
        is authoritative rather than inferred, and it cannot be confused
        with a notification that merely overtook its predecessor. It is
        registered only where the counter is missing, so the path the
        counted generation has always used is left exactly as it was.

        The callback runs on a native BLE background thread, so it records
        state and nothing else -- no logging, no library calls, no
        exceptions. Acting on it happens in :meth:`_assemble`, under the
        callback lock, before the next samples are placed.

        Args:
            device: Opened amplifier instance.
        """
        if self._cnt_idx is not None:
            return

        states = getattr(ble.Amplifier, "ConnectionState", None)
        setter = getattr(device, "set_connection_state_callback", None)
        if states is None or setter is None:
            self.log(
                "This amplifier reports no sample counter, and the "
                "installed gtec-ble has no link-state notification "
                "either, so a reconnect cannot be detected: the timeline "
                "will not advance across an outage. Upgrade gtec-ble.",
                type=Constants.LogTypes.WARNING,
            )
            return

        def on_state(state) -> None:
            if state == states.DISCONNECTED:
                self._link_up = False
                # Only the FIRST drop of an outage is stamped. A link that
                # bounces reports several drops before it settles, and
                # re-stamping each one measures the last bounce instead of
                # the outage -- silently shortening the gap the timeline is
                # advanced by, which misplaces every sample after it.
                if self._link_down_at is None:
                    self._link_down_at = channels.stamp_clock()
            elif state == states.CONNECTED:
                self._link_up = True
                self._link_reconnected = True

        try:
            setter(on_state)
        except Exception as exc:
            self.log(
                f"Could not register the link-state callback, so a "
                f"reconnect will not advance the timeline: {exc}",
                type=Constants.LogTypes.WARNING,
            )

    def _reconnect_gap_s(self, arrival: float) -> float:
        """Length of the outage that just ended, in seconds.

        Prefers the moment the driver reported the link down, which brackets
        the outage at both ends. Falls back to the last arrival, which
        overstates it by up to one notification -- still far better than
        treating the outage as instantaneous.

        Args:
            arrival: Stamp of the notification that ended the outage.

        Returns:
            Seconds the link was down, never negative.
        """
        down_at = self._link_down_at
        self._link_down_at = None
        reference = down_at if down_at is not None else self._last_arrival
        if reference is None:
            return 0.0
        return max(0.0, arrival - reference)

    def _apply_reconnect(self, gap_s: float) -> None:
        """Advance the master timeline across an outage.

        Nothing was emitted while the link was down, so the outage occupies
        no timeline positions yet and the estimate can be applied directly.
        Monotonicity is guaranteed because the gap is never negative.

        Args:
            gap_s: Length of the outage in seconds, as estimated from host
                time. The device cannot report it.
        """
        gap = max(0, int(round(gap_s * self._sampling_rate)))
        # Before the epoch moves: a dropout still open when the link went
        # down belongs to the epoch it started in, and closing it afterwards
        # would describe both halves of one outage as a single short
        # contiguous run.
        if self._burst_samples:
            self._close_burst()
        self._epoch += 1
        self._master_index += gap
        # Counted as lost, as the forward-jump branch counts what never
        # arrived. Otherwise an outage ending in a reconnect left the
        # running total untouched, and every later note reported a handful
        # of samples lost on a recording missing minutes of signal.
        self._lost_samples += gap
        self._gap_samples += gap
        self.log(
            f"BLE link re-established (epoch {self._epoch}). About "
            f"{gap_s:.2f} s of data is missing; the timeline was "
            f"advanced by {gap} sample(s), estimated from host time, "
            f"so alignment across this gap is accurate to roughly "
            f"{1000.0 / self._sampling_rate:.0f} ms rather than to "
            f"the sample.",
            type=Constants.LogTypes.WARNING,
        )

    def _service_link(self, now: float) -> None:
        """Report a stream that has gone quiet, and say why.

        A stalled stream was completely silent: the link read connected,
        no samples arrived, nothing was logged and nothing raised, and the
        recording simply ended without saying so. Measured twice on a
        Senso2, each time for over a minute, before the driver-side cause
        was found.

        This runs on the worker rather than in the acquisition callback,
        because the whole point is to notice when that callback has
        stopped being called. It reads shared state without locking and
        mutates no timeline state -- the callback owns that, and taking
        its lock here to read two floats would put the worker behind the
        radio. It must never raise: the worker is a daemon with no handler
        around it.

        Args:
            now: Current stamp, from the clock arrivals are stamped with.
        """
        try:
            since = self._stream_since
            if since is None:
                return
            last = self._last_arrival
            reference = since if last is None else max(since, last)
            silent = now - reference

            if silent < self.STALL_TIMEOUT_S:
                if self._stalled:
                    self._stalled = False
                    self.log("Amplifier data resumed.")
                return

            if now - self._last_stall_log < self.LOSS_LOG_INTERVAL_S:
                return
            self._last_stall_log = now
            self._stalled = True

            if self._link_up is False:
                self.log(
                    f"No data for {silent:.1f} s: the driver reports the "
                    f"BLE link is down. It reconnects by itself, and the "
                    f"timeline is advanced across the gap once samples "
                    f"return.",
                    type=Constants.LogTypes.WARNING,
                )
            else:
                self.log(
                    f"No data for {silent:.1f} s while the BLE link "
                    f"reports connected. The amplifier has stopped "
                    f"delivering; check that it is powered and in range, "
                    f"and restart the pipeline if it does not resume.",
                    type=Constants.LogTypes.WARNING,
                )
        except Exception:
            # A watchdog that can kill the worker is worse than no
            # watchdog: the acquisition would stop with it.
            pass

    def master_candidacy(self):
        """Claim the master timeline, at the rate the device actually runs.

        The pipeline elects its master before it opens any handle, and the
        election reads whatever rate the source declares. For a device
        whose rate is fixed in hardware that is fine. A gCore amplifier is
        runtime-configurable, so the only honest way to declare its rate
        is to have asked it -- which is why this connects first when no
        rate was given.

        Connecting here moves the same work a few lines earlier inside the
        same ``start()`` call: ``_connect()`` is idempotent and ``start()``
        reuses the handle it opens. A failure is swallowed rather than
        raised, because ``start()`` reports it with far more context a
        moment later; declining candidacy here would only mean a synthetic
        source took the timeline before that happened.

        Returns:
            ``(rate, DEVICE)``, or None if no usable rate is known.
        """
        if not self._rate_declared and self._device is None:
            try:
                self._connect()
            except Exception:
                pass
        rate = self._sampling_rate
        if not rate or rate <= 0:
            return None
        return float(rate), MasterPriority.DEVICE

    def attach_timeline(self, timeline) -> None:
        """Bind this source to the pipeline's master timeline.

        Args:
            timeline: Timeline manager owned by the pipeline.
        """
        self._timeline = timeline

    @staticmethod
    def _channel_of(types: list, role_name: str) -> Optional[int]:
        """Return the index of the first channel of a device channel type.

        Args:
            types: Channel types as reported by the device.
            role_name: Name of the gtec-ble ChannelType member.

        Returns:
            Channel index, or None if the device does not report it.
        """
        wanted = getattr(ble.Amplifier.ChannelType, role_name, None)
        wanted = getattr(wanted, "value", wanted)
        if wanted is None:
            return None
        for i, t in enumerate(types):
            if getattr(t, "value", t) == wanted:
                return i
        return None

    def _reset_timeline(self) -> None:
        """Reset the master timeline and its accounting.

        Called when the node starts and whenever a fresh device
        connection begins.
        """
        #: Position on the master timeline of the next sample to arrive.
        #: Generated here rather than taken from the device counter: it
        #: must stay monotonic across reconnects, must not inherit the
        #: counter's 2**24 precision ceiling, and must exist on devices
        #: that report no counter channel at all.
        self._master_index: int = 0
        #: Link generation; incremented on every detected reconnect.
        self._epoch: int = 0
        #: Last counter value seen, for reset detection.
        self._cnt_last: Optional[float] = None
        #: Host time of the most recent arrival, for gap estimation.
        self._last_arrival: Optional[float] = None
        #: Samples reported as fabricated by the device validity channel.
        self._lost_samples: int = 0
        #: Samples skipped over because the link dropped.
        self._gap_samples: int = 0
        self._last_loss_log: float = 0.0
        self._last_loss_note: float = 0.0
        self._last_gap_log: float = 0.0
        #: Dropouts not individually reported since the last message,
        #: and the largest of them, so a suppressed one still surfaces.
        self._suppressed_bursts: int = 0
        self._suppressed_max: int = 0
        #: Largest dropout reported so far this run. A new worst always
        #: gets past the rate limit, so the worst event of an episode
        #: is never the one nobody hears about.
        self._worst_reported: int = 0
        #: Fabricated samples in the dropout currently being received.
        self._burst_samples: int = 0
        #: Set by the link-state callback once the driver reports the link
        #: back up. Read and cleared in _assemble, under the callback lock.
        self._link_reconnected: bool = False
        #: Stamp taken when the link was reported down, if it is down.
        self._link_down_at: Optional[float] = None
        #: The driver's last reported link state, or None before it has
        #: said anything. Distinguishes "no data because the radio is
        #: down" from "no data while the radio says it is fine", which
        #: are different faults with different remedies.
        self._link_up: Optional[bool] = None
        #: Stamp taken when acquisition began, so silence can be measured
        #: before the first sample has ever arrived.
        self._stream_since: Optional[float] = None
        #: True while a stall is being reported, so its end can be too.
        self._stalled: bool = False
        #: Rate limit for the stall warning.
        self._last_stall_log: float = 0.0
        #: Host times of recently ended dropouts, for the repeat test.
        self._burst_times: deque = deque()

    def _resolve_channels(self, device) -> None:
        """Determine which columns of a device sample carry EEG data.

        The amplifier streams EEG (EXG) interleaved with accelerometer,
        battery, counter and link-quality channels. Their order is
        reported by the device and must not be assumed.

        Args:
            device: Opened amplifier instance.

        Raises:
            ConnectionError: If the device exposes too few EEG channels.
        """
        try:
            types = list(device.channel_types)
        except Exception:
            types = []

        exg = ble.Amplifier.ChannelType.EXG
        exg = getattr(exg, "value", exg)
        idx = [i for i, t in enumerate(types) if getattr(t, "value", t) == exg]

        if idx:
            self._sample_width = len(types)
        else:
            # Device did not report a usable map: keep the historical
            # behaviour of taking the leading channels.
            idx = list(range(self._num_eeg_channels))
            self._sample_width = len(types) or None

        if len(idx) < self._num_eeg_channels:
            raise ConnectionError(
                f"Amplifier reports {len(idx)} EEG channels but "
                f"{self._num_eeg_channels} were requested."
            )

        self._eeg_idx = np.asarray(idx[: self._num_eeg_channels], dtype=int)

        # The device counter and the validity flag serve different jobs.
        #
        # The counter cannot measure loss: gtec-ble back-fills sub-second
        # gaps with repeated samples numbered contiguously, so the counter
        # is gap-free by construction. What it does reveal is a *reset* to
        # 1, which the library performs on every automatic reconnect.
        #
        # VALID is the only view of real loss: 0 marks a library-
        # fabricated sample, 1 a genuinely acquired one.
        self._cnt_idx = self._channel_of(types, "CNT")
        self._valid_idx = self._channel_of(types, "VALID")

        if self._valid_idx is None:
            self.log(
                "Amplifier reports no validity channel; samples lost on "
                "the BLE link cannot be detected.",
                type=Constants.LogTypes.WARNING,
            )

        self._reset_timeline()

        rate = int(getattr(device, "sampling_rate", 0) or 0)
        if rate and rate != self._sampling_rate:
            if not self._rate_declared:
                # Nobody asserted a rate, so the device is the authority.
                # This is the ordinary path for a gCore amplifier, whose
                # rate is a runtime property nothing else can know.
                self.log(
                    f"Amplifier reports {rate} Hz; following the device "
                    f"rather than the {self._sampling_rate} Hz default."
                )
                self._sampling_rate = rate
            else:
                # Refused, not warned. The recording that follows a
                # mismatch is correctly framed and wrongly timed: every
                # filter is detuned by the ratio, every marker lands in
                # the wrong place, and the file states a duration it does
                # not have. Warning and continuing produced exactly that,
                # and a warning in a log is not a defence against a
                # plausible-looking result.
                raise ConnectionError(
                    f"Amplifier reports {rate} Hz but this node was told "
                    f"{self._sampling_rate} Hz, so timing would be wrong "
                    f"by {rate / self._sampling_rate:.3f}x with nothing "
                    f"to show for it. Pass sampling_rate={rate}, or omit "
                    f"it and the device is followed."
                )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts for BCI Core-8 data streams.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with 250 Hz sampling rate.
        """
        port_context_out = super().setup(data, port_context_in)

        # A rate discovered at connect cannot be written back to the
        # configuration -- it is read-only by now, which is the same
        # reason the discovered serial travels as provenance rather than
        # as config. The context is what every downstream node actually
        # reads, so the discovered rate is published here.
        eeg_ctx = port_context_out[PORT_OUT]
        if eeg_ctx.get(Constants.Keys.SAMPLING_RATE) != self._sampling_rate:
            frame_size = eeg_ctx[Constants.Keys.FRAME_SIZE]
            eeg_ctx[Constants.Keys.SAMPLING_RATE] = self._sampling_rate
            eeg_ctx[Constants.Keys.FRAME_RATE] = (
                self._sampling_rate / frame_size
            )

        # Describe the EEG channels so downstream nodes need not infer
        # meaning from channel position.
        #
        # An explicit montage always wins. Failing that, an eight-channel
        # acquisition is the device's own headset and gets the montage it
        # ships with -- see DEFAULT_MONTAGE_8, which also records why this
        # is not applied to any other channel count.
        eeg = port_context_out[PORT_OUT]
        n_eeg = self._num_eeg_channels
        montage = self._montage
        if montage is None and n_eeg == len(self.DEFAULT_MONTAGE_8):
            # Built through Montage rather than used as a bare list, so
            # the electrode names are normalised and the 10-20 system is
            # derived exactly as it is for an authored montage. A default
            # that skipped that would be the one montage in the package
            # whose labels no lookup could resolve.
            montage = Montage(list(self.DEFAULT_MONTAGE_8))
        roles = [Constants.ChannelRoles.SIGNAL] * n_eeg
        labels = None if montage is None else list(montage.labels)[:n_eeg]
        system = None if montage is None else montage.system

        # The timeline channels are new, so labelling them renames
        # nothing. They must be labelled: without names, a consumer can
        # only find them by position, which is what roles exist to avoid.
        roles += [
            Constants.ChannelRoles.INDEX,
            Constants.ChannelRoles.QUALITY,
            Constants.ChannelRoles.TIMESTAMP,
        ]
        if labels is None:
            labels = channels.labels_of({Constants.Keys.CHANNEL_COUNT: n_eeg})
        labels = labels + list(self.TIMELINE_LABELS)

        eeg.update(channels.describe(roles, labels, system))

        # The master-timeline claim itself is made by AmplifierSource,
        # which every g.tec amplifier shares, so all of them outrank a
        # counted stream rather than only the ones that number their own
        # samples. It also publishes the outcome into the context above.
        #
        # What stays here is the reason the claim belongs to the source at
        # all: the arrival times the rate estimate is built from are only
        # knowable in the acquisition callback, so the source has to be
        # the one feeding observe().

        return port_context_out

    def stop(self):
        """Stop BCI Core-8 amplifier and clean up resources.

        Stops data acquisition, terminates background thread, and disconnects
        from amplifier hardware.
        """
        # Detach the device first so the data callback stops feeding the
        # buffers while the pipeline is being torn down.
        device, self._device = self._device, None

        # Stop amplifier data acquisition
        if device is not None:
            try:
                device.stop()
            except Exception:
                pass

        # A dropout still open when the stream ends has nothing valid
        # arriving after it to close the run, so it would never be
        # reported. After device.stop(), so no further notification can
        # reopen it, and under the callback lock, because one already in
        # flight owns this state until it returns.
        with self._callback_lock:
            if self._burst_samples:
                self._close_burst()

        # Stop the worker before the base class, not after: the worker
        # drives cycle() on this node, and a node that has already been
        # stopped must not be cycled again.
        if self._running:
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=10)  # Wait up to 10 seconds
            self._thread = None

        # Call parent stop method
        super().stop()

        # Release the BLE connection so the device is free for the next
        # acquisition. The BLE runtime itself is only shut down at process
        # exit - tearing it down here would break every later scan.
        if device is not None:
            try:
                device.close()
            except Exception:
                pass

        # Under the lock: an in-flight notification is reading these, and
        # clearing them from another thread would have it index with None.
        with self._callback_lock:
            self._eeg_idx = None
            self._sample_width = None

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Emit the frame the worker took off the queue.

        There is no schedule to satisfy, so there is nothing to fill in:
        when no frame is waiting this returns None rather than fabricating
        one.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Frame with the timeline appended, or None if nothing is ready.
        """
        pending = self._pending
        self._pending = None
        if pending is None:
            return None
        frame, meta = pending

        # Positions are tracked unbounded in-process and only wrapped on
        # the way out, where the float32 channel can no longer represent
        # large integers exactly. Consumers recover them with
        # channels.IndexUnwrapper.
        meta = meta.copy()
        meta[:, 0] = channels.wrap_indices(meta[:, 0])
        return {PORT_OUT: np.hstack((frame, meta))}

    def _detect_reconnect(self, counters: np.ndarray) -> None:
        """Advance the master timeline across an automatic reconnect.

        gtec-ble reconnects by itself when the link returns, but restarts
        the device counter at 1. Left unhandled, the timeline would jump
        backwards and every later sample would be misplaced.

        The gap cannot be measured: the restarted counter carries no
        information about how long the link was down, so its length is
        estimated from host arrival times. That estimate is good to
        roughly one connection interval at each edge; alignment across a
        reconnect is therefore approximate, while alignment within a
        connected stretch stays exact.

        Args:
            counters: Counter-channel values of the samples just received.
        """
        first = float(counters[0])

        # A forward jump inside one connection is exact information. The
        # driver back-fills gaps only up to one second; anything longer is
        # left as a jump, so the counter states precisely how many samples
        # never arrived. No host-time estimate is needed, and unlike
        # back-filled loss there is no validity flag to reveal it - which
        # is why it has to be read here or not at all.
        if self._cnt_last is not None and first > self._cnt_last + 1:
            missing = int(first - self._cnt_last) - 1
            self._master_index += missing
            self._gap_samples += missing
            self._lost_samples += missing
            now = time.monotonic()
            # Its own budget, not the dropout warning's: a skipped
            # timeline and a held signal are different diagnoses with
            # different remedies, and one shared timestamp let either
            # starve the other for as long as both kept recurring.
            if now - self._last_gap_log >= self.LOSS_LOG_INTERVAL_S:
                self._last_gap_log = now
                self.log(
                    f"{missing} sample(s) never arrived "
                    f"({missing / self._sampling_rate * 1000:.0f} ms of "
                    f"signal); the gap is longer than the driver "
                    f"back-fills, so the timeline skips it rather than "
                    f"inventing data. {self._lost_samples} lost since "
                    f"start.",
                    type=Constants.LogTypes.WARNING,
                )

        if self._cnt_last is not None and first < self._cnt_last:
            # stamp_clock(), not monotonic(): _last_arrival is written from
            # stamp_clock, and the two are different clocks with different
            # epochs on Windows. Subtracting one from the other does not
            # raise -- it yields a plausible number that is wrong, which is
            # the whole class of defect stamping exists to remove.
            now = channels.stamp_clock()
            gap_s = 0.0
            if self._last_arrival is not None:
                gap_s = max(0.0, now - self._last_arrival)

            # A backwards counter has two very different causes, and
            # treating them alike corrupts the stream. A genuine reconnect
            # restarts the device counter near 1 after an outage long
            # enough to notice. A notification that merely overtook its
            # predecessor steps back a few counts with no elapsed time at
            # all: the driver delivers from a thread pool, and the lock
            # around assembly excludes but does not order.
            #
            # Read as a reconnect, a reordered block advances the timeline
            # by a phantom gap, reports loss that never happened, and
            # leaves _cnt_last behind so the next in-order block looks
            # like a forward jump too. The samples still get monotonic
            # positions, so nothing downstream can see that the signal
            # was reordered - which is the worst possible outcome.
            restarted = first <= self.RECONNECT_COUNTER_MAX
            stalled = gap_s >= self.RECONNECT_MIN_GAP_S
            if not (restarted or stalled):
                self._reordered_blocks += 1
                if now - self._last_reorder_log >= self.LOSS_LOG_INTERVAL_S:
                    self._last_reorder_log = now
                    self.log(
                        f"Notification out of order: device counter went "
                        f"from {self._cnt_last:.0f} back to {first:.0f} "
                        f"after {gap_s * 1000:.1f} ms. The samples are "
                        f"kept, in arrival order; "
                        f"{self._reordered_blocks} such block(s) since "
                        f"start.",
                        type=Constants.LogTypes.WARNING,
                    )
                # Keep the highest counter seen, so the next in-order
                # block is not mistaken for a forward jump.
                self._cnt_last = max(self._cnt_last, float(counters[-1]))
                return
            self._apply_reconnect(gap_s)

        self._cnt_last = float(counters[-1])

    def _account_validity(self, valid: np.ndarray) -> None:
        """Count samples the library fabricated to fill a short gap.

        The device validity channel is the only view of real radio loss:
        gtec-ble back-fills gaps up to one second with repeated samples,
        numbered contiguously, marked invalid.

        The driver delivers one sample per callback, so a dropout
        arrives as a run of separately flagged samples rather than as
        one block. The run is accumulated and reported only once it
        ends. Reporting the first sample of it instead states a loss of
        one sample and the rate limit then swallows the rest of that
        same dropout -- which is how a lost payload of four came to be
        logged as "1 sample(s) lost".

        Args:
            valid: Validity-channel values of the samples just received.
        """
        invalid = np.asarray(valid) == 0
        if not (invalid.any() or self._burst_samples):
            return

        self._lost_samples += int(np.count_nonzero(invalid))

        # Walked sample by sample rather than counted, so a block that
        # holds the end of one dropout and the start of the next is
        # reported as two. Blocks are a handful of samples long.
        for flagged in invalid:
            if flagged:
                self._burst_samples += 1
            elif self._burst_samples:
                self._close_burst()

    def _close_burst(self) -> None:
        """Report the dropout that has just ended.

        A single back-filled payload is what an ordinary 2.4 GHz link
        produces and is not worth alarming anyone about, so it is a
        note. Two things earn a warning: one dropout long enough to
        matter on its own, and dropouts arriving repeatedly, however
        short each one is -- that is a link degrading rather than a
        link being a radio.

        Warnings and notes are rate-limited separately, but a warning
        inside its own budget is demoted to a note rather than dropped:
        rate-limiting the warnings alone made the biggest dropouts the
        ones most likely to be reported nowhere while trivial ones in
        the same window still printed. Whatever is suppressed is
        carried forward and named by the next message that gets out.
        """
        burst = self._burst_samples
        self._burst_samples = 0

        now = time.monotonic()
        self._burst_times.append(now)
        while (
            self._burst_times
            and now - self._burst_times[0] > self.LOSS_WARN_WINDOW_S
        ):
            self._burst_times.popleft()

        large = burst > self.LOSS_WARN_SAMPLES
        repeated = len(self._burst_times) >= self.LOSS_WARN_BURSTS
        warn = large or repeated

        # A dropout worse than any reported yet gets through whatever
        # the budget says. Without that, an episode whose events all
        # fall inside one interval reports only its first and mildest:
        # measured at 30, 250, 4 then 200 samples across 7 s, the two
        # worst produced no output at all.
        record = warn and burst > self._worst_reported
        if warn and (
            record or now - self._last_loss_log >= self.LOSS_LOG_INTERVAL_S
        ):
            level = Constants.LogTypes.WARNING
        elif now - self._last_loss_note >= self.LOSS_LOG_INTERVAL_S:
            level = Constants.LogTypes.INFO
        else:
            self._suppressed_bursts += 1
            self._suppressed_max = max(self._suppressed_max, burst)
            return
        self._worst_reported = max(self._worst_reported, burst)

        # A warning spends the note budget too. A 16 ms note landing
        # straight after a 120 ms warning reads as the link recovering.
        self._last_loss_note = now
        if level == Constants.LogTypes.WARNING:
            self._last_loss_log = now

        ms = burst / self._sampling_rate * 1000.0
        if large:
            what = (
                f"{burst} sample(s) ({ms:.0f} ms) lost on the BLE link "
                f"in one dropout"
            )
        elif repeated:
            what = (
                f"{len(self._burst_times)} dropouts on the BLE link in "
                f"the last {self.LOSS_WARN_WINDOW_S:.0f} s, the most "
                f"recent of {burst} sample(s) ({ms:.0f} ms)"
            )
        else:
            what = f"{burst} sample(s) ({ms:.0f} ms) lost on the BLE link"

        extra = ""
        if self._suppressed_bursts:
            held = self._suppressed_max / self._sampling_rate * 1000.0
            extra = (
                f" {self._suppressed_bursts} further dropout(s) went "
                f"unreported since the last message, the largest of "
                f"{self._suppressed_max} sample(s) ({held:.0f} ms)."
            )
        self._suppressed_bursts = 0
        self._suppressed_max = 0

        advice = ""
        if level == Constants.LogTypes.WARNING:
            advice = (
                " Move the amplifier closer to the adapter or reduce "
                "2.4 GHz interference."
            )
        self.log(
            f"{what} and back-filled by the driver; the signal holds "
            f"its previous value there. {self._lost_samples} sample(s) "
            f"lost since start.{extra}{advice}",
            type=level,
        )

    def _data_callback(self, data: np.ndarray):
        """Callback function for incoming amplifier data.

        Processes individual samples, assembles them into frames, and manages
        buffer queues for real-time processing.

        Args:
            data: Sample data array from amplifier, laid out as one or more
                consecutive samples of ``sample_width`` channels each. Only
                the EEG channels resolved at connect time are used.
        """
        # The arrival time has to be taken here. By the time a frame
        # reaches a downstream node it has queued and been scheduled, and
        # the rate estimate is built from the arrival *floor* - pipeline
        # latency would be indistinguishable from link latency.
        #
        # channels.stamp_clock() and not time.monotonic(): Sync unwraps
        # this against its own reading of the same function, and the two
        # must be one clock. See channels.stamp_clock.
        arrival = channels.stamp_clock()

        # Safety check for buffer initialization
        if self._sample_buffer is None or self._eeg_idx is None:
            return

        # gtec-ble delivers notifications from a thread pool - four
        # distinct threads were observed in a 30 s acquisition. The
        # assembly below numbers samples and then queues a copy of the
        # shared frame buffers, so without this lock a thread preempted
        # between those two steps can have its row overwritten by another
        # thread and the queued copies then carry positions out of order.
        # That surfaces as an index that moves backwards, which is
        # indistinguishable downstream from a corrupted timeline.
        with self._callback_lock:
            self._assemble(block=np.asarray(data), arrival=arrival)

    def _assemble(self, block: np.ndarray, arrival: float) -> None:
        """Number the samples of one notification and queue whole frames.

        Runs under the callback lock, so numbering and queueing cannot be
        interleaved with another notification.

        Args:
            block: Raw sample data of one notification.
            arrival: Host time at which the notification was received.
        """

        # A single callback may carry more than one sample; anything beyond
        # the first would be silently dropped if the array were treated as
        # one sample, starving the pipeline.
        width = self._sample_width
        if width and block.size >= width and block.size % width == 0:
            block = block.reshape(-1, width)
        else:
            block = block.reshape(1, -1)

        if block.shape[1] <= int(self._eeg_idx[-1]):
            return

        # A reconnect must be handled before the samples are placed, so
        # the timeline is already advanced when they are numbered.
        #
        # Two signals, never both: a counted generation restarts its
        # counter, and one without a counter is reported by the driver's
        # link-state notification instead. _register_link_state subscribes
        # only in the second case, so this flag cannot be set for a device
        # whose counter is already being read.
        if self._link_reconnected:
            self._link_reconnected = False
            self._apply_reconnect(self._reconnect_gap_s(arrival))
        if self._cnt_idx is not None and block.shape[1] > self._cnt_idx:
            self._detect_reconnect(block[:, self._cnt_idx])
        if self._valid_idx is not None and block.shape[1] > self._valid_idx:
            self._account_validity(block[:, self._valid_idx])

        for sample in block:
            # Determine position within current frame
            idx_in_frame = self._in_sample_counter % self._frame_size

            # Store sample data (only the resolved EEG channels)
            self._sample_buffer[idx_in_frame, :] = sample[self._eeg_idx]
            self._meta_buffer[idx_in_frame, 0] = self._master_index
            self._meta_buffer[idx_in_frame, 1] = (
                1.0
                if self._valid_idx is None
                else float(sample[self._valid_idx])
            )
            # Every sample of one notification shares its arrival
            # time, which is the same premise the single observe()
            # below rests on: the notification cannot exist until
            # its last sample has been acquired.
            self._meta_buffer[idx_in_frame, 2] = channels.wrap_time(arrival)
            self._in_sample_counter += 1
            self._master_index += 1

            self._last_arrival = arrival

            # Check if frame is complete
            if self._in_sample_counter % self._frame_size == 0:
                try:
                    # Queue completed frame for processing
                    self._frame_buffer.put_nowait(
                        (
                            self._sample_buffer.copy(),
                            self._meta_buffer.copy(),
                        )
                    )

                except queue.Full:
                    # The frame is discarded, which puts a step
                    # discontinuity into the signal. Never let that pass
                    # silently.
                    self._in_sample_counter -= self._frame_size
                    self._overflow_count += 1
                    now = time.monotonic()
                    if (
                        now - self._last_overflow_log
                        >= self.LOSS_LOG_INTERVAL_S
                    ):
                        self._last_overflow_log = now
                        self.log(
                            f"Buffer overflow - {self._overflow_count} "
                            f"frame(s) discarded since start; the signal "
                            f"is discontinuous. Reduce downstream "
                            f"processing load.",
                            type=Constants.LogTypes.WARNING,
                        )

        # One observation per notification, on the last sample in it.
        # Every sample of a notification shares a single arrival time, so
        # feeding each of them would present identical times at distinct
        # positions and the fit would degenerate. The last sample is the
        # principled representative: the notification cannot exist until
        # that sample has been acquired.
        if self._is_master_source and self._timeline is not None:
            self._timeline.observe(self._master_index - 1, arrival)

    def _worker_function(self):
        """Drive pipeline cycles as data arrives.

        There is no sampling clock to regenerate and nothing to keep in
        step with the device, so this simply forwards what has arrived.
        The pipeline therefore advances at the transport's own cadence -
        in bursts under a packetised link, which is harmless because
        samples remain uniformly spaced in device time.

        It exists so the pipeline never runs on the native BLE callback
        thread: blocking there would stall the radio.
        """
        poll_s = 1.0 / (self._sampling_rate * 4)
        while self._running:
            try:
                self._pending = self._frame_buffer.get(timeout=poll_s)
            except queue.Empty:
                # No data is not an error here: there is no schedule to
                # miss, so there is nothing to fill in. It is, however,
                # the only place a stall is observable at all -- the
                # acquisition callback cannot report that it stopped
                # being called.
                self._service_link(channels.stamp_clock())
                continue
            self.cycle()
            self._service_link(channels.stamp_clock())


class BCICore(ioc.OChain):
    """BCI Core amplifier chain for wireless EEG acquisition.

    This is an OChain that contains:
    - _BCICoreCore: the acquisition node
    - Link: bridge for distributed operation (absent in standalone)
    - Oscar: optional artifact removal
    - Sync: places the stream on the pipeline's master timeline and
      strips the in-band timeline channels again

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Interface to the g.tec BCI Core family of wireless EEG amplifiers
    over BLE, at 250 Hz.

    The node is named for the family, not for one member of it. It was
    ``BCICore8``, which read as the name of the 8-channel product --
    but a BCI Core-4 speaks the same protocol, reports itself the same
    way and is driven by exactly this code, so the 8 named a device
    rather than a node and made the node look inapplicable to half the
    devices it already supported. ``BCICore8`` still resolves, with a
    DeprecationWarning; see ``bci_core8.py``.
    """

    # Re-export constants from core
    SCANNING_TIMEOUT_S = _BCICoreCore.SCANNING_TIMEOUT_S
    SAMPLING_RATE = _BCICoreCore.SAMPLING_RATE
    MAX_NUM_CHANNELS = _BCICoreCore.MAX_NUM_CHANNELS
    #: Re-exported so an author, and a catalog, can read the electrode
    #: names an unlabelled eight-channel recording will carry without
    #: having to reach into the private core to find them.
    DEFAULT_MONTAGE_8 = _BCICoreCore.DEFAULT_MONTAGE_8

    def __init__(
        self,
        serial: Optional[str] = None,
        channel_count: Optional[int] = None,
        frame_size: Optional[int] = None,
        montage: Optional[Montage] = None,
        sampling_rate: Optional[int] = None,
        enable_oscar: bool = False,
        **kwargs,
    ):
        """Initialize BCI Core-8 amplifier chain.

        Args:
            serial: Serial number of target device. Uses first discovered
                if None.
            channel_count: Number of EEG channels (1-8). Defaults to 8,
                or to the size of the montage if one is given.
            frame_size: Samples per processing frame.
            montage: Electrode labels for the EEG channels.
            sampling_rate: Rate the device runs at, in Hz. Leave it
                unset and the device is asked; give a value only to
                assert one. See the acquisition node.
            enable_oscar: Enable OSCAR artifact removal processing.
            **kwargs: Additional arguments.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "serial": serial,
            "channel_count": channel_count,
            "frame_size": frame_size,
            "montage": montage,
            "sampling_rate": sampling_rate,
        }
        self._core_params.update(strip_chain_keys(kwargs))
        self._enable_oscar = enable_oscar

        # Initialize OChain (calls create_internal_nodes)
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration()],
        )
        # A Montage is an object, and a configuration has to be
        # JSON-representable or the pipeline cannot be serialized at all.
        # The labels are the representable form, and the core already
        # reads them back under this key and rebuilds the Montage -- that
        # is what MONTAGE_LABELS exists for. Forwarding the object put a
        # Montage in the chain's own configuration and broke serialize()
        # outright for any amplifier configured with one.
        labels_key = _BCICoreCore.Configuration.OptionalKeys.MONTAGE_LABELS
        if montage is not None:
            kwargs.setdefault(labels_key, list(montage.labels))

        ioc.OChain.__init__(
            self,
            serial=serial,
            channel_count=channel_count,
            frame_size=frame_size,
            sampling_rate=sampling_rate,
            enable_oscar=enable_oscar,
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
            where the pipeline is distributed, OSCAR where it is
            enabled, and Sync last.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        # Recorded, replaced, or simply built, as the launch
        # configuration says. Contributes nothing under server
        # residency: the core lives on the edge, and so does
        # anything recording or replaying it.
        nodes.extend(
            raw.source_stage(self, lambda: _BCICoreCore(**self._core_params))
        )
        if residency != Constants.Residency.STANDALONE:
            nodes.append(
                Link(
                    sender=Constants.Residency.EDGE,
                    receiver=Constants.Residency.SERVER,
                    stream_id=self._link_stream_id,
                )
            )
        if self._enable_oscar:
            nodes.append(Oscar())
        # Sync is last, after OSCAR: OSCAR does not need to be called
        # uniformly, and when enabled its delay widens the window for
        # placing late observations from other sources.
        nodes.append(Sync())
        return nodes
