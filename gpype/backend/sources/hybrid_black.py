from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

from ...common._private import channels, device_probe
from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core.o_port import OPort
from .base.amplifier_source import AmplifierSource, apply_channel_units

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN

#: The gtec_unicorn module, or None until something needs it. Not
#: imported at module scope: a document naming HybridBlack is
#: deserialised in every residency, and importing gtec_unicorn loads its
#: native library, which a server has no use for and an unsupported
#: platform refuses outright.
unicorn = None


def _load() -> None:
    """Import gtec_unicorn, once, on first use.

    Only fills the name if it is still None, so a test that has replaced
    ``unicorn`` with a stand-in keeps it.

    Raises:
        RuntimeError: If gtec_unicorn is not installed, or cannot load
            its native library on this platform.
    """
    global unicorn
    if unicorn is not None:
        return
    try:
        import gtec_unicorn
    except (ImportError, OSError) as error:
        raise RuntimeError(
            f"HybridBlack needs gtec_unicorn, which could not be loaded "
            f"({type(error).__name__}: {error}). It is in the 'devices' "
            f'extra: pip install "gpype[devices]". Its wheels cover '
            f"Windows x64, Linux x86_64 and macOS arm64."
        ) from error
    unicorn = gtec_unicorn


class HybridBlack(AmplifierSource):
    """Unicorn Hybrid Black amplifier for wireless EEG acquisition.

    Interface to g.tec Unicorn Hybrid Black wireless EEG amplifier using
    Bluetooth. Supports 8-channel EEG acquisition at 250 Hz, plus optional
    accelerometer, gyroscope, battery, counter, and validation channels.

    On macOS the main thread must run an event loop while the node
    acquires, as ``MainApp.run()`` does: the driver receives every
    Bluetooth callback there, and the node reads on its own thread. A
    script whose main thread only sleeps gets read errors naming that
    cause.
    """

    #: The optional data streams, all off unless asked for.
    DEFAULT_INCLUDE_ACCEL: bool = False
    DEFAULT_INCLUDE_GYRO: bool = False
    DEFAULT_INCLUDE_AUX: bool = False
    DEFAULT_TEST_SIGNAL: bool = False

    #: Fixed sampling rate for Unicorn Hybrid Black amplifier in Hz
    SAMPLING_RATE = 250
    #: Number of EEG channels
    NUM_EEG_CHANNELS = 8
    #: Number of accelerometer channels (X, Y, Z)
    NUM_ACCEL_CHANNELS = 3
    #: Number of gyroscope channels (X, Y, Z)
    NUM_GYRO_CHANNELS = 3
    #: Number of auxiliary channels (battery, counter, validation)
    NUM_AUX_CHANNELS = 3
    #: Channels appended to carry the first-sight stamp. One instant per
    #: block, declared on the output port rather than in the
    #: configuration -- see setup().
    NUM_TIMELINE_CHANNELS = 1
    #: Total number of acquired channels (EEG + Accel + Gyro + Battery +
    #: Counter + Validation)
    TOTAL_ACQUIRED_CHANNELS = 17
    # Measured Bluetooth latency of this device, in milliseconds. Kept
    # as a recorded property of the hardware; nothing applies it. Under
    # timestamp-based synchronisation a fixed device delay is a constant
    # offset of the stream, which belongs in the timeline relation
    # rather than in a delay line here.
    DEVICE_DELAY_MS = 40
    # How long a whole frame may be waiting in the driver after every
    # read before that is reported as falling behind, in seconds. A
    # duration, not a count of reads: Bluetooth classic delivers in
    # bursts of up to 8 samples (measured 2026-09-24), so a read that
    # leaves a frame waiting is what an on-time reader of this device
    # looks like at a small frame size. A backlog that lasts a whole
    # second is not a burst.
    BEHIND_BACKLOG_S = 1.0
    #: Consecutive failed reads tolerated before acquisition is given
    #: up. One failure is a Bluetooth hiccup the next read recovers
    #: from, so failing on the first would end a run over nothing;
    #: three in a row is a device that has stopped answering, and
    #: carrying on then means a pipeline that emits nothing while
    #: reporting Healthy.
    NUM_ACQUISITION_ERRORS_ALLOWED = 3

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for Unicorn Hybrid Black specific parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration keys for Unicorn Hybrid Black settings."""

            #: Number of EEG channels, as requested. Recorded on its own
            #: because the port's channel_count is the total including the
            #: optional accelerometer, gyroscope and auxiliary channels:
            #: deriving the EEG count back out of that total would add
            #: them a second time on every round trip.
            EEG_CHANNEL_COUNT = "eeg_channel_count"
            INCLUDE_ACCEL = "include_accel"
            INCLUDE_GYRO = "include_gyro"
            INCLUDE_AUX = "include_aux"
            TEST_SIGNAL = "test_signal"

        class OptionalKeys(AmplifierSource.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Serial number of the target device. Recorded because it
            #: selects which amplifier is opened; absent means the first
            #: one discovered.
            SERIAL = "serial"

    def __init__(
        self,
        serial: Optional[str] = None,
        channel_count: Optional[int] = None,
        frame_size: Optional[int] = None,
        include_accel: Optional[bool] = None,
        include_gyro: Optional[bool] = None,
        include_aux: Optional[bool] = None,
        test_signal: Optional[bool] = None,
        enable_oscar: bool = False,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize Unicorn Hybrid Black amplifier source.

        Nothing reaches the device or its driver here: the device is
        opened when the pipeline starts, so the node constructs in every
        residency and under ``load_from``.

        Args:
            serial: Serial number of target device. Uses first discovered
                if None.
            channel_count: Number of EEG channels (1-8). Defaults to 8.
            frame_size: Samples per processing frame.
            include_accel: Include accelerometer channels (3 channels).
            include_gyro: Include gyroscope channels (3 channels).
            include_aux: Include auxiliary channels (battery, counter,
                validation).
            test_signal: Enable test signal mode instead of live data.
            enable_oscar: Run OSCAR artifact removal on this stream.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments for parent AmplifierSource.
        """
        # Validate and set channel count (1-8 EEG channels supported)
        keys = self.Configuration.Keys
        # A restored configuration binds to the named parameters, in the
        # per-port list form it was stored as, so normalise them here.
        channel_count = self.scalar(channel_count)
        frame_size = self.scalar(frame_size)
        # Read the EEG count back from its own key rather than from the
        # port's derived total, which already includes the extras.
        restored = self.scalar(kwargs.pop(keys.EEG_CHANNEL_COUNT, None))
        if restored is not None:
            channel_count = restored

        if channel_count is None:
            channel_count = self.NUM_EEG_CHANNELS
        channel_count = max(1, min(channel_count, self.NUM_EEG_CHANNELS))

        # Set default values for optional parameters
        if include_accel is None:
            include_accel = self.DEFAULT_INCLUDE_ACCEL
        if include_gyro is None:
            include_gyro = self.DEFAULT_INCLUDE_GYRO
        if include_aux is None:
            include_aux = self.DEFAULT_INCLUDE_AUX
        if test_signal is None:
            test_signal = self.DEFAULT_TEST_SIGNAL

        # Calculate total output channels based on configuration
        total_channels = channel_count
        if include_accel:
            total_channels += self.NUM_ACCEL_CHANNELS
        if include_gyro:
            total_channels += self.NUM_GYRO_CHANNELS
        if include_aux:
            total_channels += self.NUM_AUX_CHANNELS

        # Configure output ports
        kwargs.setdefault(keys.OUTPUT_PORTS, [OPort.Configuration()])
        # Recorded, None included, as the chain this replaced recorded it:
        # a document that lost it would open whichever amplifier answered
        # first.
        kwargs[self.Configuration.OptionalKeys.SERIAL] = serial or None

        # Initialize parent amplifier source with Hybrid Black specifications
        # Properties of the device, so a stored copy carries no
        # information; dropping it keeps them from arriving twice.
        kwargs.pop(keys.SAMPLING_RATE, None)
        kwargs.pop(keys.DECIMATION_FACTOR, None)

        # One cycle already carries one whole frame. The acquisition
        # thread asks the driver for frame_size samples and calls cycle()
        # once for the block, so there is nothing left to decimate: this
        # must stay 1.
        #
        # Generator looks like it does the opposite and does not.
        # FixedRateSource paces Generator per *sample*, so it needs
        # decimation_factor = frame_size to emit one frame per frame_size
        # cycles. Copying that here made the source emit one frame in
        # frame_size and drop the rest -- silently, because step() returns
        # None on a non-decimation cycle and _current_frame is overwritten
        # by the next packet. At frame_size 4 that is 75% of the recording.
        # Invisible until now only because frame_size defaulted to 1.
        super().__init__(
            enable_oscar=enable_oscar,
            channel_count=total_channels,
            eeg_channel_count=channel_count,
            sampling_rate=self.SAMPLING_RATE,
            frame_size=frame_size,
            decimation_factor=1,
            include_accel=include_accel,
            include_gyro=include_gyro,
            include_aux=include_aux,
            test_signal=test_signal,
            edge_id=edge_id,
            **kwargs,
        )

        self._frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]

        # Store device configuration
        self._target_sn = serial
        self._eeg_channel_count = channel_count
        self._total_channels = total_channels
        self._include_accel = include_accel
        self._include_gyro = include_gyro
        self._include_aux = include_aux
        self._test_signal = test_signal

        # The gtec_unicorn.Amplifier, opened before the entitlement gate
        # (open_for_attestation) or by start(), and closed by stop().
        self._device = None
        #: Why an open ahead of start() failed, kept until this start()
        #: has raised it. See _connect_early.
        self._connect_failure: Optional[Exception] = None
        #: The serial this process registered as held, for release.
        self._held_serial: Optional[str] = None

        # Initialize threading components
        self._running: bool = False
        self._acquisition_thread: Optional[threading.Thread] = None

        # Current frame for passing data from acquisition to step()
        self._current_frame: Optional[np.ndarray] = None

        # Underrun tracking
        self._underrun_counter: int = 0

        #: Consecutive failed reads, cleared by the next good one.
        self._acquisition_error_counter: int = 0

    def attach_timeline(self, timeline) -> None:
        """Bind to the pipeline's timeline, and forget a previous failure.

        The pipeline calls this as each ``start()`` begins, so an open
        that failed in the previous start is tried again in this one.

        Args:
            timeline: Timeline manager owned by the pipeline.
        """
        super().attach_timeline(timeline)
        self._connect_failure = None

    def open_for_attestation(self) -> None:
        """Open the device before the entitlement gate. See the base class.

        Safe to call twice: the handle opened here is the one ``start()``
        uses. Never raises. A failure -- the device off, no licence -- is
        logged once and raised again by ``start()``, without a second
        open.
        """
        if self._device is not None:
            return
        try:
            self._connect_early()
        except Exception as error:  # noqa: BLE001
            self.log(
                f"{node_label(self)}: could not open the Unicorn before "
                f"the entitlement gate: {error}",
                type=Constants.LogTypes.WARNING,
            )

    def _connect_early(self) -> None:
        """Open ahead of ``start()``, at most once per start.

        Raises:
            Exception: This start's open failure, new or remembered.
        """
        if self._connect_failure is not None:
            raise self._connect_failure
        try:
            self._connect()
        except Exception as error:
            self._connect_failure = error
            raise

    def _connect(self) -> None:
        """Open and configure the Unicorn.

        A ``gtec_unicorn.LicenseError`` -- no active licence for the
        product it names -- is raised unchanged: it is not a connection
        problem, and asking again cannot change the answer (D-NODE-64).

        Raises:
            RuntimeError: If gtec_unicorn cannot be loaded.
            ConnectionError: If no Unicorn is available, or the device
                acquires a width other than the one configured.
            ValueError: If ``frame_size`` exceeds what one read may ask.
        """
        _load()
        serial = self._target_sn
        if serial is None:
            devices = list(unicorn.Amplifier.get_connected_devices() or [])
            if not devices:
                raise ConnectionError(
                    "No Unicorn device available. Please pair with a "
                    "Unicorn first."
                )
            serial = str(devices[0])
            self.log(f"Using first available device: {serial}")

        # A second open of a serial this process already holds returns
        # that same session, and whichever closes first ends it for both
        # (gtec-unicorn-c W-7). Refuse rather than share it; this node's
        # own hold is the session it already has.
        if device_probe._is_held(serial) and serial != self._held_serial:
            raise ConnectionError(
                f"Unicorn {serial} is already open in this process, by "
                "another node or pipeline."
            )

        device = unicorn.Amplifier(serial)
        try:
            # The device always sends all 17 channels; this chooses which
            # ones each scan carries, in configuration order -- EEG,
            # accelerometer, gyroscope, then battery, counter and
            # validation -- which is the layout setup() declares.
            device.configure(
                eeg_channels=self._eeg_channel_count,
                accelerometer=bool(self._include_accel),
                gyroscope=bool(self._include_gyro),
                battery=bool(self._include_aux),
                counter=bool(self._include_aux),
                validation=bool(self._include_aux),
            )
            width = int(device.no_of_acquired_channels)
            if width != self._total_channels:
                raise ConnectionError(
                    f"The Unicorn acquires {width} channels where "
                    f"{self._total_channels} were configured."
                )
            limit = int(device.max_scans_per_read)
            if self._frame_size > limit:
                raise ValueError(
                    f"frame_size {self._frame_size} is more than one read "
                    f"of this Unicorn may ask for ({limit} scans with "
                    f"{width} channels)."
                )
        except Exception:
            try:
                device.close()
            except Exception:
                pass
            raise

        self._device = device
        # Pinned to the unit that answered, so a restart reopens it.
        self._target_sn = str(device.serial_number)
        # A second open of this serial in this process would share this
        # session, and its close would end it (gtec-unicorn-c W-7). The
        # probe reads this and leaves the serial alone.
        self._held_serial = self._target_sn
        device_probe.hold(self._held_serial)
        self.log(f"Connected to Unicorn Hybrid Black: {self._target_sn}")

    def start(self) -> None:
        """Start Unicorn Hybrid Black amplifier and begin data acquisition.

        Opens the device unless the entitlement gate already did, starts
        acquisition, and starts the thread that reads blocks and drives
        the pipeline via cycle().

        Raises:
            RuntimeError: If gtec_unicorn cannot be loaded.
            ConnectionError: If no Unicorn is available.
            gtec_unicorn.LicenseError: If no licence for the product it
                names is active on this machine.
            gtec_unicorn.DeviceError: If the device cannot be opened or
                started.
        """
        # Initialize current frame holder
        self._current_frame = None
        self._underrun_counter = 0
        self._acquisition_error_counter = 0

        # An open the gate already tried and failed is not tried again:
        # its error is this start's error. Taken, so the next start()
        # looks afresh.
        if self._device is None:
            failure, self._connect_failure = self._connect_failure, None
            if failure is not None:
                raise failure
            self._connect()

        # Bring the base class up before the thread starts cycling, so
        # the thread never drives a node that is not fully started.
        super().start()

        # Started here rather than on the thread, so a device that
        # refuses to start fails start() instead of a thread nobody
        # watches.
        self._device.start(test_signal=bool(self._test_signal))

        if not self._running:
            self._running = True
            self._acquisition_thread = threading.Thread(
                target=self._acquisition_function, daemon=True
            )
            self._acquisition_thread.start()

    def channel_units(self) -> Optional[list]:
        """One 'uV' per EEG channel, each enabled block's own unit, and
        None for the arrival stamp.

        Accelerometer is 'g', gyroscope 'deg/s', and the auxiliary
        triple is battery '%', then counter and the validation flag --
        gtec_unicorn's own device configuration reports both of those
        as '-' (gtec-unicorn-py's ``DEFAULT_CHANNELS``, matching
        ``Amplifier.get_configuration()``'s per-channel ``unit``), which
        this passes on rather than inventing g.Pype's own 'count' for a
        channel the driver itself declines to give a physical unit.
        """
        units = ["uV"] * self._eeg_channel_count
        if self._include_accel:
            units += ["g"] * self.NUM_ACCEL_CHANNELS
        if self._include_gyro:
            units += ["deg/s"] * self.NUM_GYRO_CHANNELS
        if self._include_aux:
            units += ["%", "-", "-"]
        units += [None] * self.NUM_TIMELINE_CHANNELS
        return units

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Describe the channel layout, and stamp after it.

        The arrival stamp is appended after every acquired block and
        counted apart from them: the blocks below are the channels the
        device delivers, which the stamp is no part of.

        The stamp is declared on the **output port** and nowhere else, so
        ``config[CHANNEL_COUNT]`` still equals the width the device was
        asked for. Core-8 raises the configuration instead, and survives
        it only because it keeps ``eeg_channel_count`` beside it; here the
        total is also the width the device is configured to and checked
        against at open, so widening the configuration would ask the
        device for a channel it does not have.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with 250 Hz sampling rate.
        """
        port_context_out = super().setup(data, port_context_in)

        # Describe the channels so a filter does not treat an
        # accelerometer as EEG. The layout configure() selects: EEG
        # first, then accelerometer, then gyroscope, then the auxiliary
        # triple, each block present only if it was asked for.
        #
        # The auxiliary triple is battery, counter and validation. The
        # counter is the device's own sample number and the validation
        # flag is a validity indicator, so INDEX and QUALITY would name
        # them more precisely -- but Sync consumes both of those roles,
        # which would silently remove channels the user turned on with
        # include_aux. They are carried as auxiliary data instead, which
        # is what the user asked for.
        roles = [Constants.ChannelRoles.SIGNAL] * self._eeg_channel_count
        for present, count in (
            (self._include_accel, self.NUM_ACCEL_CHANNELS),
            (self._include_gyro, self.NUM_GYRO_CHANNELS),
            (self._include_aux, self.NUM_AUX_CHANNELS),
        ):
            if present:
                roles += [Constants.ChannelRoles.AUXILIARY] * count

        # Roles only, no labels. Naming the stamp would mean naming the
        # measured channels too, and this device reports no montage, so
        # the stamp is found by role -- never by position or by name.
        stamped = self.NUM_TIMELINE_CHANNELS
        roles += [Constants.ChannelRoles.TIMESTAMP] * stamped

        units = self.channel_units()
        for context in port_context_out.values():
            # Read before the count is raised: the blocks above count the
            # device's channels, and the stamp is not one of them.
            total = channels.channel_count(context)
            context[Constants.Keys.CHANNEL_COUNT] = total + stamped
            context.update(channels.describe(roles))
            apply_channel_units(context, units)
        return port_context_out

    def _release_device(self) -> None:
        """Close the device, and let the probe and other nodes open it.

        Every path that releases the handle drops the hold with it, not
        stop() alone: a hold outliving its session would make another
        node refuse a Unicorn nobody has open.
        """
        super()._release_device()
        if self._held_serial is not None:
            device_probe.release(self._held_serial)
            self._held_serial = None

    def stop(self):
        """Stop Unicorn Hybrid Black amplifier and clean up resources.

        Stops the acquisition thread, then stops and closes the device.
        """
        # Stop background thread and wait for completion. A read in
        # flight returns within one frame, or the driver's 2 s timeout.
        if self._running:
            self._running = False
            acq_thread = self._acquisition_thread
            if acq_thread and acq_thread.is_alive():
                acq_thread.join(timeout=10)

        # A run is over; the next one opens the device again.
        self._connect_failure = None
        self._release_device()

        # Call parent stop method
        super().stop()

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Retrieve processed data frames from the amplifier.

        Returns data frames when decimation step is active.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing EEG data, or None if not a decimation step.
        """
        if self.is_decimation_step():
            # Return current frame (set by acquisition thread before cycle())
            if self._current_frame is not None:
                out_data = {PORT_OUT: self._current_frame}
                self._current_frame = None
                return out_data
            else:
                # Provide zero-filled frame if no data available. The
                # configuration carries the device's width, and the stamp
                # is declared only on the port, so this has to add the
                # stamp column itself or the frame is one channel
                # narrower than the port it goes out of.
                frame_size = self.config[self.Configuration.Keys.FRAME_SIZE][0]
                cc_key = self.Configuration.Keys.CHANNEL_COUNT
                channel_count = self.config[cc_key][0]
                zero_frame = np.zeros(
                    (frame_size, channel_count + self.NUM_TIMELINE_CHANNELS),
                    dtype=Constants.DATA_TYPE,
                )
                # The samples are fabricated; the instant is not. A zero
                # in the stamp column is not "no reading", it is a
                # reading of the epoch -- Sync would unwrap it into a
                # plausible instant tens of seconds out and fit the
                # timeline to it. Stamp when the frame is emitted.
                zero_frame[:, channel_count:] = channels.wrap_time(
                    channels.stamp_clock()
                )
                return {PORT_OUT: zero_frame}
        else:
            # Not a decimation step, return None
            return None

    def behind_reads(self) -> int:
        """Consecutive backlogged reads that mean a backlog, not a burst."""
        frames_per_s = self.SAMPLING_RATE / float(self._frame_size)
        return max(1, int(round(self.BEHIND_BACKLOG_S * frames_per_s)))

    def _acquisition_function(self):
        """Background thread: read a block, stamp it, cycle the pipeline.

        ``get_data`` blocks until the frame has arrived and releases the
        GIL while it waits, so the loop reads back to back with no
        pacing sleep and stamps each block the moment the read returns.
        The sleep UnicornPy needed -- its GetData held the GIL, and a
        reader that blocked in it starved every other thread -- is gone
        with it (D-NODE-63).
        """
        # Resolved once, outside the loop: every message below names
        # the node the author wrote, by the name they gave it.
        label = node_label(self)
        device = self._device
        if device is None:
            return

        frame_length = self._frame_size
        width = self._total_channels
        # Filled in place by every read; the published frame is a copy.
        receive_buffer = np.empty((frame_length, width), dtype=np.float32)
        frame_period_s = frame_length / float(self.SAMPLING_RATE)
        behind_reads = self.behind_reads()

        while self._running:
            try:
                block = device.get_data(frame_length, out=receive_buffer)
                # First sight: get_data returns only once the block is
                # complete, so this is the earliest instant it exists on
                # the host -- taken before the block is copied or
                # scheduled. channels.stamp_clock(), because Sync
                # compares the stamp against its own reading of it.
                arrival = channels.wrap_time(channels.stamp_clock())

                # A whole frame already waiting after the read means the
                # reader is behind the device. A burst does that too, so
                # only a backlog lasting BEHIND_BACKLOG_S is reported.
                waiting = device.get_stream_status().available_scans
                if waiting >= frame_length:
                    self._underrun_counter += 1
                else:
                    self._underrun_counter = 0

                if self._underrun_counter >= behind_reads:
                    self.log(
                        f"Falling behind the amplifier: a whole frame has "
                        f"been waiting after every read for "
                        f"{self.BEHIND_BACKLOG_S:.0f} s. Reduce the work "
                        f"per frame, or raise frame_size to cycle less "
                        f"often.",
                        type=Constants.LogTypes.WARNING,
                    )
                    self._underrun_counter = 0

                frame = np.empty(
                    (frame_length, width + self.NUM_TIMELINE_CHANNELS),
                    dtype=Constants.DATA_TYPE,
                )
                frame[:, :width] = block
                # One value for the whole block: the block cannot exist
                # until its last sample has been acquired, and that last
                # sample is the one Sync reads the instant back from.
                frame[:, width:] = arrival

                # Set current frame and trigger pipeline cycle immediately
                self._current_frame = frame
                self.cycle()

                # Consecutive, so a single hiccup the next read recovers
                # from never counts towards the give-up threshold.
                self._acquisition_error_counter = 0

            except Exception as error:
                if not self._running:
                    # Ordinary shutdown: stop() closed the device out
                    # from under a read that was already in flight.
                    break

                self._acquisition_error_counter += 1
                remaining = (
                    self.NUM_ACQUISITION_ERRORS_ALLOWED
                    - self._acquisition_error_counter
                )
                if remaining > 0:
                    self.log(
                        f"{label}: a read from the amplifier failed "
                        f"({error}); {remaining} more consecutive "
                        f"failure(s) will end the acquisition.",
                        type=Constants.LogTypes.WARNING,
                    )
                    # A read that fails at once would otherwise be
                    # retried at once, and the retry learns nothing.
                    time.sleep(frame_period_s)
                    continue

                # ERROR, and the loop ends. This used to be a WARNING
                # for every failure for ever: the loop kept going, the
                # condition stayed Healthy, and a device that had
                # stopped answering produced a repeating warning beside
                # a scope that never updated. AmplifierSource already
                # reports this class of failure at ERROR
                # (_report_teardown), so the two disagreed.
                #
                # _running goes false before the break because it is the
                # flag start() tests: left true, a later start() would
                # spawn no new acquisition thread and the node would be
                # permanently dead across the restart.
                self._running = False
                self._imp._condition = Constants.Conditions.ERROR
                self.log(
                    f"{label}: the amplifier stopped delivering data -- "
                    f"{self.NUM_ACQUISITION_ERRORS_ALLOWED} consecutive "
                    f"reads failed, the last with: {error}. Acquisition "
                    f"has ended and this node emits nothing further. "
                    f"Check that the device is powered and in range, "
                    f"then start the pipeline again.",
                    type=Constants.LogTypes.ERROR,
                )
                break

    @staticmethod
    def get_available_devices() -> list[str]:
        """Get list of available Unicorn Hybrid Black devices.

        Returns:
            Serial numbers the host can open: the paired devices on
            Windows, the discoverable ones elsewhere. Empty where
            gtec_unicorn cannot be loaded, or discovery fails.
        """
        try:
            _load()
            devices = unicorn.Amplifier.get_connected_devices()
            return [str(serial) for serial in devices or []]
        except Exception:
            return []
