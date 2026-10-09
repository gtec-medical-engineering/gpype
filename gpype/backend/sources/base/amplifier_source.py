from __future__ import annotations

import threading
from typing import Any, Optional

import ioiocore as ioc
import numpy as np

from ....common._private import channels
from ....common.constants import Constants
from ...core._private import assembly
from ...core._private.timeline import MasterPriority
from ...core.o_port import OPort
from . import raw
from .source import Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


def apply_channel_units(context: dict, units: Optional[list]) -> None:
    """Declare a context's per-channel units, once its layout is final.

    For a :meth:`AmplifierSource.channel_units` override to call from
    its own ``setup()``, after it has widened ``context`` with whatever
    extras and arrival-stamp channel it appends: only then does
    ``context`` describe the same channels ``units`` was built for.

    Args:
        context: Output port context, changed in place.
        units: One unit string or None per channel, or None to declare
            nothing.
    """
    if units is None:
        return
    count = channels.channel_count(context)
    if len(units) != count:
        # Declaring a unit list at the wrong width would be believed
        # for the wrong channels, which is worse than saying nothing.
        return
    context[Constants.Keys.CHANNEL_UNITS] = list(units)


def as_whole_number(value, name: str):
    """Return an integral value as an int, for a driver struct field.

    The GDS amplifiers store the sampling rate, the frame size and the
    channel count in integer struct fields -- ``uint32_t SamplingRate``,
    ``uint16_t``/``size_t NumberOfScans`` -- and cffi refuses a float
    there outright. The driver's own validation does not catch it,
    because it tests ``rate not in {256: 8, ...}`` and ``256.0`` hashes
    equal to ``256``: the float passes, then fails several statements
    later as ``TypeError: an integer is required`` from inside cffi,
    naming neither the parameter nor the field.

    A rate written ``256.0`` is the obvious way to write one, and any
    computed rate is a float in Python 3 -- ``4800 / 4`` is ``1200.0``.
    So an integral float is converted, and a genuinely fractional one is
    refused by name, because it cannot configure the device at all.

    Args:
        value: Value to convert, or None to leave unset.
        name: Parameter name, for the error message.

    Returns:
        The value as an int, or None.

    Raises:
        ValueError: If the value is not integral.
    """
    if value is None:
        return None
    as_int = int(value)
    if as_int != value:
        raise ValueError(
            f"{name} must be a whole number, got {value!r}. The device "
            f"stores it in an integer field, so a fractional value "
            f"cannot be configured."
        )
    return as_int


def opens_no_device(edge_id: Optional[str] = None) -> bool:
    """Whether an amplifier built in this process leaves its device shut.

    True on a server, where the device is on the edge; on an edge the
    amplifier is not assigned to, where the device is on another; and
    under ``load_from``, where the recording stands in for it. The chain
    runs no core in any of them, so a constructor that opened the
    device would hold an exclusive handle nothing ever releases -- and
    on the wrong edge, one that belongs to somebody else's run.

    Args:
        edge_id: The amplifier's edge id, as its constructor was given
            it. Asked before the configuration exists.

    Returns:
        True where the device must not be opened.
    """
    return not assembly.builds_core_for(edge_id) or raw.is_replaying()


def unopened_shape(node, sampling_rate, channel_count, frame_size):
    """The shape of an amplifier this process does not open.

    A value the author left to the device is stood in for and written
    back as given (``raw.stand_in``), on a server, on an edge the
    amplifier is not assigned to, and under ``load_from``. None of them
    runs the node, so nothing reads the stand-in: a
    replay takes the shape from the recording, and a server from the
    context the edge sends, which its receiving Link adopts before
    anything downstream sets up (D-CORE-70). Every amplifier example
    leaves the frame size to the device, and so does every 4.0.x
    document of one.

    Args:
        node: The amplifier being built.
        sampling_rate: As given, or None.
        channel_count: As given, or None.
        frame_size: As given, or None.

    Returns:
        ``(sampling_rate, channel_count, frame_size)`` to build with.
    """
    # None for the counts: Source's own defaults stand in for them.
    return tuple(
        raw.stand_in(
            node,
            sampling_rate=(sampling_rate, Constants.INHERITED),
            channel_count=(channel_count, None),
            frame_size=(frame_size, None),
        )
    )


def _stream_error_of(device) -> Any:
    """Why the driver's last stream ended, or None. Never raises.

    Read through getattr because gtec_ble has no such property, and
    because a g.Pype release must not require a specific driver version.

    Args:
        device: The driver handle, or None.

    Returns:
        The driver's ``stream_error``, or None.
    """
    if device is None:
        return None
    try:
        return getattr(device, "stream_error", None)
    except Exception:
        return None


class AmplifierSource(Source):
    """Base class for amplifier-based data acquisition sources.

    Provides hardware device management, sampling rate configuration,
    and multi-channel data acquisition setup for BCI applications.

    Not every amplifier numbers its samples in band, and the asymmetry is
    deliberate rather than an oversight -- it reads like one, so:

    BCI Core-8 appends an INDEX and a QUALITY channel to every frame, so
    its stream carries its own position. It can, because the BLE driver
    reports a per-sample counter and a validity flag, which is what makes
    loss visible: a gap in the numbering is the only evidence that
    samples never arrived.

    The GDS amplifiers have no equivalent. Their counter *replaces* a
    measured channel rather than adding one (channel 1 on g.HIamp,
    channel 16 on g.USBamp, and there only when all 16 are acquired), it
    is appended only on g.Nautilus, and it overruns at 1,000,000 rather
    than at ``channels.INDEX_MODULUS``. So an index for them could only
    be synthesised by counting arrivals here -- which is exactly
    ``MasterPriority.COUNTED``: a valid timeline, but not a loss-aware
    one, and dressing it as DEVICE would claim an accuracy it does not
    have while costing a real electrode.

    They therefore reach Sync as counted streams and are numbered there.
    Both families drive the timeline identically regardless: whoever
    holds mastership feeds the estimator, and Sync fills in when the
    holder is in another process. See Sync._report.
    """

    class Configuration(Source.Configuration):
        """Configuration class for AmplifierSource parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for amplifier source settings."""

            #: Sampling rate configuration key
            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE
            #: Whether OSCAR artifact removal runs on this stream
            ENABLE_OSCAR = "enable_oscar"

        def __init__(self, sampling_rate: float, **kwargs):
            """Initialize configuration with sampling rate validation.

            Args:
                sampling_rate: Sampling rate in Hz. Must be positive or
                    Constants.INHERITED for runtime determination.
                **kwargs: Additional configuration parameters.

            Raises:
                ValueError: If sampling_rate is not positive and not INHERITED.
            """
            # Validate sampling rate (allow INHERITED for runtime config)
            if sampling_rate != Constants.INHERITED and sampling_rate <= 0:
                raise ValueError("sampling_rate must be greater than zero.")
            super().__init__(sampling_rate=sampling_rate, **kwargs)

    # Class attributes for device management
    _device: Any

    #: Timeline of the pipeline this source belongs to, once bound.
    _timeline: Any = None
    #: Whether this source won the master-timeline election.
    _is_master_source: bool = False

    #: How often a running source asks its driver whether acquisition
    #: has ended on an error, in seconds. See `_start_stream_watch`.
    STREAM_WATCH_INTERVAL_S = 0.5
    #: The thread doing the asking, and the event that ends it.
    _stream_watch: Any = None
    _stream_watch_stop: Any = None
    #: The stream error already reported, so teardown does not repeat it.
    _reported_stream_error: Any = None

    def attach_timeline(self, timeline) -> None:
        """Bind this source to the pipeline's timeline.

        Args:
            timeline: Timeline manager owned by the pipeline.
        """
        self._timeline = timeline

    def __init__(
        self,
        sampling_rate: float,
        channel_count: int,
        frame_size: int,
        enable_oscar: bool = False,
        **kwargs,
    ):
        """Initialize amplifier source with acquisition parameters.

        Args:
            sampling_rate: Sampling rate in Hz. Must be positive or
                Constants.INHERITED for runtime determination.
            channel_count: Number of data channels to acquire.
            frame_size: Number of samples per data frame.
            enable_oscar: Run OSCAR artifact removal on this stream. A
                real typed parameter on the class that is actually
                constructed, so it is recognised, catalogued and
                serialised -- which is what makes
                ``GNautilus(enable_oscar=True)`` silently building no
                OSCAR structurally impossible rather than merely
                unlikely. See ``middle_nodes``.
            **kwargs: Additional arguments for parent Source class.
        """
        # Extract output_ports from kwargs with default configuration
        op_key = AmplifierSource.Configuration.Keys.OUTPUT_PORTS
        output_ports: list[OPort.Configuration] = kwargs.pop(
            op_key, [OPort.Configuration()]
        )

        # Initialize parent Source with amplifier configuration
        Source.__init__(
            self,
            output_ports=output_ports,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            enable_oscar=bool(enable_oscar),
            **kwargs,
        )
        #: The OSCAR instance ``middle_nodes`` built, or None where this
        #: amplifier runs without one. Set once, at chain assembly;
        #: read by :attr:`oscar_delay_samples`.
        self._oscar_node = None

    @property
    def oscar_delay_samples(self) -> Optional[int]:
        """OSCAR's total added delay, in samples at the input rate.

        None where this amplifier was not built with ``enable_oscar``,
        and None before the pipeline's ``start()`` has run OSCAR's own
        ``setup()`` -- the delay is a property of the fitted model, not
        of the configuration, so it cannot be known any earlier. Once
        set, it is exact: gtec_oscar's own
        ``delay_samples_input_fs``, not ``K``, the decimation factor
        that at 250 Hz reads 2 while the real delay is 83 samples (see
        ``Oscar.setup``, D-NODE-65).

        Returns:
            The delay, or None.
        """
        node = self._oscar_node
        return None if node is None else node.delay_samples_input_fs

    def middle_nodes(self) -> list:
        """Nodes this amplifier wants between its Link and its Sync.

        The OSCAR slot, and the one part of a source's shape the wrapper
        cannot derive from a port or a configuration key -- because it
        holds *nodes*, not a boolean. `wrapping.SourceChain` asks every
        core it carries; only an amplifier answers with anything.

        The position is load-bearing in both directions, and
        ``assembly.source_nodes`` records why: **before** `Sync`, because
        the model still needs the in-band ``master_index``, ``valid`` and
        ``host_time`` channels that Sync consumes; **after** `Link`,
        because a sending Link returns nothing, so the work lands on the
        server.

        Returns:
            ``[Oscar()]`` where this amplifier was asked for artifact
            removal, otherwise nothing.
        """
        key = AmplifierSource.Configuration.Keys.ENABLE_OSCAR
        if not self.config.get(key):
            return []
        # Imported here rather than at module scope: oscar.py reaches
        # the compiled hot path and pulls scipy on first real use, and
        # an amplifier that was never asked for OSCAR should pay for
        # neither.
        from ...core._private.oscar import Oscar

        self._oscar_node = Oscar()
        return [self._oscar_node]

    def open_for_attestation(self) -> None:
        """Open the device early, so the entitlement gate can challenge it.

        Default: reopen a handle a previous ``stop()`` released, where the
        source recorded how (``_device_factory``, the GDS sources), and
        otherwise do nothing. A source that cannot open its handle
        cheaply, or has not been taught to, is left exactly as it was --
        so adding this hook can only make a gate verdict *more*
        informed, never less.

        **The GDS sources need only the reopen.** They open their device
        in the constructor, so the gate finds a handle on a first start
        -- the bench recorded ``FULL`` for a g.USBamp, a g.HIamp and a
        g.Nautilus on 2026-09-24. But ``stop()`` releases it and
        ``start()`` reopened it only after the gate, so a pipeline
        started a second time was resolved with no handle and marked.

        **Why it exists.** `Pipeline.start()` resolves both entitlement
        gates before any node starts, and it is right to: a sink writes
        its header at start(), so a verdict reached later would stamp
        the wrong mark into the artifact, silently. But the device gate
        challenges `self._device`, and every amplifier opens its handle
        inside its own `start()` -- which runs *after* the gate. So the
        challenge list was empty by construction, every run.

        Measured 2026-09-06 on a BCI Core-8: a streaming, attestable
        amplifier on a licensed machine still resolved
        `attestation: limited`, reason "no g.tec amplifier in this
        pipeline" -- `evaluate()`'s empty-list branch -- while the same
        handle challenged by hand answered `verified=True, detail=OK`.
        Zero challenges were issued during start().

        `_begin_deferred_device_check`'s docstring already states the
        intended behaviour: "*The amplifier is a source.* Its handle is
        challenged inside start(), before any sink writes a header.
        Blocking is right: the pipeline has nothing to do until the
        device answers anyway." This hook is what makes that true.

        **Implement it only where the open is idempotent.** Every
        implementation must be safe to call twice, because the source's
        own `start()` will run afterwards and must reuse the handle
        rather than open a second one -- an amplifier is an exclusive
        resource and a second handle is refused.

        Never raises. A device that cannot be opened here is not a
        licensing decision: the run continues unattested and the source's
        own `start()` reports the failure properly, with its own error
        type and message. Swallowing it here would replace a clear
        connection error with a confusing entitlement one.
        """
        if getattr(self, "_device", None) is not None:
            return
        if getattr(self, "_device_factory", None) is None:
            return
        try:
            self._reopen_device()
        except Exception as error:  # noqa: BLE001
            self.log(
                f"could not reopen the amplifier before the entitlement "
                f"gate, so this run is unattested: {error}",
                type=Constants.LogTypes.WARNING,
            )

    def _reopen_device(self) -> None:
        """Reopen a device a previous ``stop()`` released.

        A pipeline can be started again after ``stop()`` -- that is the
        documented contract, and logging resources survive a stop for
        exactly that reason. The GDS sources open their device in the
        constructor, because the port configuration is read back from
        the device that actually opened -- except on a server, which
        opens nothing and records no factory. ``stop()`` then releases the
        exclusive handle through :meth:`_release_device`. A second
        ``start()`` found ``_device`` set to ``None`` and failed with
        "'NoneType' object has no attribute 'start'". Measured
        2026-09-24 on a g.USBamp by the integration testbench's D4.

        A source that can be reopened records how, in
        ``_device_factory``: a zero-argument callable opening the same
        unit with the same parameters. The data callback is registered
        again, because it belongs to the handle and not to the source.
        Idempotent -- a handle that is still open is left alone.

        All or nothing: the handle is kept only once its callback is
        registered. Kept half-registered, the next call would find it
        open and leave it alone, and the run would start on a device
        whose data reaches nothing.
        """
        if getattr(self, "_device", None) is not None:
            return
        factory = getattr(self, "_device_factory", None)
        if factory is None:
            return
        device = factory()
        try:
            device.set_data_callback(self._data_callback)
        except Exception:
            try:
                device.close()
            except Exception:
                pass
            raise
        self._device = device

    def _start_stream_watch(self) -> None:
        """Report a driver whose acquisition died, while the run is live.

        The GDS drivers acquire on a thread of their own. When it ends on
        an error -- the amplifier switched off, the cable pulled, a data
        callback that raised -- they store the reason in ``stream_error``
        and log it on their own logger, and the data callback simply stops
        being called. g.Pype read that reason only at teardown, in
        :meth:`_release_device`, so for the rest of the run the pipeline
        stayed healthy with no data arriving. Measured 2026-09-24 on a
        g.USBamp switched off mid-run (the testbench's M2): the driver
        logged "acquisition stopped: Timeout occurred. No data after
        10 s", and the condition read Healthy for the 30 s that followed.

        So a daemon thread asks. Not the data callback, which is exactly
        what stops, and not ``step()``, which only runs when the callback
        delivers. It logs at ERROR once per stream, which is what drives
        the condition -- sticky, like the BCI Core's stall error, because
        a device whose acquisition died does not resume by itself.

        Idempotent. Stopped by :meth:`_release_device`, so the watch never
        outlives the handle it reads.
        """
        thread = self._stream_watch
        if thread is not None and thread.is_alive():
            return
        self._reported_stream_error = None
        self._stream_watch_stop = threading.Event()
        self._stream_watch = threading.Thread(
            target=self._watch_stream,
            args=(self._stream_watch_stop,),
            name=f"{type(self).__name__}-stream-watch",
            daemon=True,
        )
        self._stream_watch.start()

    def _stop_stream_watch(self) -> None:
        """End the watch. Idempotent, and safe when it never started."""
        if self._stream_watch_stop is not None:
            self._stream_watch_stop.set()
        thread = self._stream_watch
        self._stream_watch = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=4 * self.STREAM_WATCH_INTERVAL_S)

    def _watch_stream(self, stop: threading.Event) -> None:
        """Poll the driver until the stream errors or the watch is ended.

        Args:
            stop: Set by :meth:`_stop_stream_watch`.
        """
        while not stop.wait(self.STREAM_WATCH_INTERVAL_S):
            if self._check_stream_error() or self._check_link():
                return

    def _check_link(self) -> bool:
        """Report a device-side link loss the driver does not raise.

        Default: nothing to check. A source whose device flags its own
        samples -- the g.Nautilus validation indicator -- overrides this.
        Runs on the watch thread, like :meth:`_check_stream_error`, so it
        must never raise.

        Returns:
            True once the link has been given up on, ending the watch.
        """
        return False

    def _check_stream_error(self) -> bool:
        """Log the driver's stream error at ERROR, if it has one.

        Never raises: it runs on a daemon thread with nothing around it.

        Returns:
            True once an error has been reported, since a stream that has
            died has nothing further to report.
        """
        reason = _stream_error_of(getattr(self, "_device", None))
        if not reason:
            return False
        self._reported_stream_error = reason
        try:
            self.log(
                f"Acquisition stopped: {reason}. The amplifier is no "
                f"longer delivering and will not resume by itself: check "
                f"that it is powered and connected, then restart the "
                f"pipeline.",
                type=Constants.LogTypes.ERROR,
            )
        except Exception:
            # Losing the message is bad; killing the thread over it
            # would lose it just the same.
            pass
        return True

    def _release_device(self) -> None:
        """Close the driver handle explicitly and forget it.

        The device is an exclusive resource: while a handle holds it, a
        second one is refused with "an already existing data acquisition
        session cannot be opened exclusively" -- so a handle that outlives
        its pipeline locks out the next run and the next process.

        Closing used to be left to ``del self._device``, which only works
        while nothing else holds a reference. A traceback, a callback
        closure or a debugger frame is enough to keep the object alive, and
        then ``__del__`` never runs and the device stays claimed until the
        interpreter exits. Observed in practice: a run that errored left a
        g.HIamp locked, and every later attempt failed on the exclusive
        session.

        Idempotent, and never raises: this runs on the teardown path, where
        a second failure would mask the first. But "never raises" used to
        mean "never reports", which is worse than it sounds: a device that
        failed mid-stream was released silently and the run looked clean.

        The GDS drivers now record why acquisition ended and keep it
        readable until the next start(), so an error that stop() swallows
        is logged here instead of being lost -- unless the stream watch
        already reported it while the run was live.
        """
        self._stop_stream_watch()
        device = getattr(self, "_device", None)
        self._device = None
        if device is None:
            return

        for method in ("stop", "close"):
            action = getattr(device, method, None)
            if callable(action):
                try:
                    action()
                except Exception as error:
                    # Still swallowed -- teardown must complete -- but no
                    # longer unreported.
                    self._report_teardown(f"{method}() failed: {error}")

        # Why the stream ended, if the driver knows. This survives stop()
        # deliberately on the driver side, so it is readable here.
        reason = _stream_error_of(device)
        if reason and reason is not self._reported_stream_error:
            self._report_teardown(f"acquisition ended with: {reason}")

    def _report_teardown(self, message: str) -> None:
        """Log a teardown observation without ever raising.

        Args:
            message: What happened, for the operator.
        """
        try:
            self.log(message, type=ioc.Constants.LogTypes.ERROR)
        except Exception:
            # No logger, or logging already torn down. Losing the message
            # is bad; raising from the release path is worse.
            pass

    def device_serial(self) -> str:
        """Return the serial of the device this source actually opened.

        Prefers the open handle over the stored configuration, because
        they answer different questions: the configuration says which
        device was *asked* for, and for ``serial=None`` -- the common case
        -- that is nothing at all, while the handle knows which one
        answered.

        Returns:
            The serial, or "" when no device is open and none was
            configured. Never raises: this runs on the setup path, and a
            source with no serial to report is a normal state rather than
            an error.
        """
        handle = getattr(self, "_device", None)
        reported = getattr(handle, "serial_number", None)
        if reported:
            return str(reported)
        key = getattr(self.Configuration.Keys, "SERIAL", None) or getattr(
            self.Configuration.OptionalKeys, "SERIAL", None
        )
        if key is not None:
            configured = self.config.get(key)
            if configured:
                return str(configured)
        # BLE sources resolve their target during start() and keep it
        # here; it is the discovered serial once discovery has run.
        target = getattr(self, "_target_sn", None)
        return str(target) if target else ""

    def channel_units(self) -> Optional[list]:
        """Physical unit of each channel this source emits, or None.

        Beside :meth:`device_serial`: a hook a concrete amplifier
        overrides, read once its own ``setup()`` has resolved the final
        channel layout -- the measured channels plus whatever extras are
        enabled plus the appended arrival-stamp channel -- so the list
        this returns has one entry per channel of *that* layout, not of
        the configured ``channel_count`` alone.

        The base implementation declares nothing, which is what every
        amplifier did before this existed. An override must read its
        numbers from the driver's own documented scaling (a vendor
        header, an SDK constant), never from memory: a wrong unit is
        worse than none, because it is believed. A channel with no fixed
        physical unit -- a bit-encoded flag, a validity indicator -- is
        None, same as one this method simply does not know.

        Returns:
            One unit string or None per channel, or None to declare
            nothing at all.
        """
        return None

    def master_candidacy(self):
        """Claim as a device.

        An amplifier is the better clock in any pipeline containing one,
        and it must win regardless of how fast it starts producing.

        Returns:
            ``(rate, DEVICE)``, or None if no usable rate is set.
        """
        rate = self.scalar(
            self.config.get(self.Configuration.Keys.SAMPLING_RATE)
        )
        if not rate or rate <= 0:
            return None
        return float(rate), MasterPriority.DEVICE

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with sampling rate information.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with sampling_rate information.
        """
        # Call parent setup to initialize base contexts
        port_context_out = super().setup(data, port_context_in)

        # Get actual sampling rate (may have been resolved from INHERITED)
        sampling_rate = self.config[self.Configuration.Keys.SAMPLING_RATE]
        frame_size = port_context_out[Constants.Defaults.PORT_OUT][
            Constants.Keys.FRAME_SIZE
        ]
        frame_rate = sampling_rate / frame_size
        out_ports = self.config[self.Configuration.Keys.OUTPUT_PORTS]

        # Which physical device this stream came from. Read here rather
        # than at construction because a BLE source discovers its device
        # during start(): its Configuration exists and is read-only by
        # then, so the discovered serial has nowhere to go -- but this
        # runs after the handle is open, so the answer is available.
        #
        # Provenance, deliberately separate from configuration. The
        # configuration says which device to open; this says which one
        # actually answered, and for `serial=None` those are not the same
        # question.
        serial = self.device_serial()

        # Add sampling rate context to each output port
        for i in range(len(out_ports)):
            # Create context with sampling rate information
            context = {
                Constants.Keys.SAMPLING_RATE: sampling_rate,
                Constants.Keys.FRAME_RATE: frame_rate,
            }
            if serial:
                context[Constants.Keys.DEVICE_SERIAL] = serial

            # Get port name and update its context
            port_name = out_ports[i][OPort.Configuration.Keys.NAME]
            port_context_out[port_name].update(context)

        # An amplifier is the better clock in any pipeline that contains
        # one, so it claims the master timeline here rather than letting
        # the downstream Sync claim on its behalf. Sync can only claim as
        # a counted stream -- it sees arrivals, not the device -- which
        # would leave a user's higher-rate source outranking real
        # hardware.
        #
        # The claim is made here and not in __init__ because the timeline
        # is bound by Pipeline.start(), which runs before setup.
        if self._timeline is not None and not self._is_master_source:
            self._is_master_source = self._timeline.claim_master(
                self, sampling_rate, MasterPriority.DEVICE
            )
        if self._is_master_source:
            # Travels downstream in the context, so Sync learns which
            # stream is master without holding a reference to this node.
            # The rung travels with it: a receiving Link turns the pair
            # into an instruction to claim again in *that* process, and
            # a claim needs a rung to land on. Without it a remote
            # amplifier would be re-elected as a counted stream and lose
            # to any local source with a higher rate.
            for i in range(len(out_ports)):
                port_name = out_ports[i][OPort.Configuration.Keys.NAME]
                port_context_out[port_name][
                    Constants.Keys.MASTER_TIMELINE
                ] = True
                port_context_out[port_name][
                    Constants.Keys.MASTER_PRIORITY
                ] = MasterPriority.DEVICE

        return port_context_out
