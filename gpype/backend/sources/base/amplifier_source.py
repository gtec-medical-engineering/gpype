from __future__ import annotations

from typing import Any

import ioiocore as ioc
import numpy as np

from ....common.constants import Constants
from ...core._private.timeline import MasterPriority
from ...core.o_port import OPort
from .source import Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


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
        **kwargs,
    ):
        """Initialize amplifier source with acquisition parameters.

        Args:
            sampling_rate: Sampling rate in Hz. Must be positive or
                Constants.INHERITED for runtime determination.
            channel_count: Number of data channels to acquire.
            frame_size: Number of samples per data frame.
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
            **kwargs,
        )

    def open_for_attestation(self) -> None:
        """Open the device early, so the entitlement gate can challenge it.

        Default: do nothing. A source that cannot open its handle
        cheaply, or has not been taught to, is left exactly as it was --
        so adding this hook can only make a gate verdict *more*
        informed, never less.

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
        return None

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
        is logged here instead of being lost. Read through getattr because
        gtec_ble has no such property, and because a g.Pype release must
        not require a specific driver version.
        """
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
        try:
            reason = getattr(device, "stream_error", None)
        except Exception:
            reason = None
        if reason:
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
        key = getattr(self.Configuration.Keys, "SERIAL", None)
        if key is not None:
            configured = self.config.get(key)
            if configured:
                return str(configured)
        # BLE sources resolve their target during start() and keep it
        # here; it is the discovered serial once discovery has run.
        target = getattr(self, "_target_sn", None)
        return str(target) if target else ""

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
