from __future__ import annotations

import time
from typing import List

import ioiocore as ioc
import numpy as np

from ...common._private import channels, driver_usage
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.oscar import Oscar
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.amplifier_source import AmplifierSource, as_whole_number

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN

#: Channel groups on a g.USBamp: four groups of four channels.
GROUP_COUNT = 4


def _as_group_flags(value, name: str):
    """Return exactly one boolean per channel group.

    Args:
        value: Sequence of flags, one per group, or None to leave the
            choice to the driver -- which writes all-False, i.e. no
            common ground and no common reference. It does *not* preserve
            whatever the device had.
        name: Parameter name, for the error message.

    Returns:
        A list of GROUP_COUNT bools, or None.

    Raises:
        ValueError: If the sequence does not have one entry per group.
    """
    if value is None:
        return None
    flags = [bool(flag) for flag in value]
    if len(flags) != GROUP_COUNT:
        raise ValueError(
            f"{name} needs one entry per channel group, i.e. "
            f"{GROUP_COUNT}, got {len(flags)}. A shorter list is accepted "
            f"by the driver but leaves the remaining groups on their "
            f"previous setting, which the recording does not show."
        )
    return flags


class _GUSBampCore(AmplifierSource):
    """Internal node implementing g.USBamp amplifier acquisition logic.

    This is the actual g.USBamp node (pure ONode inheritance). It is
    wrapped by the GUSBamp chain for distributed operation.

    Interface to g.tec's g.USBamp wired biosignal amplifier.
    """

    #: One channel appended after the device's own, carrying the
    #: instant each block was first seen on this host.
    #:
    #: The stamp rides in band because it has to survive a Link. The
    #: Sync at the end of this chain builds the master-timeline
    #: relation from (position, instant) pairs, and under EDGE/SERVER
    #: residency that Sync runs in the server process, where reading a
    #: clock measures when the frame finished crossing the network
    #: rather than when the amplifier delivered it. Sync consumes the
    #: channel again, so nothing downstream sees a wider frame.
    #:
    #: Declared on the port and not in the configuration, unlike
    #: Core-8's: a GDS configuration's channel_count is the number
    #: handed to the driver, so widening it would ask the amplifier
    #: for one more acquisition channel.
    NUM_TIMELINE_CHANNELS = 1

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for g.USBamp amplifier parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration key constants for the g.USBamp amplifier.

            Every key here is mandatory, so only settings the device
            reports back belong in it -- those are always resolved to a
            concrete value before the configuration is built.
            """

            #: Configuration key for the device serial number
            SERIAL = "serial"
            #: Configuration key for enabling the digital trigger channel
            ENABLE_TRIGGER = "enable_trigger"
            #: Configuration key for enabling counter channel
            ENABLE_COUNTER = "enable_counter"
            #: Configuration key for per-group common ground
            COMMON_GROUND = "common_ground"
            #: Configuration key for per-group common reference
            COMMON_REFERENCE = "common_reference"
            #: Configuration key for the g.USBamp shortcut
            SHORTCUT_ENABLED = "shortcut_enabled"

        class OptionalKeys(AmplifierSource.Configuration.OptionalKeys):
            """Configuration keys that may legitimately be absent."""

            #: Configuration key for per-channel bipolar derivation.
            #:
            #: Optional rather than required because the driver exposes no
            #: read-back for it and None does not mean "unipolar": it means
            #: the device's own per-channel setting is left untouched.
            #: Substituting a list of zeros to satisfy a mandatory key
            #: would quietly change that to "set every channel unipolar".
            BIPOLAR_CHANNELS = "bipolar_channels"

    def __init__(
        self,
        serial: str = None,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        enable_trigger: bool = None,
        enable_counter: bool = None,
        common_ground: list = None,
        common_reference: list = None,
        shortcut_enabled: bool = None,
        bipolar_channels: list = None,
        **kwargs,
    ):
        """Initialize g.USBamp amplifier interface.

        Args:
            serial: Device serial number. Uses first available if None.
            sampling_rate: Sampling frequency in Hz.
            channel_count: Number of channels to acquire.
            frame_size: Samples per data frame (NumberOfScans).
            enable_trigger: Enable the digital trigger channel, which is
                appended after the acquired channels.
            enable_counter: Enable the counter, which overwrites a channel
                rather than adding one. Measured to occupy physical
                channel 16, so it is only visible when all 16 channels
                are acquired.
            common_ground: Four booleans, one per channel group.
            common_reference: Four booleans, one per channel group.
            shortcut_enabled: Enable the g.USBamp shortcut.
            bipolar_channels: Reference channel number per channel, 0 to
                derive that channel unipolar.
            **kwargs: Additional parameters for AmplifierSource.

        Raises:
            NotImplementedError: If not running on Windows.
            RuntimeError: If GDS library unavailable or device init fails.
            ValueError: If a rate, frame size or channel count is not a
                whole number, or if a group list does not have one entry
                per channel group.
        """
        # No platform guard, deliberately. The refusal that stood here
        # said the driver was Windows-only, and it had outlived the
        # limitation: gtec_gds 1.6.0 publishes cp310-cp314 wheels for
        # Linux (x86_64) and macOS (x86_64 and arm64) as well as Windows
        # -- verified by wheel tag on 2026-09-05 and recorded in
        # pyproject.toml, where the same check found a stale win32 marker
        # had been silently excluding GDS from every non-Windows install.
        #
        # So a user on Linux was told g.USBamp was unsupported on a
        # platform its driver ships for. The import below stays lazy: a
        # missing driver is still a missing driver, and it must not break
        # `import gpype` or the node catalogue.

        # Import gtec_gds only when actually needed (lazy import)
        try:
            import gtec_gds as gds

            # The driver refuses to construct a handle until the
            # local usage key is registered.
            driver_usage.register()
        except ImportError as e:
            raise RuntimeError(
                f"GDS library not available: {e}. "
                "This may be expected in CI environments where the GDS "
                "library is not installed."
            ) from e

        # A restored configuration binds these named parameters in the
        # per-port list form they were stored as, and they are handed
        # straight to the native driver below, which expects numbers.
        channel_count = self.scalar(channel_count)
        frame_size = self.scalar(frame_size)
        sampling_rate = self.scalar(sampling_rate)

        # The driver writes these into uint32_t and size_t struct fields,
        # which reject a float outright -- and it accepts the float on the
        # way in, because its own check is `rate not in {256: 8, ...}` and
        # 256.0 hashes equal to 256. So a float survives validation and
        # dies later as "TypeError: an integer is required" from inside
        # cffi, naming no parameter. A sampling rate written 256.0 is the
        # obvious way to write one, so convert it here instead.
        sampling_rate = as_whole_number(sampling_rate, "sampling_rate")
        frame_size = as_whole_number(frame_size, "frame_size")
        channel_count = as_whole_number(channel_count, "channel_count")

        # Fixed-length hardware arrays. cffi accepts a short list and
        # leaves the remaining elements at whatever the struct held, so
        # common_ground=[True, True] configures groups A and B and leaves
        # C and D on their previous setting while reporting success --
        # measured: assigning [True, False] over an all-true struct
        # yields [1, 0, 1, 1]. A montage referenced two groups short is
        # not visible in the data, so it is refused here.
        common_ground = _as_group_flags(common_ground, "common_ground")
        common_reference = _as_group_flags(
            common_reference, "common_reference"
        )

        #: g.USBamp device interface instance
        #
        # The digital trigger parameter is `enable_trigger` here, not the
        # `enable_di` that g.HIamp uses. The distinction matters more than
        # a name usually would: the driver accepts **kwargs, so a wrong
        # name is not rejected -- it is swallowed, the trigger silently
        # stays off, and only the read-back below keeps the declared
        # channel count from drifting one channel wide of the data.
        self._device = gds.GUSBamp(
            serial=serial,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            enable_trigger=enable_trigger,
            enable_counter=enable_counter,
            common_ground=common_ground,
            common_reference=common_reference,
            shortcut_enabled=shortcut_enabled,
            bipolar_channels=bipolar_channels,
        )

        # Take the configuration from the device rather than from the
        # arguments: the driver resolves None to a default, clamps the
        # frame size to one the sampling rate supports, and reports the
        # serial of the amplifier it actually opened.
        serial = self._device.serial_number
        sampling_rate = self._device.sampling_rate
        channel_count = self._device.channel_count
        frame_size = self._device.frame_size
        enable_trigger = self._device.enable_trigger
        enable_counter = self._device.enable_counter
        common_ground = self._device.common_ground
        common_reference = self._device.common_reference
        shortcut_enabled = self._device.shortcut_enabled

        # The measured channels are the ones the device acquires.
        # Recorded before the trigger is appended so setup() can tell the
        # two apart.
        self._eeg_channel_count = channel_count

        # Add the digital trigger channel if enabled
        if enable_trigger:
            channel_count += 1

        # Note: enable_counter overwrites a channel, it does not add one,
        # so the arithmetic above is unaffected either way.
        #
        # The GDS header states where: "a sample counter [...] applied on
        # channel 16 instead of the measured signal if selected for
        # acquisition (overrun at 1,000,000 samples)". So it reaches the
        # stream only when channel 16 is among the acquired channels, and
        # it wraps -- a single large negative step every million samples
        # is the counter, not a fault.
        #
        # Matches what a UB-2016.03.16 did: at channel_count=16 the last
        # column counted; at channel_count=4 no column did, because
        # channel 16 was not acquired. g.HIamp differs -- there the
        # counter takes channel 1.

        # Set up data callback for real-time streaming
        self._device.set_data_callback(self._data_callback)

        # Initialize parent AmplifierSource with final configuration
        #
        # The serial is stored, unlike in the other GDS sources, because a
        # configuration restored without it opens whichever amplifier
        # happens to be first. That is a different device from the one the
        # session was recorded with, and attestation binds a signature to
        # the serial the handle reports -- so the substitution would be
        # invisible in the data and wrong in the verdict.
        super().__init__(
            serial=serial,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            enable_trigger=enable_trigger,
            enable_counter=enable_counter,
            common_ground=common_ground,
            common_reference=common_reference,
            shortcut_enabled=shortcut_enabled,
            bipolar_channels=bipolar_channels,
            **kwargs,
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Mark the digital trigger as a trigger, and stamp beside it.

        The trigger is the only channel appended after the acquired ones,
        so its position is unambiguous. Saying it is a trigger is what
        keeps a filter off it: a bandpass turns an edge into a decaying
        oscillation, and nothing about the result announces that it was
        ever an event.

        The arrival stamp is appended after both, and counted apart from
        them: the trigger block is derived from the width the device
        reports, which the stamp is no part of.

        Enabling the counter overwrites a measured channel rather than
        appending one, so it stays inside the measured block and the
        trigger's position is unaffected either way. Its role is left as
        signal: the driver does not report that a channel was overwritten,
        and inferring the position from ``enable_counter`` would relabel a
        real electrode -- measured, the counter reaches the stream only
        when all 16 channels are acquired, and g.HIamp puts it somewhere
        else again.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Output port contexts describing the channel layout.
        """
        port_context_out = super().setup(data, port_context_in)
        stamped = self.NUM_TIMELINE_CHANNELS
        for context in port_context_out.values():
            # Read before the count is raised: everything below counts
            # the device's channels, and the stamp is not one of them.
            total = channels.channel_count(context)
            n_eeg = min(self._eeg_channel_count, total)
            roles = [Constants.ChannelRoles.SIGNAL] * n_eeg
            roles += [Constants.ChannelRoles.TRIGGER] * (total - n_eeg)
            roles += [Constants.ChannelRoles.TIMESTAMP] * stamped
            context[Constants.Keys.CHANNEL_COUNT] = total + stamped
            # Roles only. Naming the stamp would mean naming the
            # measured channels too, and an amplifier that reports no
            # montage has no names to give them; the stamp is found by
            # role, never by position or name.
            context.update(channels.describe(roles))
        return port_context_out

    def start(self) -> None:
        """Start g.USBamp data acquisition.

        Initiates hardware data streaming and activates the amplifier for
        real-time biosignal processing.
        """
        # Start hardware data acquisition
        self._device.start()
        # Start parent source processing
        super().start()

    def stop(self):
        """Stop g.USBamp data acquisition and cleanup resources.

        Stops hardware streaming and ensures proper shutdown of the
        amplifier connection.
        """
        # Stop the parent first, then release the exclusive handle.
        super().stop()
        self._release_device()

    def _data_callback(self, data: np.ndarray):
        """Stamp one block of g.USBamp data and forward it.

        Callback invoked by the GDS library when new data is available.
        Forwards data through the g.Pype pipeline using the cycle
        mechanism.

        Args:
            data: Raw data with shape (frame_size, channel_count).
        """
        # First sight: the earliest instant this block exists on the
        # host, taken before it is widened, queued or scheduled.
        # time.monotonic() and not perf_counter(), because Sync
        # compares the stamp against its own reading of that clock.
        # One value for the whole block -- the block cannot exist
        # until its last sample has been acquired, which is the sample
        # Sync reads it back from.
        arrival = channels.wrap_time(channels.stamp_clock())
        stamp = np.full(
            (data.shape[0], self.NUM_TIMELINE_CHANNELS),
            arrival,
            dtype=Constants.DATA_TYPE,
        )
        # Forward data through the pipeline using the input port
        self.cycle(data={PORT_IN: np.hstack((data, stamp))})

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one step of data through the g.USBamp source.

        Args:
            data: Input data dictionary with PORT_IN key containing data.

        Returns:
            Output data dictionary with PORT_OUT key containing data.
        """
        # Pass through data from input to output port
        return {PORT_OUT: data[PORT_IN]}


class GUSBamp(ioc.OChain):
    """g.USBamp biosignal amplifier chain for real-time data acquisition.

    This is an OChain that contains:
    - _GUSBampCore: The actual g.USBamp acquisition node
    - Link: Bridge for distributed operation (passthrough in standalone)
    - Oscar: OSCAR artifact removal processing

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Interface to g.tec's g.USBamp wired biosignal amplifier.
    """

    def __init__(
        self,
        serial: str = None,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        enable_trigger: bool = None,
        enable_counter: bool = None,
        common_ground: list = None,
        common_reference: list = None,
        shortcut_enabled: bool = None,
        bipolar_channels: list = None,
        enable_oscar: bool = False,
        **kwargs,
    ):
        """Initialize g.USBamp amplifier chain.

        Args:
            serial: Device serial number. Uses first available if None.
            sampling_rate: Sampling frequency in Hz.
            channel_count: Number of channels to acquire.
            frame_size: Samples per data frame (NumberOfScans).
            enable_trigger: Enable the digital trigger channel.
            enable_counter: Enable the counter channel.
            common_ground: Four booleans, one per channel group.
            common_reference: Four booleans, one per channel group.
            shortcut_enabled: Enable the g.USBamp shortcut.
            bipolar_channels: Reference channel number per channel.
            enable_oscar: Enable OSCAR artifact removal processing.
            **kwargs: Additional arguments.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "serial": serial,
            "sampling_rate": sampling_rate,
            "channel_count": channel_count,
            "frame_size": frame_size,
            "enable_trigger": enable_trigger,
            "enable_counter": enable_counter,
            "common_ground": common_ground,
            "common_reference": common_reference,
            "shortcut_enabled": shortcut_enabled,
            "bipolar_channels": bipolar_channels,
        }
        self._core_params.update(strip_chain_keys(kwargs))
        self._enable_oscar = enable_oscar

        # Initialize OChain (calls create_internal_nodes)
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration()],
        )
        ioc.OChain.__init__(
            self,
            serial=serial,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            enable_trigger=enable_trigger,
            enable_counter=enable_counter,
            common_ground=common_ground,
            common_reference=common_reference,
            shortcut_enabled=shortcut_enabled,
            bipolar_channels=bipolar_channels,
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
            raw.source_stage(self, lambda: _GUSBampCore(**self._core_params))
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
        # Sync places this stream on the master timeline. It carries no
        # device counter, so Sync numbers it by counting samples: a valid
        # timeline, but not a loss-aware one.
        nodes.append(Sync())
        return nodes
