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


class _GNautilusCore(AmplifierSource):
    """Internal node implementing g.Nautilus amplifier acquisition logic.

    This is the actual g.Nautilus node (pure ONode inheritance).
    It is wrapped by the GNautilus chain for distributed operation.

    Interface to g.tec's g.Nautilus wireless EEG amplifier system.
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
        """Configuration class for g.Nautilus amplifier parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration key constants for the g.Nautilus amplifier."""

            #: Configuration key for the device serial number
            SERIAL = "serial"
            #: Configuration key for amplifier sensitivity setting
            SENSITIVITY = "sensitivity"
            #: Configuration key for enabling the digital trigger input
            ENABLE_DI = "enable_di"
            #: Configuration key for enabling counter channel
            ENABLE_COUNTER = "enable_counter"
            #: Configuration key for noise reduction
            NOISE_REDUCTION = "noise_reduction"
            #: Configuration key for Common Average Reference
            CAR = "car"
            #: Configuration key for acceleration data
            ACCELERATION_DATA = "acceleration_data"
            #: Configuration key for link quality
            LINK_QUALITY = "link_quality"
            #: Configuration key for battery level
            BATTERY_LEVEL = "battery_level"
            #: Configuration key for validation indicator
            VALIDATION_INDICATOR = "validation_indicator"

    def __init__(
        self,
        serial: str = None,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        sensitivity: float = None,
        enable_di: bool = None,
        enable_counter: bool = None,
        noise_reduction: bool = None,
        car: bool = None,
        acceleration_data: bool = None,
        link_quality: bool = None,
        battery_level: bool = None,
        validation_indicator: bool = None,
        **kwargs,
    ):
        """Initialize g.Nautilus amplifier interface.

        Args:
            serial: Device serial number. Uses first available if None.
            sampling_rate: Sampling frequency in Hz (250 or 500).
            channel_count: Number of EEG channels to acquire.
            frame_size: Samples per data frame (NumberOfScans).
            sensitivity: Channel sensitivity (get_supported_sensitivities()).
            enable_di: Enable digital I/O channel.
            enable_counter: Enable counter channel.
            noise_reduction: Enable noise reduction.
            car: Enable Common Average Reference.
            acceleration_data: Enable accelerometer data (adds 3 channels).
            link_quality: Enable link quality information channel.
            battery_level: Enable battery level channel.
            validation_indicator: Enable validation indicator channel.
            **kwargs: Additional parameters for AmplifierSource.

        Raises:
            NotImplementedError: If not running on Windows.
            RuntimeError: If GDS library unavailable or device init fails.
            ValueError: If the sampling rate, frame size or channel
                count is not a whole number.
        """
        # No platform guard, deliberately. The refusal that stood here
        # said the driver was Windows-only, and it had outlived the
        # limitation: gtec_gds 1.6.0 publishes cp310-cp314 wheels for
        # Linux (x86_64) and macOS (x86_64 and arm64) as well as Windows
        # -- verified by wheel tag on 2026-09-05 and recorded in
        # pyproject.toml, where the same check found a stale win32 marker
        # had been silently excluding GDS from every non-Windows install.
        #
        # So a user on Linux was told g.Nautilus was unsupported on a
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

        #: g.Nautilus device interface instance
        # A restored configuration binds these named parameters in
        # the per-port list form they were stored as, and they are
        # handed straight to the native driver below, which expects
        # numbers.
        channel_count = self.scalar(channel_count)
        frame_size = self.scalar(frame_size)
        sampling_rate = self.scalar(sampling_rate)

        # The driver stores these in integer struct fields and cffi
        # refuses a float, while the driver's own check passes one:
        # it tests `rate not in {256: 8, ...}` and 256.0 hashes
        # equal to 256, so the failure surfaces later as an
        # unattributed 'an integer is required'. Any computed rate
        # is a float in Python 3 -- 4800 / 4 is 1200.0.
        sampling_rate = as_whole_number(sampling_rate, "sampling_rate")
        frame_size = as_whole_number(frame_size, "frame_size")
        channel_count = as_whole_number(channel_count, "channel_count")
        self._device = gds.GNautilus(
            serial=serial,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            sensitivity=sensitivity,
            enable_di=enable_di,
            enable_counter=enable_counter,
            noise_reduction=noise_reduction,
            car=car,
            acceleration_data=acceleration_data,
            link_quality=link_quality,
            battery_level=battery_level,
            validation_indicator=validation_indicator,
        )

        # Update parameters with actual device configuration
        serial = self._device.serial_number
        channel_count = self._device.channel_count
        frame_size = self._device.frame_size
        sensitivity = self._device.sensitivity
        enable_di = self._device.enable_di
        enable_counter = self._device.enable_counter
        noise_reduction = self._device.noise_reduction
        car = self._device.car
        acceleration_data = self._device.acceleration_data
        link_quality = self._device.link_quality
        battery_level = self._device.battery_level
        validation_indicator = self._device.validation_indicator

        # The measured channels are the ones the device exposes as
        # Channels; everything below is appended after them. Recorded
        # before the additions so setup() can tell the two blocks apart.
        self._eeg_channel_count = channel_count

        # Add digital input channel if enabled
        if enable_di:
            channel_count += 1

        # Note: enable_counter replaces channel 1 with counter data,
        # it does not add an additional channel

        # Add acceleration channels if enabled (x, y, z)
        if acceleration_data:
            channel_count += 3

        # Add link quality channel if enabled
        if link_quality:
            channel_count += 1

        # Add battery level channel if enabled
        if battery_level:
            channel_count += 1

        # Add validation indicator channel if enabled
        if validation_indicator:
            channel_count += 1

        # Set up data callback for real-time streaming
        self._device.set_data_callback(self._data_callback)

        # Initialize parent AmplifierSource with final configuration
        #
        # The serial is the one the device reported, not the one that was
        # asked for, and it is stored: a configuration restored without it
        # opens whichever amplifier happens to be first, which is a
        # different device from the one the session was recorded with.
        # That substitution leaves no trace in the data, and attestation
        # binds a signature to the serial the handle reports, so the
        # verdict would be about the wrong device too.
        super().__init__(
            serial=serial,
            sampling_rate=sampling_rate,
            channel_count=channel_count,
            frame_size=frame_size,
            sensitivity=sensitivity,
            enable_di=enable_di,
            enable_counter=enable_counter,
            noise_reduction=noise_reduction,
            car=car,
            acceleration_data=acceleration_data,
            link_quality=link_quality,
            battery_level=battery_level,
            validation_indicator=validation_indicator,
            **kwargs,
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Separate the measured channels from the appended extras.

        Digital input, accelerometer, link quality, battery level and
        validation indicator are appended after the measured channels.
        Marking them keeps a filter from processing them as if they were
        EEG, which is otherwise invisible: a filtered battery level is
        still a plausible-looking number.

        The digital input is named as a trigger when it is the only extra
        enabled, because then its position is fixed whatever order the
        driver uses. With several enabled the order matters and is
        decided inside g.NEEDaccess: it cannot be derived from the
        channel arithmetic here, which only ever adds to a total.
        Guessing it would attach roles to the wrong channels, and a wrong
        role is not merely uninformative - it is acted on.

        Link quality and the validation indicator are marked auxiliary
        rather than quality, even though that is what they report. Sync
        consumes the quality role, so naming them that way would delete
        channels the user switched on deliberately.

        The counter is a further known gap: enabling it replaces one of
        the measured channels rather than appending one, and which one is
        not exposed here, so it stays in the measured block exactly as it
        has always been.

        The arrival stamp is appended after every extra, and counted
        apart from them. It has to be: the lone-digital-input branch
        below turns on there being exactly one extra, and counting the
        stamp as a second one would silently demote a real trigger to
        auxiliary.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Output port contexts describing the channel layout.
        """
        port_context_out = super().setup(data, port_context_in)
        keys = self.Configuration.Keys
        roles_c = Constants.ChannelRoles

        # Every block appended after the measured channels.
        extras = [
            keys.ENABLE_DI,
            keys.ACCELERATION_DATA,
            keys.LINK_QUALITY,
            keys.BATTERY_LEVEL,
            keys.VALIDATION_INDICATOR,
        ]
        only_di = [k for k in extras if self.config.get(k)] == [keys.ENABLE_DI]
        stamped = self.NUM_TIMELINE_CHANNELS

        for context in port_context_out.values():
            # Read before the count is raised: n_extra counts the
            # device's appended channels, and the stamp is not one of
            # them.
            total = channels.channel_count(context)
            n_eeg = min(self._eeg_channel_count, total)
            n_extra = total - n_eeg
            extra_role = (
                roles_c.TRIGGER
                if only_di and n_extra == 1
                else roles_c.AUXILIARY
            )
            roles = [roles_c.SIGNAL] * n_eeg
            roles += [extra_role] * n_extra
            roles += [roles_c.TIMESTAMP] * stamped
            context[Constants.Keys.CHANNEL_COUNT] = total + stamped
            # Roles only. Naming the stamp would mean naming the
            # measured channels too, and an amplifier that reports no
            # montage has no names to give them; the stamp is found by
            # role, never by position or name.
            context.update(channels.describe(roles))
        return port_context_out

    def start(self) -> None:
        """Start g.Nautilus data acquisition.

        Initiates hardware data streaming and activates the amplifier for
        real-time EEG data processing.
        """
        # Start hardware data acquisition
        self._device.start()
        # Start parent source processing
        super().start()

    def stop(self):
        """Stop g.Nautilus data acquisition and cleanup resources.

        Stops hardware streaming and ensures proper shutdown of amplifier
        connection.
        """
        # Stop hardware data acquisition
        # Stop the parent first, then release the exclusive handle.
        super().stop()
        self._release_device()

    def _data_callback(self, data: np.ndarray):
        """Stamp one block of g.Nautilus data and forward it.

        Callback invoked by GDS library when new EEG data is available.
        Forwards data through g.Pype pipeline using cycle mechanism.

        Args:
            data: Raw EEG data with shape (frame_size, channel_count).
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
        """Process one step of data through g.Nautilus source.

        Args:
            data: Input data dictionary with PORT_IN key containing EEG data.

        Returns:
            Output data dictionary with PORT_OUT key containing EEG data.
        """
        # Pass through data from input to output port
        return {PORT_OUT: data[PORT_IN]}


class GNautilus(ioc.OChain):
    """g.Nautilus EEG amplifier chain for real-time data acquisition.

    This is an OChain that contains:
    - _GNautilusCore: The actual g.Nautilus acquisition node
    - Link: Bridge for distributed operation (passthrough in standalone)
    - Oscar: OSCAR artifact removal processing

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Interface to g.tec's g.Nautilus wireless EEG amplifier system.
    """

    def __init__(
        self,
        serial: str = None,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        sensitivity: float = None,
        enable_di: bool = None,
        enable_counter: bool = None,
        noise_reduction: bool = None,
        car: bool = None,
        acceleration_data: bool = None,
        link_quality: bool = None,
        battery_level: bool = None,
        validation_indicator: bool = None,
        enable_oscar: bool = False,
        **kwargs,
    ):
        """Initialize g.Nautilus amplifier chain.

        Args:
            serial: Device serial number. Uses first available if None.
            sampling_rate: Sampling frequency in Hz (250 or 500).
            channel_count: Number of EEG channels to acquire.
            frame_size: Samples per data frame (NumberOfScans).
            sensitivity: Channel sensitivity.
            enable_di: Enable digital I/O channel.
            enable_counter: Enable counter channel.
            noise_reduction: Enable noise reduction.
            car: Enable Common Average Reference.
            acceleration_data: Enable accelerometer data (adds 3 channels).
            link_quality: Enable link quality information channel.
            battery_level: Enable battery level channel.
            validation_indicator: Enable validation indicator channel.
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
            "sensitivity": sensitivity,
            "enable_di": enable_di,
            "enable_counter": enable_counter,
            "noise_reduction": noise_reduction,
            "car": car,
            "acceleration_data": acceleration_data,
            "link_quality": link_quality,
            "battery_level": battery_level,
            "validation_indicator": validation_indicator,
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
            sensitivity=sensitivity,
            enable_di=enable_di,
            enable_counter=enable_counter,
            noise_reduction=noise_reduction,
            car=car,
            acceleration_data=acceleration_data,
            link_quality=link_quality,
            battery_level=battery_level,
            validation_indicator=validation_indicator,
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
            raw.source_stage(self, lambda: _GNautilusCore(**self._core_params))
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
        # Sync places this stream on the master timeline. It
        # carries no device counter, so Sync numbers it by counting
        # samples: a valid timeline, but not a loss-aware one.
        nodes.append(Sync())
        return nodes
