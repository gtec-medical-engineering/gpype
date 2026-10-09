from __future__ import annotations

import functools
import time
from typing import Optional

import numpy as np

from ...common._private import channels, driver_usage
from ...common.constants import Constants
from .base.amplifier_source import (
    AmplifierSource,
    apply_channel_units,
    as_whole_number,
    opens_no_device,
    unopened_shape,
)

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN

# Attempts at opening the device, and the pause between them. The GDS
# connect to a g.Nautilus fails now and then with "Couldn't open
# device" and succeeds when simply asked again: 2 of about 25 opens on
# 2026-09-24, with the device free and every retry by hand succeeding.
# See :func:`_open` for the reopen's second transient.
OPEN_ATTEMPTS = 3
OPEN_RETRY_DELAY_S = 2.0


def _open(factory, reopen: bool = False):
    """Open the device, asking again after a transient failure.

    "Couldn't open device" is asked again on every open. On a reopen
    after ``stop()``, so is a device the service does not list at all:
    its enumeration missed the g.Nautilus this source had released 2 s
    before, once on 2026-10-06, and ``start()`` opened it 6 ms later. A
    first open raises that at once, because there absence is usually
    real -- a headset switched off -- and each attempt enumerates for
    2-3 s. Anything else -- a configuration the device turns down, a
    session another process holds -- is raised at once, because asking
    again cannot change the answer.

    Args:
        factory: Zero-argument callable opening the device.
        reopen: True when this source has had the device open before.

    Returns:
        The open driver handle.
    """
    for attempt in range(1, OPEN_ATTEMPTS + 1):
        try:
            return factory()
        except Exception as error:
            transient = "Couldn't open device" in str(error) or (
                reopen and _not_listed(error)
            )
            if not transient or attempt == OPEN_ATTEMPTS:
                raise
            time.sleep(OPEN_RETRY_DELAY_S)


def _not_listed(error: Exception) -> bool:
    """Whether the driver found the device absent from the service's list.

    gtec_gds raises ``DeviceNotFoundError`` for that, and its subclass
    ``SerialNotConnectedError`` when a serial was asked for, which a
    reopen always is. Matched by name, so a stand-in driver module needs
    no import.
    """
    return any(
        cls.__name__ == "DeviceNotFoundError" for cls in type(error).__mro__
    )


class GNautilus(AmplifierSource):
    """g.Nautilus EEG amplifier for real-time data acquisition.

    Interface to g.tec's g.Nautilus wireless EEG amplifier system.
    """

    # One channel appended after the device's own, carrying the
    # instant each block was first seen on this host.
    #
    # The stamp rides in band because it has to survive a Link. The
    # Sync at the end of this chain builds the master-timeline
    # relation from (position, instant) pairs, and under EDGE/SERVER
    # residency that Sync runs in the server process, where reading a
    # clock measures when the frame finished crossing the network
    # rather than when the amplifier delivered it. Sync consumes the
    # channel again, so nothing downstream sees a wider frame.
    #
    # Declared on the port and not in the configuration, unlike
    # Core-8's, and so are the appended extras: a GDS configuration's
    # channel_count is the number handed to the driver, which is what
    # every document since 4.0.0 records under that key (D-CORE-67).
    NUM_TIMELINE_CHANNELS = 1

    # Continuous invalid samples before the headset link is reported
    # lost, and then given up on, in seconds. Switched off at the
    # headset, a g.Nautilus goes on streaming at the full rate with the
    # last real sample held, and the driver raises nothing -- measured
    # 2026-09-24: 100% repeated rows for 45 s, the pipeline Healthy. The
    # one thing that changes is the device's validation indicator, which
    # drops from 1 to 0 within 2.5 s and stays there; switching the
    # headset back on did not bring it back.
    #
    # Out of range it does come back: M12 walked a headset away and the
    # link returned 15.8 s after it dropped (2026-09-24). A switch-off
    # and a walk away look identical until then, so the error waits as
    # long as the BCI Core's does (`STALL_ERROR_S`, where the longest
    # walk away that recovered was 38 s): a switch-off still ends the run,
    # a minute later, and a walk away is not ended at 10 s -- the first
    # value here, which would have ended that run 6 s before the headset
    # returned.
    LINK_WARNING_S = 2.0
    LINK_ERROR_S = 60.0
    #: The flag has to stay 1 this long before a loss counts as over. At
    #: the edge of range it flickers -- 18 short blips before one real
    #: loss, and valid blocks inside it -- and a single valid block used
    #: to restart the clock, so a headset mostly out of range could hold
    #: the error off for ever.
    LINK_VALID_CONFIRM_S = 1.0

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for g.Nautilus amplifier parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration key constants for the g.Nautilus amplifier."""

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

        class OptionalKeys(AmplifierSource.Configuration.OptionalKeys):
            """Configuration keys that may legitimately be absent."""

            #: The serial of the unit that opened, wherever one did. A
            #: process that opens no device -- a server, or a replay --
            #: keeps None, which means what it meant to the author:
            #: whichever amplifier answers first.
            SERIAL = "serial"
            #: The sensitivity the device reported, wherever one opened.
            #: None where none opened -- a server, a replay -- means the
            #: driver's own choice, the device's lowest.
            SENSITIVITY = "sensitivity"

    # Channels each enabled extra appends after the measured ones.
    EXTRA_CHANNELS = {
        Configuration.Keys.ENABLE_DI: 1,
        Configuration.Keys.ACCELERATION_DATA: 3,
        Configuration.Keys.LINK_QUALITY: 1,
        Configuration.Keys.BATTERY_LEVEL: 1,
        Configuration.Keys.VALIDATION_INDICATOR: 1,
    }

    #: Unit of each extra's channel(s), from the vendor header
    #: (``vendor/gds-headers/GDSClientAPI_gNautilus.h`` in
    #: gtec-gds-dev): acceleration is +/- 6 g, link quality and battery
    #: level are 0-100 percent, and the digital input and validation
    #: indicator are bit-encoded flags with no physical unit.
    EXTRA_CHANNEL_UNITS = {
        Configuration.Keys.ENABLE_DI: [None],
        Configuration.Keys.ACCELERATION_DATA: ["g", "g", "g"],
        Configuration.Keys.LINK_QUALITY: ["%"],
        Configuration.Keys.BATTERY_LEVEL: ["%"],
        Configuration.Keys.VALIDATION_INDICATOR: [None],
    }

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
        edge_id: Optional[str] = None,
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
            enable_oscar: Run OSCAR artifact removal on this stream.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
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

        # The validation indicator is the only evidence of a lost headset
        # link (see LINK_ERROR_S), so it is opened even when not asked
        # for, and stripped again before the block goes out. Only where
        # its column is certain: with no other extra enabled it is the
        # sole appended channel. With several, g.NEEDaccess decides their
        # order (see setup), so it is watched only when the user asked for
        # it, and nothing is added behind their other extras.
        requested = bool(validation_indicator)
        others = any(
            bool(x)
            for x in (
                enable_di,
                enable_counter,
                acceleration_data,
                link_quality,
                battery_level,
            )
        )
        self._watch_link = requested or not others
        self._strip_validation = self._watch_link and not requested
        self._invalid_since = None
        self._valid_since = None
        self._link_warned = False
        self._link_given_up = False
        self._link_unreadable = ""

        if opens_no_device(edge_id):
            # **This process does not open the amplifier**: a server,
            # another edge's process, or a replay. See GUSBamp: the shape
            # comes from the arguments (unopened_shape), an unset flag
            # takes the value the driver gives None, and the serial and
            # the sensitivity stay as given.
            sampling_rate, channel_count, frame_size = unopened_shape(
                self, sampling_rate, channel_count, frame_size
            )
            self._device = None
            self._device_factory = None
            enable_di = bool(enable_di)
            enable_counter = bool(enable_counter)
            noise_reduction = bool(noise_reduction)
            car = bool(car)
            acceleration_data = bool(acceleration_data)
            link_quality = bool(link_quality)
            battery_level = bool(battery_level)
            validation_indicator = bool(validation_indicator)
        else:
            # Import gtec_gds only when actually needed (lazy import)
            try:
                import gtec_gds as gds

                # The driver refuses to construct a handle until the
                # local usage key is registered.
                driver_usage.register()
            except ImportError as e:
                raise RuntimeError(
                    f"GDS library not available: {e}. "
                    "This may be expected in CI environments where the "
                    "GDS library is not installed."
                ) from e

            open_kwargs = dict(
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
                validation_indicator=(
                    True if self._watch_link else (validation_indicator)
                ),
            )
            self._device = _open(
                functools.partial(gds.GNautilus, **open_kwargs)
            )

            # Update parameters with actual device configuration
            serial = self._device.serial_number
            # Pinned to the unit that actually opened, so a bench with
            # two amplifiers of this model reopens the same one after a
            # stop(). See AmplifierSource._reopen_device.
            open_kwargs["serial"] = serial
            self._device_factory = functools.partial(
                _open,
                functools.partial(gds.GNautilus, **open_kwargs),
                reopen=True,
            )
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
            # What the user sees is what the user asked for: the column
            # opened only for watching is not theirs, and is not counted
            # below.
            if self._strip_validation:
                validation_indicator = False

        # channel_count stays the channels the device exposes as
        # Channels; setup() appends the extras on the port. The counter
        # replaces channel 1 rather than adding one.

        # Set up data callback for real-time streaming
        if self._device is not None:
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
            enable_oscar=enable_oscar,
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
            edge_id=edge_id,
            **kwargs,
        )

    def channel_units(self) -> Optional[list]:
        """One 'uV' per EEG channel, each enabled extra's own unit, None
        for the arrival stamp -- see :meth:`EXTRA_CHANNEL_UNITS`."""
        n_eeg = self.scalar(self.config[self.Configuration.Keys.CHANNEL_COUNT])
        enabled = [k for k in self.EXTRA_CHANNELS if self.config.get(k)]
        units = ["uV"] * n_eeg
        for key in enabled:
            units += self.EXTRA_CHANNEL_UNITS[key]
        units += [None] * self.NUM_TIMELINE_CHANNELS
        return units

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
        enabled = [k for k in self.EXTRA_CHANNELS if self.config.get(k)]
        n_extra = sum(self.EXTRA_CHANNELS[k] for k in enabled)
        only_di = enabled == [keys.ENABLE_DI]
        stamped = self.NUM_TIMELINE_CHANNELS
        units = self.channel_units()

        for context in port_context_out.values():
            # The configuration counts the channels handed to the
            # driver; the extras and the stamp are added here.
            n_eeg = channels.channel_count(context)
            extra_role = (
                roles_c.TRIGGER
                if only_di and n_extra == 1
                else roles_c.AUXILIARY
            )
            roles = [roles_c.SIGNAL] * n_eeg
            roles += [extra_role] * n_extra
            roles += [roles_c.TIMESTAMP] * stamped
            context[Constants.Keys.CHANNEL_COUNT] = n_eeg + n_extra + stamped
            # Roles only. Naming the stamp would mean naming the
            # measured channels too, and an amplifier that reports no
            # montage has no names to give them; the stamp is found by
            # role, never by position or name.
            context.update(channels.describe(roles))
            apply_channel_units(context, units)
        return port_context_out

    def start(self) -> None:
        """Start g.Nautilus data acquisition.

        Initiates hardware data streaming and activates the amplifier for
        real-time EEG data processing.
        """
        # Start hardware data acquisition
        self._reopen_device()
        self._device.start()
        # Start parent source processing
        super().start()
        # The driver's own thread can die mid-run; see
        # AmplifierSource._start_stream_watch.
        self._start_stream_watch()

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
        # channels.stamp_clock(), because Sync compares the stamp
        # against its own reading of that same clock.
        # One value for the whole block -- the block cannot exist
        # until its last sample has been acquired, which is the sample
        # Sync reads it back from.
        arrival = channels.wrap_time(channels.stamp_clock())
        if self._watch_link:
            self._note_validity(data[:, -1])
            if self._strip_validation:
                data = data[:, :-1]
        stamp = np.full(
            (data.shape[0], self.NUM_TIMELINE_CHANNELS),
            arrival,
            dtype=Constants.DATA_TYPE,
        )
        self.cycle(data={PORT_IN: np.hstack((data, stamp))})

    def _note_validity(self, flags: np.ndarray) -> None:
        """Track how long the headset link has delivered invalid samples.

        Runs on the driver's data thread, so it only records; the watch
        thread reports (see :meth:`_check_link`).

        Args:
            flags: The block's validation indicator, 1 valid and 0 not.
        """
        if self._link_unreadable:
            return
        if not np.all((flags == 0) | (flags == 1)):
            # Not a validation indicator: the column is somewhere else.
            # Watching it would report link loss that never happened.
            self._link_unreadable = (
                "the last column is not a validation indicator (values "
                "%g..%g)" % (float(np.min(flags)), float(np.max(flags)))
            )
            return
        now = time.monotonic()
        if flags[-1]:
            if self._valid_since is None:
                self._valid_since = now
            if now - self._valid_since >= self.LINK_VALID_CONFIRM_S:
                self._invalid_since = None
        else:
            self._valid_since = None
            if self._invalid_since is None:
                self._invalid_since = now

    def _check_link(self) -> bool:
        """Warn, then give up, on a headset that delivers invalid samples.

        Returns:
            True once given up on, which ends the watch.
        """
        try:
            if self._link_unreadable:
                if not self._link_warned:
                    self._link_warned = True
                    self.log(
                        "Cannot watch the headset link: %s. A switched-off "
                        "headset would go unnoticed." % self._link_unreadable,
                        type=Constants.LogTypes.WARNING,
                    )
                return False
            since = self._invalid_since
            if since is None:
                if self._link_warned:
                    self._link_warned = False
                    self.log("The headset link is valid again.")
                return False
            lost = time.monotonic() - since
            if lost >= self.LINK_ERROR_S and not self._link_given_up:
                self._link_given_up = True
                self.log(
                    "The headset has delivered no valid sample for "
                    "%.0f s: it is switched off, out of range or out of "
                    "battery, and the base station is repeating its last "
                    "sample. Acquisition cannot continue; switch the "
                    "headset on and restart the pipeline." % lost,
                    type=Constants.LogTypes.ERROR,
                )
                return True
            if lost >= self.LINK_WARNING_S and not self._link_warned:
                self._link_warned = True
                self.log(
                    "The headset has delivered no valid sample for %.0f s; "
                    "the base station is repeating its last sample." % lost,
                    type=Constants.LogTypes.WARNING,
                )
        except Exception:
            pass
        return False

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one step of data through g.Nautilus source.

        Args:
            data: Input data dictionary with PORT_IN key containing EEG data.

        Returns:
            Output data dictionary with PORT_OUT key containing EEG data.
        """
        # Pass through data from input to output port
        return {PORT_OUT: data[PORT_IN]}
