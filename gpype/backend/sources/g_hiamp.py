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


class GHIamp(AmplifierSource):
    """g.HIamp EEG amplifier for real-time data acquisition.

    Interface to g.tec's g.HIamp wired EEG amplifier system.
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
    # Core-8's, and so is the digital input: a GDS configuration's
    # channel_count is the number handed to the driver, which is what
    # every document since 4.0.0 records under that key (D-CORE-67).
    NUM_TIMELINE_CHANNELS = 1

    class Configuration(AmplifierSource.Configuration):
        """Configuration class for g.HIamp amplifier parameters."""

        class Keys(AmplifierSource.Configuration.Keys):
            """Configuration key constants for the g.HIamp amplifier."""

            #: Configuration key for enabling the digital trigger input
            ENABLE_DI = "enable_di"
            #: Configuration key for enabling counter channel
            ENABLE_COUNTER = "enable_counter"

        class OptionalKeys(AmplifierSource.Configuration.OptionalKeys):
            """Configuration keys that may legitimately be absent."""

            #: The serial of the unit that opened, wherever one did. A
            #: process that opens no device -- a server, or a replay --
            #: keeps None, which means what it meant to the author:
            #: whichever amplifier answers first.
            SERIAL = "serial"

    def __init__(
        self,
        serial: str = None,
        sampling_rate: float = None,
        channel_count: int = None,
        frame_size: int = None,
        enable_di: bool = None,
        enable_counter: bool = None,
        enable_oscar: bool = False,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize g.HIamp amplifier interface.

        Args:
            serial: Device serial number. Uses first available if None.
            sampling_rate: Sampling frequency in Hz.
            channel_count: Number of EEG channels to acquire.
            frame_size: Samples per data frame (NumberOfScans).
            enable_di: Enable digital trigger input channel.
            enable_counter: Enable counter channel (increments each block).
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
        # So a user on Linux was told g.HIamp was unsupported on a
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

        if opens_no_device(edge_id):
            # **This process does not open the amplifier**: a server,
            # another edge's process, or a replay. See GUSBamp: the shape
            # comes from the arguments (unopened_shape), an unset flag
            # takes the value the driver gives None, and the serial stays
            # as given.
            sampling_rate, channel_count, frame_size = unopened_shape(
                self, sampling_rate, channel_count, frame_size
            )
            self._device = None
            self._device_factory = None
            enable_di = bool(enable_di)
            enable_counter = bool(enable_counter)
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
                enable_di=enable_di,
                enable_counter=enable_counter,
            )
            self._device = gds.GHIamp(**open_kwargs)

            # Update parameters with actual device configuration
            serial = self._device.serial_number
            # Pinned to the unit that actually opened, so a bench with
            # two amplifiers of this model reopens the same one after a
            # stop(). See AmplifierSource._reopen_device.
            open_kwargs["serial"] = serial
            self._device_factory = functools.partial(gds.GHIamp, **open_kwargs)
            channel_count = self._device.channel_count
            frame_size = self._device.frame_size
            enable_di = self._device.enable_di
            enable_counter = self._device.enable_counter

        # channel_count stays the acquired count; setup() appends the
        # digital input on the port. enable_counter replaces channel 1
        # with counter data rather than adding one.

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
            enable_di=enable_di,
            enable_counter=enable_counter,
            edge_id=edge_id,
            **kwargs,
        )

    def channel_units(self) -> Optional[list]:
        """One 'uV' per EEG channel (vendor/gds-headers's
        ``GDSClientAPI_gHIamp.h``: ``GDS_GHIAMP_SCALING.Offset`` is in
        uV), None for the digital input when enabled, and None for the
        arrival stamp. A channel enabling ``enable_counter`` overwrites
        a measured channel at a position this node does not track (see
        ``setup``'s docstring), so it is still declared 'uV'."""
        n_eeg = self.scalar(self.config[self.Configuration.Keys.CHANNEL_COUNT])
        trigger = int(bool(self.config[self.Configuration.Keys.ENABLE_DI]))
        return (
            ["uV"] * n_eeg
            + [None] * trigger
            + [None] * self.NUM_TIMELINE_CHANNELS
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Mark the digital input as a trigger, and stamp beside it.

        The digital input is the only channel appended after the measured
        ones, so its position is unambiguous. Saying it is a trigger is
        what keeps a filter off it: a bandpass turns an edge into a
        decaying oscillation, and nothing about the result announces that
        it was ever an event.

        The configuration counts the channels handed to the driver, so
        the digital input is added here, on the port, and the arrival
        stamp after it.

        Enabling the counter replaces one of the measured channels rather
        than appending one, and which one is not exposed here, so it
        stays in the measured block exactly as it has always been.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Output port contexts describing the channel layout.
        """
        port_context_out = super().setup(data, port_context_in)
        trigger = int(bool(self.config[self.Configuration.Keys.ENABLE_DI]))
        stamped = self.NUM_TIMELINE_CHANNELS
        units = self.channel_units()
        for context in port_context_out.values():
            n_eeg = channels.channel_count(context)
            roles = [Constants.ChannelRoles.SIGNAL] * n_eeg
            roles += [Constants.ChannelRoles.TRIGGER] * trigger
            roles += [Constants.ChannelRoles.TIMESTAMP] * stamped
            context[Constants.Keys.CHANNEL_COUNT] = n_eeg + trigger + stamped
            # Roles only. Naming the stamp would mean naming the
            # measured channels too, and an amplifier that reports no
            # montage has no names to give them; the stamp is found by
            # role, never by position or name.
            context.update(channels.describe(roles))
            apply_channel_units(context, units)
        return port_context_out

    def start(self) -> None:
        """Start g.HIamp data acquisition.

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
        """Stop g.HIamp data acquisition and cleanup resources.

        Stops hardware streaming and ensures proper shutdown of amplifier
        connection.
        """
        # Stop the parent first, then release the exclusive handle.
        super().stop()
        self._release_device()

    def _data_callback(self, data: np.ndarray):
        """Stamp one block of g.HIamp data and forward it.

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
        stamp = np.full(
            (data.shape[0], self.NUM_TIMELINE_CHANNELS),
            arrival,
            dtype=Constants.DATA_TYPE,
        )
        # Forward data through the pipeline using the input port
        self.cycle(data={PORT_IN: np.hstack((data, stamp))})

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one step of data through g.HIamp source.

        Args:
            data: Input data dictionary with PORT_IN key containing EEG data.

        Returns:
            Output data dictionary with PORT_OUT key containing EEG data.
        """
        # Pass through data from input to output port
        return {PORT_OUT: data[PORT_IN]}
