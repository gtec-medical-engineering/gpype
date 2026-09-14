from __future__ import annotations

import os
import sys
import threading
import time
from typing import List, Optional

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.oscar import Oscar
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
from .base.amplifier_source import AmplifierSource

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT
#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN


def _get_unicorn_lib_path() -> str:
    """Get the path to the Unicorn Python library.

    The UnicornPy module requires the Unicorn.dll to be accessible.
    This function returns the path to the Lib folder in the user's
    Documents folder where the Unicorn Suite is typically installed.

    Returns:
        Path to the Unicorn Python Lib directory.
    """
    # Get user documents folder on Windows
    documents_path = os.path.join(os.environ["USERPROFILE"], "Documents")
    unicorn_lib_path = os.path.join(
        documents_path,
        "gtec",
        "Unicorn Suite",
        "Hybrid Black",
        "Unicorn Python",
        "Lib",
    )
    return unicorn_lib_path


def _ensure_unicorn_path():
    """Ensure the Unicorn Python library path is in sys.path and PATH.

    Adds the Unicorn Lib directory to sys.path for module import and
    to the PATH environment variable for DLL loading.
    """
    lib_path = _get_unicorn_lib_path()

    # Add to sys.path if not already present
    if lib_path not in sys.path:
        sys.path.insert(0, lib_path)

    # Add to PATH for DLL loading
    current_path = os.environ.get("PATH", "")
    if lib_path not in current_path:
        os.environ["PATH"] = lib_path + os.pathsep + current_path


class _HybridBlackCore(AmplifierSource):
    """Internal node implementing Unicorn Hybrid Black amplifier acquisition.

    This is the actual Hybrid Black node (pure ONode inheritance).
    It is wrapped by the HybridBlack chain for distributed operation.

    Interface to g.tec Unicorn Hybrid Black wireless EEG amplifier using
    Bluetooth. Supports 8-channel EEG acquisition at 250 Hz, plus optional
    accelerometer, gyroscope, battery, counter, and validation channels.
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
    #: Measured Bluetooth latency of this device, in milliseconds. Kept
    #: as a recorded property of the hardware; nothing applies it. Under
    #: timestamp-based synchronisation a fixed device delay is a constant
    #: offset of the stream, which belongs in the timeline relation
    #: rather than in a delay line here.
    DEVICE_DELAY_MS = 40
    #: Maximum allowed consecutive buffer underruns before warning
    NUM_UNDERRUNS_ALLOWED = 5
    #: Consecutive failed reads tolerated before acquisition is given
    #: up. One failure is a Bluetooth hiccup the next read recovers
    #: from, so failing on the first would end a run over nothing;
    #: three in a row is a device that has stopped answering, and
    #: carrying on then means a pipeline that emits nothing while
    #: reporting Healthy.
    NUM_ACQUISITION_ERRORS_ALLOWED = 3
    #: Poll interval as a fraction of one frame period, when on time.
    #: Slightly under 1.0 so the loop stays just ahead of the device.
    #: These were absolute seconds (0.0039 and 0.003) tuned to a single
    #: sample at 250 Hz, which silently became the wrong cadence for any
    #: other frame size; as fractions they hold at every frame size and
    #: reproduce the original numbers exactly at frame_size 1.
    WAIT_ON_TIME_RATIO = 0.975
    #: Poll interval as a fraction of one frame period, when behind.
    WAIT_BEHIND_RATIO = 0.75
    #: Threshold for determining if we're behind (GetData blocking time)
    BEHIND_THRESHOLD_S = 0.001

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

    def __init__(
        self,
        serial: Optional[str] = None,
        channel_count: Optional[int] = None,
        frame_size: Optional[int] = None,
        include_accel: Optional[bool] = None,
        include_gyro: Optional[bool] = None,
        include_aux: Optional[bool] = None,
        test_signal: Optional[bool] = None,
        **kwargs,
    ):
        """Initialize Unicorn Hybrid Black amplifier source.

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
            **kwargs: Additional arguments for parent AmplifierSource.

        Raises:
            NotImplementedError: If not running on Windows.
        """
        # Platform check - Hybrid Black is only supported on Windows
        if sys.platform != "win32":
            raise NotImplementedError(
                "HybridBlack amplifier is only supported on Windows."
            )

        # Ensure UnicornPy is importable
        _ensure_unicorn_path()

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
            channel_count=total_channels,
            eeg_channel_count=channel_count,
            sampling_rate=self.SAMPLING_RATE,
            frame_size=frame_size,
            decimation_factor=1,
            include_accel=include_accel,
            include_gyro=include_gyro,
            include_aux=include_aux,
            test_signal=test_signal,
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

        # Initialize device connection (will be established in start())
        self._device = None

        # Initialize threading components
        self._running: bool = False
        self._acquisition_thread: Optional[threading.Thread] = None

        # Current frame for passing data from acquisition to step()
        self._current_frame: Optional[np.ndarray] = None

        # Underrun tracking
        self._underrun_counter: int = 0

        #: Consecutive failed reads, cleared by the next good one.
        self._acquisition_error_counter: int = 0

    def start(self) -> None:
        """Start Unicorn Hybrid Black amplifier and begin data acquisition.

        Establishes Bluetooth connection and starts background thread that
        acquires data and drives the pipeline via cycle().

        Raises:
            ConnectionError: If amplifier connection fails.
            RuntimeError: If background thread creation fails.
        """
        # Import UnicornPy when actually needed (lazy import)
        try:
            import UnicornPy
        except ImportError as e:
            raise RuntimeError(
                f"UnicornPy library not available: {e}. "
                "Please ensure the Unicorn Suite is installed and the "
                "library path is correct."
            ) from e

        # Initialize current frame holder
        self._current_frame = None
        self._underrun_counter = 0
        self._acquisition_error_counter = 0

        # Initialize and connect to Unicorn Hybrid Black amplifier
        if self._device is None:
            # Get available devices if no serial specified
            if self._target_sn is None:
                device_list = UnicornPy.GetAvailableDevices(True)
                if len(device_list) <= 0 or device_list is None:
                    raise ConnectionError(
                        "No Unicorn device available. Please pair with a "
                        "Unicorn first."
                    )
                self._target_sn = device_list[0]
                print(f"Using first available device: {self._target_sn}")

            # Connect to the device
            self._device = UnicornPy.Unicorn(self._target_sn)
            print(f"Connected to Unicorn Hybrid Black: {self._target_sn}")

        # Start single acquisition thread (handles both data and timing)
        if not self._running:
            self._running = True
            self._acquisition_thread = threading.Thread(
                target=self._acquisition_function, daemon=True
            )
            self._acquisition_thread.start()

        # Call parent start method
        super().start()

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
        total is also what ``_acquisition_function`` sizes its receive
        buffer from, so widening the configuration would ask the device
        for a channel it does not have.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with 250 Hz sampling rate.
        """
        port_context_out = super().setup(data, port_context_in)

        # Describe the channels so a filter does not treat an
        # accelerometer as EEG. The layout _acquisition_function builds:
        # EEG first, then accelerometer, then gyroscope, then the
        # auxiliary triple, each block present only if it was asked for.
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

        for context in port_context_out.values():
            # Read before the count is raised: the blocks above count the
            # device's channels, and the stamp is not one of them.
            total = channels.channel_count(context)
            context[Constants.Keys.CHANNEL_COUNT] = total + stamped
            context.update(channels.describe(roles))
        return port_context_out

    def stop(self):
        """Stop Unicorn Hybrid Black amplifier and clean up resources.

        Stops data acquisition, terminates background thread, and disconnects
        from amplifier hardware.
        """
        # Stop background thread and wait for completion
        if self._running:
            self._running = False
            acq_thread = self._acquisition_thread
            if acq_thread and acq_thread.is_alive():
                acq_thread.join(timeout=10)

        # Stop amplifier data acquisition
        if self._device is not None:
            try:
                self._device.StopAcquisition()
            except Exception:
                pass  # Device may already be stopped
            # Clean up device connection
            del self._device
            self._device = None

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

    def _acquisition_function(self):
        """Background thread function for data acquisition and pipeline timing.

        Uses adaptive timing: if GetData is fast (<1ms), we're behind and
        use shorter wait. If GetData blocks (>=1ms), we're on time and use
        normal wait interval.
        """
        try:
            import UnicornPy
        except ImportError:
            return

        # Resolved once, outside the loop: every message below has to
        # name the node the author wrote, and type(self).__name__ here
        # is the chain's private core.
        label = node_label(self)

        # Get number of acquired channels from device
        num_acquired_channels = self._device.GetNumberOfAcquiredChannels()

        # Acquisition frame size matches output frame size
        acq_frame_length = self._frame_size
        receive_buffer_length = acq_frame_length * num_acquired_channels * 4
        receive_buffer = bytearray(receive_buffer_length)

        # Build column indices for channel selection
        col_indices = list(range(self._eeg_channel_count))
        if self._include_accel:
            start = self.NUM_EEG_CHANNELS
            col_indices.extend(range(start, start + 3))
        if self._include_gyro:
            start = self.NUM_EEG_CHANNELS + self.NUM_ACCEL_CHANNELS
            col_indices.extend(range(start, start + 3))
        if self._include_aux:
            start = self.NUM_EEG_CHANNELS + self.NUM_ACCEL_CHANNELS + 3
            col_indices.extend(range(start, num_acquired_channels))

        # Check if we can use fast contiguous slice (EEG only, all 8 channels)
        use_slice = (
            not self._include_accel
            and not self._include_gyro
            and not self._include_aux
            and self._eeg_channel_count == self.NUM_EEG_CHANNELS
        )

        # Start data acquisition from amplifier
        self._device.StartAcquisition(self._test_signal)

        # Initialize adaptive timing. One GetData returns one frame, so
        # the cadence to hold is one frame period -- not one sample.
        frame_period_s = acq_frame_length / float(self.SAMPLING_RATE)
        wait_on_time_s = self.WAIT_ON_TIME_RATIO * frame_period_s
        wait_behind_s = self.WAIT_BEHIND_RATIO * frame_period_s
        wait_interval_s = wait_on_time_s
        t_last_call = time.perf_counter()

        while self._running:
            try:
                # Calculate expected next call time and sleep if needed
                t_expected_next = t_last_call + wait_interval_s
                t_now = time.perf_counter()
                t_remaining = t_expected_next - t_now
                if t_remaining > 0:
                    time.sleep(t_remaining)

                # GetData call with timing
                t_pre_get = time.perf_counter()
                self._device.GetData(
                    acq_frame_length, receive_buffer, receive_buffer_length
                )
                t_post_get = time.perf_counter()
                # First sight: GetData returns only once the block is
                # complete, so this is the earliest instant it exists on
                # the host -- taken before the block is reshaped, copied,
                # selected from or scheduled.
                #
                # time.monotonic() and not perf_counter(), because Sync
                # compares the stamp against its own reading of that
                # clock. t_post_get stays on perf_counter and is left
                # alone: it drives the adaptive wait interval, and the
                # two clocks are not interchangeable for either purpose.
                arrival = channels.wrap_time(channels.stamp_clock())
                t_last_call = t_post_get

                # Adaptive timing: adjust wait based on GetData blocking
                # time. The counter follows the *behind* branch, which is
                # the fast return: the driver already had a block waiting,
                # so this loop is not keeping up with the device.
                #
                # It used to be the other way round, and so reported the
                # opposite of what happened: the warning fired after six
                # consecutive healthy cycles and was cleared by the case
                # it exists to report.
                blocking_time_s = t_post_get - t_pre_get
                if blocking_time_s < self.BEHIND_THRESHOLD_S:
                    # Fast return = buffer had data = we're behind
                    wait_interval_s = wait_behind_s
                    self._underrun_counter += 1
                else:
                    # Slow return = had to wait for data = we're on time
                    wait_interval_s = wait_on_time_s
                    self._underrun_counter = 0

                if self._underrun_counter > self.NUM_UNDERRUNS_ALLOWED:
                    self.log(
                        f"Falling behind the amplifier: the driver had a "
                        f"block ready on "
                        f"{self.NUM_UNDERRUNS_ALLOWED + 1} consecutive "
                        f"reads. Reduce the work per frame, or raise "
                        f"frame_size to cycle less often.",
                        type=Constants.LogTypes.WARNING,
                    )
                    self._underrun_counter = 0

                # Convert to numpy array
                raw_data = np.frombuffer(
                    receive_buffer,
                    dtype=np.float32,
                    count=num_acquired_channels * acq_frame_length,
                ).reshape(acq_frame_length, num_acquired_channels)

                # Extract selected channels
                if use_slice:
                    frame_data = raw_data[:, : self.NUM_EEG_CHANNELS].copy()
                else:
                    frame_data = raw_data[:, col_indices].copy()

                # One value for the whole block: the block cannot exist
                # until its last sample has been acquired, and that last
                # sample is the one Sync reads the instant back from.
                stamp = np.full(
                    (frame_data.shape[0], self.NUM_TIMELINE_CHANNELS),
                    arrival,
                    dtype=Constants.DATA_TYPE,
                )

                # Set current frame and trigger pipeline cycle immediately
                self._current_frame = np.hstack((frame_data, stamp))
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
                    # The wait at the top of the loop is measured from
                    # t_last_call, which a failed read leaves in the
                    # past, so the retry slept for nothing at all: a
                    # permanently failing GetData spun this thread flat
                    # out rather than pacing itself.
                    t_last_call = time.perf_counter()
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

        # Clean up receive buffer
        del receive_buffer

    @staticmethod
    def get_available_devices() -> list[str]:
        """Get list of available Unicorn Hybrid Black devices.

        Returns:
            List of device serial numbers that are available for connection.
            Returns empty list on non-Windows platforms.
        """
        if sys.platform != "win32":
            return []

        _ensure_unicorn_path()
        try:
            import UnicornPy

            device_list = UnicornPy.GetAvailableDevices(True)
            return device_list if device_list else []
        except ImportError:
            return []
        except Exception:
            return []


class HybridBlack(ioc.OChain):
    """Unicorn Hybrid Black amplifier chain for wireless EEG acquisition.

    This is an OChain that contains:
    - _HybridBlackCore: The actual Hybrid Black acquisition node
    - Link: Bridge for distributed operation (passthrough in standalone)
    - Oscar: OSCAR artifact removal processing

    The chain structure enables distributed edge/server operation while
    keeping node inheritance clean (no chain mixing in node path).

    Interface to g.tec Unicorn Hybrid Black wireless EEG amplifier using
    Bluetooth. Supports 8-channel EEG acquisition at 250 Hz, plus optional
    accelerometer, gyroscope, battery, counter, and validation channels.
    """

    # Re-export constants from core
    DEFAULT_INCLUDE_ACCEL = _HybridBlackCore.DEFAULT_INCLUDE_ACCEL
    DEFAULT_INCLUDE_GYRO = _HybridBlackCore.DEFAULT_INCLUDE_GYRO
    DEFAULT_INCLUDE_AUX = _HybridBlackCore.DEFAULT_INCLUDE_AUX
    DEFAULT_TEST_SIGNAL = _HybridBlackCore.DEFAULT_TEST_SIGNAL
    SAMPLING_RATE = _HybridBlackCore.SAMPLING_RATE
    NUM_EEG_CHANNELS = _HybridBlackCore.NUM_EEG_CHANNELS
    NUM_ACCEL_CHANNELS = _HybridBlackCore.NUM_ACCEL_CHANNELS
    NUM_GYRO_CHANNELS = _HybridBlackCore.NUM_GYRO_CHANNELS

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
        **kwargs,
    ):
        """Initialize Unicorn Hybrid Black amplifier chain.

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
            enable_oscar: Enable OSCAR artifact removal processing.
            **kwargs: Additional arguments.

        Raises:
            NotImplementedError: If not running on Windows.
        """
        # Store parameters for create_internal_nodes
        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "serial": serial,
            "channel_count": channel_count,
            "frame_size": frame_size,
            "include_accel": include_accel,
            "include_gyro": include_gyro,
            "include_aux": include_aux,
            "test_signal": test_signal,
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
            channel_count=channel_count,
            frame_size=frame_size,
            include_accel=include_accel,
            include_gyro=include_gyro,
            include_aux=include_aux,
            test_signal=test_signal,
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
            raw.source_stage(
                self, lambda: _HybridBlackCore(**self._core_params)
            )
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
        # Sync places this stream on the master timeline. The core
        # stamps each block at first sight, so Sync fits the relation to
        # the amplifier's arrival instants rather than to its own clock
        # -- which matters under EDGE/SERVER residency, where that Sync
        # runs in the server process and would otherwise be fitting to
        # the transport. The stamp supplies the *instant* only: this
        # stream still carries no device counter, so Sync numbers it by
        # counting samples, and it remains a valid timeline but not a
        # loss-aware one.
        nodes.append(Sync())
        return nodes

    @staticmethod
    def get_available_devices() -> list[str]:
        """Get list of available Unicorn Hybrid Black devices.

        Returns:
            List of device serial numbers that are available for connection.
            Returns empty list on non-Windows platforms.
        """
        return _HybridBlackCore.get_available_devices()
