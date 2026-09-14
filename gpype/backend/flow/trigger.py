from __future__ import annotations

import numpy as np

from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode
from ..core.o_port import OPort

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Trigger(IONode):
    """Event-triggered data extraction node for BCI applications.

    Monitors trigger events and extracts time-locked data segments around
    trigger occurrences. Maintains a rolling buffer of input data and outputs
    complete data epochs when target trigger values are detected. Commonly
    used in event-related potential (ERP) analysis.
    """

    #: The marker value a trigger reacts to. A tuple, copied to a list
    #: at use: the constructor coerces a scalar into a list, so this is
    #: the already-coerced shape.
    DEFAULT_TARGET: tuple = (1,)

    #: Two inputs, so the port names are declared rather than left to
    #: the default "one input called in". These are the same names the
    #: constructor builds its IPort configurations from.
    INPUT_PORT_NAMES = ("in", "trigger")

    #: Default pre-trigger window in seconds
    DEFAULT_TIME_PRE = 0.7
    #: Default post-trigger window in seconds
    DEFAULT_TIME_POST = 0.2

    #: Port name for trigger input
    PORT_TRIGGER = "trigger"

    class Configuration(IONode.Configuration):
        """Configuration class for Trigger parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for the Trigger."""

            #: Pre-trigger time configuration key
            TIME_PRE = "time_pre"
            #: Post-trigger time configuration key
            TIME_POST = "time_post"
            #: Target trigger value configuration key
            TARGET = "target"

    def __init__(
        self,
        time_pre: float = None,
        time_post: float = None,
        target: float = None,
        **kwargs,
    ):
        """Initialize the Trigger node with timing and target configurations.

        Args:
            time_pre: Time in seconds before trigger to include in epoch.
                Must be > 0. Defaults to 0.7 seconds.
            time_post: Time in seconds after trigger to include in epoch.
                Must be > 0. Defaults to 0.2 seconds.
            target: Trigger value(s) that cause epoch extraction. Can be
                single value or list. Defaults to [1].
            **kwargs: Additional configuration parameters passed to IONode.

        Raises:
            ValueError: If time_pre or time_post is <= 0.
        """
        # Set default values if not provided
        if time_pre is None:
            time_pre = self.DEFAULT_TIME_PRE
        if time_post is None:
            time_post = self.DEFAULT_TIME_POST
        if target is None:
            target = list(self.DEFAULT_TARGET)

        # Ensure target is always a list for consistent handling
        if type(target) is not list:
            target = [target]

        # Validate timing parameters
        if time_pre <= 0:
            raise ValueError("time_pre must be greater than 0.")
        if time_post <= 0:
            raise ValueError("time_post must be greater than 0.")

        # Configure input ports: data port and trigger port
        input_ports = [
            IPort.Configuration(),  # Main data input
            IPort.Configuration(
                name=self.PORT_TRIGGER, timing=Constants.Timing.INHERITED
            ),
        ]
        input_ports = kwargs.pop(Constants.Keys.INPUT_PORTS, input_ports)

        # Configure output port with asynchronous timing (trigger-dependent)
        output_ports = [OPort.Configuration(timing=Constants.Timing.ASYNC)]
        output_ports = kwargs.pop(Constants.Keys.OUTPUT_PORTS, output_ports)

        # Initialize parent IONode with all configurations
        super().__init__(
            time_pre=time_pre,
            time_post=time_post,
            target=target,
            input_ports=input_ports,
            output_ports=output_ports,
            **kwargs,
        )

        # Initialize internal state variables
        self._buf_input = None  # Rolling input data buffer
        self._buf_output = None  # Output buffer (legacy, unused)
        self._frame_size = None  # Total epoch frame size
        self._target = None  # Target trigger values
        self._countdown = None  # List of active countdown timers
        self._previous = None  # Last trigger value seen, for edge detection
        self._samples_pre = None  # Pre-trigger samples count
        self._samples_post = None  # Post-trigger samples count

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Set up the Trigger node and initialize internal buffers.

        Validates input requirements, calculates buffer sizes based on
        sampling rate and timing parameters, and initializes the rolling
        input buffer for data collection.

        Args:
            data: Initial data dictionary for port configuration.
            port_context_in: Input port context with sampling rates, frame
                sizes, and channel counts.

        Returns:
            Output port context with updated frame size and timing
            information for the extracted epochs.

        Raises:
            ValueError: If input frame size is not 1 or sampling rate is
                not provided.
        """
        # Call parent setup to get base output context
        port_context_out = super().setup(data, port_context_in)

        # Frames of any length are accepted. This used to require one
        # sample per frame, because the trigger line was read as a single
        # value per cycle and there was no way to tell *which* sample of
        # a longer frame it belonged to. An ERP paradigm at 500 Hz
        # therefore ran the whole graph 500 times a second.
        #
        # Placement removed that limit: a sparse trigger now arrives as a
        # grid aligned to this frame, one row per sample, so the sample
        # an event happened on is simply its row.
        frame_size_in = port_context_in[PORT_IN][Constants.Keys.FRAME_SIZE]
        if isinstance(frame_size_in, list):
            frame_size_in = frame_size_in[0]
        self._frame_size_in = int(frame_size_in or 1)

        # Get configuration parameters
        tpre_key = self.Configuration.Keys.TIME_PRE
        tpost_key = self.Configuration.Keys.TIME_POST
        time_pre = self.config[tpre_key]
        time_post = self.config[tpost_key]

        # Get sampling rate for time-to-samples conversion
        sampling_rate = port_context_in[PORT_IN][Constants.Keys.SAMPLING_RATE]
        if sampling_rate is None:
            raise ValueError("Sampling rate must be provided in context.")

        # Convert time windows to sample counts
        self._samples_pre = int(round(time_pre * sampling_rate))
        self._samples_post = int(round(time_post * sampling_rate))
        frame_size_out = self._samples_pre + self._samples_post

        # Update output port context with epoch specifications
        cc_key = Constants.Keys.CHANNEL_COUNT
        fsz_key = Constants.Keys.FRAME_SIZE
        timing_key = OPort.Configuration.Keys.TIMING

        # Get channel count and timing from input context. The merge keeps
        # a key that differs between ports as a per-port mapping and
        # collapses it to a bare value when the ports agree, so both forms
        # have to be accepted: a synchronous trigger line makes the two
        # ports agree, and indexing a bare value raises.
        channel_count = port_context_out[PORT_OUT][cc_key]
        timing = port_context_out[PORT_OUT][timing_key]
        if isinstance(timing, dict):
            timing = timing[PORT_IN]

        # Set output context values
        port_context_out[PORT_OUT][cc_key] = channel_count
        port_context_out[PORT_OUT][fsz_key] = frame_size_out
        port_context_out[PORT_OUT][tpre_key] = time_pre
        port_context_out[PORT_OUT][tpost_key] = time_post
        port_context_out[PORT_OUT][timing_key] = timing

        # Initialize internal buffers and state
        self._buf_input = np.zeros(shape=(frame_size_out, channel_count))
        self._buf_output = []  # Legacy buffer, kept for compatibility
        self._frame_size = frame_size_out
        self._target = self.config[self.Configuration.Keys.TARGET]
        self._countdown = []  # List of active trigger countdowns
        #: Epochs already complete but not yet emitted. Snapshotted the
        #: moment they complete rather than deferred, because a longer
        #: frame can complete two epochs in one cycle and the rolling
        #: buffer would have moved past the earlier one by the time the
        #: second was asked for.
        self._ready = []
        self._previous = None  # No edge before the first observation

        # Whether an epoch starts on a transition or on every arrival.
        # A sparse trigger stream delivers one value per event, so the
        # arrival is the event. A continuously sampled one carries a level
        # that persists for as long as the line is asserted, and matching
        # the level would restart an epoch on every sample.
        trigger_ctx = port_context_in.get(self.PORT_TRIGGER, {})
        timing = trigger_ctx.get(IPort.Configuration.Keys.TIMING)
        self._edge_triggered = timing != Constants.Timing.ASYNC

        return port_context_out

    def _trigger_at(self, triggers, row: int, rows: int):
        """Return the trigger value for one sample of the frame.

        Args:
            triggers: The trigger port's frame, or None if it produced
                nothing this cycle.
            row: Index of the sample within the data frame.
            rows: Length of the data frame.

        Returns:
            The value, or None when this sample has no trigger to read.
        """
        if triggers is None:
            return None

        array = np.asarray(triggers)
        if array.ndim == 0:
            # A bare value for the whole cycle, which is what a caller
            # driving the node directly supplies.
            return array.item() if row == 0 else None
        if array.ndim == 1:
            if array.shape[0] == rows:
                return array[row]
            return array[0] if row == 0 and array.size else None
        if array.ndim == 2:
            if array.shape[0] == rows:
                # Aligned: one trigger sample per data sample, which is
                # what placement produces and what a continuous trigger
                # line is.
                return array[row, 0]
            if array.shape[0] == 1 and row == 0 and array.shape[1]:
                # A single unaligned observation. Attributed to the first
                # sample of the frame rather than spread over it: the
                # frame is the best resolution available, and naming a
                # specific later sample would invent precision.
                return array[0, 0]
        return None

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one frame of data and check for trigger events.

        Updates the rolling input buffer with new data, monitors trigger
        changes, and extracts complete epochs when countdown timers expire.
        Multiple triggers can be active simultaneously.

        Args:
            data: Dictionary containing input data arrays. Must include
                both the main data port and trigger port data.

        Returns:
            Dictionary containing extracted epoch data when a trigger
            countdown completes, None otherwise. Epoch has shape
            (samples_pre + samples_post, channel_count).
        """

        frame = data.get(PORT_IN)
        triggers = data.get(self.PORT_TRIGGER)
        rows = (
            frame.shape[0]
            if frame is not None and getattr(frame, "ndim", 0) == 2
            else 0
        )

        if rows == 0:
            # A trigger arriving on its own cycle, with no frame to place
            # it in. That is what an unplaced sparse source does -- it
            # cycles the graph when the event happens -- and the event
            # must still be recorded: it starts a countdown at the
            # current buffer position and completes on later data
            # cycles. Dropping it here silently lost every event from an
            # unplaced trigger source.
            value = self._trigger_at(triggers, 0, 1)
            if value is not None:
                if value in self._target:
                    if not self._edge_triggered or value != self._previous:
                        self._countdown.append(self._samples_post)
                self._previous = value
            if self._ready:
                return {PORT_OUT: self._ready.pop(0)}
            return None

        # One sample at a time. The per-sample logic is what makes an
        # epoch land on the right sample, and a loop over ten rows costs
        # far less than ten trips through the graph -- which is the whole
        # reason for accepting longer frames.
        for row in range(rows):
            value = self._trigger_at(triggers, row, rows)
            if value is not None:
                # An epoch starts on the transition into a target value: a
                # hardware digital input holds its level for as long as
                # the line is asserted, and matching the level would
                # start a fresh epoch on every sample for the whole
                # duration. A sparse stream placed onto the grid returns
                # to its fill value between observations, so a transition
                # is exactly one sample wide.
                if value in self._target:
                    if not self._edge_triggered or value != self._previous:
                        self._countdown.append(self._samples_post)
                self._previous = value

            self._buf_input[:-1] = self._buf_input[1:]
            self._buf_input[-1] = frame[row]

            for i in reversed(range(len(self._countdown))):
                self._countdown[i] -= 1
                if self._countdown[i] <= 0:
                    self._countdown.pop(i)
                    self._ready.append(self._buf_input.copy())

        if self._ready:
            return {PORT_OUT: self._ready.pop(0)}

        # No epochs ready - return None (asynchronous behavior)
        return None
