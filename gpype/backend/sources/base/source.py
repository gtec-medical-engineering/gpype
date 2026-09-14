from __future__ import annotations

import json
from typing import Optional, Union

import numpy as np

from ....common.constants import Constants
from ...core.o_node import ONode
from ...core.o_port import OPort

#: Default output port identifier
OUT_PORT = Constants.Defaults.PORT_OUT


class Source(ONode):
    """Base class for data source nodes in a pipeline.

    Provides foundation for all data source nodes that generate or acquire
    data. Sources have only output ports and serve as pipeline entry points.
    Handles validation of output ports, channel counts, and frame sizes.
    """

    #: What this source's sample positions advance against. Live sources
    #: keep the default; a reader that replays stored data overrides it,
    #: which is what lets a pipeline reject the two being mixed before
    #: anything runs. See Constants.TimeBase.
    TIME_BASE: str = Constants.TimeBase.WALL_CLOCK

    #: How a pipeline containing this source is driven. Live sources keep
    #: the default; a reader asked for one monolithic block overrides it
    #: per instance. Declared here rather than derived from TIME_BASE,
    #: because a recording *paced to the clock* is an ordinary realtime
    #: pipeline -- the two properties are independent.
    EXECUTION_MODE: str = Constants.ExecutionMode.REALTIME

    #: Timeline of the pipeline this source belongs to, once bound.
    _timeline = None

    #: Whether this kind of source derives a frame size from its rate
    #: when the author names none. False for sources where a frame is
    #: not a slice of a live stream: an event source emits one row when
    #: something happens, and a batch source's frame *is* the recording.
    DERIVE_FRAME_SIZE: bool = True

    #: Samples per second a frame should nominally span, as a divisor.
    #: 62.5 puts one frame at 16 ms and reproduces the frame sizes the
    #: hardware family already uses: 250 Hz -> 4, 500 and 512 -> 8,
    #: 1200 -> 16, 2400 -> 32.
    FRAME_RATE_DIVISOR: float = 62.5

    def _default_frame_size(self, kwargs: dict) -> int:
        """Frame size to use when the author named none.

        Derived from the sampling rate, so a frame spans roughly the same
        stretch of time whatever the rate. Falls back to
        ``Constants.Defaults.FRAME_SIZE`` when there is no rate to derive
        from, or when this kind of source does not take slices of a live
        stream at all.

        Args:
            kwargs (dict): The constructor's remaining keyword arguments,
                which is where ``sampling_rate`` arrives.

        Returns:
            int: Frame size, at least 1.
        """
        if not self.DERIVE_FRAME_SIZE:
            return Constants.Defaults.FRAME_SIZE
        return self.frame_size_for(kwargs.get(Constants.Keys.SAMPLING_RATE))

    @staticmethod
    def frame_size_for(sampling_rate) -> int:
        """Derived frame size, or the fallback if the rate is unusable.

        Never raises, and that is the point. A rate of -1 is the
        *rate's* problem, and the source's own validation says so
        precisely; deriving first and raising here would answer a
        question about the sampling rate with a message about frame
        sizes, which is exactly the kind of misdirection that costs a
        user their afternoon.

        A rate may also arrive per port, or be inherited rather than
        declared. Neither is something to derive from.

        Args:
            sampling_rate: Rate in Hz, a per-port list of them,
                ``Constants.INHERITED``, or None.

        Returns:
            int: Frame size, at least 1.
        """
        rate = sampling_rate
        if isinstance(rate, (list, tuple)):
            rate = rate[0] if rate else None
        if not isinstance(rate, (int, float)):
            return Constants.Defaults.FRAME_SIZE
        if rate == Constants.INHERITED or rate <= 0:
            return Constants.Defaults.FRAME_SIZE
        return Source.default_frame_size(rate)

    @staticmethod
    def default_frame_size(sampling_rate: float) -> int:
        """Frame size a source should use at this rate, if none is given.

        The power of two nearest ``sampling_rate / 62.5``, which holds one
        frame at roughly 13 to 20 ms across the whole supported range
        while keeping the size a power of two.

        Applied by :meth:`_default_frame_size` whenever a source is
        built without one and knows its rate. Event sources and batch
        sources opt out through ``DERIVE_FRAME_SIZE``.

        Two details are deliberate. The clamp: below about 44.2 Hz the
        unclamped expression returns a *fraction* (0.5 at 32 Hz), which
        ``Source.__init__`` rejects with a message about integers rather
        than about rates, and this repository really does use 1, 10 and
        50 Hz. And ``floor(x + 0.5)`` rather than ``round``: Python
        rounds halves to even, so a tie would resolve inconsistently
        between one rate and the next.

        Args:
            sampling_rate (float): Rate in Hz. Must be positive.

        Returns:
            int: Frame size, at least 1.

        Raises:
            ValueError: If ``sampling_rate`` is not positive.
        """
        if sampling_rate is None or sampling_rate <= 0:
            raise ValueError(
                f"sampling_rate must be positive to derive a frame size; "
                f"got {sampling_rate}."
            )
        exponent = np.log2(sampling_rate / Source.FRAME_RATE_DIVISOR)
        return max(1, int(2 ** int(np.floor(exponent + 0.5))))

    def attach_timeline(self, timeline) -> None:
        """Bind this source to the pipeline's timeline.

        Args:
            timeline: Timeline manager owned by the pipeline.
        """
        self._timeline = timeline

    def _publish_mastership(self, port_context_out: dict) -> None:
        """Tell downstream nodes if this source owns the timeline.

        The election is settled before any data flows, and its winner is
        a *source*. But the node that feeds the rate estimator is the
        Sync at the end of the chain, and it only does so when the
        context says this stream is the master. Without this, an elected
        source never gets observed: the timeline keeps a master and a
        rate but never becomes ready, so nothing can be placed on it.
        Only amplifiers used to publish it, which is why moving the
        election earlier broke every counted stream.

        Travels in the context rather than by reference, so a server
        learns which stream is master with no extra machinery. It does
        not survive a Link unchanged, and must not: the flag means "a
        node upstream of you already claimed", which is false in a
        process the claimant cannot reach. The receiving Link translates
        it into REMOTE_MASTER, an instruction to claim locally instead.

        The rung travels with it, so that translated claim lands on the
        same step of the ladder the source won on rather than a guessed
        one.

        Args:
            port_context_out: Contexts to annotate, modified in place.
        """
        timeline = self._timeline
        if timeline is None or timeline.master is not self:
            return
        for context in port_context_out.values():
            context[Constants.Keys.MASTER_TIMELINE] = True
            context[Constants.Keys.MASTER_PRIORITY] = timeline.master_priority

    @property
    def is_exhausted(self) -> bool:
        """Whether this source has emitted everything it will.

        False for a live source, because a clock does not end. A reader
        overrides it, which is what lets a batch driver know when to stop
        asking -- without it, the driver would have to read a reader's
        private state or guess a cycle count.

        Returns:
            True when nothing further will be emitted.
        """
        return False

    def master_candidacy(self):
        """Return this source's claim to own the master timeline.

        Declared at construction, so the pipeline can settle the
        election before any data flows. That timing is the whole point:
        a source that only claims when its first block arrives loses to
        whichever source starts producing soonest, and a synthetic
        source always beats real hardware that has to open a device
        first. Measured against a g.HIamp: the amplifier lost every
        time.

        Returns:
            ``(sampling_rate, priority)``, or None for a source that
            cannot serve as master. Sparse sources return None: they
            have no rate of their own and are placed onto the master's
            timeline rather than defining one.
        """
        return None

    @staticmethod
    def scalar(value):
        """Return a per-port configuration value as a single number.

        Values like channel_count and frame_size are stored per output
        port, so they are normalised to a list on the way into the
        configuration. A source rebuilt from that configuration receives
        the list form back, and constructors that expect a number would
        otherwise fail or, worse, pass the list on unnoticed.

        Args:
            value: A number, a one-element sequence of numbers, or None.

        Returns:
            The number, or None if there was none.
        """
        if isinstance(value, (list, tuple)):
            return value[0] if value else None
        return value

    class Configuration(ONode.Configuration):
        """Configuration class for Source parameters."""

        class Keys(ONode.Configuration.Keys):
            """Configuration keys for source-specific settings."""

            #: Configuration key for number of channels per port
            CHANNEL_COUNT = Constants.Keys.CHANNEL_COUNT
            #: Configuration key for samples per frame
            FRAME_SIZE = Constants.Keys.FRAME_SIZE

    def __init__(
        self,
        output_ports: Optional[list] = None,
        channel_count: Optional[Union[list, int]] = None,
        frame_size: Optional[Union[list, int]] = None,
        **kwargs,
    ):
        """Initialize source with output port configuration.

        Args:
            output_ports: List of output port configurations. Required.
            channel_count: Number of channels per port. Can be int (all ports)
                or list (per port). Defaults to 1. Must be >= 1 or INHERITED.
            frame_size: Samples per frame. Can be int (all ports) or list
                (per port). Defaults to 1. Must be >= 1 or INHERITED.
            **kwargs: Additional arguments for parent ONode class.

        Raises:
            ValueError: If validation fails or input_ports specified.
        """
        # Validate that output_ports is provided (required for sources)
        if output_ports is None:
            raise ValueError("output_ports must be defined.")

        # Validate and normalize channel_count parameter
        if channel_count is None:
            # Default to 1 channel per output port
            channel_count = [1] * len(output_ports)
        elif isinstance(channel_count, int):
            # Convert single int to list for all ports
            channel_count = [channel_count]

        # Validate channel_count values
        if not all(isinstance(c, int) for c in channel_count):
            raise ValueError("All elements of channel_count must be integers.")
        if not all(c == Constants.INHERITED or c >= 1 for c in channel_count):
            raise ValueError(
                "All elements of channel_count must be greater or equal 1."
            )
        if len(output_ports) != len(channel_count):
            raise ValueError(
                "output_ports and channel_count must have the same length."
            )

        # Validate and normalize frame_size parameter
        if frame_size is None:
            frame_size = [self._default_frame_size(kwargs)] * len(output_ports)
        elif isinstance(frame_size, int):
            # Convert single int to list for all ports
            frame_size = [frame_size] * len(output_ports)

        # Validate frame_size values
        if not all(isinstance(f, int) for f in frame_size):
            raise ValueError("All elements of frame_size must be integers.")
        if not all(f == Constants.INHERITED or f >= 1 for f in frame_size):
            raise ValueError(
                "All elements of frame_size must be greater or equal 1."
            )

        # Check frame_size consistency (all non-inherited values must be equal)
        non_inherited_frames = [
            fsz for fsz in set(frame_size) if fsz != Constants.INHERITED
        ]
        if len(non_inherited_frames) != 1:
            raise ValueError("All elements of frame_size must be equal.")
        if len(output_ports) != len(frame_size):
            raise ValueError(
                "output_ports and frame_size must have the same length."
            )

        # Sources cannot have input ports by design
        if "input_ports" in kwargs:
            raise ValueError("Source must not have input ports.")

        # Initialize parent ONode with validated parameters
        ONode.__init__(
            self,
            output_ports=output_ports,
            channel_count=channel_count,
            frame_size=frame_size,
            **kwargs,
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output port contexts with channel and frame information.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Dictionary of output port contexts with channel_count and
            frame_size information for each output port.
        """
        # Call parent setup method to initialize base contexts
        port_context_out = super().setup(data, port_context_in)

        # Get configuration parameters
        channel_count = self.config[self.Configuration.Keys.CHANNEL_COUNT]
        frame_size = self.config[self.Configuration.Keys.FRAME_SIZE]
        out_ports = self.config[self.Configuration.Keys.OUTPUT_PORTS]

        # Configure context for each output port
        for i in range(len(out_ports)):
            # Create context with channel and frame information
            context = {
                Constants.Keys.CHANNEL_COUNT: channel_count[i],
                Constants.Keys.FRAME_SIZE: frame_size[i],
                Constants.Keys.EXECUTION_MODE: self.EXECUTION_MODE,
            }

            # Published where the source knows it, so a node declaring
            # its output shape in setup() can size a whole-record result
            # without waiting for the data. A live source never knows.
            total = getattr(self, "sample_count", None)
            if total is not None:
                context[Constants.Keys.SAMPLE_COUNT] = int(total)

            # Get port name and update its context
            port_name = out_ports[i][OPort.Configuration.Keys.NAME]
            port_context_out[port_name].update(context)

        self._publish_mastership(port_context_out)

        self.log(f"Source setup complete with {json.dumps(port_context_out)}")
        return port_context_out
