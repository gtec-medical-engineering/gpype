from __future__ import annotations

import time

import numpy as np

from ....common._private import channels
from ....common.constants import Constants
from ...core.o_port import OPort
from .source import Source

# Convenience constant for default output port name
PORT_OUT = Constants.Defaults.PORT_OUT


class EventSource(Source):
    """Event-driven source for asynchronous data generation.

    Generates events in response to external triggers.
    """

    #: An event source emits one row when something happens, so a frame
    #: is not a slice of time here and there is nothing to derive a size
    #: from. See Source.DERIVE_FRAME_SIZE.
    DERIVE_FRAME_SIZE: bool = False

    def __init__(self, **kwargs):
        """Initialize event source with asynchronous output configuration.

        Args:
            **kwargs: Additional arguments for parent Source class including
                channel_count, frame_size, and other configuration parameters.
        """
        # Extract output_ports from kwargs with default async configuration
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        output_ports: list[OPort.Configuration] = kwargs.pop(
            op_key, [OPort.Configuration(timing=Constants.Timing.ASYNC)]
        )

        # Initialize parent Source with configuration
        Source.__init__(self, output_ports=output_ports, **kwargs)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Declare every output channel as a trigger.

        An event source emits occurrences, not a measured signal. Saying
        so is what stops a filter from smearing the edges it depends on:
        without a role every channel counts as signal, and a keypress run
        through a bandpass comes out as a decaying oscillation that still
        looks like data.

        Args:
            data: Input data arrays (empty for source nodes).
            port_context_in: Input port contexts (empty for source nodes).

        Returns:
            Output port contexts describing trigger channels.
        """
        port_context_out = super().setup(data, port_context_in)
        for context in port_context_out.values():
            count = channels.channel_count(context)
            # Declared as well as emitted: the frame trigger() builds
            # is this wide, and a context that disagrees is a shape
            # error in whichever node reads it next.
            #
            # Unconditional since 2026-08-26. This used to be opt-in,
            # because a two-channel trigger port beside an eight-channel
            # signal gave IONode.setup three distinct widths where it
            # allowed two. That check now exempts sparse ports, and an
            # event source's output is ASYNC, so there is nothing left
            # for the switch to protect -- and a source should stamp its
            # own samples the same way in every residency rather than
            # depending on which wrapper happened to build it.
            context[Constants.Keys.CHANNEL_COUNT] = count + 1
            roles = [Constants.ChannelRoles.TRIGGER] * count + [
                Constants.ChannelRoles.TIMESTAMP
            ]
            context.update(channels.describe(roles))
        return port_context_out

    def start(self):
        """Start event source."""

        # Call parent start method first
        Source.start(self)

        # Trigger initial cycle with empty data
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        cc_key = self.Configuration.Keys.CHANNEL_COUNT
        name_key = OPort.Configuration.Keys.NAME
        data = {}

        for op, cc in zip(self.config[op_key], self.config[cc_key]):
            # Create zero-filled array with proper data type
            data[op[name_key]] = None

        self.cycle(data)

    def stop(self):
        """Stop event source."""

        # Call parent stop method
        Source.stop(self)

    def trigger(self, value, port_name=PORT_OUT):
        """Trigger an event with the specified value.

        The event is emitted immediately, on the calling thread. It is
        deliberately not delayed to line up with a buffered signal
        stream: the observation time is what carries the event's timing,
        and a Sync node places it on the master timeline from that. A
        delay here would shift the event away from when it happened and
        destroy the very information the placement needs.

        The instant is read here, on the calling thread, and travels
        with the value. Reading it in the Sync instead times whenever
        the framework got round to the event -- which, in a distributed
        pipeline, is after it has crossed a Link.

        Args:
            port_name: Name of the output port to trigger.
            value: Event value to be transmitted. Converted to appropriate
                data type and formatted as single-sample array.
        """
        # Create data array with the event value
        data = {}
        if not isinstance(port_name, list):
            # The caller's port name, not the default: replacing it here
            # silently sent every event to the default port instead.
            port_name = [port_name]
            value = [value]
        stamp = channels.wrap_time(channels.stamp_clock())
        for i in range(len(port_name)):
            pn = port_name[i]
            pv = value[i]
            row = [pv, stamp]
            data[pn] = np.array([row], dtype=Constants.DATA_TYPE)

        self.cycle(data)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return current event data if available.

        Args:
            data: Input data dictionary (unused for source nodes).

        Returns:
            Dictionary containing event data if event is active, empty dict
            otherwise.
        """
        return data
