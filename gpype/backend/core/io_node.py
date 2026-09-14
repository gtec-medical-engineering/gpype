from __future__ import annotations

import copy

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core.i_port import IPort
from ._private.placement import PlacingMixin
from .node import Node


class IONode(PlacingMixin, ioc.IONode, Node):
    """Abstract base class for input/output nodes in the g.Pype pipeline.

    Combines ioiocore.IONode and Node functionality for signal processing
    nodes with input and output ports. Handles validation and setup logic
    for port contexts. Subclasses must implement the abstract step() method.
    """

    def __init__(
        self,
        input_ports: list[ioc.IPort.Configuration] = None,
        output_ports: list[ioc.OPort.Configuration] = None,
        **kwargs,
    ):
        """Initialize the IONode with input and output port configurations.

        Args:
            input_ports: List of input port configurations or None.
            output_ports: List of output port configurations or None.
            **kwargs: Additional arguments passed to parent classes.
        """
        ioc.IONode.__init__(
            self, input_ports=input_ports, output_ports=output_ports, **kwargs
        )
        Node.__init__(self, target=self)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the node before processing begins.

        Validates input port configurations and creates output port contexts.
        Checks for consistent sampling rates, compatible channel counts,
        matching frame sizes, and compatible types.

        Args:
            data: Dictionary mapping port names to numpy arrays.
            port_context_in: Dictionary mapping input port names to contexts.

        Returns:
            Dictionary mapping output port names to context dictionaries.

        Raises:
            ValueError: If validation fails for any configuration parameter.
        """
        # The name the author typed, not the class that is running: a
        # message naming "_BandpassCore" sends the reader looking for a
        # symbol the public API does not have. self.name defaults to the
        # class name, so it adds something only once an author has set
        # one -- and then it is the only way to tell two Bandpasses in
        # the same pipeline apart.
        label = node_label(self)

        # Validate required keys are present in all input port contexts.
        # Reported per port and per key: "in context" left the reader to
        # work out which of a Router's eight inputs arrived incomplete.
        required = (
            Constants.Keys.CHANNEL_COUNT,
            Constants.Keys.FRAME_SIZE,
            IPort.Configuration.Keys.TIMING,
        )
        for port, context in port_context_in.items():
            for key in required:
                if key not in context:
                    got = ", ".join(sorted(context)) or "an empty context"
                    raise ValueError(
                        f"{key} must be provided in context; port "
                        f"'{port}' of {label} got {got}. That context "
                        f"is what the node upstream returned from its "
                        f"own setup(); add {key} to the dictionary it "
                        f"returns."
                    )

        # Validate sampling rates - all ports must have the same rate.
        #
        # The two ways this fails read nothing alike, so they say
        # different things: a graph where no port declares a rate is
        # missing its source, while a graph declaring two is the
        # 250 Hz/500 Hz mix that needs a Decimator. Both name the ports,
        # because "all ports" identifies no pair once a pipeline has
        # more than two of them.
        sr_key = Constants.Keys.SAMPLING_RATE
        rates = {
            name: md[sr_key]
            for name, md in port_context_in.items()
            if md.get(sr_key) is not None
        }
        if not rates:
            ports = ", ".join(f"'{n}'" for n in port_context_in)
            raise ValueError(
                f"No input port of {label} declares a sampling rate; "
                f"got {ports or 'no connected input ports'}. The rate "
                f"enters a pipeline at its source and travels "
                f"downstream in the port context, so a node without "
                f"one is not reading from a source. Connect {label} "
                f"downstream of a source such as Generator or an "
                f"amplifier."
            )
        if len(set(rates.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in rates.items())
            raise ValueError(
                f"All ports must have the same sampling rate; {label} "
                f"got {detail}. One step consumes one frame from every "
                f"input at once, so the inputs cannot run at different "
                f"rates. Insert a Decimator on the faster branch, or "
                f"take both inputs from the same source."
            )

        # Validate and normalize channel counts of the continuous ports
        # Allow broadcasting: single-channel ports can be broadcast to
        # multi-channel ports
        #
        # Sparse ports are exempt from both, because their width
        # annotates their own events rather than the signal they
        # accompany: a 2-channel marker beside an 8-channel signal gives
        # the three counts {8, 2, 1} and was refused outright. Exempting
        # them from the broadcast is observable too - such a port used to
        # report the signal's width once setup() had run and now keeps
        # the one it declared. Sparse here means ASYNC, not "anything but
        # SYNC": a port left at INHERITED is continuous once the graph
        # resolves its timing, and exempting Trigger's trigger port left
        # the output count a per-port map that np.zeros refused.
        cc_key = Constants.Keys.CHANNEL_COUNT
        tim_key = IPort.Configuration.Keys.TIMING

        # Declared timing as well as the context's, because by the time
        # this runs the context may no longer say what the port is.
        # PlacingMixin.pre_setup rewrites a *placed* sparse port to SYNC
        # -- that rewrite is what puts it on the signal's grid, and it
        # has to happen before setup() so a node sizes its output from
        # what it will actually receive. The side effect was that the
        # port arrived here looking continuous, so a placed 2-channel
        # marker bank beside an 8-channel signal was refused for
        # disagreeing about a width it was never describing.
        #
        # Only ASYNC counts, not "anything but SYNC": a port left at
        # INHERITED is continuous once the graph resolves its timing, and
        # exempting Trigger's INHERITED trigger port made the output
        # count a per-port map that np.zeros refused.
        declared = {}
        for cfg in (
            self.config.get(self.Configuration.Keys.INPUT_PORTS, []) or []
        ):
            try:
                declared[cfg[IPort.Configuration.Keys.NAME]] = cfg.get(tim_key)
            except (TypeError, KeyError):  # pragma: no cover
                continue

        def _is_sparse(name, md) -> bool:
            if declared.get(name) == Constants.Timing.ASYNC:
                return True
            return md[tim_key] == Constants.Timing.ASYNC

        # Keyed by port, because the message below has to name the two
        # widths that disagree; "all continuous ports" names none.
        continuous = {
            name: md
            for name, md in port_context_in.items()
            if not _is_sparse(name, md)
        }
        counts = {
            name: md[cc_key]
            for name, md in continuous.items()
            if md.get(cc_key) is not None
        }
        channel_counts = list(counts.values())
        # None when every input is sparse, which keeps such a node's own
        # width below instead of overwriting it with the broadcast unit.
        width = max(channel_counts) if channel_counts else None
        channel_counts.append(1)  # add single channel for comparison
        if len(set(channel_counts)) > 2:
            # More than 2 unique values means incompatible multi-channel
            # ports: one real width plus the broadcastable 1 is the most
            # that can be reconciled.
            detail = ", ".join(f"{n}={v}" for n, v in counts.items())
            raise ValueError(
                f"All continuous ports must have the same channel "
                f"count; {label} got {detail}. A port carrying one "
                f"channel is broadcast to the widest port, but two "
                f"different multi-channel widths cannot be combined. "
                f"Give the ports the same channel count, or select "
                f"channels with a Router before {label}."
            )

        # Broadcast single channels to maximum channel count
        for md in continuous.values():
            if md.get(cc_key) is not None:
                md[cc_key] = max(channel_counts)  # set to max (broadcast)

        # Validate frame sizes - all SYNC ports must have the same one.
        fsz_key = Constants.Keys.FRAME_SIZE
        sizes = {
            name: md[fsz_key]
            for name, md in port_context_in.items()
            if md[IPort.Configuration.Keys.TIMING] == Constants.Timing.SYNC
            and md.get(fsz_key) is not None
        }
        if len(set(sizes.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in sizes.items())
            raise ValueError(
                f"All ports must have the same frame size; {label} got "
                f"{detail}. Every input contributes one frame to one "
                f"step, so frames of different lengths cannot be "
                f"combined. Give both sources the same frame_size, or "
                f"insert a Framer on the branch with the shorter one."
            )

        # Validate port types - all ports must have compatible types.
        # "Any" is compatible with everything, so it is left out of the
        # comparison and out of the message.
        type_key = IPort.Configuration.Keys.TYPE
        types = {
            name: md[type_key]
            for name, md in port_context_in.items()
            if md.get(type_key) not in (None, "Any")
        }
        if len(set(types.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in types.items())
            raise ValueError(
                f"All ports must have the same type; {label} got "
                f"{detail}. One step reads every input into the same "
                f"array, so the ports have to carry the same payload "
                f"type. Connect ports that declare the same type, or "
                f"declare 'Any' on the port that should accept "
                f"whatever arrives."
            )

        # Build output port contexts by merging input port contexts
        port_context_out: dict[str, dict] = {}
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        name_key = IPort.Configuration.Keys.NAME
        context = {}

        # Get all unique keys from all input port contexts.
        #
        # The per-channel description is excluded and merged separately
        # below. The generic rule here falls back to a {port: value} map
        # when ports disagree, which is right for a key like timing that
        # is genuinely per-port, and wrong for a key that has to be one
        # list per channel: every reader of channel_roles would get a
        # dict of two entries where it expects one entry per channel.
        all_keys = set().union(*port_context_in.values())
        all_keys -= channels.DESCRIPTION_KEYS

        # For each key, determine how to merge values from different ports
        for key in all_keys:
            values = {}
            for port, config in port_context_in.items():
                if key in config:
                    values[port] = config[key]

            value_list = list(values.values())

            # Case 1: Key exists in only one port
            # → use that value directly
            if len(values) == 1:
                context[key] = value_list[0]

            # Case 2: Key exists in multiple ports with identical values
            # → use the common value
            elif all(value == value_list[0] for value in value_list[1:]):
                context[key] = copy.deepcopy(value_list[0])

            # Case 3: Key exists in multiple ports with different values
            # → keep per-port mapping
            else:
                context[key] = values

        # The output is a continuous stream, so it carries the continuous
        # width. The generic merge above turns channel_count into a
        # {port: value} map as soon as a sparse port declares a different
        # width, and a consumer sizing a buffer from that - Trigger does -
        # fails with "'dict' object cannot be interpreted as an integer".
        if width is not None:
            context[cc_key] = width

        # Merge the per-channel description by its own rules, against the
        # channel count the output actually has.
        context.update(
            channels.reconcile(
                list(port_context_in.values()),
                context.get(cc_key, 0),
            )
        )

        # Apply the merged context to all output ports, each getting its
        # own copy. Sharing one dict makes every port - and, for a key
        # taken from a single input, the input context itself - alias the
        # same lists, so a node that adjusts its own description would
        # reach back and rewrite its neighbours'.
        for op in self.config[op_key]:
            port_context_out[op[name_key]] = copy.deepcopy(context)
        return port_context_out
