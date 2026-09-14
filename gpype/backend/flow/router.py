from __future__ import annotations

import copy
from typing import Union

import ioiocore as ioc
import numpy as np

from ...common._private import channels
from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode
from ..core.o_port import OPort


class Router(IONode):
    """Channel routing and selection node for flexible data flow management.

    Provides channel selection and routing capabilities for BCI data pipelines.
    Allows selecting specific channels from multiple input ports and routing
    them to multiple output ports. Supports both simple channel selection
    (index lists) and complex multi-port routing (dictionary mapping).
    """

    #: Both port sets come from the keys of the channel maps: given
    #: ``input_channels={"a": [0], "b": [1]}`` the inputs are 'a' and
    #: 'b'. With neither map the node has the default single in/out,
    #: which is the shape the catalog records.
    INPUT_PORTS_FROM = "input_channels"
    OUTPUT_PORTS_FROM = "output_channels"

    #: Special constant for selecting all available channels
    ALL: list = [-1]

    #: One port carrying every channel, which is what a Router does
    #: when told nothing. Copied at use, not shared: a class attribute
    #: handed out as a default would be mutated by whoever received it.
    DEFAULT_INPUT_CHANNELS: dict = {Constants.Defaults.PORT_IN: ALL}
    DEFAULT_OUTPUT_CHANNELS: dict = {Constants.Defaults.PORT_OUT: ALL}

    # Type annotation for the internal routing map
    _map: dict

    # Type annotation for output channel counts
    _channel_count_out: dict

    #: Per output port, the input ports it draws from. Readiness is
    #: decided per port from this, not globally: one unrelated input
    #: with nothing to say used to silence every output.
    _contributors: dict

    class Configuration(ioc.IONode.Configuration):
        """Configuration class for Router parameters."""

        class Keys(ioc.IONode.Configuration.Keys):
            """Configuration key constants for the Router."""

            #: Input channels configuration key
            INPUT_CHANNELS = "input_channels"
            #: Output channels configuration key
            OUTPUT_CHANNELS = "output_channels"

    def __init__(
        self,
        input_channels: Union[list, dict] = None,
        output_channels: Union[list, dict] = None,
        **kwargs,
    ):
        """Initialize the Router node with channel selection configurations.

        Args:
            input_channels: Specification for input channel selection. Can be
                None (all channels), list (channel indices), or dict (port
                name to channel indices mapping).
            output_channels: Specification for output channel selection.
                Same format as input_channels.
            **kwargs: Additional configuration parameters passed to IONode.

        Raises:
            ValueError: If input_channels or output_channels is empty.
        """
        # Set default input channels to all channels on default port
        if input_channels is None:
            input_channels = dict(Router.DEFAULT_INPUT_CHANNELS)

        # Convert list format to dictionary format for input channels
        if type(input_channels) is list:
            if len(input_channels) == 0:
                raise ValueError("input_channels must not be empty.")
            # Convert single list to list of lists if needed
            if type(input_channels[0]) is not list:
                input_channels = [input_channels]
            # Create port mappings
            if len(input_channels) == 1:
                input_channels = {
                    Constants.Defaults.PORT_IN: input_channels[0]
                }
            else:
                input_channels = {
                    f"in{i + 1}": val for i, val in enumerate(input_channels)
                }

        # Create input port configurations
        input_ports = [
            IPort.Configuration(name=name, timing=Constants.Timing.INHERITED)
            for name in input_channels.keys()
        ]
        input_ports = kwargs.pop(
            Router.Configuration.Keys.INPUT_PORTS, input_ports
        )

        # Set default output channels to all channels on default port
        if output_channels is None:
            output_channels = dict(Router.DEFAULT_OUTPUT_CHANNELS)

        # Convert list format to dictionary format for output channels
        if type(output_channels) is list:
            if len(output_channels) == 0:
                raise ValueError("output_channels must not be empty.")
            # Convert single list to list of lists if needed
            if type(output_channels[0]) is not list:
                output_channels = [output_channels]
            # Create port mappings
            if len(output_channels) == 1:
                output_channels = {
                    Constants.Defaults.PORT_OUT: output_channels[0]
                }
            else:
                output_channels = {
                    f"out{i + 1}": val for i, val in enumerate(output_channels)
                }

        # Create output port configurations
        output_ports = [
            OPort.Configuration(name=name) for name in output_channels.keys()
        ]
        output_ports = kwargs.pop(
            Router.Configuration.Keys.OUTPUT_PORTS, output_ports
        )

        # Initialize internal routing map
        self._map = {}

        # Store output channel counts for use in step()
        self._channel_count_out: dict = {}

        # Which inputs feed each output, filled in during setup.
        self._contributors: dict = {}

        # Initialize parent IONode with all configurations
        IONode.__init__(
            self,
            input_channels=input_channels,
            output_channels=output_channels,
            input_ports=input_ports,
            output_ports=output_ports,
            **kwargs,
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Set up the Router node and build the internal channel mapping.

        Creates the internal routing map that defines which input channels
        are connected to which output channels. Validates that all input
        ports have compatible sampling rates, frame sizes, and data types.

        Args:
            data: Initial data dictionary for port configuration.
            port_context_in: Input port context information with channel
                counts, sampling rates, frame sizes, and data types.

        Returns:
            Output port context with routing information and updated
            channel counts for each output port.

        Raises:
            ValueError: If input ports have incompatible parameters.
        """
        # Get commonly used configuration keys
        cc_key = Constants.Keys.CHANNEL_COUNT
        name_key = IPort.Configuration.Keys.NAME

        # Build input channel mapping
        input_map: list = []
        ip_key = Router.Configuration.Keys.INPUT_PORTS
        ic_key = Router.Configuration.Keys.INPUT_CHANNELS

        # Process each input port to build channel mapping
        for k in range(len(self.config[ip_key])):
            port = self.config[ip_key][k]
            name = port[name_key]
            sel = self.config[ic_key][name]

            # Expand "ALL" to actual channel range
            available = port_context_in[name].get(cc_key)
            if sel == Router.ALL:
                sel = range(available)
            elif available is not None:
                # Reject an out-of-range index here rather than letting the
                # copy fail later: the copy is per group, so one bad index
                # silently left a whole output port full of zeros.
                bad = [n for n in sel if n < 0 or n >= available]
                if bad:
                    raise ValueError(
                        f"input_channels for port '{name}' selects "
                        f"channel(s) {bad}, but the port has "
                        f"{available} channel(s)."
                    )
            # Add each selected channel to the input map
            input_map.extend([{name: [n]} for n in sel])

        # Build output port mapping using input map
        op_key = Router.Configuration.Keys.OUTPUT_PORTS
        oc_key = Router.Configuration.Keys.OUTPUT_CHANNELS

        for k in range(len(self.config[op_key])):
            port = self.config[op_key][k]
            name = port[name_key]
            sel = self.config[oc_key][name]

            # Expand "ALL" to full input map range
            if sel == Router.ALL:
                sel = range(len(input_map))
            # Map selected input channels to this output port
            map = [input_map[n] for n in sel]
            map_grouped = []
            from itertools import groupby

            for key, group in groupby(map, key=lambda x: list(x.keys())[0]):
                values = []
                for item in group:
                    values.extend(item[key])
                map_grouped.append({key: values})
            self._map[name] = map_grouped

        # An asynchronous input reaching a router is a configuration
        # problem, not something to paper over.
        #
        # Placement rewrites a positioned async port to SYNC *before*
        # setup() runs, so anything still declaring ASYNC here is a
        # sparse stream that carries no position -- and there is no
        # honest value to route from it between observations. This node
        # used to fill the gap with zeros and then repeat the last value
        # forever, which reported a trigger on every frame after the
        # first one.
        timing_key = IPort.Configuration.Keys.TIMING
        unplaced = sorted(
            name
            for name, context in port_context_in.items()
            if context.get(timing_key, Constants.Timing.SYNC)
            == Constants.Timing.ASYNC
        )
        if unplaced:
            raise ValueError(
                f"Router received asynchronous input on "
                f"{', '.join(repr(n) for n in unplaced)}, which carries "
                f"no master-timeline position and so cannot be placed on "
                f"a sample grid: between observations there is no value "
                f"to route, and inventing one reports an event at a time "
                f"it did not happen. Every source built on EventSource "
                f"gets a position from the Sync in its chain and is "
                f"placed automatically; a node emitting sparse frames "
                f"without one cannot be combined with a continuous "
                f"stream."
            )

        # Which inputs each output actually draws from, so readiness is
        # decided per output port.
        self._contributors = {
            name: {port for group in mapping for port in group}
            for name, mapping in self._map.items()
        }

        # Every port is synchronous by this point -- the check above
        # refused anything else -- so these apply to all of them rather
        # than to a subset.
        # Router keeps its own copies of the port-contract checks
        # because it accepts inputs IONode would have refused. They name
        # the same things IONode's do: a message that says only "all
        # ports" identifies no pair once a Router has eight of them, and
        # a Router is the node most users reach a mismatch through.
        label = node_label(self)

        sr_key = Constants.Keys.SAMPLING_RATE
        rates = {
            name: md[sr_key]
            for name, md in port_context_in.items()
            if md.get(sr_key) is not None
        }
        if len(set(rates.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in rates.items())
            raise ValueError(
                f"All ports must have the same sampling rate; {label} "
                f"got {detail}. One step reads one frame from every "
                f"input at once, so the inputs cannot run at different "
                f"rates. Insert a Decimator on the faster branch, or "
                f"take both inputs from the same source."
            )
        sr = next(iter(rates.values()), None)

        fsz_key = Constants.Keys.FRAME_SIZE
        sizes = {
            name: md[fsz_key]
            for name, md in port_context_in.items()
            if md.get(fsz_key) is not None
        }
        if len(set(sizes.values())) > 1:
            detail = ", ".join(f"{n}={v}" for n, v in sizes.items())
            raise ValueError(
                f"All ports must have the same frame size; {label} got "
                f"{detail}. Frames are combined row for row, so a "
                f"shorter one has nothing to pair with. Set the same "
                f"frame_size on both sources, or put a Framer on the "
                f"branch with the smaller frame."
            )
        fsz = next(iter(sizes.values()), 1)

        # Validate data type consistency across all input ports
        type_key = IPort.Configuration.Keys.TYPE
        types = [md.get(type_key, None) for md in port_context_in.values()]
        types = [tp for tp in types if (tp != "Any" and tp is not None)]
        if len(set(types)) > 1:
            raise ValueError("All ports must have the same type.")
        tp = types[0] if len(types) > 0 else "Any"

        # Build output port context information
        port_context_out: dict[str, dict] = {}
        cc_key = Constants.Keys.CHANNEL_COUNT
        op_key = self.Configuration.Keys.OUTPUT_PORTS
        name_key = OPort.Configuration.Keys.NAME
        timing_key = OPort.Configuration.Keys.TIMING

        for op in self.config[op_key]:
            context = {}
            # Get all input ports referenced by this output
            in_ports = {key for d in self._map[op[name_key]] for key in d}

            # Copy context from input ports with unique naming
            for key1 in in_ports:
                full_key = self.name + "_" + key1
                context[full_key] = {}
                for key2 in port_context_in[key1]:
                    # Skip ID and NAME keys from input context
                    if key2 in [
                        IPort.Configuration.Keys.ID,
                        IPort.Configuration.Keys.NAME,
                    ]:
                        continue
                    # Deep copy to prevent reference leakage
                    context[full_key][key2] = port_context_in[key1][key2]

            # Set output port context with validated values
            context[cc_key] = sum(
                len(d[key]) for d in self._map[op[name_key]] for key in d
            )
            context[sr_key] = sr
            context[fsz_key] = fsz
            context[type_key] = tp
            context[timing_key] = op[timing_key]

            # Carry the per-channel description across the mapping. This
            # node exists to select and reorder channels, so losing what
            # each channel is would strip every downstream consumer -- a
            # file header, a stream's channel names, a filter deciding
            # what it may touch -- of the only thing that identifies them.
            roles: list[str] = []
            labels: list = []
            systems = set()
            any_labelled = False
            for entry in self._map[op[name_key]]:
                for port_in, ch_in in entry.items():
                    src = port_context_in[port_in]
                    src_roles = channels.roles_of(src)
                    roles.extend(src_roles[c] for c in ch_in)
                    if channels.has_labels(src):
                        any_labelled = True
                        src_labels = channels.labels_of(src)
                        labels.extend(src_labels[c] for c in ch_in)
                        systems.add(channels.montage_system(src))
                    else:
                        # Nothing to carry for these, but the inputs that
                        # do have names must not lose them: one
                        # unlabelled input used to blank the whole
                        # output, so merging a montage with a trigger
                        # threw away Fz..Pz. Numbered by their place in
                        # this output once the rest is known, which is
                        # also what keeps the name unique -- an event
                        # stream's own "Ch01" collided with the
                        # amplifier's.
                        labels.extend(None for _ in ch_in)
            if any_labelled:
                labels = [
                    name if name is not None else f"Ch{i + 1:02d}"
                    for i, name in enumerate(labels)
                ]
            system = systems.pop() if len(systems) == 1 else None
            context.update(
                channels.describe(
                    roles,
                    labels if any_labelled and labels else None,
                    system if any_labelled else None,
                )
            )
            port_context_out[op[name_key]] = context
            self._channel_count_out[op[name_key]] = context[cc_key]

        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Route channels from the inputs to each output port.

        Readiness is per output port. An output is emitted when every
        input it draws from has data this cycle, and omitted otherwise --
        so two outputs reading disjoint inputs are independent, and one
        unrelated input with nothing to say no longer silences the rest.

        Nothing is fabricated. There is no fill value, no held previous
        sample and no queue, because a router has no basis for inventing
        any of them: by the time a frame arrives here, whatever had to be
        placed on the grid has been.

        Args:
            data: Input frames keyed by port name.

        Returns:
            One frame per output port whose inputs are all present.
            Empty when none are.
        """
        data_out: dict = {}

        for port_out, mapping in self._map.items():
            sources = self._contributors[port_out]
            if any(data.get(name) is None for name in sources):
                # Not this port's turn. Emitting a partial frame would
                # mean inventing the missing half.
                continue

            frame_size = data[next(iter(sources))].shape[0]
            output_array = np.empty(
                (frame_size, self._channel_count_out[port_out]),
                dtype=Constants.DATA_TYPE,
            )

            col_idx = 0
            for group in mapping:
                for port_in, ch_in in group.items():
                    num_ch = len(ch_in)
                    output_array[:, col_idx : col_idx + num_ch] = data[
                        port_in
                    ][:, ch_in]
                    col_idx += num_ch

            data_out[port_out] = output_array

        return data_out
