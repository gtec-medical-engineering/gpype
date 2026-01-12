from __future__ import annotations

from typing import Union

import ioiocore as ioc
import numpy as np

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

    #: Special constant for selecting all available channels
    ALL: list = [-1]

    # Type annotation for the internal routing map
    _map: dict

    # Type annotation for output channel counts
    _channel_count_out: dict

    # Type annotation for SYNC/ASYNC port tracking
    _sync_ports: set
    _async_ports: set

    # Type annotation for ASYNC data buffers
    _async_buffers: dict

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
            input_channels = {Constants.Defaults.PORT_IN: Router.ALL}

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

        # Set default output channels to all channels on default port
        if output_channels is None:
            output_channels = {Constants.Defaults.PORT_OUT: Router.ALL}

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

        # Initialize internal routing map
        self._map = {}

        # Store output channel counts for use in step()
        self._channel_count_out: dict = {}

        # Initialize SYNC/ASYNC port tracking
        self._sync_ports: set = set()
        self._async_ports: set = set()

        # Initialize ASYNC data buffers
        self._async_buffers: dict = {}

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
            if sel == Router.ALL:
                sel = range(port_context_in[name][cc_key])
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

        # Identify SYNC vs ASYNC ports and initialize buffers
        timing_key = IPort.Configuration.Keys.TIMING
        self._sync_ports = set()
        self._async_ports = set()
        self._async_buffers = {}

        for port_name, context in port_context_in.items():
            port_timing = context.get(timing_key, Constants.Timing.SYNC)
            if port_timing == Constants.Timing.ASYNC:
                self._async_ports.add(port_name)
                # Initialize buffer with zeros based on channel count
                cc = context.get(Constants.Keys.CHANNEL_COUNT, 1)
                self._async_buffers[port_name] = np.zeros(
                    (1, cc), dtype=Constants.DATA_TYPE
                )
            else:
                self._sync_ports.add(port_name)

        # Validate sampling rate consistency across SYNC input ports only
        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rates = [
            md.get(sr_key, None)
            for name, md in port_context_in.items()
            if name in self._sync_ports
        ]
        sampling_rates = [sr for sr in sampling_rates if sr is not None]
        if len(set(sampling_rates)) > 1:
            err_msg = "All SYNC ports must have the same sampling rate."
            raise ValueError(err_msg)
        sr = sampling_rates[0] if sampling_rates else None

        # Validate frame size consistency across SYNC input ports only
        fsz_key = Constants.Keys.FRAME_SIZE
        frame_sizes = [
            md.get(fsz_key, None)
            for name, md in port_context_in.items()
            if name in self._sync_ports
        ]
        frame_sizes = [fsz for fsz in frame_sizes if fsz is not None]
        if len(set(frame_sizes)) > 1:
            raise ValueError("All SYNC ports must have the same frame size.")
        fsz = frame_sizes[0] if frame_sizes else 1

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
                    context[full_key][key2] = port_context_in[key1][key2]

            # Set output port context with validated values
            context[cc_key] = sum(
                len(d[key]) for d in self._map[op[name_key]] for key in d
            )
            context[sr_key] = sr
            context[fsz_key] = fsz
            context[type_key] = tp
            port_context_out[op[name_key]] = context
            self._channel_count_out[op[name_key]] = context[cc_key]

        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one frame of data by routing channels according to mapping.

        Routes channels from input ports to output ports based on the channel
        mapping established during setup. Each output port receives data from
        its configured subset of input channels.

        For mixed SYNC/ASYNC inputs:
        - ASYNC data is buffered when it arrives
        - Output is only produced when SYNC data is available
        - Buffered ASYNC data is used when SYNC data triggers output

        Args:
            data: Dictionary containing input data arrays for each port.
                Keys are port names, values are arrays with shape
                (frame_size, channel_count).

        Returns:
            Dictionary containing output data arrays for each output port.
            Keys are output port names, values are arrays with shape
            (frame_size, selected_channel_count). Returns empty dict if
            no SYNC data is available.
        """
        # Update ASYNC buffers with any new ASYNC data
        for port_name in self._async_ports:
            if port_name in data and data[port_name] is not None:
                self._async_buffers[port_name] = data[port_name]

        # Check if any SYNC data is available
        sync_data_available = any(
            port_name in data and data[port_name] is not None
            for port_name in self._sync_ports
        )

        # Only produce output when SYNC data is available
        # (or if there are no SYNC ports at all)
        if not sync_data_available and len(self._sync_ports) > 0:
            return {}

        # Build merged data dict: SYNC data + buffered ASYNC data
        merged_data = {}
        for port_name in data:
            if port_name in self._sync_ports:
                merged_data[port_name] = data[port_name]
            elif port_name in self._async_ports:
                merged_data[port_name] = self._async_buffers[port_name]

        # Also include any ASYNC ports not in current data
        for port_name in self._async_ports:
            if port_name not in merged_data:
                merged_data[port_name] = self._async_buffers[port_name]

        data_out: dict = {}

        # Process each output port mapping
        for port_out, mapping in self._map.items():
            # Calculate total output channels
            channel_count = self._channel_count_out[port_out]

            # Get frame size from first available SYNC input, or default to 1
            frame_size = 1
            for port_name in self._sync_ports:
                if (
                    port_name in merged_data
                    and merged_data[port_name] is not None
                ):
                    frame_size = merged_data[port_name].shape[0]
                    break

            # Pre-allocate output array
            output_array = np.zeros(
                (frame_size, channel_count), dtype=Constants.DATA_TYPE
            )

            # Fill output array using direct slicing
            col_idx = 0
            for m in mapping:
                for port_in, ch_in in m.items():
                    try:
                        num_ch = len(ch_in)
                        port_data = merged_data.get(port_in)
                        if port_data is not None:
                            # Broadcast ASYNC data (1 row) to match frame_size
                            if port_data.shape[0] == 1 and frame_size > 1:
                                output_array[:, col_idx:col_idx + num_ch] = (
                                    np.broadcast_to(
                                        port_data[:, ch_in],
                                        (frame_size, num_ch)
                                    )
                                )
                            else:
                                output_array[:, col_idx:col_idx + num_ch] = (
                                    port_data[:, ch_in]
                                )
                        col_idx += num_ch
                    except Exception:
                        # Handle missing data with zeros
                        col_idx += len(ch_in)

            data_out[port_out] = output_array

        return data_out
