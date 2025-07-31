from __future__ import annotations

from typing import Union

import ioiocore as ioc
import numpy as np

from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode
from ..core.o_port import OPort


class Router(IONode):
    """
    Channel routing and selection node for flexible data flow management.

    The Router node provides sophisticated channel selection and routing
    capabilities for BCI data pipelines. It allows users to select specific
    channels from multiple input ports and route them to multiple output
    ports in flexible configurations.

    The router supports both simple channel selection (list of indices) and
    complex multi-port routing (dictionary mapping). It can aggregate
    channels from multiple sources and distribute them to different outputs
    based on user-defined routing rules.

    Features:
        - Multi-input, multi-output channel routing
        - Flexible channel selection with index lists
        - Automatic port naming for multiple inputs/outputs
        - Channel aggregation and distribution
        - Validation of sampling rates and frame sizes
        - Support for "ALL" channels selection

    Note:
        All input ports must have the same sampling rate, frame size,
        and data type. The router validates these constraints during setup.
    """

    # Special constant for selecting all available channels
    ALL: list = [-1]  # Take all channels

    # Type annotation for the internal routing map
    _map: dict

    class Configuration(ioc.IONode.Configuration):
        """Configuration class for Router parameters."""

        class Keys(ioc.IONode.Configuration.Keys):
            """Configuration key constants for the Router."""

            INPUT_SELECTOR = "input_selector"
            OUTPUT_SELECTOR = "output_selector"

    def __init__(
        self,
        input_selector: Union[list, dict] = None,
        output_selector: Union[list, dict] = None,
        **kwargs,
    ):
        """
        Initialize the Router node with channel selection configurations.

        Args:
            input_selector (Union[list, dict], optional): Specification for
                input channel selection. Can be:

                - None: Use all channels from default input port
                - list: Channel indices for single port or list of lists for
                  multiple ports. In case of multiple ports, their names
                  are automatically generated as 'in1', 'in2', etc.
                - dict: Port name to channel indices mapping

            output_selector (Union[list, dict], optional): Specification for
                output channel selection. Same format as input_selector.
            **kwargs: Additional configuration parameters passed to IONode.

        Raises:
            ValueError: If input_selector or output_selector is empty.
        """
        # Set default input selector to all channels on default port
        if input_selector is None:
            input_selector = {Constants.Defaults.PORT_IN: Router.ALL}

        # Convert list format to dictionary format for input selector
        if type(input_selector) is list:
            if len(input_selector) == 0:
                raise ValueError("input_selector must not be empty.")
            # Convert single list to list of lists if needed
            if type(input_selector[0]) is not list:
                input_selector = [input_selector]
            # Create port mappings
            if len(input_selector) == 1:
                input_selector = {
                    Constants.Defaults.PORT_IN: input_selector[0]
                }
            else:
                input_selector = {
                    f"in{i + 1}": val for i, val in enumerate(input_selector)
                }

        # Create input port configurations
        input_ports = [
            IPort.Configuration(name=name, timing=Constants.Timing.INHERITED)
            for name in input_selector.keys()
        ]

        # Set default output selector to all channels on default port
        if output_selector is None:
            output_selector = {Constants.Defaults.PORT_OUT: Router.ALL}

        # Convert list format to dictionary format for output selector
        if type(output_selector) is list:
            if len(output_selector) == 0:
                raise ValueError("output_selector must not be empty.")
            # Convert single list to list of lists if needed
            if type(output_selector[0]) is not list:
                output_selector = [output_selector]
            # Create port mappings
            if len(output_selector) == 1:
                output_selector = {
                    Constants.Defaults.PORT_OUT: output_selector[0]
                }
            else:
                output_selector = {
                    f"out{i + 1}": val for i, val in enumerate(output_selector)
                }

        # Create output port configurations
        output_ports = [
            OPort.Configuration(name=name) for name in output_selector.keys()
        ]

        # Initialize internal routing map
        self._map = {}

        # Initialize parent IONode with all configurations
        IONode.__init__(
            self,
            input_selector=input_selector,
            output_selector=output_selector,
            input_ports=input_ports,
            output_ports=output_ports,
            **kwargs,
        )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """
        Set up the Router node and build the internal channel mapping.

        Creates the internal routing map that defines which input channels
        are connected to which output channels. Validates that all input
        ports have compatible sampling rates, frame sizes, and data types.

        Args:
            data (dict): Initial data dictionary for port configuration.
            port_context_in (dict): Input port context information containing
                channel counts, sampling rates, frame sizes, and data types.

        Returns:
            dict: Output port context with routing information and updated
                channel counts for each output port.

        Raises:
            ValueError: If input ports have incompatible sampling rates,
                frame sizes, or data types.
        """
        # Get commonly used configuration keys
        cc_key = Constants.Keys.CHANNEL_COUNT
        name_key = IPort.Configuration.Keys.NAME

        # Build input channel mapping
        input_map: list = []
        ip_key = Router.Configuration.Keys.INPUT_PORTS
        is_key = Router.Configuration.Keys.INPUT_SELECTOR

        # Process each input port to build channel mapping
        for k in range(len(self.config[ip_key])):
            port = self.config[ip_key][k]
            name = port[name_key]
            sel = self.config[is_key][name]

            # Expand "ALL" to actual channel range
            if sel == Router.ALL:
                sel = range(port_context_in[name][cc_key])
            # Add each selected channel to the input map
            input_map.extend([{name: [n]} for n in sel])

        # Build output port mapping using input map
        op_key = Router.Configuration.Keys.OUTPUT_PORTS
        os_key = Router.Configuration.Keys.OUTPUT_SELECTOR

        for k in range(len(self.config[op_key])):
            port = self.config[op_key][k]
            name = port[name_key]
            sel = self.config[os_key][name]

            # Expand "ALL" to full input map range
            if sel == Router.ALL:
                sel = range(len(input_map))
            # Map selected input channels to this output port
            self._map[name] = [input_map[n] for n in sel]

        # Validate sampling rate consistency across all input ports
        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rates = [
            md.get(sr_key, None) for md in port_context_in.values()
        ]
        sampling_rates = [sr for sr in sampling_rates if sr is not None]
        if len(set(sampling_rates)) != 1:
            raise ValueError("All ports must have the same sampling rate.")
        sr = sampling_rates[0]

        # Validate frame size consistency across all input ports
        fsz_key = Constants.Keys.FRAME_SIZE
        frame_sizes = [
            md.get(fsz_key, None) for md in port_context_in.values()
        ]
        frame_sizes = [fsz for fsz in frame_sizes if fsz is not None]
        if len(set(frame_sizes)) != 1:
            raise ValueError("All ports must have the same frame size.")
        fsz = frame_sizes[0]

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
            context[cc_key] = len(self._map[op[name_key]])
            context[sr_key] = sr
            context[fsz_key] = fsz
            context[type_key] = tp
            port_context_out[op[name_key]] = context

        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Process one frame of data by routing channels according to the mapping.

        Routes channels from input ports to output ports based on the channel
        mapping established during setup. Each output port receives data from
        its configured subset of input channels, maintaining frame structure
        and data types.

        Args:
            data (dict): Dictionary containing input data arrays for each port.
                Keys are port names, values are NumPy arrays with shape
                (frame_size, channel_count).

        Returns:
            dict: Dictionary containing output data arrays for each output
                port. Keys are output port names, values are NumPy arrays
                with shape (frame_size, selected_channel_count).
        """
        data_out: dict = {}

        # Process each output port mapping
        for port_out, mapping in self._map.items():
            # Build channel data list for this output port
            channel_arrays = []
            for m in mapping:
                # Extract data for each channel mapping
                for port_in, ch_in in m.items():
                    # Get the specific channel data
                    try:
                        channel_arrays.append(data[port_in][:, ch_in])
                    except Exception:
                        channel_arrays.append(np.zeros((1, 1)))

            # Horizontally stack all selected channels
            # Shape: (frame_size, output_channel_count)
            data_out[port_out] = np.hstack(channel_arrays)

        return data_out
