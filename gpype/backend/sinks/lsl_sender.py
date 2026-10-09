from __future__ import annotations

from typing import Optional

import numpy as np
from pylsl import StreamInfo, StreamOutlet, cf_double64

from ...common._private import channels
from ...common._private.entitlement import MARK
from ...common.constants import Constants
from ..core.i_node import INode
from .base.sink import Sink

#: Channel role to the XDF channel type a consumer expects. A consumer
#: that knows a column is a marker rather than a measured signal can plot
#: and store it accordingly; without this every channel looks like EEG.
_LSL_TYPES = {
    Constants.ChannelRoles.SIGNAL: "EEG",
    Constants.ChannelRoles.TRIGGER: "Markers",
    Constants.ChannelRoles.AUXILIARY: "AUX",
    Constants.ChannelRoles.QUALITY: "Misc",
    Constants.ChannelRoles.INDEX: "Misc",
}


class LSLSender(Sink):
    """Lab Streaming Layer (LSL) sender for real-time data streaming.

    Streams multi-channel data to the Lab Streaming Layer network,
    configuring the LSL stream from its input port's context.

    Where the input names its channels, or gives any of them a role
    other than signal, each channel's type, label and physical unit are
    written to the stream's channel description (desc/channels/channel,
    the XDF convention). A channel that was never named gets no label;
    a unit is empty rather than omitted where the channel's own unit is
    not recorded.
    """

    #: Default LSL stream name for g.Pype data streams
    DEFAULT_STREAM_NAME = "gpype_lsl"

    class Configuration(INode.Configuration):
        """Configuration class for LSLSender parameters."""

        class Keys(INode.Configuration.Keys):
            """Configuration keys for LSL sender settings."""

            #: Stream name configuration key
            STREAM_NAME = "stream_name"

    def __init__(
        self,
        stream_name: Optional[str] = None,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the LSL sender with specified stream name.

        Args:
            stream_name: Name for the LSL stream. If None, uses the default
                stream name. Used for stream identification on the LSL network.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments passed to parent INode class.
        """
        # Use default stream name if none provided
        if stream_name is None:
            stream_name = LSLSender.DEFAULT_STREAM_NAME

        # Initialize parent INode with configuration
        INode.__init__(
            self, stream_name=stream_name, edge_id=edge_id, **kwargs
        )

        # Initialize LSL components (created during setup)
        self._lsl_info = None  # LSL stream metadata
        self._lsl_outlet = None  # LSL data outlet
        self._frame_size = None  # Samples per processing frame

    def stop(self):
        """Stop the LSL sender and clean up resources.

        Properly releases LSL resources by setting outlet and info objects
        to None, allowing them to be garbage collected.
        """
        # Release LSL resources
        self._lsl_outlet = None
        self._lsl_info = None

        # Call parent stop method
        super().stop()

    #: Whether this run must mark its output. Set by the pipeline
    #: before anything starts.
    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup the LSL stream based on input port context.

        Creates LSL StreamInfo and StreamOutlet objects using metadata from
        the input port context. Stream is configured with appropriate
        parameters for EEG data transmission.

        Args:
            data: Dictionary of input data arrays from connected ports.
            port_context_in: Context information from input ports with
                channel count, sampling rate, and frame size.

        Returns:
            Dictionary returned from parent setup method.
        """
        # Extract context information from default input port
        context = port_context_in[Constants.Defaults.PORT_IN]
        channel_count = context[Constants.Keys.CHANNEL_COUNT]
        stream_name = self.config[self.Configuration.Keys.STREAM_NAME]
        sampling_rate = context[Constants.Keys.SAMPLING_RATE]
        self._frame_size = context[Constants.Keys.FRAME_SIZE]

        # Create LSL stream info with EEG configuration
        self._lsl_info = StreamInfo(
            name=stream_name,
            channel_count=channel_count,
            type="EEG",  # Standard LSL type for EEG
            nominal_srate=sampling_rate,
            channel_format=cf_double64,  # Double prec
            source_id=stream_name,
        )  # Unique identifier

        # Describe the channels if the source told us what they are, so
        # consumers see electrode names instead of positions and a
        # trigger channel as a marker. This has to happen before the
        # outlet is created: liblsl copies the stream info into the
        # outlet, so anything added afterwards is written to a copy
        # nobody ever sees.
        #
        # Roles and labels are guarded separately, as a recording's
        # metadata is: a stream with roles but no names still says which
        # channel is the trigger, and gets no label rather than an
        # invented "Ch01". Roles that are all signal say nothing a
        # receiver does not already assume, and a plain Generator
        # declares exactly those, so its description stays as it was.
        named = channels.has_labels(context)
        roles = channels.roles_of(context)
        signal = Constants.ChannelRoles.SIGNAL
        if named or any(role != signal for role in roles):
            desc = self._lsl_info.desc().append_child("channels")
            units = channels.units_of(context)
            labels = channels.labels_of(context) if named else None
            for index, role in enumerate(roles):
                entry = desc.append_child("channel")
                if named:
                    entry.append_child_value("label", str(labels[index]))
                entry.append_child_value("type", _LSL_TYPES.get(role, "Misc"))
                # The XDF convention: desc/channels/channel/unit, beside
                # label and type. Empty rather than omitted where this
                # channel's own unit is not recorded, so a reader sees a
                # channel description either way.
                unit = units[index] if units is not None else None
                entry.append_child_value("unit", str(unit) if unit else "")

        # The mark travels in the stream description, beside the channels,
        # for the same reason and with the same constraint: liblsl copies
        # the info into the outlet, so it has to be here rather than after.
        #
        # It propagates further than the session: a recorder that persists
        # stream metadata carries the mark into files this process never
        # touches. (Whether LabRecorder does persist desc subtrees is
        # asserted in the design and still unmeasured.)
        if self._marked:
            licence = self._lsl_info.desc().append_child("licence")
            licence.append_child_value("terms", MARK)

        # Which physical amplifier is behind this stream, under the same
        # constraint as the mark: before the outlet exists, because
        # liblsl copies the info in. A consumer several hops away
        # otherwise has no way to tell which device it is looking at --
        # the stream name describes the pipeline, not the hardware.
        serial = context.get(Constants.Keys.DEVICE_SERIAL)
        if serial:
            device = self._lsl_info.desc().append_child("device")
            device.append_child_value("serial", str(serial))

        # Create LSL outlet for data transmission
        self._lsl_outlet = StreamOutlet(self._lsl_info)

        # Timestamps come from the pipeline timeline when one is
        # available. Without it, LSL stamps at push time, which is only
        # correct while the source paces its output at the sampling rate.
        # Under a packetised transport several samples are pushed within
        # microseconds of each other, and arrival time then says nothing
        # about acquisition time.
        self._sample_period = 1.0 / sampling_rate if sampling_rate else 0.0

        # Call parent setup method
        return super().setup(data, port_context_in)

    def attach_timeline(self, timeline) -> None:
        """Bind this sink to the pipeline's master timeline.

        Args:
            timeline: Timeline manager owned by the pipeline.
        """
        self._timeline = timeline

    def _timestamp(self) -> float:
        """Return the LSL timestamp for the frame being pushed.

        Returns:
            Host time of the frame's last sample, or 0.0 to let LSL stamp
            at push time when no timeline relation is available yet.
        """
        timeline = getattr(self, "_timeline", None)
        if timeline is None:
            return 0.0
        position = timeline.position
        if position is None:
            return 0.0
        stamp = timeline.time_of(position)
        return 0.0 if stamp is None else float(stamp)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process and stream data through LSL outlet.

        Sends incoming data to the LSL network using either single-sample
        or chunk-based transmission depending on frame size. Automatically
        converts numpy arrays to lists for LSL compatibility.

        Args:
            data: Dictionary containing input data arrays. Uses the default
                input port to retrieve data for streaming.

        Returns:
            None as this is a sink node with no output data.
        """
        # Get data from default input port
        d = data[Constants.Defaults.PORT_IN]

        # Stream data if LSL outlet is available
        if self._lsl_outlet:
            # LSL applies the timestamp to the last sample of what is
            # pushed, which is the position Sync published for this
            # frame. Passing 0.0 makes LSL stamp at push time instead,
            # which is only right while the source paces its output.
            stamp = self._timestamp()
            if self._frame_size == 1:
                # Single-sample streaming for minimal latency
                self._lsl_outlet.push_sample(d[0].tolist(), stamp)
            else:
                # Chunk streaming for efficiency with larger frames
                self._lsl_outlet.push_chunk(d.tolist(), stamp)

        # No output data for sink nodes
        return None
