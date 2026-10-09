from __future__ import annotations

import threading
from typing import Optional

import numpy as np
from pylsl import StreamInlet, resolve_byprop

from ...common._private import channels
from ...common.constants import Constants
from ..core._private import assembly
from ..core.o_port import OPort
from .base.source import Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT

#: XDF channel type to the role it corresponds to. The inverse of what
#: LSLSender writes, so a g.Pype stream read back keeps its roles.
_ROLE_OF_TYPE = {
    "EEG": Constants.ChannelRoles.SIGNAL,
    "Markers": Constants.ChannelRoles.TRIGGER,
    "AUX": Constants.ChannelRoles.AUXILIARY,
    "Misc": Constants.ChannelRoles.AUXILIARY,
}


class LslReceiver(Source):
    """Reads a Lab Streaming Layer stream into a pipeline.

    The mirror of LSLSender, and the way another vendor's amplifier,
    MATLAB, a stimulus program or a second g.Pype process gets its data
    in. Channel names, types and units are read from the stream
    description, so a g.Pype stream read back keeps what it was
    published with. Types and units are kept even where the channels
    are not all named; the names only where every channel has one.

    What arrives this way is somebody else's data: LSL carries no
    per-sample authentication, so a stream says what it is and cannot
    prove it.
    """

    #: How long to wait for the stream to appear, in seconds.
    RESOLVE_TIMEOUT_S = 5.0

    class Configuration(Source.Configuration):
        """Configuration class for LslReceiver parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for LslReceiver settings."""

            pass

        class OptionalKeys(Source.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Stream name to resolve
            STREAM_NAME = "stream_name"
            #: Stream type to resolve, when no name is given
            STREAM_TYPE = "stream_type"

    def __init__(
        self,
        stream_name: Optional[str] = None,
        stream_type: Optional[str] = None,
        channel_count: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        edge_id: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the receiver.

        Args:
            stream_name: Name of the stream to resolve.
            stream_type: Type to resolve, when no name is given.
            channel_count: Expected channel count. Read from the stream
                when not given.
            sampling_rate: Expected rate. Read from the stream when not
                given.
            edge_id: Which edge runs this node, matched against the
                edge process's --edge-id. None, the default, is every
                edge; ignored when the pipeline is not distributed.
            **kwargs: Additional arguments for the parent Source.

        Raises:
            ValueError: If neither a name nor a type is given, or if the
                stream shape is missing on a server.
            RuntimeError: If the shape has to be read from a stream that
                does not appear.
        """
        if not stream_name and not stream_type:
            raise ValueError("Give a stream_name or a stream_type to resolve.")
        # A rebuilt source receives the per-port list form of these
        # back out of its own configuration, so normalise before use.
        channel_count = self.scalar(channel_count)
        sampling_rate = self.scalar(sampling_rate)
        if channel_count is None or sampling_rate is None:
            # Asked once, here, and recorded: a stored configuration then
            # rebuilds with no publisher running. Not where this process
            # does not run the node -- a server, or an edge it is not
            # assigned to -- where the stream is on the owning edge's
            # network: the shape must come from the document, as a
            # reader's does.
            if not assembly.builds_core_for(edge_id):
                raise ValueError(
                    "channel_count and sampling_rate must come from the "
                    "document where this LslReceiver is not built -- on a "
                    "server, or on an edge it is not assigned to: the "
                    "stream is resolved where it is published, on its "
                    "own edge."
                )
            info = self._resolve(stream_name, stream_type)
            if channel_count is None:
                channel_count = int(info.channel_count())
            if sampling_rate is None:
                sampling_rate = float(info.nominal_srate())
        opt = self.Configuration.OptionalKeys
        if stream_name:
            kwargs.setdefault(opt.STREAM_NAME, str(stream_name))
        if stream_type:
            kwargs.setdefault(opt.STREAM_TYPE, str(stream_type))

        self._stream_name = str(stream_name) if stream_name else None
        self._stream_type = str(stream_type) if stream_type else None

        # Nothing touches the network here. The inlet, and the channel
        # description that only arrives with it, are opened in start().
        # start() runs before setup(), so the description is available
        # by the time the port context is built.
        self._inlet = None
        self._labels: Optional[list] = None
        self._roles: Optional[list] = None
        self._units: Optional[list] = None

        kwargs.setdefault("frame_size", 1)
        kwargs.setdefault("output_ports", [OPort.Configuration()])
        kwargs.setdefault("channel_count", int(channel_count))
        super().__init__(
            sampling_rate=float(sampling_rate), edge_id=edge_id, **kwargs
        )

        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._pending: Optional[np.ndarray] = None

    @classmethod
    def _resolve(cls, name: Optional[str], type_: Optional[str]):
        """Find the stream on the network.

        Args:
            name: Stream name, or None.
            type_: Stream type, used when no name is given.

        Returns:
            The resolved StreamInfo.

        Raises:
            RuntimeError: If no matching stream appears in time.
        """
        prop, value = ("name", name) if name else ("type", type_)
        found = resolve_byprop(prop, value, 1, cls.RESOLVE_TIMEOUT_S)
        if not found:
            raise RuntimeError(
                f"No LSL stream with {prop} '{value}' appeared within "
                f"{cls.RESOLVE_TIMEOUT_S:g} s."
            )
        return found[0]

    @staticmethod
    def _describe(info) -> tuple:
        """Read channel names, roles and units out of a stream description.

        Args:
            info: The resolved StreamInfo.

        Returns:
            A tuple of (labels, roles, units); every element is None
            when the publisher did not describe its channels, and labels
            alone is None when it described them without naming them all.
        """
        labels: list = []
        roles: list = []
        units: list = []
        try:
            channel = info.desc().child("channels").child("channel")
            while not channel.empty():
                label = channel.child_value("label")
                kind = channel.child_value("type")
                # The XDF convention: desc/channels/channel/unit, beside
                # label and type. LSLSender writes "" for a channel with
                # no unit of its own, read back as None.
                unit = channel.child_value("unit")
                labels.append(label or "")
                roles.append(
                    _ROLE_OF_TYPE.get(kind, Constants.ChannelRoles.SIGNAL)
                )
                units.append(unit or None)
                channel = channel.next_sibling()
        except Exception:
            # A publisher is not obliged to describe its channels, and a
            # missing description must not stop the stream being read.
            return None, None, None

        # A description of some other number of channels describes
        # nothing here, and would fail later as a length mismatch.
        if not roles or len(roles) != int(info.channel_count()):
            return None, None, None
        # Roles and units stand without names: LSLSender describes a
        # stream with a trigger channel whether or not it is named.
        if not all(labels):
            labels = None
        return labels, roles, units

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Publish what the stream said about its channels.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.
        """
        port_context_out = super().setup(data, port_context_in)

        # The stream paces itself, so this is not a FixedRateSource and
        # nothing else publishes the rate. Without it downstream Sync has
        # nothing to claim the master timeline with, and holds every
        # sample pending instead of placing it.
        rate = self.config[Constants.Keys.SAMPLING_RATE]
        frame_size = port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE]
        if isinstance(frame_size, list):
            frame_size = frame_size[0]
        port_context_out[PORT_OUT][Constants.Keys.SAMPLING_RATE] = rate
        port_context_out[PORT_OUT][Constants.Keys.FRAME_RATE] = (
            rate / frame_size
        )

        if self._roles:
            port_context_out[PORT_OUT].update(
                channels.describe(self._roles, self._labels, None)
            )
            if self._units is not None:
                port_context_out[PORT_OUT][Constants.Keys.CHANNEL_UNITS] = (
                    list(self._units)
                )
        return port_context_out

    def start(self):
        """Resolve the stream, open the inlet, and begin pulling.

        This is where the publisher has to exist. Resolving here rather
        than in the constructor is what allows a saved pipeline to be
        rebuilt while the stream is offline.

        Raises:
            RuntimeError: If no matching stream appears in time.
        """
        if self._inlet is None:
            info = self._resolve(self._stream_name, self._stream_type)
            # A resolved StreamInfo carries only the header; the channel
            # description arrives with the inlet.
            self._inlet = StreamInlet(info, max_buflen=1)
            self._labels, self._roles, self._units = self._describe(
                self._inlet.info()
            )
        Source.start(self)
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._pull, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop pulling and close the inlet."""
        Source.stop(self)
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        if self._inlet is not None:
            self._inlet.close_stream()
            self._inlet = None

    def _pull(self):
        """Pull samples from the inlet and cycle them into the pipeline."""
        while self._running:
            try:
                chunk, _ = self._inlet.pull_chunk(timeout=0.1)
            except Exception:
                if not self._running:
                    break
                raise
            if not chunk:
                continue
            block = np.asarray(chunk, dtype=Constants.DATA_TYPE)
            for row in block:
                self._pending = row.reshape(1, -1)
                self.cycle()

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Emit the sample the pulling thread handed over.

        Returns:
            One sample, or None when nothing is waiting.
        """
        pending = getattr(self, "_pending", None)
        if pending is None:
            return None
        self._pending = None
        return {PORT_OUT: pending}
