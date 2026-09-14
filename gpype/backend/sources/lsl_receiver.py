from __future__ import annotations

import threading
from typing import List, Optional

import ioiocore as ioc
import numpy as np
from pylsl import StreamInlet, resolve_byprop

from ...common._private import channels
from ...common.constants import Constants
from ..core._private.chain_params import stream_id_for, strip_chain_keys
from ..core._private.link import Link
from ..core._private.sync import Sync
from ..core.o_port import OPort
from .base import raw
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


class _LslReceiverCore(Source):
    """Internal node pulling samples from an LSL inlet."""

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
        **kwargs,
    ):
        """Initialize the receiver core.

        Args:
            stream_name: Name of the stream to resolve.
            stream_type: Type to resolve, when no name is given.
            channel_count: Expected channel count. Read from the stream
                when not given.
            sampling_rate: Expected rate. Read from the stream when not
                given.
            **kwargs: Additional arguments for the parent Source.

        Raises:
            ValueError: If neither a name nor a type is given, or if the
                stream shape is missing.
        """
        if not stream_name and not stream_type:
            raise ValueError("Give a stream_name or a stream_type to resolve.")
        # A rebuilt source receives the per-port list form of these
        # back out of its own configuration, so normalise before use.
        channel_count = self.scalar(channel_count)
        sampling_rate = self.scalar(sampling_rate)
        if channel_count is None or sampling_rate is None:
            raise ValueError(
                "channel_count and sampling_rate are required. "
                "LslReceiver resolves them from the stream once and "
                "passes them in, which is what lets a stored "
                "configuration be rebuilt with no publisher running."
            )
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

        kwargs.setdefault("frame_size", 1)
        kwargs.setdefault("output_ports", [OPort.Configuration()])
        kwargs.setdefault("channel_count", int(channel_count))
        super().__init__(sampling_rate=float(sampling_rate), **kwargs)

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
        """Read channel names and roles out of a stream description.

        Args:
            info: The resolved StreamInfo.

        Returns:
            A tuple of (labels, roles); either may be None when the
            publisher did not describe its channels.
        """
        labels: list = []
        roles: list = []
        try:
            channel = info.desc().child("channels").child("channel")
            while not channel.empty():
                label = channel.child_value("label")
                kind = channel.child_value("type")
                labels.append(label or "")
                roles.append(
                    _ROLE_OF_TYPE.get(kind, Constants.ChannelRoles.SIGNAL)
                )
                channel = channel.next_sibling()
        except Exception:
            # A publisher is not obliged to describe its channels, and a
            # missing description must not stop the stream being read.
            return None, None

        if not labels or not all(labels):
            return None, None
        return labels, roles

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

        if self._labels:
            port_context_out[PORT_OUT].update(
                channels.describe(self._roles, self._labels, None)
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
            self._labels, self._roles = self._describe(self._inlet.info())
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


class LslReceiver(ioc.OChain):
    """Reads a Lab Streaming Layer stream into a pipeline.

    The mirror of LSLSender, and the way another vendor's amplifier,
    MATLAB, a stimulus program or a second g.Pype process gets its data
    in. Channel names and types are read from the stream description, so
    a g.Pype stream read back keeps the names it was published with.

    What arrives this way is somebody else's data: LSL carries no
    per-sample authentication, so a stream says what it is and cannot
    prove it.
    """

    def __init__(
        self,
        stream_name: Optional[str] = None,
        stream_type: Optional[str] = None,
        channel_count: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the receiver chain.

        Args:
            stream_name: Name of the stream to resolve.
            stream_type: Type to resolve, when no name is given.
            channel_count: Expected channel count.
            sampling_rate: Expected sampling rate.
            **kwargs: Additional arguments.

        Raises:
            ValueError: If neither a name nor a type is given.
        """
        if not stream_name and not stream_type:
            raise ValueError("Give a stream_name or a stream_type to resolve.")
        self._link_stream_id = stream_id_for(kwargs)

        # Resolve the stream header once, here, when its shape is not
        # already known. It belongs at this level rather than in the
        # core for two reasons:
        #
        #  * the chain's configuration is what survives serialisation --
        #    the internal node is rebuilt from these parameters, not from
        #    its own stored config -- so recording the discovered channel
        #    count and rate here is what lets a saved pipeline be rebuilt
        #    with no publisher running. Without it, `deserialize()` had
        #    to find the stream live or fail;
        #  * it happens once per construction rather than once per
        #    internal node.
        channel_count = _LslReceiverCore.scalar(channel_count)
        sampling_rate = _LslReceiverCore.scalar(sampling_rate)
        if channel_count is None or sampling_rate is None:
            info = _LslReceiverCore._resolve(stream_name, stream_type)
            if channel_count is None:
                channel_count = int(info.channel_count())
            if sampling_rate is None:
                sampling_rate = float(info.nominal_srate())

        self._core_params = {
            "stream_name": stream_name,
            "stream_type": stream_type,
            "channel_count": channel_count,
            "sampling_rate": sampling_rate,
        }
        self._core_params.update(strip_chain_keys(kwargs))
        if stream_name:
            kwargs.setdefault("stream_name", stream_name)
        if stream_type:
            kwargs.setdefault("stream_type", stream_type)
        # Recorded so a rebuild does not have to ask the network again.
        kwargs.setdefault("channel_count", int(channel_count))
        kwargs.setdefault("sampling_rate", float(sampling_rate))
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [OPort.Configuration()],
        )
        # stream_name, stream_type, channel_count and sampling_rate
        # already reach kwargs via the setdefault calls above.
        ioc.OChain.__init__(
            self,
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
            where the pipeline is distributed, and Sync last.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        # Recorded, replaced, or simply built, as the launch
        # configuration says. Contributes nothing under server
        # residency: the core lives on the edge, and so does
        # anything recording or replaying it.
        # SYNC, unlike the other receivers: this core declares a plain
        # output port and the chain ends in a plain Sync(), so the
        # stream is continuous however irregularly LSL delivers it.
        # Declaring the tap ASYNC here would have put an ASYNC port
        # between two SYNC ones -- invisible until save_as was actually
        # switched on, because the timing is only read when it is.
        nodes.extend(
            raw.source_stage(
                self, lambda: _LslReceiverCore(**self._core_params)
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
        nodes.append(Sync())
        return nodes
