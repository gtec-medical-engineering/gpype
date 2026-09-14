"""TimeSeriesScope: the chain, with no Qt in it.

Importable in any residency, which is the point: deserializing a document
that names this scope must not pull a GUI toolkit into a server process
that will never draw anything. The widget half lives in
``_cores.time_series_scope`` and is imported only where it is built.
"""

import time
from typing import List

import ioiocore as ioc

from ...backend.core._private.chain_params import (
    stream_id_for,
    strip_chain_keys,
)
from ...backend.core._private.link import Link
from ...backend.core.i_port import IPort
from ...common.constants import Constants
from ...common.launch_config import LaunchConfig
from ._validation import check_bounds


class Markers(dict):
    """Container for event marker configuration.

    Stores marker properties including color, label, channel association,
    and trigger value for event-based visualization.

    Args:
        color: Hex color code or color name for marker visualization.
        label: Text label for marker identification.
        channel: Channel index where marker should be displayed.
        value: Trigger value that activates the marker.
    """

    def __init__(self, color: str, label: str, channel: int, value: int):
        """Initialize marker configuration with display properties."""
        super().__init__()
        self["color"] = color
        self["label"] = label
        self["channel"] = channel
        self["value"] = value


class TimeSeriesScope(ioc.IChain):
    """Real-time oscilloscope widget for multi-channel time series.

    IChain containing [Link, _TimeSeriesScopeCore] for distributed operation.
    Bridges data from the SERVER residency to the EDGE for local display.
    """

    #: Defaults for the parameters whose value the core would
    #: otherwise apply. Declared on the public class, which is the one an
    #: author types and the only one a catalog can read: a private core
    #: is not in this class's MRO. The core reads them from here, so
    #: there is still exactly one literal per value.
    DEFAULT_TIME_WINDOW: int = 10
    DEFAULT_AMPLITUDE_LIMIT: float = 50.0

    #: Repaints per second when the author does not choose one.
    #:
    #: This is no longer a copy of Widget.DEFAULT_REFRESH_RATE but this
    #: display's own, deliberately higher. A sweep moves continuously,
    #: so its frame rate is what the motion looks like; the other
    #: widgets only redraw when something arrives and 10 Hz is enough
    #: for them. The private core reads this value from here, exactly as
    #: it reads DEFAULT_TIME_WINDOW and DEFAULT_AMPLITUDE_LIMIT, so
    #: there is one literal and it lives in the Qt-free module a server
    #: can import.
    #:
    #: 20 rather than 30 for two measured reasons. The interval is
    #: int(round(1000/rate)) ms, so 20 Hz is exactly 50 ms while 30 Hz
    #: is 33 ms and delivers 30.3. And the redraw is not the whole cost:
    #: _update() itself is 0.415 ms at 8 channels, but a full frame
    #: including Qt's rasterisation costs 8.85 ms at 8 channels, 20.0 ms
    #: at 32 and 32.8 ms at 64 -- so at 64 channels a 30 Hz frame would
    #: not fit inside its own interval. 20 Hz costs roughly 17% of one
    #: core at 8 channels and 63% at 64.
    DEFAULT_REFRESH_RATE: float = 20.0

    #: Accepted range for ``amplitude_limit``, in microvolts. Named so
    #: that a form can offer a slider with the right bounds: these used to
    #: be literals inside an ``if`` in a private core, where nothing but
    #: the exception text could reach them.
    MIN_AMPLITUDE_LIMIT: float = 1.0
    MAX_AMPLITUDE_LIMIT: float = 5_000.0

    #: Accepted range for ``time_window``, in seconds, exclusive at both
    #: ends. The docstring said "(1-120)" for a long time while the code
    #: accepted 1 < x < 240, which is why the numbers live here now and
    #: the prose does not restate them.
    MIN_TIME_WINDOW: int = 1
    MAX_TIME_WINDOW: int = 240

    #: Marker styling helper, re-exported from the boundary node so that
    #: gp.TimeSeriesScope.Markers keeps working now that the widget is a
    #: chain rather than the node itself.
    Markers = Markers

    def __init__(
        self,
        time_window: int = None,
        amplitude_limit: float = None,
        markers: list = None,
        hidden_channels: list = None,
        refresh_rate: float = None,
        show_amplitude_control: bool = True,
        show_time_window_control: bool = True,
        name: str = None,
        **kwargs,
    ):
        """Initialize the time series scope chain.

        Args:
            time_window: Display window duration in seconds; longer than 1 and
                shorter than 240.
            amplitude_limit: Y-axis scale limit in microvolts (1-5000).
            markers: List of marker configurations for event visualization.
            hidden_channels: List of channel indices to hide from display.
            show_amplitude_control: Whether the display offers the
                viewer buttons for the amplitude scale. What they
                change is the display; the configured value is what
                a document round-trips.
            show_time_window_control: Whether the display offers the
                viewer buttons for the time window. Changing it
                re-cuts the sweep buffer, so the sweep restarts.
            name: Optional name for this node. Shown as the widget's
                group box caption, and used to refer to the node from a
                document's connections.
            **kwargs: Additional arguments forwarded to parent classes.
        """
        # Checked here, on the chain, because the core that used to
        # check is not built in server residency -- so a server accepted a
        # 9999-second window and reported the document loaded.
        check_bounds(
            "time_window",
            time_window,
            self.MIN_TIME_WINDOW,
            self.MAX_TIME_WINDOW,
            unit="s",
            inclusive=False,
        )
        check_bounds(
            "amplitude_limit",
            amplitude_limit,
            self.MIN_AMPLITUDE_LIMIT,
            self.MAX_AMPLITUDE_LIMIT,
            unit="uV",
        )

        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "time_window": time_window,
            "amplitude_limit": amplitude_limit,
            "markers": markers,
            "hidden_channels": hidden_channels,
            "refresh_rate": refresh_rate,
            "show_amplitude_control": show_amplitude_control,
            "show_time_window_control": show_time_window_control,
            "name": name,
        }
        ip_key = self.Configuration.Keys.INPUT_PORTS
        core_kwargs = strip_chain_keys(kwargs)
        user_ports = kwargs.get(ip_key)
        if user_ports:
            # Only the port shape reaches the core, rebuilt without the
            # stored ids. The timing has to travel: ioiocore compares
            # chain and boundary ports by name alone, so a port the user
            # made asynchronous would otherwise be silently rebuilt as a
            # synchronous one and refuse to connect to an event source.
            name_key = IPort.Configuration.Keys.NAME
            timing_key = IPort.Configuration.Keys.TIMING
            core_kwargs[ip_key] = [
                IPort.Configuration(name=p[name_key], timing=p[timing_key])
                for p in user_ports
            ]
        self._core_params.update(core_kwargs)
        self._scope_core = None
        kwargs.setdefault(ip_key, [IPort.Configuration()])

        # `name` reaches the chain as well as the core, so it names the
        # node and not only the group box.
        #
        # It used to be withheld, on the grounds that it was "only naming
        # the internal core widget as it does today". The consequence was
        # that a TimeSeriesScope's node name was always the class name,
        # so two of them in one document collided -- and since the loader
        # refuses to resolve a name two nodes share, a document with two
        # scopes connected by name was rejected outright with
        # "'TimeSeriesScope' is ambiguous". Measured; it is what an
        # authoring tool produces, because names are the connection form
        # meant for one.
        #
        # This node was also the only one that did it. The other two
        # scopes declare no `name` parameter at all, so theirs flows
        # through kwargs and names the node, like every other node in the
        # package. Consistency is the fix, not a second key.
        if name is not None:
            kwargs["name"] = name

        ioc.IChain.__init__(
            self,
            time_window=time_window,
            amplitude_limit=amplitude_limit,
            markers=markers,
            hidden_channels=hidden_channels,
            refresh_rate=refresh_rate,
            show_amplitude_control=show_amplitude_control,
            show_time_window_control=show_time_window_control,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create internal node chain: [Link, _TimeSeriesScopeCore].

        Returns:
            List containing Link bridge and time series scope core.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.STANDALONE:
            # edge or server
            nodes.append(
                Link(
                    sender=Constants.Residency.SERVER,
                    receiver=Constants.Residency.EDGE,
                    stream_id=self._link_stream_id,
                )
            )
        if residency != Constants.Residency.SERVER:
            from ._cores.time_series_scope import _TimeSeriesScopeCore

            self._scope_core = _TimeSeriesScopeCore(**self._core_params)
            nodes.append(self._scope_core)
        return nodes

    @property
    def widget(self):
        """Qt widget for display in the main application."""
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            return self._scope_core.widget
        else:
            return None

    def run(self):
        """Start the widget update timer."""
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            self._scope_core.run()

    def terminate(self):
        """Stop the widget update timer."""
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            self._scope_core.terminate()
