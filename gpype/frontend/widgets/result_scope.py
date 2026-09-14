"""ResultScope: the chain, with no Qt in it.

Importable in any residency, which is the point: deserializing a document
that names this scope must not pull a GUI toolkit into a server process
that will never draw anything. The widget half lives in
``_cores.result_scope`` and is imported only where it is built.
"""

from typing import List, Optional

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


class ResultScope(ioc.IChain):
    """Generic result endpoint: a sink a frontend subscribes to.

    IChain containing [Link, _ResultScopeCore] for distributed operation.
    Bridges data from the SERVER residency to the EDGE for local display.

    Makes no assumption about the stream. Where TimeSeriesScope needs a
    sampling rate and a time axis, this takes whatever arrives -- a
    classifier decision at 4 Hz, a sparse label, a feature vector -- and
    shows the current value per channel. A non-Python frontend
    subscribes to the same stream and renders it however it likes: this
    node's identity is its role in the pipeline, and the Qt readout is
    one per-residency implementation of that role, not the node itself.
    """

    #: The refresh rate a widget uses when the author does not choose
    #: one. Duplicated from Widget.DEFAULT_REFRESH_RATE deliberately: a
    #: chain must not import Widget, because that pulls Qt into a server
    #: process, which is the whole reason the chain and the core are
    #: separate files.
    DEFAULT_REFRESH_RATE: float = 10.0

    #: Digits drawn after the decimal point. Declared on the public
    #: class, which is the one an author types and the only one a
    #: catalog can read: a private core is not in this class's MRO. The
    #: core reads them from here, so there is one literal per value.
    DEFAULT_DECIMALS: int = 3
    #: Values kept per channel *besides* the current one. Zero means the
    #: readout shows the latest frame and nothing else.
    DEFAULT_HISTORY: int = 0

    #: Accepted range for ``decimals``. Bounded rather than trusted: the
    #: readout formats with ``f"{value:.{decimals}f}"``, and a negative
    #: precision raises ValueError from inside the format string on the
    #: GUI thread, where the number an author wrote appears nowhere.
    MIN_DECIMALS: int = 0
    MAX_DECIMALS: int = 12

    #: Accepted range for ``history``. The lower bound is what
    #: ``deque(maxlen=...)`` accepts -- a negative maxlen raises. The
    #: upper bound is a display bound, not a memory one: past values are
    #: drawn as text on one line per channel.
    MIN_HISTORY: int = 0
    MAX_HISTORY: int = 1_000

    def __init__(
        self,
        name: str = None,
        refresh_rate: float = None,
        decimals: int = DEFAULT_DECIMALS,
        history: int = DEFAULT_HISTORY,
        timing: str = None,
        **kwargs,
    ):
        """Initialize the result scope chain.

        Args:
            name: Optional name for this node. Shown as the widget's
                group box caption, and used to refer to the node from a
                document's connections.
            refresh_rate: Repaints per second. Defaults to
                DEFAULT_REFRESH_RATE.
            decimals: Digits after the decimal point in the readout.
            history: Past values shown per channel besides the current
                one. Zero shows the latest frame only.
            timing: Timing to give the input port and both ports of the
                Link, from Constants.Timing. Named explicitly rather
                than left in kwargs because timing is a *port* setting:
                as a stray keyword it lands in the node configuration
                instead, where nothing reads it and the ports stay SYNC
                -- see Link.__init__. A result stream may be sparse, and
                a SYNC port cannot be connected to an ASYNC source
                ("Ports have incompatible timings"). None leaves the
                port defaults alone.
            **kwargs: Additional arguments forwarded to parent classes.

        Raises:
            ValueError: If decimals or history is outside its range.
        """
        # Checked here, on the chain, because the core that checks too is
        # not built in server residency -- so a server would accept a
        # document with decimals=-1 and report it loaded, and the author
        # would find out on the edge or not at all.
        check_bounds(
            "decimals", decimals, self.MIN_DECIMALS, self.MAX_DECIMALS
        )
        check_bounds("history", history, self.MIN_HISTORY, self.MAX_HISTORY)

        self._link_stream_id = stream_id_for(kwargs)

        ip_key = self.Configuration.Keys.INPUT_PORTS
        name_key = IPort.Configuration.Keys.NAME
        timing_key = IPort.Configuration.Keys.TIMING
        user_ports = kwargs.get(ip_key)
        if timing is None and user_ports:
            # Ports supplied directly rather than through `timing` --
            # ResultScope(input_ports=[IPort.Configuration(timing=...)]),
            # which the other scopes accept and which an authoring tool
            # emitting ports produces. The Link has to match: an ASYNC
            # boundary port in front of a SYNC Link input is refused with
            # "Ports have incompatible timings", so the setting has to be
            # read back off the ports when it did not arrive by name.
            timing = user_ports[0][timing_key]
        #: Timing handed to the Link, so both halves of a split pipeline
        #: carry a sparse stream the same way.
        self._link_timing = timing
        port_kwargs = {} if timing is None else {"timing": timing}

        core_kwargs = strip_chain_keys(kwargs)
        if user_ports:
            # Only the port shape reaches the core, rebuilt without the
            # stored ids. The timing has to travel: ioiocore compares
            # chain and boundary ports by name alone, so a port the user
            # made asynchronous would otherwise be silently rebuilt as a
            # synchronous one and refuse to connect to an event source.
            core_kwargs[ip_key] = [
                IPort.Configuration(name=p[name_key], timing=p[timing_key])
                for p in user_ports
            ]
        else:
            # Built explicitly rather than left to the core's own
            # default, which is SYNC: a fresh ResultScope(timing=ASYNC)
            # would otherwise give the chain an ASYNC boundary port and
            # the core a SYNC one.
            core_kwargs[ip_key] = [IPort.Configuration(**port_kwargs)]

        self._core_params = {
            "decimals": decimals,
            "history": history,
            "refresh_rate": refresh_rate,
            "name": name,
        }
        self._core_params.update(core_kwargs)
        self._scope_core = None
        kwargs.setdefault(ip_key, [IPort.Configuration(**port_kwargs)])

        # `name` reaches the chain as well as the core, so it names the
        # node and not only the group box. Two unnamed chains of one
        # class both take the class name, and the loader refuses to
        # resolve a name two nodes share -- so a document with two of
        # these connected by name would be rejected outright.
        if name is not None:
            kwargs["name"] = name

        # `timing` is forwarded, unlike the same keyword on Link. The
        # two are opposite cases: on Link it is a *port* setting that
        # nothing reads once it lands in a node configuration, whereas
        # here it is a constructor parameter, so a stored value binds
        # back to it on rebuild and never reaches a port as a stray key.
        # Withholding it lost the setting outright -- the document
        # recorded ASYNC ports and no timing, and only the ports were
        # left to recover it from.
        ioc.IChain.__init__(
            self,
            decimals=decimals,
            history=history,
            refresh_rate=refresh_rate,
            timing=timing,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create internal node chain: [Link, _ResultScopeCore].

        Returns:
            List containing Link bridge and result scope core. Under
            SERVER residency the core is omitted -- there is no display
            on a server -- and under STANDALONE the Link is.
        """
        nodes = []
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.STANDALONE:
            # edge or server
            nodes.append(
                Link(
                    sender=Constants.Residency.SERVER,
                    receiver=Constants.Residency.EDGE,
                    stream_id=self._link_stream_id,
                    timing=self._link_timing,
                )
            )
        if residency != Constants.Residency.SERVER:
            from ._cores.result_scope import _ResultScopeCore

            self._scope_core = _ResultScopeCore(**self._core_params)
            nodes.append(self._scope_core)
        return nodes

    @property
    def widget(self) -> Optional[object]:
        """Qt widget for display in the main application.

        Returns:
            The core's widget, or None under SERVER residency, where no
            core was built.
        """
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            return self._scope_core.widget
        else:
            return None

    def run(self):
        """Start the widget update timer, where there is a widget."""
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            self._scope_core.run()

    def terminate(self):
        """Stop the widget update timer, where there is a widget."""
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            self._scope_core.terminate()
