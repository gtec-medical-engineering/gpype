"""TriggerScope: the chain, with no Qt in it.

Importable in any residency, which is the point: deserializing a document
that names this scope must not pull a GUI toolkit into a server process
that will never draw anything. The widget half lives in
``_cores.trigger_scope`` and is imported only where it is built.
"""

import re
from typing import List

import ioiocore as ioc

from ...backend.core._private.chain_params import (
    plot_port_names,
    stream_id_for,
    strip_chain_keys,
)
from ...backend.core._private.link import Link
from ...backend.flow.trigger import Trigger
from ...common.constants import Constants
from ._validation import check_bounds


class TriggerScope(ioc.IChain):
    """Event-triggered oscilloscope widget for analyzing signal epochs.

    IChain containing [Link, _TriggerScopeCore] for distributed operation.
    Bridges data from the SERVER residency to the EDGE for local display.

    Multi-input-port mode works in every residency: the Link declares
    one port pair per plot variable and bridges each on its own stream.
    """

    #: See TimeSeriesScope: declared here, read by the core.
    DEFAULT_AMPLITUDE_LIMIT: float = 50.0

    #: The refresh rate a widget uses when the author does not choose
    #: one. Duplicated from Widget.DEFAULT_REFRESH_RATE deliberately: a
    #: chain must not import Widget, because that pulls Qt into a server
    #: process, which is the whole reason the chain and the core are
    #: separate files. test_frontend_widget_refresh_defaults asserts they
    #: agree, so the duplication cannot drift silently.
    DEFAULT_REFRESH_RATE: float = 10.0

    #: One input per plot expression, so the names come from `plots`.
    INPUT_PORTS_FROM = "plots"

    #: Accepted range for ``amplitude_limit``, in microvolts. Named so
    #: that a form can offer a slider with the right bounds: these used to
    #: be literals inside an ``if`` in a private core, where nothing but
    #: the exception text could reach them.
    MIN_AMPLITUDE_LIMIT: float = 1.0
    MAX_AMPLITUDE_LIMIT: float = 5_000.0

    def __init__(
        self,
        amplitude_limit: float = 50,
        plots: list = None,
        hidden_channels: list = None,
        refresh_rate: float = None,
        **kwargs,
    ):
        """Initialize the trigger scope chain.

        Args:
            amplitude_limit: Y-axis scale limit in microvolts (1-5000).
            plots: List of mathematical expressions to evaluate and plot.
            hidden_channels: List of channel indices to hide from display.
            **kwargs: Additional arguments forwarded to parent classes.
        """
        # See TimeSeriesScope: the core that used to check this is
        # not built in server residency.
        check_bounds(
            "amplitude_limit",
            amplitude_limit,
            self.MIN_AMPLITUDE_LIMIT,
            self.MAX_AMPLITUDE_LIMIT,
            unit="uV",
        )

        self._link_stream_id = stream_id_for(kwargs)
        self._core_params = {
            "amplitude_limit": amplitude_limit,
            "plots": plots,
            "hidden_channels": hidden_channels,
            "refresh_rate": refresh_rate,
        }
        self._core_params.update(strip_chain_keys(kwargs))
        self._scope_core = None
        # The input ports are not forced to a single "in": the boundary
        # node derives one port per free variable of the plot
        # expressions, and 5.0.0 raises when supplied chain ports do not
        # match the boundary node's.
        ioc.IChain.__init__(
            self,
            amplitude_limit=amplitude_limit,
            plots=plots,
            hidden_channels=hidden_channels,
            refresh_rate=refresh_rate,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create internal node chain: [Link, _TriggerScopeCore].

        Returns:
            List containing Link bridge and trigger scope core.
        """
        from ...common.launch_config import LaunchConfig

        nodes = []
        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.STANDALONE:
            # edge or server
            # One Link port pair per plot variable, mirroring the
            # core's own ports. A chain exposes its first internal
            # node's input ports as its own, so a one-port Link in
            # front of a multi-plot core both misdeclared the chain and
            # left the core's other ports unwired -- which is the
            # `Input port 'in' not found.` this fixes.
            nodes.append(
                Link(
                    sender=Constants.Residency.SERVER,
                    receiver=Constants.Residency.EDGE,
                    stream_id=self._link_stream_id,
                    ports=plot_port_names(self._core_params["plots"]),
                )
            )
        if residency != Constants.Residency.SERVER:
            from ._cores.trigger_scope import _TriggerScopeCore

            self._scope_core = _TriggerScopeCore(**self._core_params)
            nodes.append(self._scope_core)
        return nodes

    @property
    def widget(self):
        """Qt widget for display in the main application."""
        from ...common.launch_config import LaunchConfig

        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            return self._scope_core.widget
        else:
            return None

    def run(self):
        """Start the widget update timer."""
        from ...common.launch_config import LaunchConfig

        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            self._scope_core.run()

    def terminate(self):
        """Stop the widget update timer."""
        from ...common.launch_config import LaunchConfig

        residency = LaunchConfig.get().residency
        if residency != Constants.Residency.SERVER:
            self._scope_core.terminate()
