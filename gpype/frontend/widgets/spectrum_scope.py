"""SpectrumScope: the chain, with no Qt in it.

Importable in any residency, which is the point: deserializing a document
that names this scope must not pull a GUI toolkit into a server process
that will never draw anything. The widget half lives in
``_cores.spectrum_scope`` and is imported only where it is built.
"""

from typing import List

import ioiocore as ioc

from ...backend.core._private.chain_params import (
    stream_id_for,
    strip_chain_keys,
)
from ...backend.core._private.link import Link
from ...backend.core.i_port import IPort
from ...common.constants import Constants
from ._validation import check_bounds


class SpectrumScope(ioc.IChain):
    """Frequency domain visualization widget for spectral analysis.

    IChain containing [Link, _SpectrumScopeCore] for distributed operation.
    Bridges data from the SERVER residency to the EDGE for local display.
    """

    #: See TimeSeriesScope: declared here, read by the core.
    DEFAULT_AMPLITUDE_LIMIT: float = 50.0
    DEFAULT_NUM_AVERAGES: int = 10

    #: The refresh rate a widget uses when the author does not choose
    #: one. Duplicated from Widget.DEFAULT_REFRESH_RATE deliberately: a
    #: chain must not import Widget, because that pulls Qt into a server
    #: process, which is the whole reason the chain and the core are
    #: separate files. test_frontend_widget_refresh_defaults asserts they
    #: agree, so the duplication cannot drift silently.
    DEFAULT_REFRESH_RATE: float = 10.0

    #: Accepted range for ``amplitude_limit``, in microvolts. Named so
    #: that a form can offer a slider with the right bounds: these used to
    #: be literals inside an ``if`` in a private core, where nothing but
    #: the exception text could reach them.
    MIN_AMPLITUDE_LIMIT: float = 1.0
    MAX_AMPLITUDE_LIMIT: float = 5_000.0

    def __init__(
        self,
        amplitude_limit: float = None,
        num_averages: int = None,
        hidden_channels: list = None,
        refresh_rate: float = None,
        **kwargs,
    ):
        """Initialize the spectrum scope chain.

        Args:
            amplitude_limit: Maximum amplitude for display scaling.
            num_averages: Number of spectra to average for smoothing.
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
            "num_averages": num_averages,
            "hidden_channels": hidden_channels,
            "refresh_rate": refresh_rate,
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
        ioc.IChain.__init__(
            self,
            amplitude_limit=amplitude_limit,
            num_averages=num_averages,
            hidden_channels=hidden_channels,
            refresh_rate=refresh_rate,
            stream_id=self._link_stream_id,
            **kwargs,
        )

    def create_internal_nodes(self) -> List[ioc.Node]:
        """Create internal node chain: [Link, _SpectrumScopeCore].

        Returns:
            List containing Link bridge and spectrum scope core.
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
            from ._cores.spectrum_scope import _SpectrumScopeCore

            self._scope_core = _SpectrumScopeCore(**self._core_params)
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
