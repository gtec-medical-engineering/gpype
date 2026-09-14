"""Collector: keeps what reaches it, so a run can hand it back."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...common.constants import Constants
from ...common.result import Result
from ..core.i_node import INode
from ..core.i_port import IPort

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN


class Collector(INode):
    """Terminates a pipeline in memory instead of in a file.

    The point of an offline run is usually to get data back into Python,
    not into a second file. ``Pipeline.run()`` returns every Collector's
    :class:`~gpype.common.result.Result`, keyed by node name.

    In a batch run this receives exactly one block -- the whole
    recording -- and keeping it costs nothing beyond the data itself. It
    also works in a realtime pipeline, where it accumulates frames and
    therefore grows for as long as the pipeline runs; that is useful for
    a short measurement and wrong for a long one.

    Args:
        **kwargs: Additional arguments for the parent INode.
    """

    #: Declares that ``Pipeline.run()`` should gather this node's
    #: ``result``. Declared rather than inferred from having a ``result``
    #: member, so a node that happens to expose one is not collected by
    #: accident -- and so a user's own sink can opt in deliberately.
    COLLECTS: bool = True

    def __init__(self, **kwargs):
        input_ports = [IPort.Configuration(name=PORT_IN)]
        input_ports = kwargs.pop(Constants.Keys.INPUT_PORTS, input_ports)
        super().__init__(input_ports=input_ports, **kwargs)
        self._blocks: list[np.ndarray] = []
        self._context: dict = {}

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Record what describes the incoming stream.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            An empty dict -- a Collector has no output ports.
        """
        self._context = dict(port_context_in.get(PORT_IN, {}))
        # A restart must not concatenate two runs into one result.
        self._blocks = []
        # Through the parent, so the incoming context is checked for what
        # every sink needs rather than being taken on trust.
        return super().setup(data, port_context_in)

    def step(self, data: dict[str, np.ndarray]) -> dict:
        """Keep the incoming block.

        Args:
            data: Input data dictionary.

        Returns:
            An empty dict.
        """
        block = data.get(PORT_IN)
        if block is not None:
            self._blocks.append(np.asarray(block))
        return {}

    @property
    def block_count(self) -> int:
        """How many blocks arrived.

        One after a batch run. More says the run was paced, which is
        worth being able to see from a test.
        """
        return len(self._blocks)

    @property
    def result(self) -> Optional[Result]:
        """What was collected, or None if nothing arrived.

        Blocks are concatenated along the time axis, so a paced run and
        a batch run of the same recording give the same array.
        """
        if not self._blocks:
            return None
        if len(self._blocks) == 1:
            data = self._blocks[0]
        else:
            data = np.concatenate(self._blocks, axis=0)
        return Result(data=data, context=self._context, name=self.name)
