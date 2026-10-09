"""Collector: keeps what reaches it, so a run can hand it back."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private.naming import node_label
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

    The input takes the timing of whatever is connected to it. A
    continuous stream is joined along the time axis, ``(time,
    channel)``. Epochs from an asynchronous node such as ``Trigger`` are
    stacked into ``(time, channel, trial)``, one trial per epoch in the
    order they arrived, even when only one arrived. The trial axis
    carries no condition labels. Epochs of differing shape are refused
    when the result is read.

    Args:
        **kwargs: Additional arguments for the parent INode.
    """

    #: Declares that ``Pipeline.run()`` should gather this node's
    #: ``result``. Declared rather than inferred from having a ``result``
    #: member, so a node that happens to expose one is not collected by
    #: accident -- and so a user's own sink can opt in deliberately.
    COLLECTS: bool = True

    def __init__(self, **kwargs):
        # Inherited, so an ASYNC producer can connect; a SYNC one keeps
        # the behaviour a SYNC port had (D-BATCH-73).
        input_ports = [
            IPort.Configuration(
                name=PORT_IN, timing=Constants.Timing.INHERITED
            )
        ]
        input_ports = kwargs.pop(Constants.Keys.INPUT_PORTS, input_ports)
        super().__init__(input_ports=input_ports, **kwargs)
        self._blocks: list[np.ndarray] = []
        self._context: dict = {}
        #: Whether blocks are trials, decided at setup from the timing
        #: the input resolved to when it was connected.
        self._trials = False

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
        # The port's resolved timing, not the context's: Trigger declares
        # an ASYNC output but writes its data input's timing into the
        # context it publishes.
        self._trials = (
            self.get_input_port(PORT_IN).timing == Constants.Timing.ASYNC
        )
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
        worth being able to see from a test. On an asynchronous input it
        is the number of trials.
        """
        return len(self._blocks)

    @property
    def result(self) -> Optional[Result]:
        """What was collected, or None if nothing arrived.

        A continuous stream is concatenated along the time axis, so a
        paced run and a batch run of the same recording give the same
        array. Epochs are stacked along a third, trial axis.

        Raises:
            ValueError: If epochs of differing shape arrived, naming the
                first shape and the first that differs.
        """
        if not self._blocks:
            return None
        if self._trials:
            data = self._stack_trials()
        elif len(self._blocks) == 1:
            data = self._blocks[0]
        else:
            data = np.concatenate(self._blocks, axis=0)
        return Result(data=data, context=self._context, name=self.name)

    def _stack_trials(self) -> np.ndarray:
        """Stack the epochs into ``(time, channel, trial)``.

        Returns:
            The stacked array.

        Raises:
            ValueError: If two epochs differ in shape.
        """
        first = self._blocks[0].shape
        for index, block in enumerate(self._blocks):
            if block.shape != first:
                raise ValueError(
                    f"{node_label(self)} cannot stack its epochs into "
                    f"trials: epoch 0 has shape {first} and epoch "
                    f"{index} has shape {block.shape}. A trial axis "
                    f"needs every epoch to cover the same samples and "
                    f"channels."
                )
        return np.stack(self._blocks, axis=2)
