"""A plain Python function as a node, for offline runs only."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Apply(IONode):
    """Runs a plain Python function over a whole recording.

    The escape hatch for offline work: a transform that is three lines of
    numpy needs no class, no ``Configuration`` holder and no port
    declaration -- ``gp.Apply(np.abs)`` is the whole node.

    Batch only; a realtime pipeline is refused by ``start()`` with this
    node named (``D-BATCH-56``). An arbitrary callable has no bound on
    its execution time, and a closure quietly accumulating state across
    frames looks identical to a pure transform at the call site. Offline
    neither applies: there is no deadline, and the whole recording
    arrives in one call.

    The function must return an array of the same shape and dtype it was
    given. Anything else is a change to the contract with the next node,
    which a function has no way to declare, so it is refused rather than
    misconfiguring whatever is downstream -- write a node class for that.
    A function is not a configuration value either, so a pipeline holding
    an ``Apply`` cannot be written to a document and rebuilt elsewhere.

    Args:
        function: Called once with the whole recording as a
            ``numpy.ndarray`` of shape ``(samples, channels)``. Must
            return an array of the same shape and dtype.
        **kwargs: Passed to :class:`IONode` -- ``name`` in particular,
            which is what a refusal will call this node.

    Raises:
        TypeError: If *function* is not callable.
    """

    #: Refused outside a batch run, by ``Pipeline._validate``. A class
    #: attribute rather than a check inside ``step``, so the refusal
    #: arrives from ``start()`` with the node named -- and not on the
    #: first cycle, by which point a source may already hold a device.
    BATCH_ONLY: bool = True

    def __init__(self, function: Callable, **kwargs):
        if not callable(function):
            raise TypeError(
                f"Apply takes a callable, not {type(function).__name__}. "
                f"Pass a function that takes one array and returns one "
                f"of the same shape -- Apply(np.abs), or Apply(lambda "
                f"block: block * 1e6)."
            )
        super().__init__(**kwargs)
        # Instance state, not configuration: see the class docstring on
        # serialisation. Putting it in a Configuration holder would also
        # widen `recognised_keys`, which is not a registry for callables.
        self._function = function

    def process(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Hand the whole recording to the function.

        ``process`` and not ``step``: this node exists only for batch
        runs, and a batch run delivers the recording in one call. There
        is deliberately no ``step``, so a realtime pipeline that somehow
        reached a cycle would fail loudly rather than silently doing
        nothing.

        Args:
            data: The complete recording, under key ``PORT_IN``.

        Returns:
            What the function returned, under key ``PORT_OUT``.

        Raises:
            ValueError: If the function returned something of a
                different shape or dtype, or something that is not an
                array at all.
        """
        block = data[PORT_IN]
        result = self._function(block)
        label = node_label(self)
        name = getattr(self._function, "__name__", repr(self._function))

        if not isinstance(result, np.ndarray):
            raise ValueError(
                f"{label}: {name} returned "
                f"{type(result).__name__}, and a node has to hand the "
                f"next one an array. Return the block you were given, "
                f"transformed -- not a list, a scalar or None."
            )
        if result.shape != block.shape:
            raise ValueError(
                f"{label}: {name} returned shape {result.shape} for an "
                f"input of {block.shape}. Apply declares its output to "
                f"look like its input, because a function cannot say "
                f"otherwise -- so a change of shape is a change to the "
                f"contract with the next node. Write a node class, "
                f"whose setup() declares the shape it produces."
            )
        if result.dtype != block.dtype:
            raise ValueError(
                f"{label}: {name} returned dtype {result.dtype} for an "
                f"input of {block.dtype}. Cast it back -- a change of "
                f"dtype is the same kind of contract change as a change "
                f"of shape, and Constants.DATA_TYPE is what the rest of "
                f"the graph is built on."
            )
        return {PORT_OUT: result}


def node(function: Callable) -> Callable:
    """Turn a function into something a pipeline can be handed.

    Sugar over :class:`Apply`, and nothing more: the decorated name
    becomes a factory, so the function stays callable in its own right
    and can be unit-tested without a pipeline anywhere near it.

    ::

        @gp.node
        def to_microvolts(block):
            return block * 1e6

        p.connect(reader, to_microvolts())
        assert to_microvolts.function(block).max() == ...

    Batch only, for the reasons in :class:`Apply`.

    Args:
        function: The transform. One array in, one of the same shape
            out.

    Returns:
        A factory taking the same ``**kwargs`` as :class:`Apply`, with
        the original function on its ``function`` attribute.
    """

    def factory(**kwargs) -> Apply:
        kwargs.setdefault("name", getattr(function, "__name__", "Apply"))
        return Apply(function, **kwargs)

    # Kept reachable so the decorator does not take the function away
    # from its author. A decorator that makes a unit test impossible
    # has cost more than it saved.
    factory.function = function
    factory.__name__ = getattr(function, "__name__", "node")
    factory.__doc__ = function.__doc__
    return factory
