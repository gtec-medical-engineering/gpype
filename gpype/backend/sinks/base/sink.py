"""A node that delivers data out of this process."""

from __future__ import annotations

from ...core.i_node import INode


class Sink(INode):
    """Base for a node whose output leaves this process.

    A file on disk, a socket, an LSL stream. What they share is not how
    they are written but *where they are*: a sink has a location, so a
    distributed pipeline has to decide which process builds it, and that
    is what a chain exists to express.

    That is the whole content of this class. It declares no parameters,
    no ports and no behaviour, and deliberately so -- it names a property
    that was previously implicit and unreadable. Before it,
    ``_CsvWriterCore`` was a ``FileWriter(INode)`` and ``_LSLSenderCore``
    a bare ``INode``, and nothing in the hierarchy separated either from
    :class:`~gpype.backend.sinks.collector.Collector`, which is an
    ``INode`` too. So a reader could not tell which nodes needed a chain,
    and neither could the ``Pipeline`` once it started building them --
    see ``assembly.kind_of``.

    **`Collector` is not one, and that is the line this class draws.** It
    "terminates a pipeline in memory instead of in a file": it hands data
    back to the caller *inside* this process, has no external endpoint,
    and no residency question to answer. A node that only touches data
    needs no chain, whether or not it happens to sit at the end of a
    pipeline.

    Public as ``gp.Sink``. Subclass it for a sink of your own; subclass
    ``INode`` directly for one that keeps its output in the process, as
    `Collector` does.
    """

    #: Whether this run's output must carry the non-commercial mark.
    #:
    #: A property of *delivering data outward*, which is why it belongs
    #: here: it was written twice before this class existed, identically,
    #: on ``FileWriter`` and on ``_LSLSenderCore``, because there was no
    #: base to put it on.
    #:
    #: False until a pipeline says otherwise, so a sink used outside one
    #: produces unmarked output rather than claiming a restriction
    #: nobody established.
    #:
    #: A sink is free to record it and never use it. ``UDPSender``
    #: does: a datagram has nowhere to put a mark, which is a fact about
    #: the format rather than about entitlement not applying to it.
    _marked: bool = False

    def attach_entitlement(self, verdict) -> None:
        """Record whether this run's output must be marked.

        Called by the pipeline before anything starts, because output
        written first would carry the wrong answer.

        Args:
            verdict: The pipeline's resolved Entitlement.
        """
        self._marked = bool(verdict.marked)
