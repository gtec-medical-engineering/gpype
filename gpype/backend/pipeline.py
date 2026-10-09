import copy
import logging
import os
import platform
import sys
import tempfile
import threading
import time
from time import perf_counter
from typing import Optional, Union

import ioiocore as ioc

from .. import _installer
from ..common import config_keys
from ..common._private import (
    attestation,
    device_probe,
    licence,
    naming,
    pipelines,
    remote_attestation,
    timebase,
)
from ..common._private.channels import stamp_clock_complaint
from ..common._private.entitlement import (
    MARK,
    Entitlement,
    Permission,
    device_required,
    discard_supplied,
    entitlement_of,
    lifted,
    meet,
    release_entitlement,
    remote_required,
    set_entitlement,
    take_supplied,
)
from ..common.constants import Constants
from ..common.launch_config import ENV_EDGE_ID, LaunchConfig
from ..common.result import Result
from .core._private.controllable import (
    action_names,
    ensure_json,
    has_control,
    merged_update,
    resolve_action,
    resolve_control,
)
from .core._private.link import CONTEXT_TIMEOUT_S
from .core._private.placement import position_gate
from .core._private.timeline import (
    TimelineManager,
    release_timeline,
    timeline_of,
)
from .core.node import Node
from .core.o_port import OPort

log = logging.getLogger(__name__)

#: Where a node's document id lives. It is a *configuration key*, not a
#: property -- there is no ``id`` property anywhere in ioiocore -- so it
#: is read out of the node's configuration dict.
_ID_KEY = ioc.Portable.Configuration.Keys.ID


def _node_id_of(node) -> Optional[str]:
    """The document id of *node*.

    Deliberately the id and not the name. Names are **not unique**: a
    node defaults its name to its class name
    (``ioiocore/processing_element.py``), and ``chain_params`` refuses to
    use names for exactly this reason, in writing -- "two unnamed chains
    of the same class both take the class name, verified, so deriving
    from the name would pair two sinks on the same stream and the
    broker's last-writer-wins would cross them silently". Addressing a
    control at a name would pick whichever of them came first, and
    succeed.

    Args:
        node: The node to read.

    Returns:
        The id, or None when the node carries no configuration. Read
        defensively so a stand-in without one cannot break a report that
        is meant to describe the whole pipeline.
    """
    config = getattr(node, "config", None)
    if not isinstance(config, dict):
        return None
    return config.get(_ID_KEY)


def _fallback_log_directory() -> Optional[str]:
    """Where to log when ioiocore may have no default.

    ioiocore knows Windows, Darwin and Linux and raises
    ``RuntimeError("Unsupported OS")`` for any other
    ``platform.system()`` -- Android and iOS included (PEP 738, PEP 730).
    A durable directory there lies in the app's container, which only
    the host app knows; the temp directory is the app's own on both, and
    the one Python checks is writable.

    Returns:
        None where ioiocore has a default. Otherwise a directory under
        the temp directory, or "" -- memory-only logging -- when none is
        writable.
    """
    if platform.system() in ("Windows", "Darwin", "Linux"):
        return None
    try:
        return os.path.join(tempfile.gettempdir(), "gtec", "gPype")
    except OSError:
        return ""


class Pipeline(ioc.Pipeline):
    """Brain-Computer Interface pipeline for real-time data processing.

    Extends ioiocore Pipeline for BCI applications with automatic logging
    to platform-specific directories. Manages node lifecycle, data flow
    connections, and real-time execution of interconnected processing nodes.
    """

    def __init__(self):
        """Initialize Pipeline with platform-specific logging directory."""
        # Determine platform-specific log directory
        if sys.platform == "win32":
            log_dir = os.path.join(os.getenv("APPDATA", ""), "gtec", "gPype")
        elif sys.platform == "darwin":
            app_support = os.path.expanduser("~/Library/Application Support")
            log_dir = os.path.join(app_support, "gtec", "gPype")
        else:
            log_dir = _fallback_log_directory()

        # Initialize parent pipeline with logging directory
        super().__init__(directory=log_dir)

        self._init_gpype_state()

    #: Names ``repr`` shows in a line before it elides the middle.
    _REPR_NAMES = 6

    def __repr__(self) -> str:
        """One line: the mode, the nodes, and the state.

        ``Pipeline(batch, 3 nodes: eeg -> bp -> out, stopped)`` for a
        pipeline that is one line; otherwise the sources and the sinks,
        ``Pipeline(realtime, 5 nodes, sources [a, b], sinks [c],
        running)``. A node is counted as its author wrote it: the chain
        the pipeline builds around it is that node. Never raises.
        """
        try:
            return self._repr_text()
        except Exception:  # pragma: no cover - a repr must not raise
            return object.__repr__(self)

    def _repr_text(self) -> str:
        """Build the text ``__repr__`` returns.

        Returns:
            The one-line description.
        """
        elements = list(getattr(self._imp, "_nodes", None) or [])
        try:
            mode = self.execution_mode()
        except ValueError:
            mode = "mixed modes" if self._sources() else "no source"

        count = len(elements)
        nodes = f"{count} node{'' if count == 1 else 's'}"
        path = self._linear_path(elements)
        if path:
            nodes += ": " + " -> ".join(self._repr_names(path))
        else:
            sources = [e for e in elements if self._repr_kind(e) == "source"]
            sinks = [e for e in elements if self._repr_kind(e) == "sink"]
            if sources:
                nodes += f", sources [{', '.join(self._repr_names(sources))}]"
            if sinks:
                nodes += f", sinks [{', '.join(self._repr_names(sinks))}]"

        if self.get_condition() == ioc.Constants.Conditions.ERROR:
            state = "failed"
        else:
            state = str(self.get_state()).lower()
        return f"Pipeline({mode}, {nodes}, {state})"

    def _linear_path(self, elements: list) -> Optional[list]:
        """The pipeline in order, where it is a single line of nodes.

        Args:
            elements: Every element in the pipeline.

        Returns:
            The elements from source to sink, or None when the pipeline
            branches, merges, has a loop, or holds a node that was never
            connected.
        """
        if len(elements) < 2 or len(elements) != len(self._elements):
            return None
        downstream = {
            key: {id(e): e for e in targets}
            for key, targets in self._cycle_adjacency().items()
        }
        upstream: dict = {}
        for targets in downstream.values():
            if len(targets) > 1:
                return None
            for key in targets:
                upstream[key] = upstream.get(key, 0) + 1
        if any(n > 1 for n in upstream.values()):
            return None
        heads = [e for e in elements if id(e) not in upstream]
        if len(heads) != 1:
            return None
        path = [heads[0]]
        while id(path[-1]) in downstream and len(path) <= len(elements):
            path.extend(downstream[id(path[-1])].values())
        return path if len(path) == len(elements) else None

    @staticmethod
    def _repr_kind(element) -> Optional[str]:
        """``source`` for outputs only, ``sink`` for inputs only."""
        if isinstance(element, ioc.Chain):
            if isinstance(element, ioc.OChain):
                return "source"
            if isinstance(element, ioc.IChain):
                return "sink"
            return None
        inputs = isinstance(element, ioc.INode)
        outputs = isinstance(element, ioc.ONode)
        if outputs and not inputs:
            return "source"
        if inputs and not outputs:
            return "sink"
        return None

    @classmethod
    def _repr_names(cls, elements: list) -> list:
        """The elements' names, the middle elided past ``_REPR_NAMES``."""
        names = [
            str(getattr(e, "name", None) or type(e).__name__) for e in elements
        ]
        if len(names) <= cls._REPR_NAMES:
            return names
        half = cls._REPR_NAMES // 2
        return names[:half] + ["..."] + names[-half:]

    @property
    def event_margin_ms(self) -> float:
        """How late an event may arrive and still be placed, in ms.

        One number for the whole pipeline -- see
        TimelineManager.DEFAULT_EVENT_MARGIN_MS for what it buys and what
        it costs. A property rather than a constructor argument because
        ``deserialize`` builds its instance with ``object.__new__`` and
        never runs ``__init__``, so a constructor-only setting would be
        absent on exactly the pipelines a document produced.

        Kept here rather than on the timeline, even though the timeline
        is where placement reads it: ``start()`` releases the timeline
        and builds a fresh one, deliberately, so that a run cannot
        inherit the previous one's master election or epoch. A margin
        written to the timeline before ``start()`` went with it --
        measured, every value from 0 to 100 ms produced the same
        10-frame hold, because placement was reading the default off an
        object that had just been replaced.

        Returns:
            The margin in milliseconds.
        """
        stored = getattr(self, "_event_margin_ms", None)
        if stored is not None:
            return stored
        if self._oscar_active():
            return 0.0
        return TimelineManager.DEFAULT_EVENT_MARGIN_MS

    def _oscar_active(self) -> bool:
        """Whether an enabled OSCAR node is in this pipeline.

        The margin exists to give a late observation somewhere to land,
        and it pays for that by holding frames. OSCAR already delays
        the signal path by far more than the margin ever buys -- 70
        samples at 250 Hz against the 10 the default asks for -- and an
        event arriving inside that delay is placed on its own position
        regardless. Holding frames on top of it would add latency and
        buy nothing, so the margin defaults to zero here. An explicit
        setting still wins: this only decides the default.

        Imported inside the method: this module is imported from
        ``gpype/__init__`` and the node package reaches back into it.

        Returns:
            True when a node in this pipeline is removing artifacts.
        """
        from .core._private.oscar import Oscar

        for element in getattr(self, "_elements", []):
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if isinstance(node, Oscar) and node.enabled:
                    return True
        return False

    @event_margin_ms.setter
    def event_margin_ms(self, value: Optional[float]) -> None:
        """Set the margin for every placing node in this pipeline.

        Args:
            value: Milliseconds, 0 to opt out, or None for the
                default.
        """
        self._event_margin_ms = (
            None if value is None else max(0.0, float(value))
        )
        # Pushed through as well, so a change lands on a pipeline that is
        # already running rather than only on its next start().
        timeline_of(self).event_margin_ms = self.event_margin_ms

    def _init_gpype_state(self) -> None:
        """Establish the state that belongs to this class, not ioiocore.

        Called from ``__init__`` *and* from ``deserialize``, which builds
        its instance with ``object.__new__`` and therefore never runs
        ``__init__`` at all. One method rather than two lists, because
        two lists drift: this one had, and every attribute added to
        ``__init__`` since was simply absent on every deserialised
        pipeline. Measured on one: ``.failure`` and ``.close()`` both
        raised ``AttributeError``, it was missing from the registry a GUI
        finds pipelines through, and it carried **zero** error handlers
        where a constructed pipeline carries one -- so a deserialised
        pipeline that died recorded nothing and told nobody.

        That is the path a remote runtime uses for every pipeline it is
        given, which is where it mattered.
        """
        #: Allowed late arrival for asynchronous observations, in
        #: milliseconds, or None for the default. Set here as well as
        #: in __init__ because deserialize() runs only this.
        self._event_margin_ms = None
        #: Elements seen by connect(), so the pipeline can find the nodes
        #: that need its master timeline. ioiocore adds nodes implicitly
        #: and exposes no enumeration, but every participant passes
        #: through connect() at least once.
        self._elements: list = []

        #: The failing log entry, once a run has failed. Recorded
        #: without anyone asking: a pipeline that has died is a thing
        #: every caller may need to know about, and requiring a
        #: registration step means the one script that forgot is the one
        #: that fails silently.
        self._failure = None

        # Registered here rather than by the caller, for the same reason.
        self.add_error_handler(self._record_failure)

        # So a GUI can find this pipeline without being handed it:
        # MainApp and Pipeline are built independently and never
        # introduced, and a windowed application may never show the
        # console the failure is printed to.
        pipelines.register(self)

        #: EDGE-side responders answering the server's challenges (C10).
        self._responders: list = []

        #: A broker this pipeline bound *early*, before its nodes
        #: started, only so a remote attestation could happen at all.
        #: Held for the run and released in stop(), which balances the
        #: extra acquire() against the Links' own.
        self._attest_broker = None

        #: Live probes, by stream id. Not part of the pipeline and not in
        #: _elements: a probe is a callable on a port, deliberately not
        #: a node -- see backend/core/_private/probe.py for why.
        self._probes: dict = {}

        #: The chain built around each world-facing core, by object id.
        #: What makes `chain_of` answer for a core the chain holds but
        #: does not run -- a source under SERVER residency. See _wrap.
        self._wrappers: dict = {}
        #: The residencies the chains were built under. A pipeline runs
        #: under the residency it was built for: built as a server and
        #: started as standalone, it would keep its broker and Links while
        #: the licence gate saw development (D-ENT-100).
        self._built_residencies: set = set()

        #: The clock-sync health this run's edge start converged on, as
        #: its source streams' contexts carry it; None when the run did
        #: not sync (STANDALONE, SERVER, batch, or no source here).
        self._clock_health: Optional[dict] = None

        #: The pins serialize() writes when given none: those it was last
        #: given, or those of the document this pipeline was rebuilt
        #: from, as it carried them (D-CORE-109).
        self._requirements = None

    def _register(self, element: Union[Node, dict]) -> None:
        """Remember an element taking part in this pipeline.

        Args:
            element: Node, chain, or ``node["port"]`` specification.
        """
        if isinstance(element, dict):
            element = element.get("node")
        if element is not None and not any(
            e is element for e in self._elements
        ):
            self._elements.append(element)

    def _wrap(self, element: Union[Node, dict]):
        """Put a chain around a world-facing node that has none.

        ``connect()`` is the only place a node enters a pipeline, so it
        is where assembly can stop being the node's own business. Handed
        a bare source, sink or widget core, this returns the chain that
        carries it -- Link, Sync, recording stage and all -- and handed
        anything else it returns what it was given.

        A chain is returned unchanged, so a hand-written one is left as
        its author built it. See ``wrapping.wrap``.

        The same core handed in twice gets the same chain, held in a map
        keyed on the node. A second
        chain would be a second Link on the same stream id, which is the
        kind of fault that shows up as a silent half-delivery rather
        than as an error.

        Args:
            element: Node, chain, or ``node["port"]`` specification.

        Returns:
            The element to connect, which may be the one passed in. A
            port specification comes back as a new dict naming the
            chain, leaving the caller's own dict untouched.
        """
        from .core._private import wrapping

        if isinstance(element, dict):
            node = element.get("node")
            wrapped = self._wrap(node)
            if wrapped is node:
                return element
            return {**element, "node": wrapped}

        if element is None or isinstance(element, ioc.Chain):
            return element

        # Keyed on the node, not on `_elements`: `add_node` puts a node
        # into the pipeline without registering it there -- deliberately, as
        # `_validate_has_signal` records -- so a pipeline that adds a core
        # and then connects it would otherwise get a second chain for the
        # same core. Which is a second Link on one stream id, and that
        # fails as a silent half-delivery rather than as an error.
        #
        # The chain holds the core, so the identity cannot be reused
        # while the entry is live.
        chain = self._wrappers.get(id(element))
        if chain is None:
            chain = wrapping.wrap(element)
            if chain is not element:
                self._wrappers[id(element)] = chain
                self._built_residencies.add(LaunchConfig.get().residency)
        return chain

    def add_node(self, node) -> None:
        """Add a node to the pipeline, wrapping it if it needs a chain.

        ``connect`` is the ordinary way in and does the same thing; this
        is the other one, and it has to agree. A core added here and
        connected afterwards would otherwise reach the pipeline twice --
        once bare and once inside its chain -- which starts the core's
        acquisition thread twice and writes it into the document twice.

        Unlike ``connect`` this does not register the node in
        ``_elements``, which is unchanged and deliberate: ``_elements``
        holds what was *connected*, and ``_validate_has_signal`` says so.

        Args:
            node: Node or chain to add.
        """
        super().add_node(self._wrap(node))

    def _timeline_consumers(self) -> list:
        """Return every node in this pipeline that wants the timeline.

        Any node exposing ``attach_timeline`` is offered it, so sinks
        that stamp outgoing data can reach it as well as Sync nodes.

        Returns:
            Nodes to bind, including those nested inside chains.
        """
        found = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if callable(
                    getattr(node, "attach_timeline", None)
                ) and not any(n is node for n in found):
                    found.append(node)
        return found

    def chain_of(self, node):
        """Return the chain carrying *node*, or None.

        The question a caller asks now that assembly is not the node's
        own business: a public node is not a chain, and the chain this
        pipeline built around it is found here. A hand-written chain
        answers for itself.

        Accepts a node this pipeline does not carry and answers None
        rather than raising: a test or a tool may reasonably ask about a
        node it has not connected yet, and an exception would make the
        question harder to ask than the reach-in it replaces.

        Args:
            node: A node handed to ``connect``, or one nested inside a
                chain that was.

        Returns:
            The chain, the node itself where it is already one, or None
            when this pipeline does not carry it.
        """
        for element in self._elements:
            if element is node:
                return element
            internal = getattr(element, "internal_nodes", None) or []
            if any(n is node for n in internal):
                return element

        # A chain the pipeline built holds its core whether or not it
        # run it, and under SERVER residency it does not: the scope
        # is not drawn and the device is not opened there. Walking
        # ``internal_nodes`` alone therefore answered None for a core
        # this pipeline is very much carrying, which is the one residency
        # where a caller most needs the answer.
        chain = self._wrappers.get(id(node))
        if chain is not None and any(e is chain for e in self._elements):
            return chain
        return None

    def _sources(self) -> list:
        """Return every source node in this pipeline.

        A source is identified by declaring a time base, which only
        Source subclasses do.

        Returns:
            Source nodes, including those nested inside chains.
        """
        found = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if getattr(node, "TIME_BASE", None) is not None and not any(
                    n is node for n in found
                ):
                    found.append(node)
        return found

    def _placement_consumers(self) -> list:
        """Return every node that places async inputs onto the grid.

        A separate sweep from the timeline one because the method has a
        separate name: a node that already defines ``attach_timeline``
        would shadow the placing mixin's version, so placement asks for
        the timeline under its own name.

        Returns:
            Nodes to inform, including those nested inside chains.
        """
        found = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if hasattr(node, "attach_placement_timeline") and not any(
                    n is node for n in found
                ):
                    found.append(node)
        return found

    def _flattened_nodes(self) -> list:
        """Return every node this pipeline runs in, chains flattened.

        Returns:
            The nodes, each once, nested chains included.
        """
        found = []
        seen = set()

        def visit(element) -> None:
            internal = getattr(element, "internal_nodes", None)
            if internal:
                for node in internal:
                    visit(node)
            elif id(element) not in seen:
                seen.add(id(element))
                found.append(element)

        for element in self._elements:
            visit(element)
        return found

    def _attach_position_gate(self) -> None:
        """Tell every node which of its outputs carry positions.

        Decided from the pipeline here, before any node runs, because a
        source's first frames leave its ``Sync`` before the node that
        places them has set up; see ``placement.position_gate``. A
        pipeline with no node that may place carries nothing, so its nodes
        run exactly as before carried positions existed.
        """
        nodes = self._flattened_nodes()
        candidates, carrying = position_gate(nodes)
        for node in nodes:
            attach = getattr(node, "attach_position_gate", None)
            if callable(attach):
                attach(carrying.get(id(node), ()), id(node) in candidates)

    def _record_failure(self, entry) -> None:
        """Remember the entry that ended the run.

        Called on ioiocore's monitoring thread, so it does as little as
        possible: storing a reference cannot fail, and anything that
        could would be happening on the only thread that reports
        failures at all.

        Args:
            entry: The failing log entry.
        """
        self._failure = entry

    @property
    def failure(self):
        """The entry that ended the run, or None if it has not failed.

        ioiocore sets ERROR, stops the pipeline and only then calls the
        handler that records this, so a poll can see ERROR first. Until
        the handler has run -- or for good, if a raising ``stop()``
        keeps it from running -- the entry is read from ioiocore's log
        instead. A failure therefore implies ERROR, but not yet STOPPED.

        Returns:
            A log entry carrying the message, the node and the source
            location.
        """
        if self._failure is not None:
            return self._failure
        if self.get_condition() != ioc.Constants.Conditions.ERROR:
            return None
        return self._consumed_error()

    def _consumed_error(self):
        """The ERROR entry ioiocore's monitor stopped the run on.

        The monitor takes the logger's latest ERROR and marks it seen by
        its timestamp. The latest moves on if stopping logs another, so
        the entry is found by that mark, newest first.

        It falls back to the latest error when the marked entry has left
        the logger's 100-entry buffer, or when ioiocore keeps no mark.
        And an error logged after the monitor's read, within one clock
        tick of the entry it read, shares the mark and is reported
        instead; a tick is 15.6 ms on Windows under Python 3.10-3.12.

        Returns:
            A log entry, or None if no error was logged.
        """
        error = ioc.Constants.LogTypes.ERROR
        latest = self.get_last_error()
        logger = getattr(self._imp, "_logger", None)
        marks = getattr(getattr(logger, "_imp", None), "_last_timestamp", None)
        mark = marks.get(error) if isinstance(marks, dict) else None
        if latest is None or mark is None or latest["timestamp"] == mark:
            return latest
        for entry in reversed(logger.get_by_type(error)):
            if entry["timestamp"] == mark:
                return entry
        return latest

    def raise_if_failed(self) -> None:
        """Raise if the run has failed, naming the node and the cause.

        Overridden to name :attr:`failure`. ioiocore's names its latest
        error, a different entry once stopping has logged another.

        Raises:
            RuntimeError: If the pipeline is in an error condition.
        """
        if self.get_condition() != ioc.Constants.Conditions.ERROR:
            return
        entry = self.failure
        error = RuntimeError("The pipeline failed.")
        if entry is not None:
            source = entry["source"] or {}
            parts = [str(entry["message"])]
            if source.get("instance"):
                parts.append(f"in node '{source['instance']}'")
            if source.get("summary"):
                parts.append(f"at {source['summary']}")
            error = RuntimeError("The pipeline failed: " + " ".join(parts))
        cause = self._first_failure([node for node, _ in self._walk_nodes()])
        if cause is not None:
            raise error from cause
        raise error

    def _links(self) -> list:
        """Return every Link in this pipeline, including nested ones.

        Same traversal as the other sweeps, and nested for the same
        reason: a Link lives inside the chain of whatever node spans the
        residency boundary.

        Returns:
            Link nodes.
        """
        from .core._private.link import Link

        found = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if isinstance(node, Link) and not any(
                    n is node for n in found
                ):
                    found.append(node)
        return found

    def _first_device_handle(self):
        """Return a handle that can attest, or None.

        One challenge carries one attestation, so an edge holding several
        amplifiers answers with the first that has a handle. Sufficient
        under the rule that ships -- the run is attested if *any* source
        attested -- and worth revisiting only if the §5.4 matrix ever
        needs a per-device remote verdict.

        Returns:
            An amplifier handle, or None if this pipeline has none.
        """
        for node in self._sources():
            handle = getattr(node, "_device", None)
            if handle is not None:
                return handle
        return None

    def _provenance_consumers(self) -> list:
        """Return every node that wants this run's provenance record.

        Any node exposing ``attach_provenance`` is offered it -- every
        :class:`~gpype.Source` does, which is how it comes to merge its
        own ``INPUT`` in and publish the whole thing under
        ``Constants.Keys.PROVENANCE`` (LQ-P2). Same shape as the
        timeline and entitlement sweeps.

        Returns:
            Nodes to inform, including those nested inside chains.
        """
        found = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if callable(
                    getattr(node, "attach_provenance", None)
                ) and not any(n is node for n in found):
                    found.append(node)
        return found

    def _entitlement_consumers(self) -> list:
        """Return every node that wants to know the run's entitlement.

        Any node exposing ``attach_entitlement`` is offered it, which is
        how a sink learns whether its output must carry the mark. Same
        shape as the timeline sweep, and it reaches inside chains for the
        same reason: every sink is a chain wrapping a core.

        Returns:
            Nodes to inform, including those nested inside chains.
        """
        found = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            for node in candidates:
                if callable(
                    getattr(node, "attach_entitlement", None)
                ) and not any(n is node for n in found):
                    found.append(node)
        return found

    def _validate(self) -> None:
        """Reject pipelines that cannot mean anything, before they run.

        Validity is not entitlement and not configuration: these are
        pipelines whose sources cannot be reconciled at all. Checked here,
        ahead of everything else, so the error names the real problem
        rather than surfacing later as a licence message or as data that
        merely looks wrong.

        Raises:
            ValueError: If sources with different time bases are mixed,
                if no source produces a continuously sampled stream, if
                the pipeline contains an illegal feedback loop, or if an
                input that inherits its timing is not connected.
        """
        self._validate_time_bases()
        self._validate_has_signal()
        self._validate_no_illegal_cycles()
        self._validate_batch_only_nodes()
        self._validate_fittable_state()
        self._validate_inputs_connected()
        self._warn_unknown_chain_keys()
        self._warn_untapped_sources()
        self._warn_edge_named_by_nothing()
        self._warn_sources_every_edge_sends()
        self._warn_edge_ids_differing_in_case()

    def _wrapped_cores(self) -> list:
        """The world-facing node each chain of this pipeline carries.

        Returns:
            The cores, in element order; an element that is not one of
            the pipeline's chains contributes nothing.
        """
        from .core._private import wrapping

        cores = [wrapping.core_of(element) for element in self._elements]
        return [core for core in cores if core is not None]

    def _warn_sources_every_edge_sends(self) -> None:
        """Say so when this edge opens a source the document leaves to all.

        A source that names no edge is built by every edge running the
        document, which is what a document written before edge ids says
        and still means. In a document that assigns other nodes to
        edges, several edges are the likely deployment, and then each
        one opens that source and sends its stream: two senders on one
        stream id. The broker keeps the first to write and refuses the
        rest (D-NODE-52), so the server's stream is one sender's -- but
        which edge's is a race, and every other edge's device was opened
        for nothing.

        Warned rather than refused: a document that assigns only its
        sinks and is run by one edge is a legitimate deployment, and an
        edge cannot count the edges that run beside it. Said on the edge,
        where the source is opened; the refusal is said on the server.
        """
        from .core._private import assembly

        if LaunchConfig.get().residency != Constants.Residency.EDGE:
            return
        cores = self._wrapped_cores()
        named = {assembly.edge_id_of(core) for core in cores} - {None}
        if not named:
            return
        unassigned = sorted(
            core.name
            for core in cores
            if assembly.kind_of(core) == assembly.SOURCE
            and assembly.edge_id_of(core) is None
        )
        if not unassigned:
            return
        one = len(unassigned) == 1
        names = ", ".join(repr(name) for name in unassigned)
        ids = ", ".join(repr(edge_id) for edge_id in sorted(named))
        self.log(
            f"{names} {'names' if one else 'name'} no edge, so every edge "
            f"that runs this document opens {'it' if one else 'them'} "
            f"and sends {'its stream' if one else 'their streams'}, while "
            f"the document assigns other nodes to {ids}. Run by two edges "
            f"that is two senders on one stream: the server keeps the "
            f"edge that connects first and refuses the other. Give each "
            f"source an edge_id.",
            type=Constants.LogTypes.WARNING,
        )

    def _warn_edge_ids_differing_in_case(self) -> None:
        """Say so when two edge ids in play differ only in case.

        Ids are matched exactly (D-NODE-49), so ``'Phone'`` and
        ``'phone'`` are two edges, and a node assigned the one is built
        by no process running as the other. When this edge's id is named
        by some other node, :meth:`_warn_edge_named_by_nothing` stays
        quiet, and the node that missed is simply not built anywhere --
        no writer, no file, no message. Surrounding whitespace, the other
        near miss, is refused wherever an id is written.

        Asked on an edge, with its own id among the spellings, and on a
        server, which sees the whole document; standalone ignores ids.
        """
        from .core._private import assembly

        launch = LaunchConfig.get()
        if launch.residency == Constants.Residency.STANDALONE:
            return
        spellings: dict = {}
        for core in self._wrapped_cores():
            edge_id = assembly.edge_id_of(core)
            if edge_id is not None:
                owners = spellings.setdefault(edge_id.casefold(), {})
                owners.setdefault(edge_id, []).append(repr(core.name))
        if launch.edge_id is not None:
            owners = spellings.setdefault(launch.edge_id.casefold(), {})
            owners.setdefault(launch.edge_id, []).append("this edge")
        for owners in spellings.values():
            if len(owners) < 2:
                continue
            described = ", ".join(
                f"{edge_id!r} ({', '.join(owners[edge_id])})"
                for edge_id in sorted(owners)
            )
            self.log(
                f"edge ids {described} differ only in case. Ids are "
                f"matched exactly, so these are different edges: a node "
                f"assigned one of them is built by no edge running as "
                f"another. Spell each edge one way.",
                type=Constants.LogTypes.WARNING,
            )

    def _warn_edge_named_by_nothing(self) -> None:
        """Say so when this edge's id is one the document never uses.

        An edge launched with a mistyped ``--edge-id``, or with none
        against a document that assigns every node, builds only the nodes
        that name no edge -- possibly none at all. It then starts, reports
        healthy and sends nothing, and the only symptom is a stream that
        never arrives at the server.

        Warned rather than refused: an edge that runs just the unassigned
        nodes -- a viewer that draws the scopes every edge draws -- is a
        legitimate deployment, and the same document serves it.
        """
        from .core._private import assembly, wrapping

        launch = LaunchConfig.get()
        if launch.residency != Constants.Residency.EDGE:
            return
        named = set()
        for element in self._elements:
            edge_id = assembly.edge_id_of(wrapping.core_of(element))
            if edge_id is not None:
                named.add(edge_id)
        if not named or launch.edge_id in named:
            return
        this = (
            f"edge {launch.edge_id!r}"
            if launch.edge_id is not None
            else "an edge with no edge id"
        )
        ids = ", ".join(repr(edge_id) for edge_id in sorted(named))
        self.log(
            f"this process is {this}, and no node in this pipeline is "
            f"assigned to it: the document assigns nodes to {ids}. It "
            f"builds only the nodes that name no edge. If this process "
            f"holds a device, check its --edge-id (or {ENV_EDGE_ID}).",
            type=Constants.LogTypes.WARNING,
        )

    def _warn_untapped_sources(self) -> None:
        """Name any source this run's ``save_as``/``load_from`` misses.

        Both are inserted by ``raw.source_stage``, which every chain the
        pipeline builds calls. A source inside a chain written by hand
        that does not call it has no such point at all, so ``save_as``
        contributes no file for it and ``load_from`` leaves its live core
        in place.

        Both halves matter, and the second one more than it looks.
        Under ``save_as``, a manifest that simply does not mention a
        stream is indistinguishable from a manifest of a pipeline that
        never had it. Under ``load_from``, that source goes on
        acquiring live data beside the replayed streams -- and since a
        replay core declares WALL_CLOCK (D-TIME-28),
        ``_validate_time_bases`` no longer refuses the mixture the way
        it refuses a file reader beside a live source. Nothing else
        would say so.

        Warned rather than refused: an unrecorded source is not a reason
        to take away a working pipeline. But it must be said, because
        the promise of both parameters is *every* source, and the run
        that discovers otherwise is the one being replayed.
        """
        config = LaunchConfig.get()
        if not config.save_as and not config.load_from:
            return

        from .sources.base.raw import _RawReplayCore, _RawTap

        staged = set()
        for element in self._elements:
            internal = getattr(element, "internal_nodes", None) or []
            if any(
                isinstance(node, (_RawTap, _RawReplayCore))
                for node in internal
            ):
                staged.update(id(node) for node in internal)

        missing = sorted(
            naming.public_name(type(node).__name__)
            for node in self._sources()
            if id(node) not in staged
        )
        if not missing:
            return
        names = ", ".join(missing)
        if config.save_as:
            self.log(
                f"save_as does not cover {names}: a source that is not a "
                f"chain has no point to tap, so this run's manifest will "
                f"not mention it and a replay of it will have no such "
                f"stream. Everything else in this pipeline is recorded.",
                type=Constants.LogTypes.WARNING,
            )
        else:
            self.log(
                f"load_from does not cover {names}: a source that is not "
                f"a chain keeps its own core, so it is acquiring live "
                f"data beside the replayed streams. Nothing refuses that "
                f"mixture -- a replay core is WALL_CLOCK, like a live "
                f"source -- so the two have no instant in common beyond "
                f"the one this run happens to give them.",
                type=Constants.LogTypes.WARNING,
            )

    def _validate_batch_only_nodes(self) -> None:
        """Refuse a node that only means something offline.

        ``Apply`` and anything else declaring ``BATCH_ONLY`` runs a
        plain callable, and D-BATCH-56 keeps that off the realtime path:
        an arbitrary function has no bound on its execution time against
        a one-frame budget, and purity is not enforceable there.

        Refused here rather than on the first cycle, and the difference
        is not cosmetic. By cycle time a source may already hold a
        device and a writer may already have created a file, so a
        refusal then leaves work to undo; and ``start()`` would have
        returned normally, which is the failure ``_await_setup`` exists
        to stop happening.

        Raises:
            ValueError: If the pipeline is not in batch mode and
                contains a node that requires one.
        """
        batch_only = [
            node
            for node, _ in self._walk_nodes()
            if getattr(type(node), "BATCH_ONLY", False)
        ]
        if not batch_only:
            return

        # Asked after the nodes, so a pipeline with none of them never
        # pays for this -- and so a pipeline with no source at all
        # still fails with "no source" rather than with a mode error.
        mode = self.execution_mode()
        if mode == Constants.ExecutionMode.BATCH:
            return

        names = ", ".join(
            sorted(naming.public_name(type(n).__name__) for n in batch_only)
        )
        raise ValueError(
            f"{names} runs only in a batch pipeline, and this one is "
            f"'{mode}'. A plain function has no bound on how long it "
            f"takes, and one frame is 4 to 16 ms -- so it is offered "
            f"offline only. Either drive this with a batch source -- "
            f"CsvReader(..., mode='batch') and run() -- or write a node "
            f"class with a step() for the realtime path."
        )

    def _refuse_batch_without_run(self) -> None:
        """Refuse a batch pipeline that start() was called on directly.

        run() drives a batch source's cycles on the caller's thread;
        start() alone never does, so the pipeline reported RUNNING and
        Healthy, logged nothing, and processed nothing. The mirror of
        run()'s refusal of a realtime pipeline.

        Kept out of :meth:`_validate`, which judges the pipeline: this
        judges how it is being driven, and run() reaches it through
        start().

        Raises:
            ValueError: If the pipeline is batch and run() is not
                driving it.
        """
        # A pipeline with no source of its own -- a server, or a Link
        # receiver -- has no mode to judge, and one whose sources
        # disagree is not this check's to refuse: start() never
        # consulted the mode for either, and still does not.
        if getattr(self, "_batch_driven", False) or not self._sources():
            return
        try:
            mode = self.execution_mode()
        except ValueError:
            return
        if mode != Constants.ExecutionMode.BATCH:
            return
        raise ValueError(
            "start() runs a realtime pipeline, and this one is 'batch': "
            "its source is read once, as fast as possible, and only "
            "run() drives it. Call run(), which returns the result -- or, "
            "for a recording reader, leave mode at its default to replay "
            "the file against the clock with start() and stop()."
        )

    def _validate_fittable_state(self) -> None:
        """Refuse a REALTIME start with a fittable node that has no state.

        Implements D-BATCH-33: a fittable node fits only in a batch run
        (:meth:`fit`), so one with nothing loaded is a deployment
        mistake, not a training run in progress. Refused here for the
        same reason as :meth:`_validate_batch_only_nodes`: by cycle time
        a source may already hold a device.

        A batch run is exempt -- that is exactly how a first fit
        happens -- and a batch run that is not fitting and finds no
        state fails later, from inside the node's own ``process``.

        Raises:
            ValueError: If the pipeline is not batch and holds a fittable
                node with no state, naming it.
        """
        unfit = [node for node in self._fittable_nodes() if node.state is None]
        if not unfit:
            return

        mode = self.execution_mode()
        if mode == Constants.ExecutionMode.BATCH:
            return

        names = ", ".join(
            sorted(
                f"{naming.public_name(type(n).__name__)} ('{n.name}')"
                for n in unfit
            )
        )
        raise ValueError(
            f"{names} has no fitted state, and this pipeline is "
            f"'{mode}'. Fit it first -- Pipeline.fit() against a batch "
            f"source -- or deserialize a document whose artifacts "
            f"section already carries one for it."
        )

    def _validate_inputs_connected(self) -> None:
        """Refuse an input that inherits its timing and is not connected.

        An input port declared ``INHERITED`` -- ``Collector``'s,
        ``Router``'s, ``Equation``'s, ``Trigger``'s trigger input -- takes
        its timing from what connects to it.
        Unconnected, it has none, and ioiocore refuses it at start with
        ``INHERITED timing not allowed on INode input ports during
        start()``, which names neither the node nor the port. The rule is
        ioiocore's; this names what it refuses (D-BATCH-74).

        Walks the pipeline's own nodes, which ``add_node`` reaches and
        ``_elements`` does not, and every chain's internal nodes.

        Raises:
            ValueError: If an input port's timing is still inherited,
                naming the port and its node.
        """
        from .core._private import wrapping

        ports_key = Constants.Keys.INPUT_PORTS
        name_key = OPort.Configuration.Keys.NAME
        loose: list = []
        seen: set = set()

        def visit(node, core) -> None:
            internal = getattr(node, "internal_nodes", None)
            if internal is not None:
                for inner in internal:
                    visit(inner, core)
                return
            config = getattr(node, "config", None) or {}
            for port in config.get(ports_key) or []:
                name = port.get(name_key)
                try:
                    timing = node.get_input_port(name).timing
                except Exception:
                    continue
                if timing != Constants.Timing.INHERITED:
                    continue
                if (id(node), name) in seen:
                    continue
                seen.add((id(node), name))
                # The node the author wrote: the core a chain of ours
                # carries, or the node itself.
                owner = core if core is not None else node
                loose.append(f"'{name}' of {naming.node_label(owner)}")

        for element in list(getattr(self._imp, "_nodes", None) or []):
            visit(element, wrapping.core_of(element))
        if not loose:
            return
        one = len(loose) == 1
        raise ValueError(
            f"{'input' if one else 'inputs'} {', '.join(loose)} "
            f"{'is' if one else 'are'} not connected. An input like that "
            f"takes its timing from what connects to it, so unconnected "
            f"it has none and the pipeline cannot start. Connect it, or "
            f"leave its node out of the pipeline."
        )

    def _warn_unknown_chain_keys(self) -> None:
        """Warn about configuration keys a chain does not recognise.

        ``Node.__init__`` already does this for nodes, and a chain never
        reaches it: a chain is not a Node, and a stray keyword lands on
        *its* configuration rather than on the core's. That is the case
        that costs debugging time. ``Generator(amplitude=100.0)`` instead
        of ``signal_amplitude`` is accepted, stored, and survives
        re-serialisation, while the pipeline reports Running and Healthy
        and emits exact zeros -- three debugging cycles in one session,
        recorded in devdoc/archive/TODO.md.

        Warned, not refused, on the owner's instruction: refusing would
        stop every stored document carrying a stray key from loading at
        all, and a key nothing reads is not a reason to refuse to run.

        The rule is shared -- ``common.config_keys`` -- with the node
        check and with the runtime's answer to "is this document valid",
        so all three say the same thing about the same document rather
        than agreeing until one of them changes.
        """
        for element in self._elements:
            # A node, not a chain: Node.__init__ has already looked, and
            # warning twice about one key reads as two problems.
            if getattr(element, "internal_nodes", None) is None:
                continue
            config = getattr(element, "config", None)
            unknown = config_keys.unrecognised_keys(
                type(element), config if isinstance(config, dict) else {}
            )
            if not unknown:
                continue
            message = config_keys.describe(type(element), unknown)
            logger = getattr(element, "log", None)
            if callable(logger):
                logger(message, type=Constants.LogTypes.WARNING)
            else:  # pragma: no cover - a chain that cannot log yet
                self.log(message, type=Constants.LogTypes.WARNING)

    def _validate_time_bases(self) -> None:
        """Reject a pipeline mixing replayed and live sources.

        Raises:
            ValueError: If sources with different time bases are mixed.
        """
        bases = {}
        for node in self._sources():
            bases.setdefault(node.TIME_BASE, []).append(type(node).__name__)
        if len(bases) > 1:
            recorded = ", ".join(
                sorted(bases.get(Constants.TimeBase.RECORDED, []))
            )
            live = ", ".join(
                sorted(bases.get(Constants.TimeBase.WALL_CLOCK, []))
            )
            raise ValueError(
                f"a file source cannot be combined with live sources: "
                f"{recorded} replays stored data while {live} "
                f"advance{'s' if len(bases.get(Constants.TimeBase.WALL_CLOCK, [])) == 1 else ''} "  # noqa: E501
                f"with the host clock, so there is no single instant "
                f"their samples both refer to."
            )

    def _validate_has_signal(self) -> None:
        """Reject a pipeline built only from event sources.

        Every position in a pipeline comes from the master timeline, and
        the master timeline comes from a continuously sampled source: it
        is what defines the sample grid that everything else is placed
        on. An event has no position of its own -- a keypress is a host
        timestamp, nothing more -- so with no continuous source there is
        no grid to place it on, and placement is where an event acquires
        a position at all.

        Such a pipeline does not fail. It starts, reports healthy, cycles
        once at start and then never again, and writes nothing. That is
        the worst possible way for it to behave, because the author has
        no symptom to work from: measured on a Marker feeding a Router,
        one cycle and no values, with every node reporting Healthy.

        Refusing at start() turns that into a sentence naming the cause.

        Raises:
            ValueError: If sources exist but none of them is continuous.
        """
        if not self._sources():
            # Nothing was connected, or everything reached the pipeline
            # through add_node(), which does not register. Not this
            # check's business either way.
            return

        timing_key = OPort.Configuration.Keys.TIMING
        ports_key = ioc.ONode.Configuration.Keys.OUTPUT_PORTS

        # Walked per element rather than over _sources(), so the message
        # can name what the author wrote: the node a chain carries, not
        # the chain -- "SourceChain" is no help to somebody who wrote
        # gp.Marker().
        from .core._private import wrapping

        event_only = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
            # The core too, run here or not. An edge that owns only a
            # Marker runs no continuous source of its own -- the
            # amplifier belongs to another edge -- and the grid its
            # markers are placed on is the server's, fed by that other
            # edge. The question is whether the *document* has a signal,
            # and a core this process does not run is still in it.
            core = wrapping.core_of(element)
            if core is not None and not any(c is core for c in candidates):
                candidates.append(core)
            cores = [
                node
                for node in candidates
                if getattr(node, "TIME_BASE", None) is not None
            ]
            if not cores:
                continue
            for core in cores:
                ports = core.config.get(ports_key) or []
                timings = [port.get(timing_key) for port in ports]
                # Anything not declared ASYNC counts as continuous. The
                # test is deliberately this way round: a false refusal
                # breaks a working pipeline, while a false acceptance
                # only leaves the behaviour that was there before.
                if any(t != Constants.Timing.ASYNC for t in timings):
                    return
            owner = wrapping.core_of(element) or element
            event_only.append(naming.public_name(type(owner).__name__))

        named = sorted(set(event_only))
        names = ", ".join(named)
        emit = "emits" if len(named) == 1 else "emit"
        raise ValueError(
            f"this pipeline has no continuously sampled source: "
            f"{names} {emit} events, and an event has no position until "
            f"something places it on a sample grid. The grid comes from "
            f"a continuous source, so there is nothing here to place "
            f"events on and the pipeline would run without producing "
            f"any output. Add a data source -- an amplifier, Generator "
            f"or CsvReader -- alongside the event source."
        )

    def _validate_no_illegal_cycles(self) -> None:
        """Reject a pipeline containing an illegal feedback loop.

        Runs here for a pipeline assembled by :meth:`deserialize`, which
        never passes through :meth:`connect` at all. For an ordinarily
        built pipeline it is mostly a second look at rules already
        applied per-edge -- see :meth:`connect` -- with one rule that
        only this call can apply: a loop must be fed from outside
        itself, which is a property of the finished pipeline and cannot be
        judged while edges are still arriving.

        Raises:
            ValueError: If the pipeline contains a loop refused by
                :meth:`_describe_component_violation`.
        """
        violation = self._cycle_violation()
        if violation is not None:
            raise ValueError(violation)

    def _cycle_owner_map(self) -> dict:
        """Map every declared input port to the element that owns it.

        Rebuilt on every call rather than cached, because the answer
        changes on every :meth:`connect` and nothing here tracks that
        invalidation. The pipelines this runs against are, at most, dozens
        of nodes -- nothing next to actually running a pipeline.

        Returns:
            ``id(port) -> element``, for every declared input port of
            every registered element that exposes one. An element that
            cannot be introspected (a malformed config, a port that will
            not resolve) is skipped rather than raised on -- this check
            must never be the reason an otherwise-working connect()
            fails. Every skip is logged, though: an edge this map misses
            is an edge the loop check cannot see, so a swallowed error
            degrades the detector towards "no loops found" -- exactly
            the silent acceptance it exists to prevent. The reason has
            to stay recoverable from a debug log instead of vanishing.
        """
        ip_key = ioc.INode.Configuration.Keys.INPUT_PORTS
        name_key = ioc.IPort.Configuration.Keys.NAME
        owners: dict = {}
        for element in self._elements:
            getter = getattr(element, "get_input_port", None)
            if getter is None:
                # Not a failure: every gpype source is an OChain, which
                # has no input ports at all, so no edge can end at it.
                continue
            try:
                specs = element.config.get(ip_key) or []
            except Exception as error:
                log.debug(
                    "cycle check: cannot read the input ports of %s "
                    "(%s): %s -- edges into it stay invisible to the "
                    "loop check",
                    getattr(element, "name", "?"),
                    type(element).__name__,
                    error,
                )
                continue
            for spec in specs:
                name = spec.get(name_key)
                try:
                    port = getter(name)
                except Exception as error:
                    log.debug(
                        "cycle check: cannot resolve input port %r of "
                        "%s: %s -- edges into that port stay invisible "
                        "to the loop check",
                        name,
                        getattr(element, "name", "?"),
                        error,
                    )
                    continue
                owners[id(port)] = element
        return owners

    def _cycle_adjacency(self) -> dict:
        """Return the directed connections built by connect() so far.

        Returns:
            ``id(element) -> [downstream elements]``, one entry per
            outgoing edge -- an element with a fan-out of two appears
            twice, pointing at each of its consumers. Elements that
            cannot be introspected are skipped and logged, for the
            reason given in :meth:`_cycle_owner_map`.
        """
        op_key = ioc.ONode.Configuration.Keys.OUTPUT_PORTS
        name_key = ioc.IPort.Configuration.Keys.NAME
        owners = self._cycle_owner_map()
        adjacency: dict = {}
        for element in self._elements:
            getter = getattr(element, "get_output_port", None)
            if getter is None:
                # Not a failure: a sink or a scope is an IChain, which
                # has no output ports, so no edge can start at it.
                continue
            try:
                specs = element.config.get(op_key) or []
            except Exception as error:
                log.debug(
                    "cycle check: cannot read the output ports of %s "
                    "(%s): %s -- edges out of it stay invisible to the "
                    "loop check",
                    getattr(element, "name", "?"),
                    type(element).__name__,
                    error,
                )
                continue
            for spec in specs:
                name = spec.get(name_key)
                try:
                    port = getter(name)
                except Exception as error:
                    log.debug(
                        "cycle check: cannot resolve output port %r of "
                        "%s: %s -- edges out of that port stay "
                        "invisible to the loop check",
                        name,
                        getattr(element, "name", "?"),
                        error,
                    )
                    continue
                for connected in list(getattr(port, "_connected_ports", ())):
                    downstream = owners.get(id(connected))
                    if downstream is not None:
                        adjacency.setdefault(id(element), []).append(
                            downstream
                        )
        return adjacency

    def _cycle_components(self, adjacency: dict) -> list:
        """Return every strongly connected component holding a cycle.

        Classifying per component rather than per cycle is the whole
        point of this detector, and the reason it stopped enumerating
        cycles. Enumeration cannot be made correct here: a depth-first
        search that slices a path out whenever it meets a node already
        on its stack skips edges into finished nodes, so it reports at
        most one cycle per branch and never sees a second cycle sharing
        nodes with the first. Measured on ``A -> B -> C -> A`` with B a
        declaring :class:`~gpype.Delay`, then adding ``A -> C``: the
        legal loop was reported, the illegal ``A -> C -> A`` -- no
        breaker in it, the exact deadlock this check exists to prevent
        -- was not, and connect() accepted it.

        A component contains *every* cycle through its nodes, so a
        property established for the component holds for all of them,
        with nothing left to enumerate. Tarjan's algorithm, written
        iteratively because recursion depth here is the pipeline's depth
        and a pipeline may be deeper than the interpreter's limit;
        linear in nodes plus edges.

        Args:
            adjacency: The connections from :meth:`_cycle_adjacency`.

        Returns:
            One list of elements per component that contains at least
            one cycle -- either more than one member, or a single member
            with an edge to itself. A component with one member and no
            self-edge is just an ordinary node and is left out. Ordered
            by each component's earliest member in ``_elements``, so
            which loop gets reported first does not wander between runs.
        """
        index: dict = {}
        low: dict = {}
        on_stack: dict = {}
        pending: list = []
        components: list = []
        counter = 0

        for root in self._elements:
            if id(root) in index:
                continue
            index[id(root)] = low[id(root)] = counter
            counter += 1
            pending.append(root)
            on_stack[id(root)] = True
            # Each frame is a node and its not-yet-followed successors;
            # the iterator is what makes resuming a frame cheap.
            work = [(root, iter(adjacency.get(id(root), ())))]
            while work:
                node, successors = work[-1]
                descended = False
                for downstream in successors:
                    did = id(downstream)
                    if did not in index:
                        index[did] = low[did] = counter
                        counter += 1
                        pending.append(downstream)
                        on_stack[did] = True
                        work.append((downstream, iter(adjacency.get(did, ()))))
                        descended = True
                        break
                    if on_stack.get(did):
                        low[id(node)] = min(low[id(node)], index[did])
                if descended:
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[id(parent)] = min(low[id(parent)], low[id(node)])
                if low[id(node)] == index[id(node)]:
                    members = []
                    while True:
                        member = pending.pop()
                        on_stack[id(member)] = False
                        members.append(member)
                        if member is node:
                            break
                    components.append(members)

        cyclic = []
        for members in components:
            if len(members) > 1:
                cyclic.append(members)
                continue
            node = members[0]
            if any(d is node for d in adjacency.get(id(node), ())):
                cyclic.append(members)

        last = len(self._elements)
        order = {id(e): i for i, e in enumerate(self._elements)}
        cyclic.sort(key=lambda ms: min(order.get(id(m), last) for m in ms))
        return cyclic

    def _first_cycle(self, nodes: set, adjacency: dict) -> list:
        """Return the first cycle lying inside nodes, or an empty list.

        A depth-first search restricted to ``nodes``, colouring an
        element grey while it sits on the current path and black once
        fully explored; an edge into a grey element closes a cycle. This
        doubles as the acyclicity test the rules below need -- an empty
        result means ``nodes`` induces no cycle at all -- and it is
        linear in nodes plus edges, so running it on every connect()
        costs nothing worth measuring.

        The path holds *nodes*, not edges. An earlier version of this
        detector kept an edge stack and named the wrong entry node for a
        self-loop (``Equation -> Equation`` was reported as
        ``Generator -> Equation -> Generator``), because an edge stack
        records where a node was reached *from*, not the node reached.

        Args:
            nodes: Ids of the elements the search may enter; an edge to
                anything else is treated as if it did not exist.
            adjacency: The connections from :meth:`_cycle_adjacency`.

        Returns:
            The cycle's elements in traversal order, the element where
            it closes first, or an empty list if there is no cycle.
        """
        grey, black = 1, 2
        colour: dict = {}
        for root in self._elements:
            if id(root) not in nodes or colour.get(id(root)):
                continue
            path: list = [root]
            colour[id(root)] = grey
            work = [iter(adjacency.get(id(root), ()))]
            while work:
                descended = False
                for downstream in work[-1]:
                    did = id(downstream)
                    if did not in nodes or colour.get(did) == black:
                        continue
                    if colour.get(did) == grey:
                        start = next(
                            i for i, n in enumerate(path) if id(n) == did
                        )
                        return path[start:]
                    colour[did] = grey
                    path.append(downstream)
                    work.append(iter(adjacency.get(did, ())))
                    descended = True
                    break
                if descended:
                    continue
                colour[id(path.pop())] = black
                work.pop()
        return []

    def _cycle_through(self, node, nodes: set, adjacency: dict) -> list:
        """Return one cycle that starts and ends at node.

        Only ever used to name a loop in a refusal, so the path a
        message prints begins at the node that message blames. Runs on
        the refusal path alone -- an accepted pipeline never calls it.

        Args:
            node: Element the returned path must start at.
            nodes: Ids of the elements the search may enter.
            adjacency: The connections from :meth:`_cycle_adjacency`.

        Returns:
            The cycle's elements, ``node`` first, or an empty list if no
            cycle through ``node`` lies inside ``nodes`` -- which cannot
            happen for a member of a cyclic component, since every node
            of one lies on a cycle with every other.
        """
        path: list = [node]
        seen = {id(node)}

        def walk(current) -> bool:
            for downstream in adjacency.get(id(current), ()):
                did = id(downstream)
                if did == id(node):
                    return True
                if did not in nodes or did in seen:
                    continue
                seen.add(did)
                path.append(downstream)
                if walk(downstream):
                    return True
                path.pop()
            return False

        return path if walk(node) else []

    def _async_input_of(self, node) -> Optional[str]:
        """Return the name of node's first connected asynchronous input.

        Args:
            node: Candidate node inside a feedback loop.

        Returns:
            The port name, or None if every connected input port is
            synchronous. An input declared ASYNC but left unconnected
            does not count -- nothing arrives on it, so it plays no part
            in this loop's timing. A port that cannot be introspected is
            skipped and logged, for the reason given in
            :meth:`_cycle_owner_map`.
        """
        ip_key = ioc.INode.Configuration.Keys.INPUT_PORTS
        name_key = ioc.IPort.Configuration.Keys.NAME
        getter = getattr(node, "get_input_port", None)
        if getter is None:
            return None
        try:
            specs = node.config.get(ip_key) or []
        except Exception as error:
            log.debug(
                "cycle check: cannot read the input ports of %s (%s): "
                "%s -- treated as fully synchronous, so an async input "
                "on it would not be refused",
                getattr(node, "name", "?"),
                type(node).__name__,
                error,
            )
            return None
        for spec in specs:
            name = spec.get(name_key)
            try:
                port = getter(name)
            except Exception as error:
                log.debug(
                    "cycle check: cannot resolve input port %r of %s: "
                    "%s -- treated as synchronous, so an async timing "
                    "on it would not be refused",
                    name,
                    getattr(node, "name", "?"),
                    error,
                )
                continue
            if not port.is_connected():
                continue
            if port.timing == Constants.Timing.ASYNC:
                return name
        return None

    @staticmethod
    def _cycle_path_text(members: list) -> str:
        """Render a loop as the round trip its author wrote.

        Args:
            members: The loop's elements, entry node first.

        Returns:
            ``"A -> B -> A"``, closing back on the entry node so the
            reader can see it is a loop and not a chain.
        """
        return " -> ".join(m.name for m in members + [members[0]])

    def _describe_component_violation(
        self, members: list, adjacency: dict, require_feed: bool = True
    ) -> Optional[str]:
        """Classify one cyclic component and refuse what cannot run.

        Three shapes are refused, all measured rather than merely
        reasoned about:

        1. Some cycle in the component has no breaker in it -- no
           :class:`~gpype.Delay` that declares its own context, which
           is the only form that can emit a frame before it has seen
           one. Nothing in that cycle ever fires, since every node is
           waiting on every other, so the pipeline reports Healthy,
           produces nothing, and grows its input queues without bound.
           A plain ``gp.Delay(num_samples=...)`` does not count: it
           derives its context from its input, so it reports
           ``BREAKS_CYCLES`` False. Tested by deleting the breakers
           and asking whether anything cyclic survives, which is exact:
           every cycle passes through a breaker if and only if removing
           the breakers leaves the rest acyclic. A component that shares
           nodes between a broken and an unbroken loop is the case that
           makes this worth doing -- see :meth:`_cycle_components`.
        2. No edge enters the component from outside it. The loop is
           then its own only producer as well as its own only consumer:
           it can never start, and if it somehow did, recursion would
           run until the process stack overflows (0xC00000FD on
           Windows, not a catchable RecursionError). Unlike the other
           two, this is a property of the *finished* pipeline rather than
           of any one edge: closing a loop before wiring the feed that
           drives it is an ordinary way to write the same pipeline. So
           it is checked at start() and skipped while edges are still
           being added -- see ``require_feed``.
        3. Some node in the component has a connected asynchronous
           input. Such a node fires as soon as its synchronous inputs
           have data, so with the loop edge as its only synchronous
           input it becomes its own clock: measured at roughly 43000
           traversals/s on a 250 Hz pipeline, 48% of one core,
           reporting Healthy throughout.

        A legal loop is therefore the complement: every cycle in it
        passes through a breaker, something outside feeds it, and every
        connected input of every node in it is synchronous.

        Args:
            members: The component's elements, in no useful order.
            adjacency: The connections from :meth:`_cycle_adjacency`.
            require_feed: Whether to apply rule 2. False while the
                pipeline is still being assembled, where the feed may
                simply not be connected yet.

        Returns:
            The refusal text naming the offending loop, or None if this
            component is legal.
        """
        member_ids = {id(m) for m in members}
        breakers = {
            id(m) for m in members if getattr(m, "BREAKS_CYCLES", False)
        }

        unbroken = self._first_cycle(member_ids - breakers, adjacency)
        if unbroken:
            path = self._cycle_path_text(unbroken)
            return (
                f"the connection closes a feedback loop: {path}. g.Pype "
                f"executes on data arrival, and a node fires only once "
                f"every one of its inputs has a frame, so a loop with "
                f"nothing in it never fires at all: the pipeline would "
                f"report Healthy, produce nothing, and grow its input "
                f"queues without bound. Insert a gp.Delay that "
                f"declares its own context -- gp.Delay(num_samples=1, "
                f"channel_count=..., sampling_rate=...) -- into this "
                f"loop: it emits one declared initial frame before it "
                f"has seen input, which is what lets the first cycle "
                f"happen, and makes the loop delay exactly num_samples "
                f"samples. A plain gp.Delay(num_samples=...) does not "
                f"count, because it derives its context from its input "
                f"and so cannot exist before the loop it would break. "
                f"A breaker elsewhere does not cover this loop either, "
                f"not even one in a loop sharing nodes with it: every "
                f"loop needs a breaker of its own."
            )

        fed_from_outside = True
        if require_feed:
            fed_from_outside = False
            for source_id, downstream in adjacency.items():
                if source_id in member_ids:
                    continue
                if any(id(d) in member_ids for d in downstream):
                    fed_from_outside = True
                    break
        if not fed_from_outside:
            cycle = self._first_cycle(member_ids, adjacency) or members
            path = self._cycle_path_text(cycle)
            return (
                f"the feedback loop {path} has no input from outside "
                f"the loop. Nothing outside it feeds any of its nodes, "
                f"so the loop is both its own only producer and its "
                f"own only consumer: it can never start, and if it ever "
                f"did it would recurse on one thread until the process "
                f"stack overflows. Feed one node in the loop from a "
                f"source outside it."
            )

        # Walked in _elements order rather than in the order Tarjan
        # popped the component, so the node a message blames is the
        # first one the author wrote, not an implementation detail.
        for node in (e for e in self._elements if id(e) in member_ids):
            port_name = self._async_input_of(node)
            if port_name is None:
                continue
            cycle = self._cycle_through(node, member_ids, adjacency) or members
            path = self._cycle_path_text(cycle)
            return (
                f"the feedback loop {path} is closed at "
                f"{node.name}, whose input port '{port_name}' "
                f"is asynchronous. A node with an asynchronous "
                f"input fires as soon as its synchronous inputs "
                f"have data, so with the loop edge as the only "
                f"synchronous input the loop becomes its own "
                f"clock: measured at 43000 traversals per second "
                f"on a 250 Hz pipeline, reporting Healthy "
                f"throughout. Every input port of every node in a "
                f"loop must be synchronous."
            )
        return None

    def _cycle_violation(self, require_feed: bool = True) -> Optional[str]:
        """Return the message for the first illegal loop, or None.

        Args:
            require_feed: Whether a loop must already be fed from
                outside itself. False while the pipeline is still being
                assembled -- see :meth:`_describe_component_violation`.

        Returns:
            Refusal text naming the offending loop, or None if every
            cycle in the pipeline built so far is legal (including none at
            all).
        """
        adjacency = self._cycle_adjacency()
        for members in self._cycle_components(adjacency):
            message = self._describe_component_violation(
                members, adjacency, require_feed=require_feed
            )
            if message is not None:
                return message
        return None

    def _reset_run_state(self) -> None:
        """Let every node set itself up again on the next run.

        ioiocore gates ``setup()`` on a per-node counter and resets that
        counter only when the previous run ended in ERROR. After an
        ordinary ``stop()`` it therefore never re-runs, and a second run
        is not a second run at all: measured on a Generator to CsvWriter
        pipeline, run 2 created no file and wrote no rows while
        ``get_condition()`` reported Healthy and the node counters climbed.
        Every sink behaves that way, because each one skips its work on the
        handle its own ``stop()`` set to None.

        So this is not new behaviour -- restart-after-error already re-runs
        setup(). It removes the inconsistency between the two restart
        paths, in the direction that works.

        Two things make it safe, and both were established by measurement
        rather than reasoning:

        It resets **ioiocore's own node list**, not ``self._elements``.
        ``_elements`` holds what passed through ``connect()``, so a node
        added by ``add_node()`` alone is invisible to it -- and a *partial*
        reset does not fail loudly, because a cleared input context is
        silently backfilled from an un-reset upstream port. Run 2 would set
        itself up against run 1's context and look plausible.

        It stays inside the caller's ``not RUNNING`` guard. ioiocore
        deliberately makes ``start()`` on a running pipeline a no-op;
        resetting there would re-run ``setup()`` mid-recording, reopen the
        *same* file path and truncate what had been captured, and destroy
        and recreate an LSL outlet under every connected consumer.
        """
        imp = getattr(self, "_imp", None)
        for node in list(getattr(imp, "_nodes", None) or []):
            reset = getattr(
                getattr(node, "_imp", None), "reset_run_state", None
            )
            if callable(reset):
                try:
                    reset()
                except Exception:
                    # A node that cannot reset must not prevent the others
                    # from doing so, nor block the start.
                    pass

    def _elect_master(self, timeline) -> None:
        """Settle the master-timeline election before any data flows.

        Candidacy is a construction-time property, so the election does not
        have to wait for anyone to produce a block -- and it must not. A
        source that claims on its first block wins by starting fastest, and
        a synthetic source always starts faster than hardware that has to
        open a device: measured against a g.HIamp, a 512 Hz Generator took
        the timeline every time and the amplifier's stronger claim arrived
        after the election had already frozen.

        Claims are offered strongest first, so the timeline's own
        displacement rule never has to run and the outcome depends only on
        which sources are present.

        Args:
            timeline: The pipeline's timeline manager.
        """
        candidates = []
        for node in self._sources():
            declare = getattr(node, "master_candidacy", None)
            candidacy = declare() if callable(declare) else None
            if candidacy is None:
                continue
            rate, priority = candidacy
            candidates.append((int(priority), float(rate), node))

        # Strongest first: highest priority, then the better clock. The
        # node is not part of the sort key -- nodes are not orderable, and
        # ties are genuinely interchangeable.
        candidates.sort(key=lambda entry: (entry[0], entry[1]), reverse=True)
        for priority, rate, node in candidates:
            timeline.claim_master(node, rate, priority)

    def _resolve_entitlement(self) -> Entitlement:
        """Resolve both gates for this run and latch the outcome.

        Runs after the sources exist and before any node starts, which is
        the only window that works: a handle exists at construction, but a
        sink writes its header at start() -- so resolving later would let a
        Device-tier run stamp an evaluation mark, or the reverse, silently.

        Latches for the run, not for the process: ``stop()`` releases the
        verdict, so a second ``start()`` resolves again. That is the
        intended split. Re-resolving *mid-run* would let a network blip or
        a radio drop downgrade a recording in progress, and a sink has
        already written its header by then -- whereas a restart is a new
        recording writing new files, so a device unplugged in between
        should be reflected in what the next one claims.

        Returns:
            The combined verdict.
        """
        current = entitlement_of(self)
        if current is not None:
            return current
        self._licence_unverified = None

        residency = LaunchConfig.get().residency
        built = getattr(self, "_built_residencies", set())
        if built and built != {residency}:
            verdict = Entitlement(
                Permission.ABORT,
                f"this pipeline was built for {', '.join(sorted(built))} "
                f"residency and is starting as {residency}; a pipeline "
                f"runs under the residency it was built for",
            )
            set_entitlement(self, verdict)
            return verdict

        # A verdict supplied from outside wins, and is taken exactly
        # once. An embedder that established entitlement by means this
        # process cannot repeat -- a runtime that challenged an admin
        # machine for a signed licence attestation and verified it --
        # has a conclusion this resolution could not reach on its own:
        # licence.evaluate() asks about *this* machine, and a container
        # holds no licence. See D-ENT-61. The mark the data itself carries
        # is not the supplier's to lift (D-ENT-97), so it still applies.
        supplied = take_supplied(self)
        # A supplied verdict carries a conclusion, not the evidence: it
        # never stands in for a device that answered (D-ENT-98).
        self._entitlement_supplied = supplied is not None
        if supplied is not None:
            data_permission, data_reason = self._data_mark()
            if data_permission < supplied.permission:
                supplied = Entitlement(
                    data_permission,
                    data_reason,
                    licence=supplied.licence,
                    attestation=supplied.attestation,
                    attestation_reason=getattr(
                        supplied, "attestation_reason", ""
                    ),
                )
            set_entitlement(self, supplied)
            return supplied

        # One query for both questions, so they cannot disagree.
        frozen = licence.is_frozen_deployment()
        state = licence.query_licence()
        licence_permission, licence_reason = licence.evaluate(
            frozen, residency, state
        )
        # A deployment whose licence cannot be determined runs only if a
        # g.tec device attests (D-ENT-100); the detail, or None.
        self._licence_unverified = licence.unverified(frozen, residency, state)

        # Ask each amplifier to open its handle first. Every source
        # opens its device inside its own start(), which runs after this
        # -- so without this loop every handle below is None and the
        # challenge list is empty by construction. Measured 2026-09-06
        # on a BCI Core-8: zero challenges issued during a real start(),
        # while the same handle challenged by hand answered
        # verified=True. See AmplifierSource.open_for_attestation, and
        # _begin_deferred_device_check's docstring, which already
        # describes the behaviour this restores.
        #
        # Sources that do not implement the hook are unchanged: they
        # contribute no handle, exactly as before.
        # Guarded here as well as in each implementation. A device that
        # will not open is a connection problem, and the source's own
        # start() reports it with its own error type and message. Letting
        # it escape *here* would turn an unreachable amplifier into an
        # entitlement failure -- a confusing error, raised from the wrong
        # gate, about the wrong thing.
        for node in self._sources():
            opener = getattr(node, "open_for_attestation", None)
            if callable(opener):
                try:
                    opener()
                except Exception as error:  # noqa: BLE001
                    self.log(
                        f"could not open a device before the entitlement "
                        f"gate, so this run is unattested: {error}",
                        type=ioc.Constants.LogTypes.WARNING,
                    )

        results = []
        for node in self._sources():
            handle = getattr(node, "_device", None)
            if handle is not None:
                results.append(attestation.attest_device(handle))
        results.extend(self._remote_attestation_results())
        attest_permission, attest_reason = attestation.evaluate(results)
        # A device that answered here is held by this run, so it cannot
        # answer the presence probe's challenge too; it has answered
        # this one (D-ENT-98).
        for result in results:
            if getattr(result, "verified", False):
                device_probe.vouch(getattr(result, "serial", ""))
        # A held licence lifts the device gate's mark, and only its mark
        # (D-ENT-96): a licensed run is not marked for want of an
        # amplifier.
        device_permission = lifted(attest_permission, licence_permission)
        device_reason = attest_reason
        # The owner: never "without our devices AND without a license". A
        # frozen standalone application's device check enforces it after
        # start(); anywhere else nothing would, so it is refused here.
        if (
            self._licence_unverified is not None
            and attest_permission is not Permission.FULL
            and not device_required(frozen=frozen, residency=residency)
        ):
            device_permission = Permission.ABORT
            device_reason = (
                f"this deployment's licence could not be verified "
                f"({self._licence_unverified}) and no g.tec device "
                f"attested; a deployment runs with a licence or with an "
                f"attested device. {licence.remedy(self._licence_unverified)}"
            )

        # A third, independent input: the mark the data itself carries.
        # meet() takes the strictest of the three, so this can only
        # tighten FULL to LIMITED -- an ABORT from either gate above
        # stays an ABORT.
        data_permission, data_reason = self._data_mark()

        permission = meet(
            (licence_permission, device_permission, data_permission)
        )
        # The reason is the restricting input's, and an unrestricted run
        # has none: a lifted device gate still has its own reason, which
        # says the output is marked.
        reason = ""
        if permission is not Permission.FULL:
            for gate, why in (
                (licence_permission, licence_reason),
                (device_permission, device_reason),
                (data_permission, data_reason),
            ):
                if gate is permission and why:
                    reason = why
                    break

        verdict = Entitlement(
            permission,
            reason,
            licence=licence_permission,
            attestation=attest_permission,
            attestation_reason=attest_reason,
        )
        set_entitlement(self, verdict)
        return verdict

    def _data_mark(self) -> tuple:
        """The mark this run's data carries, whatever the licence says.

        A loaded artifact fitted on marked data carries it into every run
        that deploys it (D-BATCH-92), and a recording a marked run wrote
        carries it into every run that replays it (D-ENT-97): the licence
        that lifts this run's own mark does not license what an
        unlicensed run produced. A source carries a recording's mark only
        as its ``mark`` property, read from the file where it is run,
        and only the mark's own text counts, so an unrelated attribute of
        that name marks nothing.

        Returns:
            ``(Permission.LIMITED, reason)`` if the data is marked, else
            ``(Permission.FULL, "")``.
        """
        fitted = sorted(
            f"'{node.name}'"
            for node in self._fittable_nodes()
            if getattr(node, "_gpype_marked", False)
        )
        if fitted:
            return Permission.LIMITED, (
                f"{', '.join(fitted)} carries an artifact fitted on marked "
                f"data, so this run carries the mark forward"
            )
        replayed = sorted(
            f"'{node.name}'"
            for node in self._sources()
            if getattr(node, "mark", None) == MARK
        )
        if replayed:
            return Permission.LIMITED, (
                f"{', '.join(replayed)} replays a recording a marked run "
                f"wrote, so this run carries the mark forward"
            )
        return Permission.FULL, ""

    def _remote_attestation_results(self) -> list:
        """Challenge the far end of every Link (C10, SERVER side).

        Called from inside the entitlement phase, which is before any
        node starts, because that is the only placement that works: a
        sink writes its header at ``start()``, so a verdict arriving
        later could not change what the artifact claims.

        That forces an awkward step. The broker is bound *here*, ahead of
        the Links that normally bind it, because the edge cannot connect
        to a port nothing is listening on -- and the extra reference is
        held for the run and released in ``stop()``.

        Returns:
            One AttestationResult per connected edge. Empty when this is
            not a server or the switch is off, so the gate sees exactly
            what it saw before.
        """
        if not remote_required():
            return []

        config = LaunchConfig.get()
        if config.residency != Constants.Residency.SERVER:
            return []

        from .core._private import threads

        if not threads.available():
            # A page connects only once start() has returned, so there is
            # no edge to challenge here (D-CORE-129).
            return [
                attestation.AttestationResult(
                    "",
                    False,
                    "no edge can connect before a server without threads "
                    "has started, so none was challenged",
                )
            ]

        from .core._private.ws.broker import WsBroker

        broker = WsBroker.get_instance()
        broker.acquire(config.endpoint)
        self._attest_broker = broker

        if not broker.wait_for_client(remote_attestation.JOIN_TIMEOUT_S):
            return [
                attestation.AttestationResult(
                    "",
                    False,
                    f"no edge connected to {config.endpoint} within "
                    f"{remote_attestation.JOIN_TIMEOUT_S:g}s",
                )
            ]

        channels = broker.attest_channels()
        if not channels:
            return [
                attestation.AttestationResult(
                    "", False, "no edge to challenge"
                )
            ]
        return remote_attestation.challenge_all(channels)

    def _start_attestation_responders(self) -> None:
        """Answer the server's challenges for this run (C10, EDGE side).

        After ``super().start()``, because a Link only creates its
        WsClient when it starts -- and unlike the server, the edge does
        not gate its own start on the handshake: its handle is local, so
        C9 already challenged it directly. The edge is only answering on
        the server's behalf.
        """
        if not remote_required():
            return
        if LaunchConfig.get().residency != Constants.Residency.EDGE:
            return

        handle = self._first_device_handle()
        for link in self._links():
            client = getattr(link, "_ws", None)
            factory = getattr(client, "attest_channel", None)
            if factory is None:
                continue
            responder = remote_attestation.Responder(factory(), handle)
            responder.start()
            self._responders.append(responder)

    def _stop_attestation(self) -> None:
        """Stop the responders and give back an early broker reference.

        Never raises: this runs on the teardown path, where losing the
        original error is worse than failing to tidy up.
        """
        responders, self._responders = self._responders, []
        for responder in responders:
            try:
                responder.stop()
            except Exception:
                pass

        broker, self._attest_broker = self._attest_broker, None
        if broker is not None:
            try:
                broker.release()
            except Exception:
                pass

    def _walk_nodes(self) -> list[tuple]:
        """Every node this pipeline owns, chain internals included.

        One walk, shared by :meth:`get_nodes` and node addressing, so
        that a node which can be reported is exactly a node which can be
        addressed. Two walks would agree until one of them changed.

        Returns:
            One ``(node, is_internal)`` pair per node, in registration
            order, each chain followed by its own internal nodes.
        """
        walked: list[tuple] = []
        for element in self._elements:
            walked.append((element, False))
            internal = getattr(element, "internal_nodes", None)
            if internal:
                walked.extend((node, True) for node in internal)
        return walked

    def _resolve_node(self, node_id: Union[str, Node]):
        """The node in *this* pipeline carrying *node_id*.

        Resolved over this pipeline's own elements rather than through
        ``PortableImp.get_by_id``, which is a **process-global**
        registry: it would hand back a node belonging to a different
        pipeline in the same process, so a control plane holding two
        sessions could drive the wrong one and get an ordinary success
        back.

        A document id names the node the author wrote. Where this
        process runs it, that node is returned; where it does not -- a
        source under server residency -- the chain carrying it is, which
        is what the id addressed before the pipeline built the chains.

        The node object itself resolves the same way, by identity: a
        script holds its nodes, and had to look each id up by name
        through :meth:`get_nodes` first. Never by name -- names are not
        unique (see :func:`_node_id_of`).

        Args:
            node_id: The node's document id, or the node itself.

        Returns:
            The node object.

        Raises:
            KeyError: If no node in this pipeline carries that id, or
                the node given is not one of this pipeline's.
        """
        from .core._private import wrapping

        if not isinstance(node_id, str):
            for node, _ in self._walk_nodes():
                if node is node_id:
                    return node
            for element in self._elements:
                if wrapping.core_of(element) is node_id:
                    return element
            raise KeyError(
                f"{naming.node_label(node_id)} is not a node of this "
                f"pipeline"
            )
        for node, _ in self._walk_nodes():
            if _node_id_of(node) == node_id:
                return node
        for element in self._elements:
            if _node_id_of(wrapping.core_of(element)) == node_id:
                return element
        raise KeyError(f"no node with id '{node_id}' in this pipeline")

    @staticmethod
    def _control_names(node_id: Union[str, Node]) -> tuple:
        """How control messages name the node they were addressed at.

        Args:
            node_id: A document id, or a node.

        Returns:
            (subject, prefix): ``node 'x'`` and ``x`` for an id, as
            before nodes were accepted; the node's label for both for a
            node, which already reads ``Knob 'knob'``.
        """
        if isinstance(node_id, str):
            return f"node '{node_id}'", node_id
        label = naming.node_label(node_id)
        return label, label

    def get_nodes(self) -> list[dict]:
        """Report every node in this pipeline and its current state.

        Intended for anything driving a pipeline from outside -- a control
        plane, a monitoring view -- which otherwise has to reach into
        private implementation attributes to answer "what is in here, and
        is it healthy". Chain internals are included, since a chain is a
        container and the node that actually failed is inside it.

        Returns:
            One entry per node, each with its ``name``, ``class``,
            whether it is an ``internal`` node of a chain, its ``state``
            and cycle ``counter`` when the node exposes them, and its
            control surface:

            * ``id`` -- the document id to address a control at, or None
              for a node carrying no configuration.
            * ``actions`` -- the declared action names, sorted, empty
              when the node declares none.
            * ``controllable`` -- whether the node has a usable control
              channel, meaning it declares *both* halves. Two fields
              rather than one, because a node may well have actions and
              no state, or state and no actions.
        """
        from .core._private import wrapping

        report: list[dict] = []
        for node, is_internal in self._walk_nodes():
            # A chain the pipeline built is reported as the node it
            # carries -- its class, document id and controls -- as a
            # public chain was in 4.0: "SourceChain" and the chain's own
            # id are in no document. The node keeps its own row among the
            # internals, so a caller that reads the internals still sees
            # the node that did the work.
            owner = wrapping.core_of(node) or node
            entry = {
                "name": getattr(node, "name", ""),
                "class": type(owner).__name__,
                "internal": is_internal,
                "id": _node_id_of(owner),
                "actions": action_names(owner),
                "controllable": has_control(owner),
            }
            for key, attr in (
                ("state", "get_state"),
                ("counter", "get_counter"),
            ):
                getter = getattr(node, attr, None)
                if callable(getter):
                    try:
                        entry[key] = getter()
                    except Exception:
                        # A node that cannot report its own state must
                        # not stop the caller from seeing the rest.
                        entry[key] = None
            report.append(entry)
        return report

    #: Duty above which a node is reported as short of headroom. Chosen
    #: below 1.0 rather than at it: a pipeline that is exactly keeping up
    #: has no margin for the next scope repaint or garbage collection,
    #: and the useful warning is the one that arrives before frames are
    #: actually late.
    LOAD_WARNING_DUTY: float = 0.70

    def get_load(self) -> dict:
        """Report how much of the real-time budget the pipeline is using.

        Each node times its own ``step()``; this divides that by the wall
        time since the previous call, which is the only denominator that
        is right for every driving mode. A rate-derived one -- frame size
        over sampling rate -- is wrong for any source paced per sample:
        ``FixedRateSource`` wakes once per sample whatever the frame
        size, so dividing by the frame period would report several times
        more headroom than exists.

        Reading resets the counters, so two calls a second apart describe
        that second. The first call after ``start()`` covers everything
        since the nodes were built, including setup, and is not
        representative.

        Returns:
            dict: ``headroom`` (1 - duty, so 1.0 is idle and 0.0 is
            saturated), ``duty``, ``elapsed_s``, ``busiest`` naming the
            node with the highest duty, ``short_of_headroom`` naming
            every node over :attr:`LOAD_WARNING_DUTY`, and ``nodes``,
            one row per node with its ``name``, ``class``, ``duty``,
            ``worst_cycle_ms`` and ``cycles``.
        """
        now = perf_counter()
        previous = getattr(self, "_load_read_at", None)
        self._load_read_at = now
        elapsed = (now - previous) if previous else 0.0

        rows: list[dict] = []
        total = 0.0
        for node, _is_internal in self._walk_nodes():
            take = getattr(node, "take_load", None)
            if not callable(take):
                continue
            busy, worst, cycles = take()
            total += busy
            rows.append(
                {
                    # The label the author would recognise. A chain's
                    # work happens in its private core, so the raw name
                    # here is "_GeneratorCore" -- which reads as an
                    # internal leak in a report whose whole job is to
                    # say which node to go and look at. The unmapped
                    # class name stays in "class" for diagnosis.
                    "name": naming.node_label(node),
                    "class": type(node).__name__,
                    "duty": (busy / elapsed) if elapsed else None,
                    "worst_cycle_ms": worst * 1000.0,
                    "cycles": cycles,
                }
            )

        duty = (total / elapsed) if elapsed else None
        busiest = None
        if rows and elapsed:
            # Self-time, because the push to downstream happens after the
            # step handler returns -- so the busiest node is the one to
            # look at, not merely the one furthest downstream.
            busiest = max(rows, key=lambda r: r["duty"] or 0.0)["name"]

        # The threshold applied here rather than left to the caller,
        # because a number without a verdict is a number nobody acts
        # on: "duty 0.72" reads as fine until you know what fine was.
        # A list rather than a bool, since the useful answer to "am I
        # short of headroom" is which node to look at.
        short = [
            row["name"]
            for row in rows
            if row["duty"] is not None and row["duty"] > self.LOAD_WARNING_DUTY
        ]

        return {
            "headroom": (1.0 - duty) if duty is not None else None,
            "duty": duty,
            "elapsed_s": elapsed,
            "busiest": busiest,
            "short_of_headroom": short,
            "nodes": rows,
        }

    def get_control(self, node_id: Union[str, Node]) -> dict:
        """Read the full control state of one node.

        Args:
            node_id: The node's document id, as reported by
                :meth:`get_nodes`, or the node itself.

        Returns:
            The node's whole control state, as a copy -- mutating it
            cannot reach the node.

        Raises:
            KeyError: If no node in this pipeline carries that id, or
                the node given is not one of this pipeline's.
            AttributeError: If the node declares no control channel.
            TypeError: If the node's answer is not a JSON-representable
                dict.
        """
        node = self._resolve_node(node_id)
        subject, prefix = self._control_names(node_id)
        if not has_control(node):
            raise AttributeError(f"{subject} declares no control channel.")
        getter, _ = resolve_control(node)
        state = ensure_json(getter(), False, f"{prefix}.get_control")
        # Copied on the way out. A node returning its own live dict would
        # otherwise let a caller mutate its state by hand -- no merge, no
        # unknown-key check, none of the node's own validation -- which is
        # the whole surface this method exists to be.
        return dict(state)

    def set_control(self, node_id: Union[str, Node], arg: dict) -> dict:
        """Apply a partial control update to one node.

        The update **merges**: *arg* names only what changes, so a
        one-field change stays one field and two clients need no
        get-before-set to avoid overwriting each other. Any key the
        node's own ``get_control`` does not report is refused **before
        the node sees it** -- the getter is the key set.

        Args:
            node_id: The node's document id, or the node itself.
            arg: The keys to change.

        Returns:
            The resulting full control state, so a set is also a get.
            A copy, as in :meth:`get_control`.

        Raises:
            KeyError: If no node carries that id, the node given is not
                one of this pipeline's, or *arg* holds a key the node
                does not accept -- that message names both the
                offending keys and the accepted ones.
            TypeError: If *arg* is not a dict, or the node's answer is
                not a JSON-representable dict.
            AttributeError: If the node declares no control channel.
            ValueError: Whatever the node raises when the resulting
                combination is not usable. Propagated deliberately: only
                the node knows its own cross-field rules.
        """
        node = self._resolve_node(node_id)
        subject, prefix = self._control_names(node_id)
        if not has_control(node):
            raise AttributeError(f"{subject} declares no control channel.")
        getter, setter = resolve_control(node)
        current = ensure_json(getter(), False, f"{prefix}.get_control")
        state = ensure_json(
            setter(merged_update(current, arg)),
            False,
            f"{prefix}.set_control",
        )
        return dict(state)  # copied on the way out -- see get_control

    def invoke_action(
        self, node_id: Union[str, Node], name: str, arg: dict
    ) -> Optional[dict]:
        """Run one declared action on one node.

        Args:
            node_id: The node's document id, or the node itself.
            name: The action name, as reported by :meth:`get_nodes`.
            arg: The action's argument; may be empty.

        Returns:
            Whatever the action reports, or None when it reports nothing.

        Raises:
            KeyError: If no node in this pipeline carries that id, or
                the node given is not one of this pipeline's.
            AttributeError: If the node declares no action of that name.
                An ordinary method is not an action, so this is also what
                an attempt to call anything else by name gets -- which is
                what keeps the rest of the class unreachable.
            TypeError: If *arg* is not a dict, or the result is not a
                JSON-representable dict or None.
            ValueError: Whatever the action itself raises.
        """
        if not isinstance(arg, dict):
            raise TypeError(
                f"action argument must be a dict, "
                f"got {type(arg).__name__}."
            )
        node = self._resolve_node(node_id)
        subject, prefix = self._control_names(node_id)
        method = resolve_action(node, name)
        if method is None:
            raise AttributeError(
                f"{subject} declares no action '{name}'; "
                f"declared actions are {action_names(node)}"
            )
        result = ensure_json(method(arg), True, f"{prefix}.{name}")
        return None if result is None else dict(result)

    def attach_probe(
        self,
        node_id: Union[str, Node],
        port: str = Constants.Defaults.PORT_OUT,
        max_rate: float = None,
    ) -> str:
        """Publish what one port emits, live, for a subscriber to watch.

        A debugging instrument. The port keeps working exactly as it
        did: a probe copies what goes past and never consumes, delays or
        modifies it.

        What arrives is a **subsample**, not the signal. The probe
        decimates to ``max_rate`` and applies no anti-aliasing filter,
        because a filter would make this a measurement instrument and
        the number read off a probe must never be mistaken for one. The
        published context says so -- it carries the delivered rate, the
        decimation factor and ``antialiased: false``.

        The returned stream id is also the *permission* to watch: it
        carries random bytes, because the broker's ``subscribe``
        capability is granted per connection rather than per stream, so
        anyone holding a session's reader token could otherwise watch
        any probe whose id they guessed. Hand it only to whoever asked.

        Costs nothing while nobody is subscribed: the probe checks
        before it copies.

        Args:
            node_id: Document id of the node to probe, as
                :meth:`get_nodes` reports it, or the node itself.
            port: Name of its output port.
            max_rate: Samples per second to deliver, at most. Defaults
                to :data:`probe.DEFAULT_MAX_RATE_HZ`.

        Returns:
            str: The stream id to subscribe to.

        Raises:
            KeyError: If no node in this pipeline carries that id, or
                the node given is not one of this pipeline's.
            ValueError: If the node has no such output port, or if this
                process has no running broker to publish on.
            RuntimeError: Where no thread can be started (Pyodide).
        """
        from .core._private import probe as probe_module
        from .core._private.ws.broker import WsBroker

        node = self._resolve_node(node_id)
        subject, _ = self._control_names(node_id)
        # The probe's record and stream id carry the document id even when
        # the node itself was given, so list_probes() stays JSON for a
        # remote client (D-CORE-126).
        if not isinstance(node_id, str):
            node_id = _node_id_of(node_id) or naming.node_label(node_id)
        # A source the pipeline wrapped is watched where downstream sees
        # it, at its chain's boundary: after the Sync, with the arrival
        # stamp stripped. The node's own port carries the raw frame.
        node = self._wrappers.get(id(node), node)
        getter = getattr(node, "get_output_port", None)
        if getter is None:
            raise ValueError(
                f"{subject} has no output ports, so there is nothing on "
                f"it to watch. A probe reads what a node emits; a sink "
                f"emits nothing."
            )
        try:
            port_imp = getter(port)
        except Exception as error:
            raise ValueError(
                f"{subject} has no output port '{port}': {error}"
            ) from error

        from .core._private import threads

        if not threads.available():
            # Before anything is published: a probe's sender is a thread.
            raise RuntimeError(
                "attach_probe needs a thread for the probe's sender, and "
                "none can be started here (Pyodide). A page reads a "
                "widget's stream through its Link instead."
            )

        broker = WsBroker.get_instance()
        if not broker.is_running:
            raise ValueError(
                "no data socket is bound in this process, so there is "
                "nowhere to publish a probe. A probe is delivered over "
                "the same broker the pipeline's own streams use, and a "
                "standalone run binds none -- nothing is listening and "
                "nothing is serving. A server-residency run has one; so "
                "does any process that acquired a broker itself."
            )

        verdict = entitlement_of(self)
        stream_id = probe_module.new_stream_id(node_id, port)
        instrument = probe_module._Probe(
            stream_id=stream_id,
            node_id=node_id,
            port_name=port,
            port_imp=port_imp,
            broker=broker,
            max_rate=(
                probe_module.DEFAULT_MAX_RATE_HZ
                if max_rate is None
                else float(max_rate)
            ),
            marked=bool(verdict is not None and verdict.marked),
        )
        instrument.start()
        self._probes[stream_id] = instrument
        self.log(
            f"probing '{node_id}' port '{port}' as {stream_id}",
            type=Constants.LogTypes.INFO,
        )
        return stream_id

    def detach_probe(self, stream_id: str) -> bool:
        """Stop and remove one probe.

        Args:
            stream_id: What :meth:`attach_probe` returned.

        Returns:
            bool: True if a probe was removed, False if there was none.
        """
        instrument = self._probes.pop(stream_id, None)
        if instrument is None:
            return False
        instrument.stop()
        self.log(f"stopped probing {stream_id}", type=Constants.LogTypes.INFO)
        return True

    def list_probes(self) -> list[dict]:
        """Report every probe attached to this pipeline.

        Returns:
            list[dict]: One entry per probe, each naming the node and
            port it watches, its decimation, how many samples it has
            sent, how many frames it dropped, and how many connections
            are currently watching it.
        """
        return [probe.status() for probe in list(self._probes.values())]

    def _detach_all_probes(self) -> None:
        """Stop every probe. Called when the run ends."""
        for stream_id in list(self._probes):
            try:
                self.detach_probe(stream_id)
            except Exception:  # pragma: no cover - teardown
                self._probes.pop(stream_id, None)

    def connect(self, source: Union[Node, dict], target: Union[Node, dict]):
        """Connect two nodes to establish data flow in the pipeline.
        Nodes are automatically added to pipeline if not already present.

        Args:
            source (Union[Node, dict]): Source node or port specification.
                Use dict for specific ports (e.g., node["port_name"]).
            target (Union[Node, dict]): Target node or port specification.
                Use dict for specific ports (e.g., node["port_name"]).

        Raises:
            ValueError: If this edge would make a feedback loop illegal
                -- see :meth:`_describe_component_violation`. The edge
                is undone before raising, so the pipeline is left exactly
                as it was before this call.
        """
        # Before registering either: a world-facing node reaches the
        # pipeline inside the chain that gives it a Link and a Sync, and it
        # is the chain that has to be the element -- ``_elements`` is
        # what every residency, timeline and entitlement walk reads.
        source = self._wrap(source)
        target = self._wrap(target)

        self._register(source)
        self._register(target)
        super().connect(source, target)

        # Checked after the edge exists, not before: the real port connections
        # -- which a bare node, a dict spec, and a chain proxying to its
        # first/last internal node all normalise to differently -- only
        # exists once ioiocore has resolved it.
        #
        # The check looks at the whole pipeline rather than at this edge,
        # and disconnecting this edge is still the right rollback. Every
        # prior connect() left the pipeline with no illegal loop in it
        # (a pipeline rebuilt by deserialize() is covered at start()
        # instead -- see _validate_no_illegal_cycles), so a component
        # that violates a rule now has to involve this edge: either it
        # closed the loop, or it connected the port that turned an
        # existing legal loop illegal -- an ASYNC input, say. Removing it
        # restores the pipeline that passed.
        #
        # What the message names is the loop, which is not always this
        # edge. That is deliberate: the loop is what the author has to
        # change.
        #
        # require_feed is off here: "something outside feeds this loop"
        # is true of a finished pipeline, not of every intermediate one.
        # Closing the loop and then wiring the source that drives it is
        # an ordinary way to write the same pipeline, so enforcing it
        # per-edge would refuse a pipeline that is about to be legal.
        # start() applies it -- see _validate_no_illegal_cycles.
        violation = self._cycle_violation(require_feed=False)
        if violation is not None:
            self._imp.disconnect(source, target)
            raise ValueError(violation)

    #: How long an edge's start waits for its clock to converge on the
    #: server's reference, in seconds. As long as a receiving Link waits
    #: for a context, so an edge started before its server keeps working.
    CLOCK_SYNC_TIMEOUT_S = CONTEXT_TIMEOUT_S

    @staticmethod
    def _running_pipelines(exclude=None) -> list:
        """Return the pipelines in this process that are running.

        Args:
            exclude: A pipeline to leave out, usually the caller.

        Returns:
            The running pipelines.
        """
        running = ioc.Constants.States.RUNNING
        return [
            p
            for p in pipelines.live()
            if p is not exclude and p.get_state() == running
        ]

    @staticmethod
    def rezero_reference() -> int:
        """Zero this process's reference clock, in a new epoch.

        The server's half of the time base (D-TIME-20, D-TIME-67). Edges
        sync their clocks to the reference this process answers clock
        probes with, and every stamp this process reads is in it. Zeroing
        it when a pipeline is loaded keeps stamps small and starts a new
        epoch, which an edge synced to the previous one is told about.

        Under SERVER residency :meth:`deserialize` calls this, which is
        how the g.Pype runtime's load zeroes it. A process that loads
        pipelines any other way calls it itself.

        Returns:
            The new epoch.

        Raises:
            RuntimeError: While any pipeline in this process is running:
                its stamps would move by the whole reference at once.
        """
        running = Pipeline._running_pipelines()
        if running:
            raise RuntimeError(
                f"Cannot zero the reference clock while {len(running)} "
                f"pipeline(s) in this process are running: every stamp "
                f"they hold would jump by the reference's age. Stop them "
                f"first."
            )
        return timebase.current().rezero()

    def _sync_edge_clock(self) -> None:
        """Relate this edge's clock to the server's reference, once.

        The edge's start gate (D-TIME-65, D-TIME-68). Before any source
        stamps: the offset is stepped in, so the first stamp is already
        in the server's base, and it then stays constant until stop().
        Nothing re-syncs during the run. Every start syncs afresh, which
        is what picks up a reference the server zeroed at a new load.

        Only on an edge that builds a source here: a sink or a widget
        writes no stamp. Not under STANDALONE or SERVER residency, and
        not in a batch run.

        Raises:
            ClockSyncError: If the clock does not converge within
                :attr:`CLOCK_SYNC_TIMEOUT_S`, naming the endpoint.
        """
        # A start() on a running pipeline must not sync again: stepping
        # the offset now would move every later stamp of this run.
        if self.get_state() == ioc.Constants.States.RUNNING:
            return
        self._clock_health = None
        launch = LaunchConfig.get()
        if launch.residency != Constants.Residency.EDGE:
            return
        links = [
            link
            for link in self._links()
            if link.residency == Constants.Residency.EDGE
            and link.sender == Constants.Residency.EDGE
        ]
        for link in links:
            link.attach_clock_sync(None)
        if getattr(self, "_batch_driven", False) or not self._sources():
            return

        from .core._private import clock_sync
        from .core._private.ws.clock_client import ClockConnection

        base = timebase.current()
        # A new run holds no stamp, unless another pipeline here runs.
        if not self._running_pipelines():
            base.reopen()
        connection = ClockConnection(launch.endpoint)
        try:
            health = clock_sync.synchronise(
                connection, self.CLOCK_SYNC_TIMEOUT_S, clock=base.clock
            )
        finally:
            connection.close()
        # What the stamps are corrected by, which is what the context
        # must report: not the estimate, if another pipeline's stood.
        health = clock_sync.apply_offset(health, base)
        self._clock_health = clock_sync.context_entry(
            health, launch.edge_id or clock_sync.EDGE_TOKEN
        )
        for link in links:
            link.attach_clock_sync(self._clock_health)
        precise = health.uncertainty <= clock_sync.CONVERGED_UNCERTAINTY_S
        self.log(
            f"Clock synced to {launch.endpoint}: this edge is "
            f"{health.offset * 1e3:.3f} ms ahead of the server's reference "
            f"(epoch {health.epoch}), known to "
            f"{health.uncertainty * 1e3:.3f} ms; constant for this run."
            + (
                ""
                if precise
                else f" Above the "
                f"{clock_sync.CONVERGED_UNCERTAINTY_S * 1e3:g} ms target: "
                f"the network's round trip is too long for better, and "
                f"events may be misplaced against other edges' by up to "
                f"that much."
            ),
            type=(
                Constants.LogTypes.INFO
                if precise
                else Constants.LogTypes.WARNING
            ),
        )

    def start(self):
        """Start the pipeline and begin real-time data processing.

        Initiates execution of all nodes according to their configured
        connections and timing. Runs continuously until stop() is called.
        This method is non-blocking.

        On an edge that builds a source, first relates its clock to the
        server's reference, and raises if that does not converge within
        :attr:`CLOCK_SYNC_TIMEOUT_S` (``ClockSyncError``).
        """
        # Startup runs in ordered phases, and the order is load-bearing.
        # Note setup() does not run here -- ioiocore defers it to each
        # node's first cycle -- so a phase may only use what is known at
        # construction.
        #
        #   1. validate  -- reject pipelines that cannot mean anything, so
        #                   the user is told the real problem
        #   2. timeline  -- bind it, so election has somewhere to happen
        #   3. start     -- nodes run, setup happens, election resolves
        #
        # Phases for tier resolution belong between 2 and 3, before any
        # sink can write.
        #
        # A run that failed is still being stopped when ``failure`` is
        # first set, with the state RUNNING, and records the failure only
        # after that. A start() in either window would be skipped or
        # would inherit that record, so it waits the run out first.
        self._await_failing_stop()
        self._validate()
        self._refuse_batch_without_run()

        # Before anything reads the stamp clock -- the raw recording's
        # origin just below included -- because the offset can be
        # stepped in only before the first reading (D-TIME-46).
        self._sync_edge_clock()

        # Anchor a replay to *this* run. The clock every replay core
        # schedules against is process-global, so a pipeline started a
        # second time would keep the first run's origin -- every
        # recorded offset would already be in the past, and the whole
        # recording would be re-emitted as fast as the loop could go,
        # which reproduces nothing and looks like a working replay.
        #
        # Here rather than in the cores: they anchor lazily on their
        # first block, so the reset has to happen while none of them is
        # running, and start() is the only such moment.
        launch = LaunchConfig.get()
        if launch.load_from:
            from .sources.base.raw import ReplayClock

            ReplayClock.reset()
        if launch.save_as:
            # The same argument on the recording side. The run object
            # lives as long as the pipeline, so without this a pipeline
            # started a second time records block times relative to the
            # *first* run's origin -- and a replay of that recording
            # sleeps out the gap between the two runs before emitting
            # anything. See RawRun.begin.
            from .sources.base.raw import RawRun

            run = RawRun.current(launch.save_as)
            run.begin()

        # Say once, per run, when the clock the samples are stamped
        # from is too coarse to mean what the documentation says.
        #
        # Every source stamps its own samples (D-NODE-24), so this is
        # not specific to a pipeline shape and there is nothing to inspect
        # to decide whether it matters. It is silent on CPython 3.13
        # and newer and on every Unix, which is why it can afford to be
        # unconditional here.
        complaint = stamp_clock_complaint()
        if complaint:
            self.log(complaint, type=Constants.LogTypes.WARNING)

        # Bind the master timeline before anything runs. It is owned per
        # pipeline rather than per process: a process routinely builds
        # several pipelines, and a shared timeline would let one inherit
        # another's master and epoch.
        #
        # A fresh one is established for every run, not merely released
        # in stop(): ioiocore stops the pipeline from inside its monitor
        # thread when a node errors, which calls its own stop() and never
        # this override, and start() then auto-resets instead of
        # refusing. A rolled-back start() failure leaves the same residue
        # without even setting the error condition. Reusing that timeline
        # would keep the previous run's master election, epoch and rate
        # anchor while the sources restart at sample zero, placing every
        # event a whole run into the past.
        if self.get_state() != ioc.Constants.States.RUNNING:
            release_timeline(self)
            release_entitlement(self)
            self._reset_run_state()
            # A fresh run must not report the previous run's failure.
            self._failure = None
        timeline = timeline_of(self)
        # After timeline_of, because the release above discards the
        # object this was last written to.
        timeline.event_margin_ms = self.event_margin_ms
        for node in self._timeline_consumers():
            node.attach_timeline(timeline)
        for node in self._placement_consumers():
            node.attach_placement_timeline(timeline)
        self._attach_position_gate()
        self._elect_master(timeline)

        # Resolve both gates before any node starts. A sink writes its
        # header at start(), so a verdict reached later would let a run
        # stamp the wrong mark -- silently, in the artifact.
        verdict = self._resolve_entitlement()
        if not verdict.allowed:
            # Give back the broker reference the remote handshake may
            # have taken. A refused start must not leave a port bound
            # until somebody remembers to call close().
            self._stop_attestation()
            # Nor an amplifier held. The gate opened handles to challenge
            # them, and no node ran, so stop() does not reach a source:
            # a BCI Core stayed connected until close(). A GDS source's
            # constructor handle goes too; the next start reopens it.
            self._release_devices()
            raise PermissionError(verdict.reason)

        # Hand the verdict to whatever produces an artifact, before
        # anything starts. A sink that learns it later has already
        # written its header.
        for node in self._entitlement_consumers():
            node.attach_entitlement(verdict)
        if verdict.marked:
            self.log(
                f"{MARK}: {verdict.reason}",
                type=ioc.Constants.LogTypes.INFO,
            )

        # Stamped once per run, in both modes, before any node's setup()
        # can run -- a source merges it, plus its own INPUT, into the
        # context it publishes (LQ-P2). Built from the pipeline as it is
        # right now: a fittable node already holding a state (a realtime
        # deployment) is named in the record; one that will only fit
        # once this run starts is not yet (D-BATCH-100).
        from ..common._private import provenance as provenance_module

        record = provenance_module.build(self)
        for node in self._provenance_consumers():
            node.attach_provenance(record)

        super().start()

        # Only now, once the pipeline is actually running: the case where
        # a device must be *reachable* and none attested as a source. See
        # _begin_deferred_device_check for why it is after start() and
        # not before.
        self._begin_deferred_device_check(verdict)

        # Also only now: the edge's Links have their WsClients, so it can
        # start answering. See _start_attestation_responders for why the
        # edge does not wait for this and the server does.
        self._start_attestation_responders()

        # Last, because it is the only phase that waits: give the first
        # cycles a moment to report a setup failure, so a pipeline that was
        # never viable fails here rather than going quiet. Skipped for a
        # batch run, which run() drives on the caller's thread -- a setup
        # failure there is raised rather than logged, and nothing has
        # cycled yet at this point, so the wait would buy nothing and
        # charge every offline run the whole grace period.
        if not getattr(self, "_batch_driven", False):
            self._await_setup()
            # And, on a server, name later what never arrived: a stream
            # waits without bound, so a wait here would be a guess.
            self._begin_stream_report()

        # Anchor the load window *here*, after everything start() waits
        # for, so the first get_load() describes the run and not the
        # starting of it.
        #
        # It used to be anchored immediately after super().start(), on
        # the stated grounds that "setup ran before this point and its
        # cost is not a steady-state duty". The first half was wrong:
        # ioiocore runs setup() on a node's first *cycle*, not in
        # start(), which is the whole reason _await_setup() exists
        # below it. So the grace period spent waiting for setup was
        # folded into the first duty -- exactly the thing the comment
        # said must not happen.
        #
        # Invisible on a fast host, where the wait is a few
        # milliseconds. On a slow one it is most of SETUP_GRACE_S:
        # measured on macOS CI, a get_load() after sleeping 0.2 s
        # reported 0.38-0.40 s, consistently, on three Python versions.
        self._load_read_at = perf_counter()

    #: How long start() waits for the first cycles to report a setup
    #: failure, in seconds. ioiocore runs setup() on a node's first
    #: cycle rather than in start(), so without a wait there is nothing
    #: to look at yet. Bounded because a node fed only by a sparse event
    #: stream sets up when its first event arrives, which may be never.
    SETUP_GRACE_S = 0.5

    #: Poll interval while waiting, so a healthy pipeline pays only the
    #: time it actually needs -- one frame period, typically.
    SETUP_POLL_S = 0.005

    def _setup_failures(self) -> list:
        """Return every node whose ``setup()`` raised.

        Asked of the node rather than inferred from its condition: a
        ``step()`` that raises on cycle 1 leaves the same condition
        behind, and that is a run that died rather than a pipeline that
        could not be built. Only the second is start()'s business.

        Returns:
            The nodes that could not set up, chain internals included.
        """
        found = []
        for node, _ in self._walk_nodes():
            failed = getattr(node, "setup_failed", None)
            if callable(failed) and failed():
                found.append(node)
        return found

    def _await_setup(self) -> None:
        """Let the first cycles run, and refuse a pipeline that cannot.

        ``setup()`` is where a node says what it cannot do -- a cutoff
        above Nyquist, a channel count that does not agree, a file
        extension it cannot write -- and ioiocore runs it on the node's
        *first cycle*, which happens after ``start()`` has returned. A
        node that fails there logs the message, marks itself, and then
        returns from every later cycle without processing anything.

        What the user saw was a scope that never updated. The monitor
        thread does stop the run and print the message, but its passes
        are ``MONITORING_INTERVAL`` (1 s) apart and the banner goes to a
        console a windowed application may not have -- and ``start()``
        had already returned normally, so a script carried on past a
        pipeline that was never going to produce anything.

        Best effort by construction: a node fed only by a sparse event
        stream sets up when its first event arrives, which may be never,
        so the wait is bounded and whatever sets up after the deadline
        stays with the monitor thread and :meth:`raise_if_failed`.

        Raises:
            RuntimeError: If a node failed to set up, naming the node
                and quoting what it wrote.
        """
        from .core._private import threads

        if not threads.available():
            # No first cycle can run while the only thread waits here,
            # so the wait would only delay start(). A setup failure then
            # reaches the monitor, as one after the deadline does.
            return
        deadline = time.monotonic() + self.SETUP_GRACE_S
        while time.monotonic() < deadline:
            if self._setup_failures():
                break
            pending = [
                node
                for node, _ in self._walk_nodes()
                if getattr(node, "get_counter", None)
                and node.get_counter() == 0
            ]
            if not pending:
                break
            time.sleep(self.SETUP_POLL_S)

        failed = self._setup_failures()
        if not failed:
            return

        # Stopped here rather than left to the monitor thread's next
        # pass: a start() that raises must not leave an acquisition
        # thread running, a device claimed and a file half written for
        # up to a second afterwards.
        try:
            self.stop()
        except Exception:
            # The setup failure is the story. A stop that also fails
            # must not replace it with its own.
            pass

        entry = self.failure or self.get_last_error()
        detail = "."
        if entry is not None:
            message = entry["message"] if "message" in entry else entry
            detail = f": {message}"
        names = ", ".join(
            sorted(naming.public_name(type(n).__name__) for n in failed)
        )
        error = RuntimeError(
            f"{names} could not set up, so the pipeline is not "
            f"running{detail} Nothing was emitted and no sink wrote a "
            f"row; fix what the node reported and start again."
        )

        # Chained to what the node actually raised, so the traceback
        # names the line in the node -- and in the user's own node, if
        # they wrote one. Until ioiocore kept the exception this could
        # only quote the logged message, which tells a caller that
        # something failed and never where.
        #
        # `from` only when there is a cause: `raise X from None`
        # suppresses the chain, which would be a worse answer than
        # saying nothing about it.
        cause = self._first_failure(failed)
        if cause is not None:
            raise error from cause
        raise error

    @staticmethod
    def _first_failure(failed: list):
        """The exception behind the first node that has one.

        Best effort by construction. `failure()` arrived in ioiocore
        5.0.0rc4, and a node built against an older one answers
        nothing -- in which case the message quoted above is all there
        ever was, and a chained `None` would not improve it.

        Args:
            failed (list): Nodes whose ``setup()`` raised.

        Returns:
            The first exception found, or None if none is available.
        """
        for node in failed:
            getter = getattr(node, "failure", None)
            if not callable(getter):
                continue
            try:
                cause = getter()
            except Exception:  # noqa: BLE001 - a probe, never a failure
                continue
            if cause is not None:
                return cause
        return None

    #: Safety stop for the batch driver. A batch source emits its whole
    #: recording in one cycle, so the loop below normally runs once; this
    #: only bounds a source that reports "more" forever, which would
    #: otherwise be an unkillable loop rather than an error.
    MAX_BATCH_CYCLES = 1_000_000

    def execution_mode(self) -> str:
        """How this pipeline will be driven, as declared by its sources.

        Read from the sources rather than from their time base: a
        recording *paced to the clock* is an ordinary realtime pipeline,
        so the two properties are independent and only one of them says
        how to drive the pipeline.

        Returns:
            A value from :class:`Constants.ExecutionMode`.

        Raises:
            ValueError: If the pipeline has no source, or if its sources
                disagree -- a pipeline cannot be half batch.
        """
        sources = self._sources()
        if not sources:
            raise ValueError(
                "This pipeline has no source, so there is no execution "
                "mode to speak of. Connect a source first."
            )
        modes = {
            getattr(
                node, "EXECUTION_MODE", Constants.ExecutionMode.REALTIME
            ): node
            for node in sources
        }
        if len(modes) > 1:
            names = ", ".join(
                f"{type(node).__name__} ({mode})"
                for mode, node in sorted(modes.items())
            )
            raise ValueError(
                f"A pipeline is either realtime or batch, never both, "
                f"but its sources disagree: {names}."
            )
        return next(iter(modes))

    def run(self) -> Union[Result, dict, None]:
        """Process a recording from end to end and return the results.

        The batch counterpart of :meth:`start`. Where a realtime
        pipeline is started, left running and stopped by the caller,
        a batch pipeline is *driven*: this call returns when the whole
        recording has passed through the last node.

        It executes on the calling thread and creates none of its own,
        which is what makes an exception in a node arrive as an ordinary
        Python traceback through the caller's line rather than as a log
        entry from a monitoring thread.

        Returns:
            The :class:`~gpype.common.result.Result` of the single
            :class:`~gpype.backend.sinks.collector.Collector` in the
            pipeline; a dict of results keyed by node name if there is more
            than one; ``None`` if there is no Collector, in which case
            the run's output went wherever its sinks put it.

        Raises:
            ValueError: If the pipeline is not a batch pipeline -- see
                :meth:`execution_mode` and :meth:`_validate_batch`.
            Exception: Whatever a node raises, unchanged. The node's
                frame and the caller's line are both in the traceback.
        """
        mode = self.execution_mode()
        if mode != Constants.ExecutionMode.BATCH:
            raise ValueError(
                f"run() drives a batch pipeline, and this one is "
                f"'{mode}'. Either build it with a source in batch mode "
                f"-- CsvReader(..., mode='batch') -- or use start() and "
                f"stop() to run it against the clock."
            )
        self._validate_batch()

        # Before start(), because a node's setup() happens on its first
        # cycle and the flag decides whether a failure there is raised or
        # logged. Set on every node, chain internals included.
        self._set_batch(True)
        try:
            self.start()
            for source in self._sources():
                cycles = 0
                while not source.is_exhausted:
                    source.cycle()
                    cycles += 1
                    if cycles >= self.MAX_BATCH_CYCLES:
                        raise RuntimeError(
                            f"{type(source).__name__} still reports more "
                            f"data after {cycles} cycles. A batch source "
                            f"must eventually report is_exhausted."
                        )
            self._raise_node_failure()
            results = self._collect()
        finally:
            self.stop()
            self._set_batch(False)
        return results

    def fit(self) -> Union[Result, dict, None]:
        """Fit every fittable node in this pipeline, then apply the result.

        The offline half of "fit offline, deploy online" (D-BATCH-16):
        drives a batch run exactly like :meth:`run`, except every
        :class:`~gpype.backend.core.fittable.Fittable` node in the pipeline
        fits on the whole recording first, instead of requiring state
        that was never loaded. A node not inheriting ``Fittable`` runs
        exactly as it would under :meth:`run`.

        Returns:
            Whatever :meth:`run` returns -- a fit run is a batch run,
            with the same single-Collector-or-dict-or-None contract.

        Raises:
            ValueError: If this pipeline holds no fittable node. Use
                :meth:`run` for an ordinary batch pipeline.
            Exception: Whatever :meth:`run` raises, unchanged.
        """
        fittable = self._fittable_nodes()
        if not fittable:
            raise ValueError(
                "this pipeline holds no fittable node, so there is "
                "nothing for fit() to learn. Use run() to drive an "
                "ordinary batch pipeline."
            )

        source_mark = self._single_source_mark()
        for node in fittable:
            node._gpype_fit_run = True
            node._gpype_external_mark = source_mark
        try:
            outcome = self.run()
        finally:
            for node in fittable:
                node._gpype_fit_run = False
        self._add_fitted_document(outcome)
        return outcome

    def _add_fitted_document(self, outcome) -> None:
        """Give a fit run's results the document that carries its fit.

        The provenance record is stamped at ``start()``, before anything
        is fitted, so ``Result.document()`` rebuilt a pipeline that
        ``run()`` refused for want of a state. ``fitted_document`` is
        this pipeline's document after the fit, artifacts included.

        Args:
            outcome: What :meth:`run` returned.
        """
        from ..common._private import provenance as provenance_module

        if isinstance(outcome, Result):
            results = [outcome]
        elif isinstance(outcome, dict):
            results = [r for r in outcome.values() if isinstance(r, Result)]
        else:
            results = []
        if not results:
            return
        document = provenance_module.fitted_document(self)
        if document is None:
            return
        key = Constants.Keys.PROVENANCE
        for result in results:
            record = result._context.get(key)
            if isinstance(record, dict):
                result._context[key] = dict(record, fitted_document=document)

    def _single_source_mark(self) -> Optional[str]:
        """The entitlement mark this run's one input file carries, if any.

        A batch run has exactly one source (see :meth:`_validate_batch`).
        Where it is a
        :class:`~gpype.backend.sources.base.recording_reader.RecordingReader`,
        its own ``mark`` answers whether the run that *wrote* the file
        was itself marked -- history the file carries forward regardless
        of what this machine's own licence says now (D-BATCH-92). Not a
        port-context key: nothing besides this has needed one there.

        Returns:
            The mark, or None where this pipeline has no single source
            or that source declares none.
        """
        sources = self._sources()
        if len(sources) != 1:
            return None
        return getattr(sources[0], "mark", None)

    #: Config key a batch reader's file name is stored under -- the
    #: literal string every reader's own constructor parameter writes
    #: (``CsvReader``, ``MatReader``, ``HDF5Reader``, ``EDFReader``,
    #: ``GtcReader``); there is no ``Constants.Keys`` entry for it.
    _FILE_NAME_KEY = "file_name"

    def run_all(self, files: list, source: Optional[str] = None) -> dict:
        """Run this pipeline's document once per file, in input order.

        The many-file counterpart of :meth:`run` (LQ-R3): this pipeline
        is never driven itself. Instead, once per file, its **document**
        is rebuilt with that file's name in place of the one this
        pipeline was constructed with, given fresh ids (D-BATCH-72, so
        the several pipelines this builds in one process do not collide
        in ioiocore's id registry), deserialized into a fresh
        :class:`Pipeline`, and run. Each result carries its own
        provenance (LQ-P2), naming that file's own input digest.

        Owner's decision (document-rebuild form, D-BATCH-104): a
        factory form -- ``run_all(build_fn, files)``, calling *build_fn*
        fresh per file -- would also run a script-only pipeline
        (``Apply``), which this form cannot. Rebuilding from the
        document is what makes every file run the identical, already-
        validated pipeline rather than trusting a callback to reconstruct
        it consistently, and it is what lets a caller build the
        template pipeline once, from a document someone else wrote,
        with no author script to call back into at all.

        The failure policy is fail-fast: the first file that raises
        stops the run, with its path in the message and the original
        exception as ``__cause__`` (D-BATCH-104, beside D-BATCH-52's
        pure propagation). Nothing before it is retried, and nothing
        after it runs.

        Args:
            files: Paths to run the document against, in order. Each
                becomes the single batch source's ``file_name``.
            source: Name of the node whose ``file_name`` to replace.
                None (the default) finds this pipeline's one source --
                a batch pipeline permits exactly one (:meth:`_validate_batch`)
                -- and needs naming only where a caller wants a specific
                node addressed by name regardless.

        Returns:
            dict: ``{file: result}``, in *files* order -- each value
            whatever :meth:`run` returns for that file (a
            :class:`~gpype.common.result.Result`, a dict of them, or
            None).

        Raises:
            ValueError: If this pipeline is not BATCH, if it does not
                have exactly one source and *source* names none, if
                *source* names a node that is not a source, if a file is
                listed twice (the results are keyed by file), or if the
                pipeline holds a function node (D-BATCH-70) -- checked
                once, before any file is read.
            Exception: Whatever the failing file's run raised, chained
                as ``__cause__`` of a ``RuntimeError`` naming that file.
        """
        mode = self.execution_mode()
        if mode != Constants.ExecutionMode.BATCH:
            raise ValueError(
                f"run_all() drives a batch pipeline once per file, and "
                f"this one is '{mode}'. Build it with a source in batch "
                f"mode -- CsvReader(..., mode='batch') -- first."
            )
        self._validate_batch()

        # Keyed by file, a file listed twice kept one result, silently.
        files = list(files)
        repeated = sorted(
            {str(f) for i, f in enumerate(files) if f in files[:i]}
        )
        if repeated:
            raise ValueError(
                f"run_all() returns one result per file, keyed by file, "
                f"and these are listed more than once: "
                f"{', '.join(repeated)}."
            )
        if source is not None:
            named = [n for n in self._sources() if n.name == source]
            if not named:
                sources = sorted(str(n.name) for n in self._sources())
                raise ValueError(
                    f"run_all(source={source!r}) must name this "
                    f"pipeline's source, whose file it replaces; the "
                    f"source is {', '.join(sources) or 'missing'}."
                )

        # Refuses a function node up front, naming it (D-BATCH-70) --
        # before any file in *files* is even opened.
        template = self.serialize()
        node_entries = list(template.get("nodes") or [])

        if source is None:
            # A source is written as its own core, not the chain that
            # wraps it (``_write_cores_not_chains``), so it is found by
            # class rather than by identity against ``self._imp._nodes``,
            # which holds the chains. Unique because a batch pipeline
            # permits exactly one source (``_validate_batch``, above).
            target = self._sources()[0]
            candidates = [
                i
                for i, entry in enumerate(node_entries)
                if entry.get("class") == type(target).__name__
                and entry.get("module") == type(target).__module__
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"run_all() could not find this pipeline's one "
                    f"source ({type(target).__name__}) uniquely in its "
                    f"own document. Pass source=<name> to say which "
                    f"node's file_name to replace."
                )
            index = candidates[0]
        else:
            index = next(
                (
                    i
                    for i, entry in enumerate(node_entries)
                    if (entry.get("config") or {}).get("name") == source
                ),
                None,
            )
            if index is None:
                names = ", ".join(
                    sorted(
                        str((entry.get("config") or {}).get("name"))
                        for entry in node_entries
                    )
                )
                raise ValueError(
                    f"run_all() found no node named {source!r} in this "
                    f"pipeline. It holds: {names}."
                )

        from ..common import document as document_module

        results: dict = {}
        for file in files:
            per_file = copy.deepcopy(template)
            nodes = list(per_file["nodes"])
            node_entry = dict(nodes[index])
            config = dict(node_entry.get("config") or {})
            config[self._FILE_NAME_KEY] = str(file)
            node_entry["config"] = config
            nodes[index] = node_entry
            per_file["nodes"] = nodes
            per_file = document_module.with_fresh_ids(per_file)

            try:
                built = Pipeline.deserialize(per_file)
                try:
                    results[file] = built.run()
                finally:
                    built.close()
            except Exception as error:
                raise RuntimeError(
                    f"run_all() failed on {file!r}: {error}"
                ) from error
        return results

    def _validate_batch(self) -> None:
        """Reject a pipeline that cannot mean anything as a batch run.

        Two rules, and both exist because of what a monolithic block is:
        a second source would carry an independent time axis with nothing
        to align it to, and a node with two fed inputs would be merging
        two blocks whose only relationship is arrival order.

        Fan-*out* is fine and stays allowed: branches never rejoin, so
        nothing can be misaligned by one.

        Raises:
            ValueError: If the pipeline has more than one source, or any
                node has more than one connected input port.
        """
        sources = self._sources()
        if len(sources) > 1:
            names = ", ".join(sorted(type(n).__name__ for n in sources))
            raise ValueError(
                f"A batch pipeline takes exactly one source, because a "
                f"second one has its own time axis and nothing to align "
                f"it to. This pipeline has {len(sources)}: {names}."
            )

        for node, _ in self._walk_nodes():
            imp = getattr(node, "_imp", None)
            ports = getattr(imp, "_input_ports", None) or {}
            fed = [
                name
                for name, port in ports.items()
                if getattr(port, "_connected_ports", None)
            ]
            if len(fed) > 1:
                raise ValueError(
                    f"{type(node).__name__} has {len(fed)} connected "
                    f"input ports ({', '.join(sorted(fed))}), which a "
                    f"batch run cannot merge: each carries a whole "
                    f"recording and their only relationship would be "
                    f"arrival order. Offline, what a realtime pipeline "
                    f"merges arrives as channels of one block."
                )

    def _set_batch(self, value: bool) -> None:
        """Tell every node whether this run is a batch run.

        Args:
            value: True while a batch run is being driven.
        """
        # Recorded on the pipeline as well as on the nodes: start() has
        # to know which way it is being driven before any node has
        # cycled, and walking the nodes there would be a search for an
        # answer run() already has.
        self._batch_driven = bool(value)
        for node, _ in self._walk_nodes():
            imp = getattr(node, "_imp", None)
            if imp is not None and hasattr(type(imp), "batch"):
                imp.batch = value

    def _raise_node_failure(self) -> None:
        """Raise if a node ended the run in an error condition.

        A backstop rather than the main path: in a batch run a node's
        exception propagates to the caller directly. This catches the
        case where something set the error condition without raising --
        a monitoring thread, or a node that reported rather than threw --
        so that a failed run cannot return an empty result and look
        successful.

        Raises:
            RuntimeError: If any node is in an error condition.
        """
        # get_condition() is new on ioiocore's ProcessingElement. Until
        # it existed the getattr below found nothing on any node -- the
        # condition was reachable only as Pipeline.get_condition and as
        # node._imp._condition -- so `failed` was always empty and this
        # backstop had never once fired. A batch run whose node reported
        # rather than threw still returned an empty result and looked
        # like a success.
        failed = [
            node
            for node, _ in self._walk_nodes()
            if getattr(node, "get_condition", None)
            and node.get_condition() == ioc.Constants.Conditions.ERROR
        ]
        if not failed:
            return
        entry = self.failure or self.get_last_error()
        detail = "."
        if entry is not None:
            detail = f": {entry['message'] if 'message' in entry else entry}"
        # The public name: a chain's core is what carries the condition,
        # so this list read "_CsvWriterCore" -- a symbol the public API
        # does not have, sending the reader after a class they cannot
        # find.
        names = ", ".join(
            sorted(naming.public_name(type(n).__name__) for n in failed)
        )
        raise RuntimeError(f"Batch run failed in {names}{detail}")

    def _collect(self) -> Union[Result, dict, None]:
        """Gather what the pipeline's collectors kept.

        A node opts in by declaring ``COLLECTS``, rather than by
        happening to expose a ``result`` member -- the same reason
        ``TIME_BASE`` and ``is_externally_fed`` are declared: an
        accidental match is a defect nobody would look for.

        Returns:
            One result, a dict of them keyed by node name, or None.
        """
        found = {}
        for node, _ in self._walk_nodes():
            if not getattr(type(node), "COLLECTS", False):
                continue
            result = node.result
            if result is not None:
                found[node.name] = result
        if not found:
            return None
        if len(found) == 1:
            return next(iter(found.values()))
        return found

    def _replays_only(self) -> bool:
        """Whether every source of this run replays a recording.

        The owner, 2026-10-03: "replay should not be device dongled"
        (D-ENT-99). A replay is one of the shipped readers -- a recording
        reader, a CSV reader, a batch source such as the GTC reader -- or
        a ``load_from`` replay core. Decided by class, not by the declared
        time base, which any source can set. Every source must replay: a
        recording reader beside a live source is refused at start(), and
        ``load_from`` replaces every source, so a mix that reaches here is
        one this method must not exempt.

        Returns:
            True if there is a source and every one of them replays.
        """
        from .sources.base.batch_source import BatchSource
        from .sources.base.raw import _RawReplayCore
        from .sources.base.recording_reader import RecordingReader
        from .sources.csv_reader import CsvReader

        replaying = (RecordingReader, CsvReader, BatchSource, _RawReplayCore)
        sources = self._sources()
        return bool(sources) and all(
            isinstance(node, replaying) for node in sources
        )

    def _begin_deferred_device_check(self, verdict=None) -> None:
        """Check for a required device without delaying start.

        Two cases, and they want opposite treatment:

        *An amplifier source attested.* Its handle was challenged inside
        start(), before any sink wrote a header, and answered: the
        requirement is met and there is nothing to look for.

        *None did* -- no amplifier is a source, or one did not answer the
        challenge -- and presence alone is the requirement. Finding a
        device costs 5-9 s, dominated by driver enumeration, and much
        more on a loaded machine, so the pipeline starts and the check
        runs behind it; if no device answers, the run is stopped a few
        seconds in and told why. A source merely holding a handle is not
        enough: a source of any make can hold one.

        The accepted residual: such a run does emit a few seconds of
        output before it is stopped. A truncated recording is not a
        usable artifact, which is what the requirement protects, so this
        is weaker than blocking and much stronger than nothing.

        Nothing happens at all unless entitlement.device_required() says
        a device is a precondition, which it does for a frozen standalone
        application only (D-ENT-98): development pays for no probe. Nor
        does a replay (D-ENT-99).

        Args:
            verdict: The run's resolved entitlement, whose ``attestation``
                says whether a device answered during start().
        """
        self._device_check = None
        if not device_required(
            frozen=licence.is_frozen_deployment(),
            residency=LaunchConfig.get().residency,
        ):
            return
        if (
            verdict is not None
            and verdict.attestation is Permission.FULL
            and not getattr(self, "_entitlement_supplied", False)
        ):
            return
        # A replay needs no device -- unless the licence cannot be
        # determined, when a device is what lets it run (D-ENT-100).
        if self._replays_only() and not getattr(
            self, "_licence_unverified", None
        ):
            return
        if self.execution_mode() == Constants.ExecutionMode.BATCH:
            # A batch run can be over before a deferred answer arrives,
            # so it waits: run() blocks its caller anyway, and nothing
            # has cycled yet. Afresh every run, as D latches for a run.
            presence = device_probe.probe(force=True)
            if not presence.reachable:
                raise PermissionError(
                    f"this application runs only with a g.tec device "
                    f"attached: {presence.detail}"
                )
            return

        def on_absent(presence) -> None:
            self.log(
                f"stopping: this application runs only with a g.tec "
                f"device attached: {presence.detail}",
                type=ioc.Constants.LogTypes.ERROR,
            )
            # Recorded as the run's failure, so an application polling
            # `failure` can tell this stop from an operator's.
            entry = self.get_last_error()
            if entry is not None:
                self._record_failure(entry)
            try:
                self.stop()
            except Exception:
                pass

        def on_present(presence) -> None:
            self.log(
                f"g.tec device reachable: {presence.serial}",
                type=ioc.Constants.LogTypes.INFO,
            )

        self._device_check = device_probe.DeferredCheck(
            on_absent=on_absent, on_present=on_present, force=True
        )
        self._device_check.start()

    def _cancel_deferred_device_check(self) -> None:
        """Stop caring about a check still in flight.

        An operator who stops the pipeline before the probe returns has
        already decided; reporting the answer then would stop an
        already-stopped pipeline and log a failure about nothing.
        """
        check = getattr(self, "_device_check", None)
        if check is not None:
            check.cancel()
            self._device_check = None

    #: How long after start() a server waits before naming the streams
    #: nothing at the far end has answered, in seconds. The Link's own
    #: context budget: the processes of one pipeline may be started in
    #: any order, and within it an edge that has not connected is still
    #: starting rather than missing.
    STREAM_REPORT_S = CONTEXT_TIMEOUT_S

    def _begin_stream_report(self) -> None:
        """On a server, schedule :meth:`_report_unanswered_streams`.

        Nothing else reports a node whose edge is not running (D-NODE-53).
        Its stream never arrives, so its receiving Link never sets up and
        its context timeout never fires; a sink's stream is pushed into
        an empty subscriber set without complaint; and a node downstream
        that merges the silent stream with a live one queues the live one
        without bound, until the only message is a generic input backlog.

        A timer rather than a wait, because start() must not charge every
        server run the budget, and once rather than repeatedly: the
        report is about the deployment, which does not change mid-run.
        """
        from .core._private import threads

        self._cancel_stream_report()
        if LaunchConfig.get().residency != Constants.Residency.SERVER:
            return
        self._stream_report_since = time.monotonic()
        if not threads.available():
            # A Timer is a thread; the event loop holds the report
            # instead, and its handle cancels the same way.
            self._stream_report = threads.call_later(
                self.STREAM_REPORT_S, self._report_unanswered_streams
            )
            return
        timer = threading.Timer(
            self.STREAM_REPORT_S, self._report_unanswered_streams
        )
        timer.daemon = True
        self._stream_report = timer
        timer.start()

    def _cancel_stream_report(self) -> None:
        """Drop a report still pending; a stopped run has nothing to say."""
        timer = getattr(self, "_stream_report", None)
        self._stream_report = None
        if timer is not None:
            timer.cancel()

    def _unanswered_streams(self) -> list:
        """Every Link stream here that nothing at the far end answered.

        Returns:
            ``(node name, edge id, stream id, sent)`` per stream, where
            ``sent`` is True for a stream this process sends and nobody
            receives, False for one it receives and nobody sends.
        """
        from .core._private import assembly, wrapping
        from .core._private.link import Link

        found = []
        for element in self._elements:
            core = wrapping.core_of(element)
            name = (core if core is not None else element).name
            edge_id = assembly.edge_id_of(core)
            nodes = [element, *(getattr(element, "internal_nodes", ()) or ())]
            for node in nodes:
                if not isinstance(node, Link):
                    continue
                sent = node.is_externally_drained()
                for stream in node.unanswered_streams():
                    found.append((name, edge_id, stream, sent))
        return found

    def _report_unanswered_streams(self) -> None:
        """Name each stream no edge has sent or subscribed to, with its node.

        Run by the timer :meth:`_begin_stream_report` starts, and
        skipped once the run is over.
        """
        if self.get_state() != ioc.Constants.States.RUNNING:
            return
        found = self._unanswered_streams()
        if not found:
            return
        parts = []
        for name, edge_id, stream, sent in found:
            where = "no edge_id" if edge_id is None else f"edge {edge_id!r}"
            what = "has no edge receiving it" if sent else "has sent nothing"
            parts.append(f"{name!r} ({where}, stream {stream!r}) {what}")
        since = getattr(self, "_stream_report_since", None)
        when = (
            "Since start"
            if since is None
            else f"{time.monotonic() - since:.0f} s after start"
        )
        self.log(
            f"{when}, "
            + "; ".join(parts)
            + ". An edge builds a node only when the node names it or no "
            "edge, with ids matched exactly, so no process is running as "
            "that edge, it has not started yet, or it was started with "
            "another id. A node fed by a stream that sends nothing waits "
            "for it without bound.",
            type=Constants.LogTypes.WARNING,
        )

    def stop(self):
        """Stop the pipeline and terminate all data processing.

        Gracefully shuts down all nodes, releasing threads and hardware
        connections. Always call stop() before program termination,
        especially when using hardware interfaces.

        Logging resources outlive a stop() so that a pipeline can be
        started again; call close() to release those as well.
        """
        self._await_failing_stop()
        # Before super().stop(), so a check that returns during teardown
        # cannot call stop() a second time from its own thread.
        self._cancel_deferred_device_check()
        self._cancel_stream_report()
        self._stop_attestation()
        # And before the nodes go: a probe holds a callable on a port,
        # and leaving it there would attach it to the *next* run of this
        # pipeline -- publishing to a stream id whose subscriber is long
        # gone, from a pipeline the operator never pointed it at.
        self._detach_all_probes()
        try:
            super().stop()
        finally:
            # Discard the timeline so a later pipeline in this process
            # starts with no master and no epoch.
            release_timeline(self)
            release_entitlement(self)
            # And a verdict supplied for a run that never started, so it
            # cannot entitle an unrelated later one (D-ENT-61).
            discard_supplied(self)

    def close(self):
        """Dispose the pipeline and release its master timeline.

        ioiocore's close() stops the pipeline through its own
        implementation, bypassing this class's stop(), so the timeline
        has to be discarded here as well. Safe to call repeatedly.
        """
        self._await_failing_stop()
        self._cancel_deferred_device_check()
        self._cancel_stream_report()
        self._stop_attestation()
        try:
            super().close()
        finally:
            release_timeline(self)
            release_entitlement(self)
            # And a verdict supplied for a run that never started, so it
            # cannot entitle an unrelated later one (D-ENT-61).
            discard_supplied(self)
            self._release_devices()

    def _await_failing_stop(self) -> None:
        """Let the monitor finish stopping a failed run first.

        :attr:`failure` is readable while ioiocore's monitor is still
        stopping the nodes (D-CORE-71), and ioiocore's ``stop()`` takes
        no lock, so a caller reacting to it stopped every node a second
        time, concurrently. Under ERROR, ioiocore's ``stop()`` and
        ``close()`` join the monitor anyway; this joins it before the
        nodes are touched. A join rather than a wait for STOPPED: a
        ``stop()`` that raises in the monitor leaves the state RUNNING,
        and this caller's stop is then the retry.

        ``start()`` joins it too (D-CORE-73). While the state reads
        RUNNING, ioiocore skips the start; once it reads STOPPED, the
        monitor has still to record the failure, and would record it on
        the new run.
        """
        if self.get_condition() != ioc.Constants.Conditions.ERROR:
            return
        thread = getattr(self._imp, "_monitor_thread", None)
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _release_devices(self) -> None:
        """Close every exclusive driver handle this pipeline opened.

        A source's own stop() releases its device on the ordinary path, but
        that path is not always taken: ioiocore stops the pipeline from its
        monitor thread when a node errors, and Pipeline.start() can raise
        before anything runs -- the entitlement gate does exactly that. A
        handle left open then locks the device out of the next run and the
        next process, because the driver refuses a second exclusive
        session.

        Called by close(), and by start() when the entitlement gate
        refuses the run. Until 2026-09-26 only close() called it, so a
        refused start kept the handle the gate had opened, and stop()
        does not reach a node that never ran.

        Never raises: this is teardown, where a second failure would mask
        the first.
        """
        for node in self._sources():
            release = getattr(node, "_release_device", None)
            if callable(release):
                try:
                    release()
                except Exception:
                    pass

    #: Key under which a package bundle travels inside the payload.
    BUNDLE_KEY = "bundle"

    #: Key under which a document's pins travel: a list of
    #: ``name==version`` strings (D-CORE-109). Absent when there are none.
    REQUIREMENTS_KEY = "requirements"

    #: Key under which every fitted node's state travels inside the
    #: document, keyed by its own content digest (D-BATCH-91, LQ-F2):
    #: ``{sha256: {encoding, data, meta}}``. A fitted node's own config
    #: carries only the digest, under ``fittable.ARTIFACT_KEY``.
    ARTIFACTS_KEY = "artifacts"

    def serialize(
        self,
        packages: Optional[list[str]] = None,
        requirements: Optional[list[str]] = None,
    ) -> dict:
        """Serialize the pipeline configuration to a dictionary.

        Args:
            packages: Top-level packages to carry along, for nodes whose
                classes are not installed on the executing host. A node
                from an uninstalled package cannot be rebuilt there,
                however complete its configuration is.
            requirements: Exact pins, ``name==version``, of what the
                document needs beyond g.Pype, written under
                :attr:`REQUIREMENTS_KEY`. A deployment that opted in
                installs them when it loads the document. The pipeline
                keeps them: a later call without them, and the provenance
                record of each run, write them again (D-CORE-109). None
                writes the pins kept, from an earlier call or the
                document this pipeline was rebuilt from, as that document
                carried them; an empty list writes none, and keeps none.

        Returns:
            dict: Dictionary containing the complete pipeline configuration,
                including nodes, connections, parameters, and metadata.

        Raises:
            ValueError: If the pipeline holds a function node -- ``Apply``
                or an ``@gp.node`` -- naming it. A document could not
                rebuild it (D-BATCH-70).
            RequirementError: If a requirement is not an exact pin, or
                names a distribution twice, naming every such entry.
        """
        self._refuse_script_only_nodes()
        if requirements is not None:
            pins = [str(pin) for pin in _installer.parse_pins(requirements)]
        else:
            pins = copy.deepcopy(getattr(self, "_requirements", None))
        data = super().serialize()
        self._write_cores_not_chains(data)
        self._write_fitted_artifacts(data)

        # ioiocore records itself in `writer` but cannot know about
        # g.Pype, and g.Pype is the half whose node behaviour has
        # actually changed under an unchanged document -- a filter's
        # initial state, a transform's scaling, a decimator gaining an
        # anti-alias stage. So the version that matters most for
        # attributing a difference is this one.
        from ioiocore.imp.pipeline_imp import PipelineImp

        writer_key = PipelineImp.SerializationKeys.WRITER
        writer = dict(data.get(writer_key) or {})
        try:
            from ..__version__ import __version__ as gpype_version

            writer["gpype"] = str(gpype_version)
        except Exception:  # pragma: no cover
            pass
        if writer:
            data[writer_key] = writer

        if packages:
            from ..common._private import bundle as bundle_module

            data[self.BUNDLE_KEY] = bundle_module.collect(packages)
        if pins:
            data[self.REQUIREMENTS_KEY] = pins
        if requirements is not None:
            # Kept once the document is written, so a refused call keeps
            # what was there.
            self._requirements = list(pins) or None
        return data

    def _script_only_nodes(self) -> list:
        """The nodes a document of this pipeline could not rebuild.

        A node declaring ``SCRIPT_ONLY`` -- ``Apply``, and so every
        ``@gp.node`` -- holds a Python function as instance state. A
        document would rebuild it without one, and the load fails with
        ``missing 1 required positional argument: 'function'``.

        The one answer for every caller that must not write or re-run a
        document of such a pipeline, ``serialize()`` first.

        The pipeline's own nodes count, and the internal nodes of a chain
        that writes them into the document (``INTERNALS_ARE_DERIVED =
        False``). A chain whose internals are derived rebuilds them from
        its configuration, so a function node it holds is its class's to
        supply.

        Returns:
            The nodes, in the order the pipeline holds them.
        """
        found: list = []

        def visit(node) -> None:
            if getattr(type(node), "SCRIPT_ONLY", False):
                found.append(node)
            internal = getattr(node, "internal_nodes", None)
            if internal and not getattr(
                type(node), "INTERNALS_ARE_DERIVED", True
            ):
                for inner in internal:
                    visit(inner)

        for node in list(getattr(self._imp, "_nodes", None) or []):
            visit(node)
        return found

    def _refuse_script_only_nodes(self) -> None:
        """Refuse to write a document that could not be rebuilt.

        Raises:
            ValueError: If the pipeline holds a function node, naming each.
        """
        found = self._script_only_nodes()
        if not found:
            return
        names = ", ".join(
            f"'{node.name}' ({naming.public_name(type(node).__name__)})"
            for node in found
        )
        raise ValueError(
            f"this pipeline cannot be written to a document: it holds a "
            f"function node, {names}. A function is not a configuration "
            f"value, so a document could not rebuild the node. Run the "
            f"pipeline from the script that builds it, or write a node "
            f"class whose parameters are configuration."
        )

    def _write_cores_not_chains(self, data: dict) -> None:
        """Record the core each chain of ours carries, not the chain.

        **The rule a distributed pipeline already lives by**: one
        document describes the whole system, and each process builds its
        own part of it. A chain *is* that per-process part -- which Link
        exists, whether the core is run at all -- so a document that
        named the chain would be describing one process's answer and
        handing it to every other. Writing the core keeps the document
        saying what the author wrote, and leaves each reader to wrap it
        for its own residency.

        **Connections become name references.** A chain exposes its
        boundary internal node's ports as its own, so a source chain's
        output port id *is* its `Sync`'s -- a node built fresh on every
        load. Recorded as an id it could never resolve again. ioiocore
        resolves ``"eeg.out"`` by name for exactly this reason, and a
        name is what the author wrote in the first place.

        Does nothing when no element is one of our chains.

        Args:
            data: The document from ``super().serialize()``, updated in
                place.
        """
        from ioiocore.imp.pipeline_imp import PipelineImp

        from .core._private import wrapping

        # Walked over ioiocore's own node list, which is what
        # ``super().serialize()`` built the document from -- element for
        # element, in that order. ``_elements`` is a different list: it
        # holds what passed through ``connect``, and ``add_node`` adds to
        # the pipeline without registering there. Indexing one against the
        # other replaced the wrong node, measured: an ``add_node``ed
        # Bandpass became the Generator core and the real chain stayed a
        # ``SourceChain``.
        elements = list(getattr(self._imp, "_nodes", None) or [])
        SK = PipelineImp.SerializationKeys
        nodes = list(data.get(SK.NODES) or [])
        if len(elements) != len(nodes):  # pragma: no cover
            # Nothing can be matched up safely; leave the document as
            # ioiocore wrote it rather than corrupt it.
            return

        # Port id -> "node.port", for every boundary port of every chain
        # we are about to replace.
        renamed: dict = {}
        replaced = False
        seen: dict = {}
        for index, element in enumerate(elements):
            core = wrapping.core_of(element)
            if core is None:
                continue
            replaced = True
            nodes[index] = core.serialize()
            name = (core.config or {}).get("name")
            seen.setdefault(name, []).append(core)
            for port_id, port_name in wrapping.boundary_port_ids(
                element
            ).items():
                renamed[port_id] = f"{name}.{port_name}"
        if not replaced:
            return
        self._refuse_ambiguous_names(seen)
        data[SK.NODES] = nodes
        data[SK.CONNECTIONS] = [
            [renamed.get(source, source), renamed.get(target, target)]
            for source, target in (data.get(SK.CONNECTIONS) or [])
        ]

    @staticmethod
    def _refuse_ambiguous_names(by_name: dict) -> None:
        """Refuse to write a document whose endpoints would collide.

        **Node names become load-bearing the moment a chain is
        assembly.** A connection to one is written as ``"eeg.out"``
        rather than as a port id, because the id belongs to a `Sync`
        rebuilt on every load (see ``_write_cores_not_chains``). Two
        nodes of one class that nobody named take ioiocore's default
        name -- their class name -- so they produce the *same* endpoint
        string, and the document stops recording which is which.

        Measured: a source feeding two unnamed `_CsvWriterCore`s writes
        ``[["eeg.out", "_CsvWriterCore.in"], ["eeg.out",
        "_CsvWriterCore.in"]]`` -- two byte-identical entries. Loading it
        does fail, because ioiocore drops an ambiguous name rather than
        picking the first, but it fails in whoever opens the file rather
        than in whoever wrote it, and by then the information is gone.

        So it is refused here, where the author can still fix it by
        naming the nodes. The same argument ``chain_params`` already
        makes about stream ids: an id is unique by construction, a name
        is not, and silently crossing two streams is worse than a
        message.

        Args:
            by_name: Node name to the cores carrying it.

        Raises:
            ValueError: If any name is carried by more than one wrapped
                node, naming the class and the count.
        """
        clashes = {
            name: cores for name, cores in by_name.items() if len(cores) > 1
        }
        if not clashes:
            return
        detail = "; ".join(
            f"{len(cores)} x {type(cores[0]).__name__} named {name!r}"
            for name, cores in sorted(
                clashes.items(), key=lambda kv: str(kv[0])
            )
        )
        raise ValueError(
            f"this pipeline cannot be written to a document: {detail}. "
            f"A connection to a node the pipeline builds a chain around "
            f"is recorded by name, because the port ids such a chain "
            f"exposes belong to internal nodes rebuilt on every load. "
            f"Two nodes sharing a name therefore produce the same "
            f"endpoint and the document stops saying which is which. "
            f"Give each of them a distinct name."
        )

    def _fittable_nodes(self) -> list:
        """Every :class:`~gpype.backend.core.fittable.Fittable` node here.

        Returns:
            The nodes, chain internals included (none are, today: a
            fittable node is a plain transform, never wrapped).
        """
        from .core.fittable import Fittable

        return [
            node
            for node, _ in self._walk_nodes()
            if isinstance(node, Fittable)
        ]

    def _write_fitted_artifacts(self, data: dict) -> None:
        """Write every fitted node's state into the document (LQ-F2).

        Each fitted node's state becomes one entry of
        ``data[Pipeline.ARTIFACTS_KEY]``, keyed by its own content
        digest, and that digest is written back into the node's own
        serialized config under ``fittable.ARTIFACT_KEY`` -- so a node
        with no state (never fitted) serializes exactly as before.

        Args:
            data: The document from ``super().serialize()``, updated in
                place.
        """
        fitted = [n for n in self._fittable_nodes() if n.state is not None]
        if not fitted:
            return

        from ..common._private import artifacts as artifacts_module
        from .core.fittable import ARTIFACT_KEY

        elements = list(getattr(self._imp, "_nodes", None) or [])
        nodes = list(data.get("nodes") or [])
        section = dict(data.get(self.ARTIFACTS_KEY) or {})

        for node in fitted:
            artifact_entry = artifacts_module.encode_state(node.state)
            sha256 = artifacts_module.digest_of(artifact_entry)
            section[sha256] = artifact_entry
            try:
                index = next(
                    i for i, element in enumerate(elements) if element is node
                )
            except StopIteration:  # pragma: no cover - defensive
                continue
            # ioiocore's own ``Configuration`` is read-only
            # (``__setitem__`` refuses), so the node's serialized entry
            # and its config are rebuilt as plain dicts rather than
            # mutated in place.
            node_entry = dict(nodes[index])
            config = dict(node_entry.get("config") or {})
            config[ARTIFACT_KEY] = sha256
            node_entry["config"] = config
            nodes[index] = node_entry

        data[self.ARTIFACTS_KEY] = section
        data["nodes"] = nodes

    @staticmethod
    def _rebuild(data: dict):
        """Rebuild the pipeline, wrapping each core on the way in.

        ``ioc.Pipeline.deserialize`` builds the nodes and wires them in
        one pass, with no point between the two where a node can be
        substituted -- and substituting is the whole job here, because a
        document records the **core** and every process has to put its
        own chain around it. So the pass is done here instead: build,
        wrap, add, then wire.

        Everything else is ioiocore's, deliberately. The format-version
        refusal, and the id-then-name endpoint resolution, are imported
        rather than reimplemented: a second implementation of *which
        documents load* is exactly the kind that drifts one release later
        and refuses a document the writer thought it had written.

        A document written before step 4 of ``chain-assembly`` names a
        class that is now the core, and records the chain's constructor
        call as its settings; it is built and wrapped like any other. A
        document naming a hand-written chain gets it back untouched.

        Args:
            data: The document.

        Returns:
            An ``ioc.Pipeline`` with every node in place.

        Raises:
            TypeError: If a connection endpoint cannot be resolved,
                naming the endpoints as the document wrote them.
        """
        from ioiocore.imp.pipeline_imp import (
            PipelineImp,
            _index_nodes,
            _resolve_port,
        )

        from .core._private import assembly, wrapping

        SK = PipelineImp.SerializationKeys

        built = ioc.Pipeline(directory=_fallback_log_directory())
        built.log("Starting pipeline deserialization...")
        PipelineImp._check_format_version(data, built)

        nodes = []
        for setting in data[SK.NODES]:
            # Constructed from what the document stored, so a node that
            # checks its settings against its world can tell them from
            # an author's (D-NODE-61).
            with assembly.rebuilding():
                node = ioc.Portable.deserialize(setting)
            node = wrapping.wrap(node)
            nodes.append(node)
            built.add_node(node)

        # Indexed on what was built, so a chain answers to the name its
        # core carried -- which is why a wrapper adopts it.
        by_name, ambiguous = _index_nodes(nodes)

        # An endpoint recorded as a port **id** names a port of the node
        # the document describes -- which, once that node is wrapped, is
        # a port *inside* the chain rather than the chain's own. It would
        # resolve, silently, to the wrong place: a consumer wired
        # upstream of the `Sync`, so the arrival-stamp channel is never
        # stripped and nothing is placed on the master timeline, with the
        # Sync left dangling and no error raised.
        #
        # Measured by rewriting a stored 4.0.0 document's class name to
        # its core, which is exactly what step 4 does to every document
        # already on disk: endpoint 2371E226... resolved to
        # `_GeneratorCore.out` while the chain's boundary `Sync.out` was
        # in no connection at all.
        #
        # Translated rather than refused, because those documents are
        # correct -- it is the meaning of "the node" that moved. Straight
        # to the chain's port, not through its name: a document that
        # connected by id may well hold two nodes of one class that
        # nobody named, and a name they share resolves to neither.
        translated = {}
        for node in nodes:
            core = wrapping.core_of(node)
            if core is None:
                continue
            for port_id, port_name in wrapping.boundary_port_ids(core).items():
                translated[port_id] = (node, port_name)

        def resolve(ref, output: bool):
            hit = translated.get(ref)
            if hit is None:
                return _resolve_port(ref, by_name, ambiguous, output=output)
            chain, port_name = hit
            if output:
                return chain.get_output_port(port_name)
            return chain.get_input_port(port_name)

        for source, target in data[SK.CONNECTIONS]:
            try:
                oport = resolve(source, output=True)
                iport = resolve(target, output=False)
                oport.connect(iport)
                try:
                    iport.connect(oport)
                except Exception:
                    oport.disconnect(iport)
                    raise
                built.log(f"Connected ports {source} and {target}.")
            except Exception as exc:
                message = (
                    f"Deserialization failed: {exc} "
                    f"(connecting {source!r} to {target!r})"
                )
                built.log(msg=message, type=Constants.LogTypes.ERROR)
                raise TypeError(message) from exc
        built.log("Deserialization complete.")
        return built

    @staticmethod
    def _named_modules(data: dict) -> list:
        """The modules a document's nodes name, in document order.

        Args:
            data: A serialized pipeline document.

        Returns:
            Every string module name, malformed entries skipped; the
            loader reports those in its own terms.
        """
        from ioiocore.imp.pipeline_imp import PipelineImp
        from ioiocore.imp.portable_imp import PortableImp

        nodes = data.get(PipelineImp.SerializationKeys.NODES) or []
        if not isinstance(nodes, (list, tuple)):
            return []
        return [
            setting.get(PortableImp.SerializationKeys.MODULE)
            for setting in nodes
            if isinstance(setting, dict)
            and isinstance(
                setting.get(PortableImp.SerializationKeys.MODULE), str
            )
        ]

    @staticmethod
    def check(data: dict, install: bool = False) -> None:
        """Refuse a document the allow-list or requirement check refuses.

        The document-level half of :meth:`deserialize`, which calls it
        first, so the order is written only here. The modules the
        document's nodes name are checked against the deployment's
        allow-list. Then, where the deployment installs a document's pins
        (``GPYPE_PACKAGE_INDEX`` is set) and the document has some, they
        are checked for form and against what is installed, and, with
        *install*, every absent one is installed. Last, for a document
        carrying packages, their code is checked for imports this host
        cannot satisfy.

        Nothing else happens: no bundle is unpacked, nothing is put on
        the import path, no module a node names is imported and no node
        is built. A validator can therefore ask whether a check would
        refuse a document, in the terms loading it uses, without loading
        it. The load can still refuse a document that passes: for its
        format revision, or with a ``BundleError`` as its bundle is
        unpacked.

        Without *install*, pip never runs, and a pin that is absent here
        may be what supplies a missing import; so while any pin is
        absent, the missing imports are not refused. The load installs
        the pins first and refuses them then.

        Args:
            data (dict): Serialized pipeline configuration dictionary.
            install (bool): Install the document's absent pins, where the
                deployment opted in. :meth:`deserialize` passes True.

        Raises:
            ModuleNotAllowedError: If the document names a module this
                deployment does not permit, naming every such module.
            RequirementError: If a pin is not exact, or a pinned wheel
                needs a distribution nothing pins or provides.
            RequirementConflictError: If a pin names a version other than
                the one installed here, or, with *install*, a pinned
                wheel needs another version of an installed distribution
                or would overwrite its files.
            InstallerError: If the pins cannot be installed.
            MissingRequirementsError: If the document's bundled code
                imports a package this host does not have, naming every
                such package.
        """
        from ..common._private import allowlist as _allowlist

        _allowlist.check_document(data)

        pinned = _installer.for_document(
            data.get(Pipeline.REQUIREMENTS_KEY), install_absent=install
        )

        if data.get(Pipeline.BUNDLE_KEY):
            from ..common._private import bundle as bundle_module

            try:
                bundle_module.check_requirements(
                    data[Pipeline.BUNDLE_KEY], Pipeline._named_modules(data)
                )
            except bundle_module.MissingRequirementsError as error:
                if pinned is not None and pinned["absent"] and not install:
                    return
                hint = _installer.missing_hint(
                    pinned, data.get(Pipeline.REQUIREMENTS_KEY)
                )
                raise bundle_module.MissingRequirementsError(
                    f"{error} {hint}"
                ) from None

    @staticmethod
    def restore_requirements() -> dict:
        """Reinstall the pins the package cache records, with no network.

        For the process that starts a deployment: in a container the
        cache is a volume and the environment is the image's, so the
        pins installed before a restart are installed again from the
        cache, each wheel copied out of it and checked against the hash
        recorded at its download (D-CORE-88, D-CORE-117). Entry by entry:
        one whose distribution is installed at another version now is
        skipped and reported, as is one whose wheel this interpreter does
        not install, is missing or was replaced, one pip refuses or that
        would overwrite another distribution's files, and one whose own
        pip call does not finish within 600 s; the others are installed.
        Does nothing where ``GPYPE_PACKAGE_INDEX`` is not set.

        Whoever can write the cache can have a wheel installed here, with
        no document involved: the cache must be writable only by the
        deployment.

        Returns:
            dict: ``{"enabled": bool, "installed": [...], "present":
            [...], "skipped": [{"requirement": str, "reason": str}]}``,
            JSON-safe.

        Raises:
            InstallerError: Before anything is installed, if the cache's
                manifest cannot be read, another install holds the cache
                for longer than 1200 s, this interpreter's wheel tags
                cannot be read, or this interpreter, its site-packages or
                its pip cannot install. At any point, possibly after some
                entries were installed, which the next call reports as
                present, if pip cannot be started or the cache or a
                private directory beside pip cannot be used.
        """
        return _installer.restore()

    @staticmethod
    def installer_config() -> dict:
        """Whether this deployment installs a document's pins, and how.

        Read once from ``GPYPE_PACKAGE_INDEX`` and
        ``GPYPE_PACKAGE_CACHE``, for a control plane to report. Nothing
        can set it but the deployment's environment.

        Returns:
            dict: ``{"enabled": bool, "index": str or None, "cache":
            str}``, JSON-safe, the index's credentials redacted.
        """
        return _installer.configuration()

    @staticmethod
    def deserialize(data: dict) -> "Pipeline":
        """Deserialize a pipeline configuration from a dictionary.

        Rebuilding a node imports the module it names, so a document is
        executable input. It is checked first (:meth:`check`, with
        ``install=True``): the modules it names against the deployment's
        allow-list; then its pins, installing every absent one where the
        deployment opted in; then any bundled code for imports this host
        cannot satisfy -- all before any bundle is written to disk and put
        on the import path, so an untrusted payload is never unpacked for
        a document that was going to be refused. Without the opt-in,
        nothing is installed for it.

        Any packages carried in the payload are then unpacked and put on
        the import path, because rebuilding a node imports its class. The
        rebuilt pipeline keeps the document's pins as the document
        carried them, and its :meth:`serialize` writes them again; they
        are validated where they are installed.

        Args:
            data (dict): Serialized pipeline configuration dictionary.

        Returns:
            Pipeline: A new Pipeline instance with the specified configuration.

        Raises:
            ModuleNotAllowedError: If the document names a module this
                deployment does not permit.
            RequirementError, RequirementConflictError, InstallerError:
                If the deployment installs pins and this document's
                cannot be installed (:meth:`check`).
            MissingRequirementsError: If the document's bundled code
                imports a package this host does not have.
            ModuleNotFoundError: If a node's module cannot be imported;
                for a document with pins where the deployment installs
                none, saying that they were not installed.
        """
        Pipeline.check(data, install=True)

        if data.get(Pipeline.BUNDLE_KEY):
            from ..common._private import bundle as bundle_module

            bundle_module.install(data[Pipeline.BUNDLE_KEY])

        # Deserialize using parent class. A node module its pins would
        # have supplied fails here where the deployment installs none,
        # so the refusal says so (D-CORE-111).
        try:
            ioc_pipeline = Pipeline._rebuild(data)
        except ModuleNotFoundError as error:
            hint = _installer.pins_not_installed(
                data.get(Pipeline.REQUIREMENTS_KEY)
            )
            if hint is None:
                raise
            raise ModuleNotFoundError(
                f"{error} {hint}", name=error.name, path=error.path
            ) from error

        # Create Pipeline instance without calling __init__
        pipeline = object.__new__(Pipeline)

        # Copy all instance attributes from deserialized pipeline
        pipeline.__dict__.update(ioc_pipeline.__dict__)

        # __init__ was skipped, so this class's own state does not exist
        # yet. Established through the same method __init__ uses, never a
        # second hand-written list -- that is exactly how the previous
        # version fell behind.
        pipeline._init_gpype_state()
        pipeline._requirements = copy.deepcopy(
            data.get(Pipeline.REQUIREMENTS_KEY)
        )

        # One thing genuinely differs from a fresh pipeline: the elements
        # come from ioiocore's own node list rather than starting empty.
        # They are what connect() would have recorded, and an empty list
        # would leave every Sync in a restored pipeline without a
        # timeline, so no event could ever be placed.
        imp = getattr(ioc_pipeline, "_imp", None)
        pipeline._elements = list(getattr(imp, "_nodes", None) or [])

        # The chains were built in _rebuild, before this pipeline
        # existed, so the map `chain_of` reads is empty unless it is
        # filled here. Without it a restored SERVER pipeline answers None
        # for every source it carries: the server runs no source core,
        # so walking `internal_nodes` finds nothing.
        from .core._private import wrapping

        for element in pipeline._elements:
            core = wrapping.core_of(element)
            if core is not None:
                pipeline._wrappers[id(core)] = element

        pipeline._load_fitted_artifacts(data)

        # Loading is when a server's reference is zeroed (D-TIME-20), so
        # a server that only deserializes -- a test, an example -- zeroes
        # it as the runtime does. Skipped while another pipeline here is
        # running rather than refused: a document read to be checked
        # must not fail, nor move a running pipeline's clock.
        launch = LaunchConfig.get()
        if (
            launch.residency == Constants.Residency.SERVER
            and not Pipeline._running_pipelines()
        ):
            Pipeline.rezero_reference()

        return pipeline

    def _load_fitted_artifacts(self, data: dict) -> None:
        """Load every fitted node's state out of the document (LQ-F2).

        Args:
            data: The document just rebuilt from.

        Raises:
            ValueError: If a node names an artifact the document does
                not carry, or one whose stored digest disagrees with its
                own content (edited or corrupted after it was written).
        """
        section = data.get(self.ARTIFACTS_KEY) or {}
        if not section:
            return

        from ..common._private import artifacts as artifacts_module
        from .core.fittable import ARTIFACT_KEY

        for node in self._fittable_nodes():
            sha256 = getattr(node, "_gpype_artifact_ref", None)
            if not sha256:
                continue
            label = naming.node_label(node)
            entry = section.get(sha256)
            if entry is None:
                raise ValueError(
                    f"{label} names artifact {sha256!r} ({ARTIFACT_KEY} "
                    f"in its configuration), but this document's "
                    f"{self.ARTIFACTS_KEY!r} section does not carry it."
                )
            state = artifacts_module.verify_and_decode(sha256, entry, label)
            node._gpype_state = state
            node._gpype_marked = bool(state.get("marked", False))
            node.load_state(state)
