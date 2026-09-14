import logging
import time
from time import perf_counter
from typing import Optional, Union

import ioiocore as ioc

from ..common import config_keys
from ..common._private import (
    attestation,
    device_probe,
    licence,
    naming,
    pipelines,
    remote_attestation,
)
from ..common._private.channels import stamp_clock_complaint
from ..common._private.entitlement import (
    MARK,
    Entitlement,
    Permission,
    device_required,
    discard_supplied,
    entitlement_of,
    meet,
    release_entitlement,
    remote_required,
    set_entitlement,
    take_supplied,
)
from ..common.constants import Constants
from ..common.launch_config import LaunchConfig
from ..common.result import Result
from .core._private.controllable import (
    action_names,
    ensure_json,
    has_control,
    merged_update,
    resolve_action,
    resolve_control,
)
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


class Pipeline(ioc.Pipeline):
    """Brain-Computer Interface pipeline for real-time data processing.

    Extends ioiocore Pipeline for BCI applications with automatic logging
    to platform-specific directories. Manages node lifecycle, data flow
    connections, and real-time execution of interconnected processing nodes.
    """

    def __init__(self):
        """Initialize Pipeline with platform-specific logging directory."""
        # Determine platform-specific log directory
        import os
        import sys

        if sys.platform == "win32":
            log_dir = os.path.join(os.getenv("APPDATA", ""), "gtec", "gPype")
        elif sys.platform == "darwin":
            app_support = os.path.expanduser("~/Library/Application Support")
            log_dir = os.path.join(app_support, "gtec", "gPype")
        else:
            log_dir = None  # Use default ioiocore directory

        # Initialize parent pipeline with logging directory
        super().__init__(directory=log_dir)

        self._init_gpype_state()

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

        #: Live probes, by stream id. Not part of the graph and not in
        #: _elements: a probe is a callable on a port, deliberately not
        #: a node -- see backend/core/_private/probe.py for why.
        self._probes: dict = {}

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

        Returns:
            A log entry carrying the message, the node and the source
            location.
        """
        return self._failure

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
        graphs whose sources cannot be reconciled at all. Checked here,
        ahead of everything else, so the error names the real problem
        rather than surfacing later as a licence message or as data that
        merely looks wrong.

        Raises:
            ValueError: If sources with different time bases are mixed,
                if no source produces a continuously sampled stream, or
                if the graph contains an illegal feedback loop.
        """
        self._validate_time_bases()
        self._validate_has_signal()
        self._validate_no_illegal_cycles()
        self._validate_batch_only_nodes()
        self._warn_unknown_chain_keys()
        self._warn_untapped_sources()

    def _warn_untapped_sources(self) -> None:
        """Name any source this run's ``save_as``/``load_from`` misses.

        Both are inserted by ``raw.source_stage``, which every source
        *chain* calls. A source that is not a chain has no such point at
        all -- ``GtcReader`` extends ``BatchSource`` directly -- so
        ``save_as`` contributes no file for it and ``load_from`` leaves
        its live core in place.

        Both halves matter, and the second one more than it looks.
        Under ``save_as``, a manifest that simply does not mention a
        stream is indistinguishable from a manifest of a graph that
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
                f"stream. Everything else in this graph is recorded.",
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
            ValueError: If the graph is not a batch pipeline and
                contains a node that requires one.
        """
        batch_only = [
            node
            for node, _ in self._walk_nodes()
            if getattr(type(node), "BATCH_ONLY", False)
        ]
        if not batch_only:
            return

        # Asked after the nodes, so a graph with none of them never
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
        """Reject a graph mixing replayed and live sources.

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
        """Reject a graph built only from event sources.

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
        # can name what the author wrote. A source is usually a chain
        # around a private core, and being told that "_MarkerCore" is the
        # problem is no help to somebody who wrote gp.Marker().
        event_only = []
        for element in self._elements:
            candidates = [element]
            internal = getattr(element, "internal_nodes", None)
            if internal:
                candidates.extend(internal)
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
            event_only.append(type(element).__name__)

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
        """Reject a graph containing an illegal feedback loop.

        Runs here for a graph assembled by :meth:`deserialize`, which
        never passes through :meth:`connect` at all. For an ordinarily
        built pipeline it is mostly a second look at rules already
        applied per-edge -- see :meth:`connect` -- with one rule that
        only this call can apply: a loop must be fed from outside
        itself, which is a property of the finished graph and cannot be
        judged while edges are still arriving.

        Raises:
            ValueError: If the graph contains a loop refused by
                :meth:`_describe_component_violation`.
        """
        violation = self._cycle_violation()
        if violation is not None:
            raise ValueError(violation)

    def _cycle_owner_map(self) -> dict:
        """Map every declared input port to the element that owns it.

        Rebuilt on every call rather than cached, because the answer
        changes on every :meth:`connect` and nothing here tracks that
        invalidation. The graphs this runs against are, at most, dozens
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
        """Return the directed node graph built by connect() so far.

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
        iteratively because recursion depth here is the graph's depth
        and a pipeline may be deeper than the interpreter's limit;
        linear in nodes plus edges.

        Args:
            adjacency: The graph from :meth:`_cycle_adjacency`.

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
            adjacency: The graph from :meth:`_cycle_adjacency`.

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
        the refusal path alone -- an accepted graph never calls it.

        Args:
            node: Element the returned path must start at.
            nodes: Ids of the elements the search may enter.
            adjacency: The graph from :meth:`_cycle_adjacency`.

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
           two, this is a property of the *finished* graph rather than
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
            adjacency: The graph from :meth:`_cycle_adjacency`.
            require_feed: Whether to apply rule 2. False while the
                graph is still being assembled, where the feed may
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
                outside itself. False while the graph is still being
                assembled -- see :meth:`_describe_component_violation`.

        Returns:
            Refusal text naming the offending loop, or None if every
            cycle in the graph built so far is legal (including none at
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

        # A verdict supplied from outside wins, and is taken exactly
        # once. An embedder that established entitlement by means this
        # process cannot repeat -- a runtime that challenged an admin
        # machine for a signed licence attestation and verified it --
        # has a conclusion this resolution could not reach on its own:
        # licence.evaluate() asks about *this* machine, and a container
        # holds no licence. See D-ENT-61.
        supplied = take_supplied(self)
        if supplied is not None:
            set_entitlement(self, supplied)
            return supplied

        licence_permission, licence_reason = licence.evaluate()

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

        permission = meet((licence_permission, attest_permission))
        reason = licence_reason if licence_permission is permission else ""
        if not reason and attest_permission is permission:
            reason = attest_reason

        verdict = Entitlement(
            permission,
            reason,
            licence=licence_permission,
            attestation=attest_permission,
        )
        set_entitlement(self, verdict)
        return verdict

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

    def _resolve_node(self, node_id: str):
        """The node in *this* pipeline carrying *node_id*.

        Resolved over this pipeline's own elements rather than through
        ``PortableImp.get_by_id``, which is a **process-global**
        registry: it would hand back a node belonging to a different
        pipeline in the same process, so a control plane holding two
        sessions could drive the wrong one and get an ordinary success
        back.

        Only nodes that exist in this process resolve at all. Under
        server residency a source or sink chain has no core here, so it
        simply does not resolve -- intended, and not a gap to forward
        around.

        Args:
            node_id: The node's document id.

        Returns:
            The node object.

        Raises:
            KeyError: If no node in this pipeline carries that id.
        """
        for node, _ in self._walk_nodes():
            if _node_id_of(node) == node_id:
                return node
        raise KeyError(f"no node with id '{node_id}' in this pipeline")

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
        report: list[dict] = []
        for node, is_internal in self._walk_nodes():
            entry = {
                "name": getattr(node, "name", ""),
                "class": type(node).__name__,
                "internal": is_internal,
                "id": _node_id_of(node),
                "actions": action_names(node),
                "controllable": has_control(node),
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

    def get_control(self, node_id: str) -> dict:
        """Read the full control state of one node.

        Args:
            node_id: The node's document id, as reported by
                :meth:`get_nodes`.

        Returns:
            The node's whole control state, as a copy -- mutating it
            cannot reach the node.

        Raises:
            KeyError: If no node in this pipeline carries that id.
            AttributeError: If the node declares no control channel.
            TypeError: If the node's answer is not a JSON-representable
                dict.
        """
        node = self._resolve_node(node_id)
        if not has_control(node):
            raise AttributeError(
                f"node '{node_id}' declares no control channel."
            )
        getter, _ = resolve_control(node)
        state = ensure_json(getter(), False, f"{node_id}.get_control")
        # Copied on the way out. A node returning its own live dict would
        # otherwise let a caller mutate its state by hand -- no merge, no
        # unknown-key check, none of the node's own validation -- which is
        # the whole surface this method exists to be.
        return dict(state)

    def set_control(self, node_id: str, arg: dict) -> dict:
        """Apply a partial control update to one node.

        The update **merges**: *arg* names only what changes, so a
        one-field change stays one field and two clients need no
        get-before-set to avoid overwriting each other. Any key the
        node's own ``get_control`` does not report is refused **before
        the node sees it** -- the getter is the key set.

        Args:
            node_id: The node's document id.
            arg: The keys to change.

        Returns:
            The resulting full control state, so a set is also a get.
            A copy, as in :meth:`get_control`.

        Raises:
            KeyError: If no node carries that id, or *arg* holds a key
                the node does not accept -- that message names both the
                offending keys and the accepted ones.
            TypeError: If *arg* is not a dict, or the node's answer is
                not a JSON-representable dict.
            AttributeError: If the node declares no control channel.
            ValueError: Whatever the node raises when the resulting
                combination is not usable. Propagated deliberately: only
                the node knows its own cross-field rules.
        """
        node = self._resolve_node(node_id)
        if not has_control(node):
            raise AttributeError(
                f"node '{node_id}' declares no control channel."
            )
        getter, setter = resolve_control(node)
        current = ensure_json(getter(), False, f"{node_id}.get_control")
        state = ensure_json(
            setter(merged_update(current, arg)),
            False,
            f"{node_id}.set_control",
        )
        return dict(state)  # copied on the way out -- see get_control

    def invoke_action(
        self, node_id: str, name: str, arg: dict
    ) -> Optional[dict]:
        """Run one declared action on one node.

        Args:
            node_id: The node's document id.
            name: The action name, as reported by :meth:`get_nodes`.
            arg: The action's argument; may be empty.

        Returns:
            Whatever the action reports, or None when it reports nothing.

        Raises:
            KeyError: If no node in this pipeline carries that id.
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
        method = resolve_action(node, name)
        if method is None:
            raise AttributeError(
                f"node '{node_id}' declares no action '{name}'; "
                f"declared actions are {action_names(node)}"
            )
        result = ensure_json(method(arg), True, f"{node_id}.{name}")
        return None if result is None else dict(result)

    def attach_probe(
        self,
        node_id: str,
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
                :meth:`get_nodes` reports it.
            port: Name of its output port.
            max_rate: Samples per second to deliver, at most. Defaults
                to :data:`probe.DEFAULT_MAX_RATE_HZ`.

        Returns:
            str: The stream id to subscribe to.

        Raises:
            KeyError: If no node in this pipeline carries that id.
            ValueError: If the node has no such output port, or if this
                process has no running broker to publish on.
        """
        from .core._private import probe as probe_module
        from .core._private.ws.broker import WsBroker

        node = self._resolve_node(node_id)
        getter = getattr(node, "get_output_port", None)
        if getter is None:
            raise ValueError(
                f"node '{node_id}' has no output ports, so there is "
                f"nothing on it to watch. A probe reads what a node "
                f"emits; a sink emits nothing."
            )
        try:
            port_imp = getter(port)
        except Exception as error:
            raise ValueError(
                f"node '{node_id}' has no output port '{port}': {error}"
            ) from error

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
        self._register(source)
        self._register(target)
        super().connect(source, target)

        # Checked after the edge exists, not before: the real port graph
        # -- which a bare node, a dict spec, and a chain proxying to its
        # first/last internal node all normalise to differently -- only
        # exists once ioiocore has resolved it.
        #
        # The check looks at the whole graph rather than at this edge,
        # and disconnecting this edge is still the right rollback. Every
        # prior connect() left the pipeline with no illegal loop in it
        # (a graph rebuilt by deserialize() is covered at start()
        # instead -- see _validate_no_illegal_cycles), so a component
        # that violates a rule now has to involve this edge: either it
        # closed the loop, or it connected the port that turned an
        # existing legal loop illegal -- an ASYNC input, say. Removing it
        # restores the graph that passed.
        #
        # What the message names is the loop, which is not always this
        # edge. That is deliberate: the loop is what the author has to
        # change.
        #
        # require_feed is off here: "something outside feeds this loop"
        # is true of a finished graph, not of every intermediate one.
        # Closing the loop and then wiring the source that drives it is
        # an ordinary way to write the same pipeline, so enforcing it
        # per-edge would refuse a graph that is about to be legal.
        # start() applies it -- see _validate_no_illegal_cycles.
        violation = self._cycle_violation(require_feed=False)
        if violation is not None:
            self._imp.disconnect(source, target)
            raise ValueError(violation)

    def start(self):
        """Start the pipeline and begin real-time data processing.

        Initiates execution of all nodes according to their configured
        connections and timing. Runs continuously until stop() is called.
        This method is non-blocking.
        """
        # Startup runs in ordered phases, and the order is load-bearing.
        # Note setup() does not run here -- ioiocore defers it to each
        # node's first cycle -- so a phase may only use what is known at
        # construction.
        #
        #   1. validate  -- reject graphs that cannot mean anything, so
        #                   the user is told the real problem
        #   2. timeline  -- bind it, so election has somewhere to happen
        #   3. start     -- nodes run, setup happens, election resolves
        #
        # Phases for tier resolution belong between 2 and 3, before any
        # sink can write.
        self._validate()

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
            # lives as long as the graph, so without this a pipeline
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
        # not specific to a graph shape and there is nothing to inspect
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

        super().start()

        # Only now, once the pipeline is actually running: the case where
        # a device must be *reachable* but is not a source. See
        # _begin_deferred_device_check for why it is after start() and
        # not before.
        self._begin_deferred_device_check()

        # Also only now: the edge's Links have their WsClients, so it can
        # start answering. See _start_attestation_responders for why the
        # edge does not wait for this and the server does.
        self._start_attestation_responders()

        # Last, because it is the only phase that waits: give the first
        # cycles a moment to report a setup failure, so a graph that was
        # never viable fails here rather than going quiet. Skipped for a
        # batch run, which run() drives on the caller's thread -- a setup
        # failure there is raised rather than logged, and nothing has
        # cycled yet at this point, so the wait would buy nothing and
        # charge every offline run the whole grace period.
        if not getattr(self, "_batch_driven", False):
            self._await_setup()

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
        behind, and that is a run that died rather than a graph that
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
        """Let the first cycles run, and refuse a graph that cannot.

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

    def _validate_batch(self) -> None:
        """Reject a graph that cannot mean anything as a batch run.

        Two rules, and both exist because of what a monolithic block is:
        a second source would carry an independent time axis with nothing
        to align it to, and a node with two fed inputs would be merging
        two blocks whose only relationship is arrival order.

        Fan-*out* is fine and stays allowed: branches never rejoin, so
        nothing can be misaligned by one.

        Raises:
            ValueError: If the graph has more than one source, or any
                node has more than one connected input port.
        """
        sources = self._sources()
        if len(sources) > 1:
            names = ", ".join(sorted(type(n).__name__ for n in sources))
            raise ValueError(
                f"A batch pipeline takes exactly one source, because a "
                f"second one has its own time axis and nothing to align "
                f"it to. This graph has {len(sources)}: {names}."
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
                    f"arrival order. Offline, what a realtime graph "
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
        """Gather what the graph's collectors kept.

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

    def _has_amplifier_source(self) -> bool:
        """Whether a device this pipeline reads from was challenged.

        A source holding a driver handle has already been attested during
        start(), synchronously, because the pipeline cannot produce data
        without it -- there is nothing to defer.

        Returns:
            True if any source exposes a driver handle.
        """
        return any(
            getattr(node, "_device", None) is not None
            for node in self._sources()
        )

    def _begin_deferred_device_check(self) -> None:
        """Check for a required-but-unused device without delaying start.

        Two cases, and they want opposite treatment:

        *The amplifier is a source.* Its handle is challenged inside
        start(), before any sink writes a header. Blocking is right: the
        pipeline has nothing to do until the device answers anyway.

        *The amplifier is not a source* and its presence alone is the
        requirement. Finding it costs 5-9 s, dominated by driver
        enumeration, and blocking start() would charge that to every run
        -- including every example and most of the test suite, none of
        which depend on the answer. So the pipeline starts and the check
        runs behind it; if no device answers, the run is stopped a few
        seconds in and told why.

        The accepted residual: such a run does emit a few seconds of
        output before it is stopped. A truncated recording is not a
        usable artifact, which is what the requirement protects, so this
        is weaker than blocking and much stronger than nothing.

        Nothing happens at all unless entitlement.device_required() says
        this run is conditional, which it does not yet -- so today this
        is inert by construction rather than by luck.
        """
        self._device_check = None
        if not device_required():
            return
        if self._has_amplifier_source():
            return

        def on_absent(presence) -> None:
            self.log(
                f"stopping: {presence.detail}",
                type=ioc.Constants.LogTypes.ERROR,
            )
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
            on_absent=on_absent, on_present=on_present
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

    def stop(self):
        """Stop the pipeline and terminate all data processing.

        Gracefully shuts down all nodes, releasing threads and hardware
        connections. Always call stop() before program termination,
        especially when using hardware interfaces.

        Logging resources outlive a stop() so that a pipeline can be
        started again; call close() to release those as well.
        """
        # Before super().stop(), so a check that returns during teardown
        # cannot call stop() a second time from its own thread.
        self._cancel_deferred_device_check()
        self._stop_attestation()
        # And before the nodes go: a probe holds a callable on a port,
        # and leaving it there would attach it to the *next* run of this
        # pipeline -- publishing to a stream id whose subscriber is long
        # gone, from a graph the operator never pointed it at.
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
        self._cancel_deferred_device_check()
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

    def _release_devices(self) -> None:
        """Close every exclusive driver handle this pipeline opened.

        A source's own stop() releases its device on the ordinary path, but
        that path is not always taken: ioiocore stops the pipeline from its
        monitor thread when a node errors, and Pipeline.start() can raise
        before anything runs -- the entitlement gate does exactly that. A
        handle left open then locks the device out of the next run and the
        next process, because the driver refuses a second exclusive
        session.

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

    def serialize(self, packages: Optional[list[str]] = None) -> dict:
        """Serialize the pipeline configuration to a dictionary.

        Args:
            packages: Top-level packages to carry along, for nodes whose
                classes are not installed on the executing host. A node
                from an uninstalled package cannot be rebuilt there,
                however complete its configuration is.

        Returns:
            dict: Dictionary containing the complete pipeline configuration,
                including nodes, connections, parameters, and metadata.
        """
        data = super().serialize()

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
        return data

    @staticmethod
    def deserialize(data: dict) -> "Pipeline":
        """Deserialize a pipeline configuration from a dictionary.

        Rebuilding a node imports the module it names, so a document is
        executable input. The modules it names are checked against the
        deployment's allow-list *first* -- before any bundle is written
        to disk and put on the import path, so an untrusted payload is
        never unpacked for a document that was going to be refused.

        Any packages carried in the payload are then unpacked and put on
        the import path, because rebuilding a node imports its class.

        Args:
            data (dict): Serialized pipeline configuration dictionary.

        Returns:
            Pipeline: A new Pipeline instance with the specified configuration.

        Raises:
            ModuleNotAllowedError: If the document names a module this
                deployment does not permit.
        """
        from ..common._private import allowlist as _allowlist

        _allowlist.check_document(data)

        if data.get(Pipeline.BUNDLE_KEY):
            from ..common._private import bundle as bundle_module

            bundle_module.install(data[Pipeline.BUNDLE_KEY])

        # Deserialize using parent class
        ioc_pipeline = ioc.Pipeline.deserialize(data)

        # Create Pipeline instance without calling __init__
        pipeline = object.__new__(Pipeline)

        # Copy all instance attributes from deserialized pipeline
        pipeline.__dict__.update(ioc_pipeline.__dict__)

        # __init__ was skipped, so this class's own state does not exist
        # yet. Established through the same method __init__ uses, never a
        # second hand-written list -- that is exactly how the previous
        # version fell behind.
        pipeline._init_gpype_state()

        # One thing genuinely differs from a fresh pipeline: the elements
        # come from ioiocore's own node list rather than starting empty.
        # They are what connect() would have recorded, and an empty list
        # would leave every Sync in a restored pipeline without a
        # timeline, so no event could ever be placed.
        imp = getattr(ioc_pipeline, "_imp", None)
        pipeline._elements = list(getattr(imp, "_nodes", None) or [])

        return pipeline
