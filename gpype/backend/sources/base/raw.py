"""Recording and replaying every source's frames at the tap point.

The tap point is a source core's own output port: after the driver, after
the source's channel selection and its in-band timeline channels, and
*before* any ``Link``, ``Oscar`` or ``Sync``. Two launch parameters put
something there:

* ``save_as`` inserts :class:`_RawTap` behind every core, which writes
  the frames the core emits, unmodified, to one HDF5 file per stream.
* ``load_from`` replaces every core with :class:`_RawReplayCore`, which
  re-emits those frames on the same block boundaries with the same
  inter-block spacing.

Both are decided in ``create_internal_nodes()`` through
:func:`source_stage`, so a chain gains this by replacing the one line
that builds its core.

Four properties are load-bearing, and each one is silent when it is
wrong. They are the reason this module exists rather than the tap being
a plain :class:`~gpype.CsvWriter` wired in by hand.

**One shifted clock, not many.** ``channels.wrap_time`` carries host time
as 100 us units modulo 2**20, so a timestamp channel repeats every 105
seconds, and ``unwrap_time`` resolves a stored value against the
*reader's* own ``perf_counter``. Replaying stored stamps verbatim offsets
every one of them by an arbitrary constant -- and when that constant
approaches the 52.4 s half-period, the reconstructed instants jump by
105 s partway through the run. So a replay computes **one** shift, at the
whole run's origin, and applies it to the emission schedule and to every
stamp column: see :class:`ReplayClock` and :func:`shift_stamps`. Because
the shift is exact modular arithmetic on the wrapped value, the intra-block
stamp structure an amplifier writes per sample survives untouched.

**Block boundaries are recorded, not inferred.** A file writer's flat
sample matrix loses them, and the stamp column cannot recover them: a
frame may carry several distinct per-sample arrivals, and two
notifications can share one at 100 us quantisation. So every emitted
frame gets a row in ``gpype_raw_blocks``, and that table is what the
replay schedules from. Without it, "as if it were live" degrades to "at
the nominal sampling rate", which for a packetised transport is a
different stream.

**Candidacy is recorded.** A replay core that does not reproduce
``master_candidacy()`` lets the election pick a different winner. Every
stream still aligns; it aligns to a different grid, and nothing says so.

**Streams are matched by manifest, never positionally.** ``stream_id_for``
mints a random id when the document carries none, so files cannot be
keyed on it. The key is the author's node name where there is one and
``<Class>_<ordinal>`` otherwise, and :func:`resolve_stream` refuses a
graph whose shape disagrees with the manifest rather than pairing two
streams that merely happen to sit in the same position.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional

import numpy as np

from ....common._private import channels
from ....common.constants import Constants
from ....common.launch_config import LaunchConfig
from ...core.i_port import IPort
from ...core.io_node import IONode
from ...core.o_port import OPort
from .source import Source

#: Default port identifiers.
PORT_IN = Constants.Defaults.PORT_IN
PORT_OUT = Constants.Defaults.PORT_OUT

#: Version tag of the run manifest, and the prefix a loader accepts.
#: Major only, for the reason ``recording_meta.META_VERSION`` records: a
#: field can be added without every existing loader needing a bump.
MANIFEST_VERSION = "gpype-raw-run/1"

#: Name of the manifest file inside a run directory.
MANIFEST_NAME = "manifest.json"

#: Dataset holding one row per emitted frame, as
#: ``(sample_offset, length, t_emit, t_stamp_last)``. Times are seconds
#: from the run origin, float64 -- deliberately not the float32 the
#: sample matrix carries, because a relative second in float32 is
#: quantised at 0.24 ms after an hour, twice the 100 us the stamp design
#: guarantees.
BLOCKS_DATASET = "gpype_raw_blocks"

#: Sub-document of ``gpype_meta`` carrying what a replay needs and an
#: ordinary recording does not. Nested inside the existing document
#: rather than beside it, so ``recording_meta.parse`` validates one
#: version tag and ``read_h5`` keeps ignoring what it does not know.
RAW_META_KEY = "raw"

#: How many frames may wait for the disk before the tap starts dropping
#: them. Bounded on purpose: an unbounded queue turns a slow disk into
#: unbounded memory growth, and a blocking put turns it into an overrun
#: on the acquisition thread. Dropping and *saying so* is the only one of
#: the three that neither lies nor takes the run down -- the same trade
#: ``_BCICoreCore`` makes for its own frame buffer.
QUEUE_LIMIT = 2048

#: Seconds between two reports of a full queue, so a sustained overrun
#: does not fill the log with one line per frame.
OVERFLOW_LOG_INTERVAL = 5.0


def _h5py():
    """Import h5py, or say what installs it.

    Returns:
        The h5py module.

    Raises:
        ImportError: If h5py is not installed.
    """
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment
        raise ImportError(
            "save_as and load_from store raw streams as HDF5, which "
            "needs h5py, provided by the 'formats' extra: "
            "pip install 'gpype[formats]'"
        ) from error
    return h5py


# -- stream keys ---------------------------------------------------------

#: How many chains of each class have been keyed since the last reset.
_ordinals: dict = {}
#: Keys already taken by a stream in the graph being built.
_claimed: set = set()
#: Guards both; chains are normally built on one thread, but a process
#: hosting two sessions is not obliged to.
_ordinal_lock = threading.Lock()


def reset_stream_keys() -> None:
    """Forget the per-class ordinals.

    Called from ``LaunchConfig.configure``, which is the one point that
    is guaranteed to run before any chain is built and after any previous
    run has finished. Without it a second pipeline in one process
    continues counting, and a ``load_from`` graph built after a
    ``save_as`` graph would ask for ``Generator_1``.
    """
    with _ordinal_lock:
        _ordinals.clear()
        # Together, always. Reissuing an ordinal while still holding the
        # claim on it makes this function a trap: the next graph asks
        # for Generator_0 again and is refused for colliding with the
        # previous graph's stream. Found by a test that reset the
        # ordinals on their own.
        _claimed.clear()


def stream_key(chain) -> str:
    """The name this chain's raw stream is filed under.

    The author's node name where there is one, because a document that
    names its nodes deserves file names it can read, and because a name
    survives inserting a node ahead of it. ``<Class>_<ordinal>``
    otherwise, assigned in construction order.

    Deliberately not the node id: ``stream_id_for`` mints one at random
    when the document carries none, so it differs between the run that
    saved and the run that loads. Deliberately not the class name alone
    either -- two `Generator`s would collide, which is the same
    silent cross-wiring ``chain_params`` refused for stream ids.

    Read out of ``_core_params`` rather than off ``chain.name``, which
    is not answerable yet: ``OChain.__init__`` calls
    ``_init_internal_nodes`` *before* ``create_config``, so at the
    moment a chain builds its internals it has no configuration and
    therefore no name. Every chain in this package stores its core's
    parameters under that attribute before chaining up, and the name is
    among them because ``strip_chain_keys`` only removes the id and the
    port lists. Measured: without this, a Marker built as
    ``gp.Marker(name="mrk")`` was keyed ``Marker_0``.

    Args:
        chain: The chain being built.

    Returns:
        A file-name-safe key.
    """
    from ....common._private.naming import public_name

    cls = public_name(type(chain).__name__)
    params = getattr(chain, "_core_params", None) or {}
    name = params.get("name") or getattr(chain, "name", None)
    if name and name != type(chain).__name__ and name != cls:
        return _sanitize(str(name))

    with _ordinal_lock:
        ordinal = _ordinals.get(cls, 0)
        _ordinals[cls] = ordinal + 1
    return f"{cls}_{ordinal}"


def claim_stream_key(key: str, owner: str) -> None:
    """Reserve *key* for one stream of this graph, or refuse it.

    Two sources can reach one key: nothing in g.Pype refuses two nodes
    with the same name, and :func:`_sanitize` additionally maps distinct
    names onto one key -- ``"left hand"`` and ``"left_hand"`` both
    become ``left_hand``. Unrefused, the second tap overwrites the
    first's manifest entry, so the run describes one stream where two
    were recorded, and a replay hands that one entry to both chains:
    one device's stream through another's chain, which is precisely
    what :func:`resolve_stream` promises to refuse and cannot detect
    here, because both nodes are the same class.

    Refused while the graph is still being built rather than left to
    the replay: the recording is the artifact, and one that cannot be
    replayed is worth failing a run to prevent.

    Args:
        key: The stream key.
        owner: How to name the offending stream in the message.

    Raises:
        ValueError: If the key is already claimed in this graph.
    """
    with _ordinal_lock:
        if key in _claimed:
            raise ValueError(
                f"two sources in this graph both file their raw stream "
                f"under '{key}' ({owner} is the second). A stream key "
                f"is the node's name, or its class and an ordinal when "
                f"it has none -- and names are not unique, so two nodes "
                f"named alike, or named 'left hand' and 'left_hand', "
                f"collide here. Give them distinct names."
            )
        _claimed.add(key)


def _sanitize(text: str) -> str:
    """Reduce *text* to something safe as a file name.

    Args:
        text: The author's node name.

    Returns:
        The name with every character outside ``[A-Za-z0-9._-]`` replaced
        by an underscore.
    """
    keep = set("._-")
    return (
        "".join(c if (c.isalnum() or c in keep) else "_" for c in text)
        or "stream"
    )


def _free_path(path: str) -> str:
    """Return *path*, or the next free name beside it.

    A pipeline may be started again after ``stop()`` -- g.Pype
    documents that and the restart re-runs ``setup()``, so this opens
    the file a second time. Opening it ``"w"`` would truncate the
    recording that had just been made, and the same is true of
    ioiocore's error-restart path, where a fast retry loop would
    overwrite what it just recorded.

    So the name is never reused. ``FileWriter._generate_file_path``
    reached the same conclusion for the same reason and states it
    plainly: losing recorded data is the one outcome worth extra code
    to prevent. The run's *directory* already carries a timestamp, so
    the names inside it can stay deterministic and only a second run
    within one process needs a suffix.

    Args:
        path: The name this stream would like.

    Returns:
        That name, or ``<stem>_2.h5``, ``<stem>_3.h5``, ... for a
        second and later recording of the same stream in one run
        directory.
    """
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    index = 2
    while os.path.exists(f"{stem}_{index}{ext}"):
        index += 1
    return f"{stem}_{index}{ext}"


# -- the run -------------------------------------------------------------


class RawRun:
    """One ``save_as`` run: a directory, a manifest, and a shared origin.

    Process-wide rather than per pipeline, because ``LaunchConfig`` is:
    the flag names one run, and every source in the process belongs to
    it. The origin in particular *must* be shared -- it is what makes the
    recorded block times comparable between streams, which is the whole
    reason to record them.
    """

    #: The run this process is recording into, if any.
    _current: Optional["RawRun"] = None
    _lock = threading.Lock()

    @classmethod
    def current(cls, save_as: str) -> "RawRun":
        """Return the run for *save_as*, creating it once.

        Args:
            save_as: The configured path stem.

        Returns:
            The shared run.
        """
        with cls._lock:
            if cls._current is None or cls._current.stem != save_as:
                cls._current = RawRun(save_as)
            return cls._current

    @classmethod
    def reset(cls) -> None:
        """Forget the current run, so a later one starts a new directory."""
        with cls._lock:
            cls._current = None

    def __init__(self, save_as: str):
        """Create the run directory.

        Args:
            save_as: Path stem. A ``.h5`` suffix is accepted and dropped:
                the stem names a *run*, and a run is a directory of
                streams rather than one file.
        """
        self.stem = save_as
        stem, ext = os.path.splitext(save_as)
        if ext.lower() in (".h5", ".hdf5", ""):
            base = stem or save_as
        else:
            base = save_as
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        directory = f"{base}_{stamp}"
        suffix = 1
        while os.path.exists(directory):
            directory = f"{base}_{stamp}_{suffix}"
            suffix += 1
        os.makedirs(directory, exist_ok=True)
        self.directory = directory

        self._entries: dict = {}
        self._origin: Optional[float] = None
        self._origin_units: Optional[int] = None
        #: Guards the origin only. Separate from the manifest's lock on
        #: purpose: ``origin()`` is reached from ``_RawTap.step``, which
        #: runs on the source's own thread -- an amplifier's acquisition
        #: callback -- while ``register`` holds its lock across a file
        #: write. One shared lock let a manifest rewrite block an
        #: acquisition callback for the length of that write, which is
        #: what this module refuses to do on the sample path and had no
        #: business doing here.
        self._anchor = threading.Lock()
        #: Guards the manifest entries and the file they go to.
        self._guard = threading.Lock()

    def origin(self) -> float:
        """Seconds on the stamp clock that this run's times count from.

        Set by whichever stream emits first, and shared from then on. A
        per-stream origin would make every stream start at zero, which
        is exactly the information a distributed alignment needs and the
        one thing a replay cannot invent back.

        Quantised to the stamp clock's own 100 us unit, so that
        :meth:`origin_units` is exact and the shift a replay applies to
        a wrapped stamp column stays integer arithmetic.

        Returns:
            The origin, on ``channels.stamp_clock()``.
        """
        # Read without the lock first: both fields are written under it
        # and never rewritten while set, so a reader seeing one sees the
        # other. This sits on the per-frame path.
        origin = self._origin
        if origin is not None:
            return origin
        with self._anchor:
            if self._origin is None:
                units = int(
                    round(
                        channels.stamp_clock() * channels.TIME_UNITS_PER_SECOND
                    )
                )
                self._origin_units = units
                self._origin = units / channels.TIME_UNITS_PER_SECOND
            return self._origin

    def origin_units(self) -> int:
        """This run's origin in whole 100 us units of the stamp clock.

        Recorded in every stream's metadata, and it has to be: a stamp
        column holds ``round(t * U) % M`` of an *absolute* reading of the
        recording process's clock. A replay wants those values to mean
        "now", which is ``(replay_origin - recording_origin)`` added
        under the same modulus -- so without this number the second term
        is missing and every stamp is displaced by an arbitrary constant.

        Measured while building this: with the term missing, the first
        marker of a replayed run landed 88 samples from where the live
        run placed it, while every later one was exact -- because the
        master's stamps carried the same wrong constant and the timeline
        relation absorbed it, right up until the point where an unwrap
        resolved one of them to a different candidate.

        Returns:
            The origin in 100 us units.
        """
        self.origin()
        return int(self._origin_units)

    def begin(self) -> None:
        """Forget the origin, so the next run anchors itself afresh.

        A pipeline may be started again after ``stop()``, and this
        object lives as long as the *graph* rather than the run -- it is
        created in ``create_internal_nodes``. Without this, run 2's
        block times stay relative to run 1's origin: a recording made
        ten seconds after the first carries ``t_emit`` around 10.5, and
        a replay of it then sits in ``time.sleep`` for ten seconds
        before emitting anything, with nothing logged and
        ``is_exhausted`` still False. Hours, if the two runs were hours
        apart.

        The mirror of :meth:`ReplayClock.reset`, called from the same
        place in ``Pipeline.start`` and for the same reason. The
        directory is deliberately not reset: it belongs to the graph,
        and ``_free_path`` gives each stream's second recording its own
        file inside it.
        """
        with self._anchor:
            self._origin = None
            self._origin_units = None

    def path_for(self, key: str) -> str:
        """Where the stream keyed *key* is written.

        Args:
            key: The stream key.

        Returns:
            Absolute-or-relative path of the HDF5 file.
        """
        return os.path.join(self.directory, f"{key}.h5")

    def register(self, entry: dict) -> None:
        """Record or update one stream's manifest entry, and rewrite it.

        Written at ``setup`` and again at ``stop``, so a run killed
        mid-recording still leaves a manifest a loader can read -- the
        same reasoning ``recording_meta.collect`` records for writing a
        header before the first sample.

        Args:
            entry: The stream's entry, carrying at least ``key``.
        """
        with self._guard:
            self._entries[entry["key"]] = entry
            document = {
                "format": MANIFEST_VERSION,
                "created": datetime.now(timezone.utc).isoformat(),
                "residency": LaunchConfig.get().residency,
                "origin_clock": "perf_counter",
                "streams": [self._entries[k] for k in sorted(self._entries)],
            }
            # Written beside and renamed rather than truncated in
            # place. Each stream rewrites this twice, so an in-place
            # write leaves a window where a killed process leaves a
            # truncated manifest -- and then every intact .h5 in the
            # directory is unloadable, because the manifest is how a run
            # is found at all.
            path = os.path.join(self.directory, MANIFEST_NAME)
            temporary = path + ".part"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
            os.replace(temporary, path)


class ReplayClock:
    """The single instant a ``load_from`` run maps its recording onto.

    One per process, for the reason :class:`RawRun` shares its origin:
    two streams anchored independently lose their relation to each
    other, and reproducing that relation is the point.

    Quantised to the stamp clock's own 100 us unit so that the shift
    applied to a wrapped stamp column is an exact integer. Rounding it
    later would spend up to half a unit of the resolution the stamp
    design guarantees, for nothing.
    """

    _origin_units: Optional[int] = None
    _lock = threading.Lock()

    @classmethod
    def origin_units(cls) -> int:
        """This run's origin, in ``channels.TIME_UNITS_PER_SECOND`` units.

        Returns:
            The origin, set by the first caller.
        """
        with cls._lock:
            if cls._origin_units is None:
                cls._origin_units = int(
                    round(
                        channels.stamp_clock() * channels.TIME_UNITS_PER_SECOND
                    )
                )
            return cls._origin_units

    @classmethod
    def origin(cls) -> float:
        """This run's origin in seconds, quantised.

        Returns:
            Seconds on the stamp clock.
        """
        return cls.origin_units() / channels.TIME_UNITS_PER_SECOND

    @classmethod
    def reset(cls) -> None:
        """Forget the origin, so the next run anchors itself afresh."""
        with cls._lock:
            cls._origin_units = None


def shift_stamps(block: np.ndarray, columns, shift_units: int) -> None:
    """Move a recorded stamp column onto this run's clock, in place.

    A wrapped stamp is ``round(t * TIME_UNITS_PER_SECOND) % TIME_MODULUS``,
    so adding a constant to *t* is adding a constant to the wrapped value
    under the same modulus. Doing it that way rather than unwrapping,
    adding and re-wrapping is not an optimisation: it is what keeps the
    per-sample structure exact, because no value ever leaves the integer
    grid the recording was written on.

    Args:
        block: Frame to modify, shape ``(samples, channels)``.
        columns: Indices of the timestamp channels.
        shift_units: Whole 100 us units to add, already reduced modulo
            ``TIME_MODULUS`` by the caller.
    """
    if not len(columns) or shift_units == 0:
        return
    modulus = channels.TIME_MODULUS
    for column in columns:
        if column >= block.shape[1]:
            continue
        values = np.rint(block[:, column].astype(np.float64))
        block[:, column] = np.mod(values + shift_units, modulus).astype(
            block.dtype
        )


# -- the tap -------------------------------------------------------------


class _RawTap(IONode):
    """Writes every frame passing through it, and passes it on unchanged.

    Sits between a source core and whatever follows it. Transparent in
    both directions: the output context is the input context, so nothing
    downstream can tell it is there.

    A node rather than a ``post_step_handler`` composed onto the core.
    The handler would cost no thread, but it never sees a port context --
    so the file could not carry the roles, labels and rate that make it
    replayable -- and it is invisible to the pipeline's
    ``attach_entitlement`` sweep, which only walks nodes, so the
    recording would not carry the run's mark.
    """

    #: Whether this run's output must carry the non-commercial mark.
    #: False until the pipeline says otherwise, exactly as
    #: ``FileWriter._marked`` is, so a tap used outside a pipeline
    #: produces an unmarked file rather than claiming a restriction
    #: nobody established.
    _marked: bool = False

    def attach_entitlement(self, verdict) -> None:
        """Record whether this run's artifacts must be marked.

        Args:
            verdict: The pipeline's resolved Entitlement.
        """
        self._marked = bool(verdict.marked)

    class Configuration(IONode.Configuration):
        """Configuration class for _RawTap parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for the raw tap."""

            #: Where this stream is written
            FILE_NAME = "file_name"
            #: The manifest key this stream is filed under
            STREAM_KEY = "stream_key"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys for the raw tap."""

            #: Class name of the chain being recorded, for the manifest
            SOURCE_CLASS = "source_class"

    def __init__(
        self,
        file_name: str,
        stream_key: str,
        run: "RawRun",
        source_class: Optional[str] = None,
        node_name: Optional[str] = None,
        node_id: Optional[str] = None,
        timing: str = Constants.Timing.SYNC,
        **kwargs,
    ):
        """Initialize the tap.

        Args:
            file_name: Path of the HDF5 file to write.
            stream_key: Manifest key for this stream.
            run: The run this stream belongs to, for the shared origin
                and the manifest.
            source_class: Public class name of the chain, recorded so a
                loader can refuse a graph that has changed shape.
            node_name: The chain's node name, recorded for the same
                reason.
            node_id: The chain's document id, recorded for the same
                reason.
            timing: Timing of both ports. A sparse stream must stay
                sparse through the tap, or every downstream node would
                wait for data on every cycle -- the same reason ``Sync``
                takes this.
            **kwargs: Additional arguments for IONode.
        """
        self._run = run
        self._node_name = node_name
        self._node_id = node_id
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_LIMIT)
        self._stop_event = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._file = None
        self._data_ds = None
        self._blocks_ds = None
        self._samples = 0
        self._blocks = 0
        self._dropped = 0
        self._last_overflow_log = 0.0
        self._time_columns: list = []
        self._entry: dict = {}
        self._candidacy = None

        kwargs.setdefault("input_ports", [IPort.Configuration(timing=timing)])
        kwargs.setdefault("output_ports", [OPort.Configuration(timing=timing)])
        IONode.__init__(
            self,
            file_name=file_name,
            stream_key=stream_key,
            source_class=source_class,
            **kwargs,
        )

    # -- lifecycle -------------------------------------------------------

    def start(self):
        """Start the background writer, once.

        Guarded like :meth:`_RawReplayCore.start` is: a second call
        would spawn a second writer thread, drop the handle to the
        first, and let the two interleave ``_samples`` and ``resize``
        on one dataset.
        """
        if self._worker is not None and self._worker.is_alive():
            return
        self._samples = 0
        self._blocks = 0
        self._dropped = 0
        self._last_overflow_log = time.monotonic()
        self._stop_event.clear()
        self._worker = threading.Thread(target=self._write_worker, daemon=True)
        self._worker.start()
        super().start()

    def stop(self):
        """Stop accepting frames, drain the queue, then close the file.

        The order is the whole of it, and two things were wrong before.

        ``super().stop()`` comes **first**, so no further frame can be
        queued once the worker has gone. It used to come last: the
        worker leaves as soon as it finds the queue empty, and the file
        close plus a manifest rewrite after it takes long enough for a
        250 Hz source to queue several more frames -- which were never
        written, never counted as dropped, and left the recording
        quietly shorter than the run it is supposed to reproduce.

        And the close cannot be allowed to escape. Neither
        ``ChainImp.stop`` nor ``PipelineImp.stop`` guards a node's
        ``stop()``, so an ``OSError`` from finalising a recording on a
        full disk would leave the *amplifier* in the same chain
        unstopped, its device handle held, and the pipeline never
        reaching STOPPED.
        """
        try:
            super().stop()
        finally:
            self._stop_event.set()
            if self._worker is not None:
                self._worker.join(timeout=10.0)
                self._worker = None
            try:
                self._close()
            except Exception as error:
                self.log(
                    f"the raw recording of "
                    f"'{self.config[self.Configuration.Keys.STREAM_KEY]}' "
                    f"could not be finalised: {error}. The samples "
                    f"already written are on disk; its manifest entry "
                    f"may understate what it holds.",
                    type=Constants.LogTypes.ERROR,
                )

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Open the file and forward the context untouched.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts.

        Returns:
            The input context, forwarded unchanged.
        """
        # Deliberately not calling IONode.setup. Its validation exists
        # for nodes that *compute* from the context, and it requires a
        # sampling rate on every input -- which a sparse stream
        # legitimately does not have. Inheriting that check made this
        # node refuse a Marker outright, which is to say it refused
        # exactly the streams save_as exists to capture. A transparent
        # node requires nothing of its input and derives nothing from
        # it: the output context *is* the input context, and any
        # difference between the two would be this node editing the
        # description of a stream it is only supposed to be watching.
        context = dict(port_context_in[PORT_IN])
        self._time_columns = channels.channels_with_role(
            context, Constants.ChannelRoles.TIMESTAMP
        ).tolist()
        self._open(context)
        return {PORT_OUT: context}

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Queue the frame for writing and pass it on.

        The frame is copied, because everything downstream is free to
        modify it and the copy is what reaches the disk a few
        milliseconds later.

        Args:
            data: Input data.

        Returns:
            The same data, unmodified.
        """
        block = None if data is None else data.get(PORT_IN)
        if block is not None and getattr(block, "size", 0):
            stamp = channels.stamp_clock() - self._run.origin()
            try:
                self._queue.put_nowait((np.asarray(block).copy(), stamp))
            except queue.Full:
                self._dropped += 1
                now = time.monotonic()
                if now - self._last_overflow_log >= OVERFLOW_LOG_INTERVAL:
                    self._last_overflow_log = now
                    self.log(
                        f"the raw recording of "
                        f"'{self.config[self.Configuration.Keys.STREAM_KEY]}'"
                        f" is behind the stream and has dropped "
                        f"{self._dropped} frame(s); the file has holes and "
                        f"a replay of it will not reproduce this run.",
                        type=Constants.LogTypes.ERROR,
                    )
        return {PORT_OUT: block}

    # -- file ------------------------------------------------------------

    def _open(self, context: dict) -> None:
        """Create the file, its metadata and the manifest entry.

        Args:
            context: The input port context describing the stream.
        """
        from ...sinks.base import recording_meta

        module = _h5py()
        keys = self.Configuration.Keys
        path = self.config[keys.FILE_NAME]
        key = self.config[keys.STREAM_KEY]
        rate = context.get(Constants.Keys.SAMPLING_RATE)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        path = _free_path(path)
        self._file = module.File(path, "w")

        meta = recording_meta.collect({PORT_IN: context}, self._marked, rate)
        meta[RAW_META_KEY] = self._raw_meta(context, rate)
        self._file.create_dataset(
            recording_meta.META_DATASET, data=json.dumps(meta, sort_keys=True)
        )

        self._entry = {
            "key": key,
            "file": os.path.basename(path),
            "source_class": self.config.get(
                self.Configuration.OptionalKeys.SOURCE_CLASS
            ),
            "node_name": self._node_name,
            "node_id": self._node_id,
            "channel_count": int(channels.channel_count(context)),
            "sampling_rate": None if rate is None else float(rate),
            "timing": context.get(IPort.Configuration.Keys.TIMING),
            "execution_mode": context.get(Constants.Keys.EXECUTION_MODE),
            "samples": 0,
            "blocks": 0,
        }
        self._run.register(dict(self._entry))

    def _raw_meta(self, context: dict, rate) -> dict:
        """What a replay needs and an ordinary recording does not.

        Args:
            context: The input port context.
            rate: Sampling rate, or None for a sparse stream.

        Returns:
            The sub-document stored under :data:`RAW_META_KEY`.
        """
        candidacy = getattr(self, "_candidacy", None)
        return {
            "format": MANIFEST_VERSION,
            # The absolute origin, without which a replayed stamp is
            # displaced by an arbitrary constant. See
            # RawRun.origin_units.
            "origin_units": self._run.origin_units(),
            "timing": context.get(IPort.Configuration.Keys.TIMING),
            "channel_count": int(channels.channel_count(context)),
            "sampling_rate": None if rate is None else float(rate),
            "frame_size": context.get(Constants.Keys.FRAME_SIZE),
            "execution_mode": context.get(Constants.Keys.EXECUTION_MODE),
            "timestamp_columns": list(self._time_columns),
            "master_candidacy": (
                None
                if candidacy is None
                else [float(candidacy[0]), int(candidacy[1])]
            ),
        }

    def attach_candidacy(self, candidacy) -> None:
        """Record the core's claim to the master timeline.

        Set by :func:`source_stage` from the core it just built, because
        the tap has no other way to see it and a replay that does not
        reproduce it lets the election pick a different winner.

        Args:
            candidacy: ``(rate, priority)``, or None.
        """
        self._candidacy = candidacy

    def _write_worker(self) -> None:
        """Drain the queue onto the disk until stopped and empty."""
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                block, stamp = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._write(block, stamp)
            except Exception as error:  # pragma: no cover - disk failure
                self.log(
                    f"the raw recording of "
                    f"'{self.config[self.Configuration.Keys.STREAM_KEY]}' "
                    f"failed: {error}",
                    type=Constants.LogTypes.ERROR,
                )
                return

    def _write(self, block: np.ndarray, stamp: float) -> None:
        """Append one frame and its block-table row.

        Args:
            block: The frame, shape ``(samples, channels)``.
            stamp: Emission time, seconds from the run origin.
        """
        if self._file is None:
            return
        n_samples, n_channels = block.shape

        if self._data_ds is None:
            self._data_ds = self._file.create_dataset(
                "data",
                shape=(n_channels + 1, 0),
                maxshape=(n_channels + 1, None),
                dtype=np.float64,
                chunks=(n_channels + 1, max(1, n_samples)),
            )
            self._blocks_ds = self._file.create_dataset(
                BLOCKS_DATASET,
                shape=(0, 4),
                maxshape=(None, 4),
                dtype=np.float64,
            )

        # Row 0 is the frame's emission time, repeated. The synthetic
        # ``n / fs`` grid a FileWriter builds would be a second opinion
        # about timing this file already states exactly, and a sparse
        # stream has no rate to build one from.
        times = np.full((n_samples, 1), float(stamp), dtype=np.float64)
        combined = np.hstack((times, block.astype(np.float64))).T

        end = self._samples + n_samples
        self._data_ds.resize((n_channels + 1, end))
        self._data_ds[:, self._samples : end] = combined

        last_stamp = stamp
        if self._time_columns:
            column = self._time_columns[0]
            if column < n_channels:
                last_stamp = self._recorded_stamp(
                    float(block[-1, column]), stamp
                )

        self._blocks_ds.resize((self._blocks + 1, 4))
        self._blocks_ds[self._blocks, :] = (
            float(self._samples),
            float(n_samples),
            float(stamp),
            float(last_stamp),
        )

        self._samples = end
        self._blocks += 1
        self._file.flush()

    def _recorded_stamp(self, wrapped: float, emitted: float) -> float:
        """Unwrap one in-band stamp into run-relative seconds.

        The stamp says when the sample was *acquired*, which precedes
        emission by however long the source buffered it. Anchoring the
        unwrap on the emission time is what makes that lag recoverable:
        both are on one clock, and the true instant is the candidate
        nearest to it.

        Args:
            wrapped: The value in the timestamp channel.
            emitted: Emission time, seconds from the run origin.

        Returns:
            Acquisition time, seconds from the run origin.
        """
        absolute = channels.unwrap_time(wrapped, emitted + self._run.origin())
        return absolute - self._run.origin()

    def _close(self) -> None:
        """Finalise the metadata, the manifest entry and the file."""
        if self._file is None:
            return
        from ...sinks.base import recording_meta

        try:
            node = self._file.get(recording_meta.META_DATASET)
            if node is not None:
                meta = recording_meta.parse(node.asstr()[()])
                meta["sample_count"] = int(self._samples)
                meta[RAW_META_KEY]["blocks"] = int(self._blocks)
                meta[RAW_META_KEY]["dropped"] = int(self._dropped)
                del self._file[recording_meta.META_DATASET]
                self._file.create_dataset(
                    recording_meta.META_DATASET,
                    data=json.dumps(meta, sort_keys=True),
                )
        finally:
            self._file.close()
            self._file = None
            self._data_ds = None
            self._blocks_ds = None

        self._entry["samples"] = int(self._samples)
        self._entry["blocks"] = int(self._blocks)
        self._entry["dropped"] = int(self._dropped)
        self._run.register(dict(self._entry))


# -- the replay ----------------------------------------------------------


def read_manifest(load_from: str) -> dict:
    """Read a run manifest.

    Args:
        load_from: The run directory, or the manifest file itself.

    Returns:
        The parsed manifest.

    Raises:
        FileNotFoundError: If there is no manifest there.
        ValueError: If it declares a version this loader does not know.
    """
    path = load_from
    if os.path.isdir(path):
        path = os.path.join(path, MANIFEST_NAME)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"load_from={load_from!r} names no raw run: expected "
            f"'{MANIFEST_NAME}' there. A run directory is what save_as "
            f"produces, and it is the directory that is loaded, not one "
            f"of the files inside it."
        )
    with open(path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    version = str(manifest.get("format", ""))
    if not version.startswith(MANIFEST_VERSION):
        raise ValueError(
            f"'{path}' is a raw run of version {version!r}, which this "
            f"build does not recognise; it understands "
            f"'{MANIFEST_VERSION}.x'."
        )
    manifest["_directory"] = os.path.dirname(os.path.abspath(path))
    return manifest


def resolve_stream(manifest: dict, key: str, source_class: str) -> dict:
    """The manifest entry for one chain, or a refusal naming the problem.

    Matched on the key, and then cross-checked on the class. Never
    positionally: pairing the *n*-th source of this graph with the
    *n*-th stream of the recording is what silently crosses two streams
    when a node has been inserted, and a crossed stream looks like
    plausible data.

    Args:
        manifest: The parsed manifest.
        key: The stream key this chain derived.
        source_class: Public class name of the chain.

    Returns:
        The entry, with ``path`` filled in.

    Raises:
        ValueError: If no stream carries that key, or if the stream that
            does was recorded from a different class.
    """
    streams = {entry["key"]: entry for entry in manifest.get("streams", [])}
    entry = streams.get(key)
    if entry is None:
        available = ", ".join(sorted(streams)) or "nothing"
        raise ValueError(
            f"this run has no raw stream '{key}' to replay; it holds "
            f"{available}. The key is the node's name where it has one "
            f"and <Class>_<ordinal> otherwise, so a graph that gained "
            f"or lost a source since the recording no longer lines up. "
            f"Name the nodes and record again, or load a matching run."
        )
    recorded = entry.get("source_class")
    if recorded and recorded != source_class:
        raise ValueError(
            f"raw stream '{key}' was recorded from {recorded} and this "
            f"graph asks for it as {source_class}. Replaying it would "
            f"feed one device's stream through another's chain."
        )
    entry = dict(entry)
    entry["path"] = os.path.join(
        manifest.get("_directory", "."), entry["file"]
    )
    return entry


class _RawReplayCore(Source):
    """Re-emits a recorded raw stream as if the device were producing it.

    One class for every source family, because at the tap point they are
    all the same thing: frames of ``Constants.DATA_TYPE`` with roles
    attached. That is the largest simplification this feature offers --
    replay needs no per-device knowledge at all, and a `Keyboard` stream
    replays on a machine with no display.

    Not a ``RecordingReader``, which is the class this looks like it
    should have been. That paces one cycle per *sample* off a
    declared rate, which cannot
    express a `Marker`: one row when something happens, on an ASYNC port,
    with no sampling rate in the context at all -- and ``Sync`` branches
    on exactly that to decide whether to count a stream or place it.
    """

    #: Paced to the host clock and re-stamped in its base, so this is a
    #: live stream in every sense the pipeline checks. Recorded data
    #: whose *positions* advance through a file is the other thing, and
    #: that is what RecordingReader declares.
    TIME_BASE = Constants.TimeBase.WALL_CLOCK

    #: A frame here is a recorded block, not a slice of a live stream.
    DERIVE_FRAME_SIZE: bool = False

    class Configuration(Source.Configuration):
        """Configuration class for _RawReplayCore parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for the replay core."""

            #: Path of the HDF5 file being replayed
            FILE_NAME = "file_name"

        class OptionalKeys(Source.Configuration.OptionalKeys):
            """Optional configuration keys for the replay core.

            ``sampling_rate`` is optional and not required, and that is
            the whole point: a sparse stream has none, and publishing a
            zero would make ``Sync`` count it as a continuous stream
            instead of placing its events on the master's grid.
            """

            #: Rate to publish, or None for a sparse stream
            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE

    def __init__(
        self,
        file_name: str,
        timing: str = Constants.Timing.SYNC,
        speed: float = 1.0,
        **kwargs,
    ):
        """Initialize the replay core.

        Args:
            file_name: Path of the HDF5 file written by a tap.
            timing: Timing of the output port, from the recording.
            speed: Replay speed relative to the recording. One is real
                time; zero emits as fast as the machine allows, which is
                for tests rather than for reproducing a run.
            **kwargs: Additional arguments for the parent source.

        Raises:
            ValueError: If the file carries no raw block table, i.e. it
                is an ordinary recording rather than a raw stream.
        """
        meta, blocks = _read_raw_file(file_name)
        raw = meta[RAW_META_KEY]

        self._blocks = blocks
        self._raw = raw
        self._meta = meta
        self._speed = float(speed)
        self._timing = timing
        self._position = 0
        self._exhausted = False
        self._reported_end = False
        self._file = None
        self._data_ds = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._time_columns = list(raw.get("timestamp_columns") or [])
        candidacy = raw.get("master_candidacy")
        self._candidacy = (
            None if not candidacy else (float(candidacy[0]), int(candidacy[1]))
        )
        #: Where the recording's clock was, in the units a stamp column
        #: is written in. The stored stamps are absolute readings of
        #: *that* clock, so moving them onto this run's clock is a
        #: subtraction as well as an addition. See
        #: RawRun.origin_units for what happens without this term.
        if raw.get("origin_units") is None:
            raise ValueError(
                f"'{file_name}' carries no recording origin, so its "
                f"stamps cannot be moved onto this run's clock: every "
                f"one of them would be displaced by an arbitrary "
                f"constant and the run would look correct. Re-record "
                f"it with this version."
            )
        self._recorded_origin_units = int(raw["origin_units"])

        rate = raw.get("sampling_rate")
        mode = raw.get("execution_mode") or Constants.ExecutionMode.REALTIME
        self.EXECUTION_MODE = mode

        frame_size = int(blocks[0][1]) if len(blocks) else 1
        if mode == Constants.ExecutionMode.BATCH:
            frame_size = int(sum(int(row[1]) for row in blocks)) or 1

        kwargs.setdefault("output_ports", [OPort.Configuration(timing=timing)])
        kwargs.setdefault("channel_count", int(raw["channel_count"]))
        kwargs.setdefault("frame_size", frame_size)
        Source.__init__(
            self,
            file_name=file_name,
            sampling_rate=None if rate is None else float(rate),
            **kwargs,
        )

    # -- description -----------------------------------------------------

    def master_candidacy(self):
        """The claim the recorded source made.

        Reproduced rather than re-derived, because the election has to
        settle the same way it did live: a different winner aligns every
        stream to a different grid and says nothing about it.

        Returns:
            ``(rate, priority)``, or None if the recorded source made no
            claim -- which is what a sparse source does.
        """
        return self._candidacy

    @property
    def sample_count(self) -> int:
        """Number of samples in the recording."""
        return int(self._meta.get("sample_count", 0)) or int(
            sum(int(row[1]) for row in self._blocks)
        )

    @property
    def is_exhausted(self) -> bool:
        """Whether every recorded block has been emitted."""
        return self._exhausted

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Publish the recorded stream's own description.

        Everything the live source put in its context has to come back,
        or the graph behind this node is set up differently than it was
        during the recording: roles decide what ``Sync`` consumes, the
        presence of a rate decides whether a stream is counted or
        placed, and the labels are what a downstream writer records.

        Args:
            data: Initial data arrays.
            port_context_in: Input port contexts (empty for a source).

        Returns:
            The recorded port context.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_out[PORT_OUT]

        rate = self.config.get(Constants.Keys.SAMPLING_RATE)
        if rate is not None:
            context[Constants.Keys.SAMPLING_RATE] = float(rate)

        roles = self._meta.get("channel_roles")
        labels = self._meta.get("channel_labels")
        if roles is not None or labels is not None:
            if roles is None:
                roles = [Constants.ChannelRoles.SIGNAL] * len(labels)
            context.update(
                channels.describe(
                    list(roles),
                    None if labels is None else list(labels),
                    self._meta.get("montage_system"),
                )
            )

        for key, name in (
            (Constants.Keys.DEVICE_SERIAL, "device_serial"),
            (Constants.Keys.CHANNEL_UNITS, "channel_units"),
            (Constants.Keys.START_TIME, "start_time"),
            (Constants.Keys.SAMPLING_RATE_EXACT, "sampling_rate_exact"),
        ):
            value = self._meta.get(name)
            if value is not None:
                context[key] = value

        context[OPort.Configuration.Keys.TIMING] = self._timing
        self._position = 0
        self._exhausted = False
        self._reported_end = False
        return port_context_out

    # -- driving ---------------------------------------------------------

    def start(self):
        """Open the recording and start the scheduler.

        A batch run has no scheduler: the driver calls ``cycle()`` once
        and the whole recording leaves in that call, exactly as a batch
        reader does.
        """
        module = _h5py()
        path = self.config[self.Configuration.Keys.FILE_NAME]
        self._file = module.File(path, "r")
        # A stream that never emitted has metadata and nothing else:
        # both datasets are created on the first frame, so a Marker
        # nobody pressed leaves a file with no ``data`` in it. That is a
        # perfectly ordinary recording of a perfectly ordinary session,
        # and reading the dataset unconditionally made the whole replay
        # fail to start -- every other stream in the run included.
        self._data_ds = self._file.get("data") if self._blocks else None
        self._position = 0
        self._exhausted = False

        if self.EXECUTION_MODE == Constants.ExecutionMode.BATCH:
            Source.start(self)
            return

        Source.start(self)
        if self._timing == Constants.Timing.ASYNC:
            # The priming cycle an EventSource sends at start, and for
            # the same reason: it announces the sparse port to whatever
            # places it before any event has to be placed. Without it
            # the *first* event of a replayed stream landed on the
            # newest row instead of its own -- measured at row 2 against
            # a computed position of 90.5877, while every later event
            # was exact to thirteen digits. Reproducing a live source
            # means reproducing what it does at start, not only what it
            # emits.
            self.cycle({PORT_OUT: None})
        if not self._running:
            self._running = True
            self._thread = threading.Thread(target=self._schedule, daemon=True)
            self._thread.start()

    def stop(self):
        """Stop the scheduler and close the recording."""
        Source.stop(self)
        if self._running:
            self._running = False
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None
            self._data_ds = None

    def _schedule(self) -> None:
        """Emit each recorded block at its recorded offset, or report why not.

        The offsets are relative to the run's origin and the origin is
        shared by every stream in the process, so the spacing *between*
        streams is reproduced as well as the spacing within one. That is
        the property the whole design turns on; see :class:`ReplayClock`.

        Wrapped, because a thread that raises here is invisible: Python
        prints the traceback to stderr and the pipeline goes on
        reporting a healthy graph that happens to carry no data.
        Measured during development -- an AttributeError in the shift
        arithmetic produced a replay of exactly nothing, with the
        pipeline reporting no failure at all.
        """
        try:
            self._run_schedule()
        except Exception as error:
            self.log(
                f"the replay of "
                f"'{self.config[self.Configuration.Keys.FILE_NAME]}' "
                f"stopped after {self._position} block(s): {error}",
                type=Constants.LogTypes.ERROR,
            )

    def _run_schedule(self) -> None:
        """Emit each recorded block at its recorded offset."""
        if not self._blocks:
            self._mark_exhausted()
            return
        origin = ReplayClock.origin()
        shift = self._stamp_shift()
        speed = self._speed

        for index in range(len(self._blocks)):
            if not self._running:
                return
            offset, length, t_emit, _ = self._blocks[index]
            if speed:
                due = origin + float(t_emit) / speed
                delay = due - channels.stamp_clock()
                if delay > 0:
                    # Woken in slices so a stop() during a long gap does
                    # not have to wait the gap out.
                    deadline = time.monotonic() + delay
                    while self._running:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        time.sleep(min(remaining, 0.05))
            if not self._running:
                return
            block = self._read_block(int(offset), int(length))
            shift_stamps(block, self._time_columns, shift)
            self._position = index + 1
            self.cycle({PORT_OUT: block})

        self._mark_exhausted()

    def _read_block(self, offset: int, length: int) -> np.ndarray:
        """Read one recorded frame.

        Read on demand rather than loading the file, which is what makes
        a raw amplifier dump replayable at all: a g.HIamp hour is 17.7 GB
        and ``RecordingReader``'s load-everything would need it resident.

        Args:
            offset: First sample of the block.
            length: Number of samples.

        Returns:
            The frame, shape ``(length, channels)``, in the pipeline's
            data type. Row 0 of the stored dataset is the emission time
            and is dropped: it is provenance, not a channel.
        """
        stored = self._data_ds[1:, offset : offset + length]
        return np.ascontiguousarray(
            np.asarray(stored).T, dtype=Constants.DATA_TYPE
        )

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Return the frame the scheduler handed over, or the whole record.

        Args:
            data: The frame, in realtime mode. Ignored in batch mode,
                where the driver cycles this node with nothing.

        Returns:
            The frame, or None once the recording is exhausted.
        """
        if self.EXECUTION_MODE == Constants.ExecutionMode.BATCH:
            if self._exhausted:
                return None
            blocks = [
                self._read_block(int(row[0]), int(row[1]))
                for row in self._blocks
            ]
            self._mark_exhausted()
            if not blocks:
                return None
            whole = np.vstack(blocks)
            shift_stamps(whole, self._time_columns, self._stamp_shift())
            return {PORT_OUT: whole}
        return data

    def _stamp_shift(self) -> int:
        """Whole 100 us units to add to every recorded stamp.

        The difference between the two runs' origins, under the stamp
        column's own modulus. Both terms matter: the recording's origin
        is what the stored values are relative to, and this run's is
        where they have to end up, which is close enough to ``now`` that
        ``unwrap_time`` resolves them without a candidate to choose
        between.

        Returns:
            The shift, already reduced modulo ``TIME_MODULUS``.
        """
        return (
            ReplayClock.origin_units() - self._recorded_origin_units
        ) % channels.TIME_MODULUS

    def _mark_exhausted(self) -> None:
        """Say once that the recording has run out.

        Going quiet rather than stopping the pipeline: a device that
        stops sending does not take the application down with it, and a
        Qt app whose pipeline stopped itself would leave a live window
        over a dead graph. ``is_exhausted`` is what an application reads
        to decide otherwise.
        """
        if self._reported_end:
            return
        self._reported_end = True
        self._exhausted = True
        name = self.config[self.Configuration.Keys.FILE_NAME]
        if not self._blocks:
            self.log(
                f"'{name}' recorded no frames at all, so this stream "
                f"stays silent for the whole replay. That is what a "
                f"source which never fired looks like.",
                type=Constants.LogTypes.INFO,
            )
            return
        self.log(
            f"replayed the whole of '{name}' "
            f"({self._samples_emitted()} sample(s) in "
            f"{len(self._blocks)} block(s)); this stream is now silent.",
            type=Constants.LogTypes.INFO,
        )

    def _samples_emitted(self) -> int:
        """Total samples in the recording, for the closing message."""
        return int(sum(int(row[1]) for row in self._blocks))


def _read_raw_file(file_name: str) -> tuple:
    """Read a raw stream's metadata and block table.

    Args:
        file_name: Path of the HDF5 file.

    Returns:
        A ``(meta, blocks)`` pair. ``blocks`` is a list of
        ``(offset, length, t_emit, t_stamp_last)`` tuples.

    Raises:
        ValueError: If the file is not a raw stream.
    """
    from ...sinks.base import recording_meta

    module = _h5py()
    with module.File(file_name, "r") as handle:
        node = handle.get(recording_meta.META_DATASET)
        if node is None:
            raise ValueError(
                f"'{file_name}' carries no g.Pype recording metadata, so "
                f"it cannot be replayed as a raw stream."
            )
        meta = recording_meta.parse(node.asstr()[()])
        if RAW_META_KEY not in meta:
            raise ValueError(
                f"'{file_name}' is an ordinary g.Pype recording, not a "
                f"raw stream: it has no block table, so the frame "
                f"boundaries and arrival times a replay needs are not "
                f"in it. Use HDF5Reader for such a file."
            )
        table = handle.get(BLOCKS_DATASET)
        blocks = (
            [tuple(float(v) for v in row) for row in np.asarray(table[()])]
            if table is not None
            else []
        )
    return meta, blocks


# -- the seam every chain calls -----------------------------------------


def source_stage(
    chain,
    factory: Callable[[], object],
    timing: str = Constants.Timing.SYNC,
) -> list:
    """The nodes that stand where a source chain's core would.

    Four answers, in the order they are decided:

    * nothing, under ``SERVER`` residency -- the core lives on the edge,
      and so does anything recording or replaying it;
    * a replay core, under ``load_from``;
    * the core and a tap behind it, under ``save_as``;
    * the core alone.

    Args:
        chain: The chain being built, for its key and its identity.
        factory: Builds the core. A callable rather than a class and its
            parameters, because a core's constructor may import a driver
            -- ``_KeyboardCore`` imports pynput -- and a replayed or
            server-side chain must not.
        timing: Timing of the stream. Sparse sources pass ASYNC, for the
            reason they already pass it to their ``Link`` and ``Sync``.

    Returns:
        The node-list fragment, ready to be extended into the chain's
        internal nodes.
    """
    config = LaunchConfig.get()
    if config.residency == Constants.Residency.SERVER:
        return []

    from ....common._private.naming import public_name

    # Nothing below this line runs with the feature off, and that
    # matters: stream_key consumes a process-global ordinal, so keying
    # unconditionally let a feature-off graph advance the counter. A
    # host that then switched recording on filed its streams as
    # Generator_3 and Generator_4, and the replay -- whose counter
    # starts clean -- asked for Generator_0 and was refused.
    if not config.save_as and not config.load_from:
        return [factory()]

    key = stream_key(chain)
    source_class = public_name(type(chain).__name__)

    if config.load_from:
        manifest = read_manifest(config.load_from)
        entry = resolve_stream(manifest, key, source_class)
        return [
            _RawReplayCore(
                file_name=entry["path"],
                timing=timing,
                name=f"{key}_replay",
            )
        ]

    core = factory()
    run = RawRun.current(config.save_as)
    # Refused here, while the graph is still being built, rather than
    # left to produce a run whose manifest describes fewer streams than
    # it holds. See RawRun.claim.
    claim_stream_key(key, f"{source_class} '{key}'")
    declare = getattr(core, "master_candidacy", None)
    tap = _RawTap(
        file_name=run.path_for(key),
        stream_key=key,
        run=run,
        source_class=source_class,
        # From the places these are actually answerable at this moment,
        # which is not the chain's own attributes: create_internal_nodes
        # runs before create_config, so `chain.name` is not set yet --
        # the ordering stream_key documents and works around, which
        # these two fields did not get. `chain.id` is worse: a chain has
        # no such attribute at any point, so that read could only ever
        # return None. Both fields were therefore null in every manifest
        # of every run, while claiming to be what a loader would check a
        # changed graph against. Found by three of the test authors
        # independently.
        node_name=(getattr(chain, "_core_params", None) or {}).get("name"),
        node_id=getattr(chain, "_link_stream_id", None),
        timing=timing,
        name=f"{key}_tap",
    )
    tap.attach_candidacy(declare() if callable(declare) else None)
    return [core, tap]


def is_replaying() -> bool:
    """Whether this process is replaying a raw run.

    Read by a source that offers a live API of its own -- ``Marker.emit``
    -- so it can decline rather than emit an event beside the recorded
    one it is meant to be reproducing.

    Returns:
        True when ``load_from`` is configured.
    """
    return bool(LaunchConfig.get().load_from)
