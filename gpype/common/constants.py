import ioiocore as ioc
import numpy as np


class Constants(ioc.Constants):
    """Application-wide constants for g.Pype BCI framework.

    Extends ioiocore Constants with g.Pype-specific constants,
    data types, and configuration keys for signal processing.
    """

    #: Default data type for numerical operations in the pipeline
    DATA_TYPE = np.float32

    #: Special value indicating inherited timing or configuration
    INHERITED = -1

    class Keys(ioc.Constants.Keys):
        """Configuration key constants for pipeline components.

        Standard key names for configuration dictionaries used
        throughout the g.Pype framework.
        """

        #: Sampling rate in Hz (samples per second)
        SAMPLING_RATE: str = "sampling_rate"

        #: Number of data channels in the signal
        CHANNEL_COUNT: str = "channel_count"

        #: Number of samples processed per frame
        FRAME_SIZE: str = "frame_size"

        #: Frame rate in Hz (frames per second, optional)
        FRAME_RATE: str = "frame_rate"

        #: Factor by which to reduce the sampling rate
        DECIMATION_FACTOR: str = "decimation_factor"

        #: Per-channel semantic roles, one entry per channel. Absent
        #: means every channel is a signal channel.
        CHANNEL_ROLES: str = "channel_roles"

        #: Per-channel labels, one entry per channel. Absent means
        #: channels are unnamed and are referred to positionally.
        CHANNEL_LABELS: str = "channel_labels"

        #: Name of the electrode system the labels come from, if any.
        MONTAGE_SYSTEM: str = "montage_system"

        #: Which edge builds a world-facing node, matched against
        #: ``LaunchConfig.edge_id``. A node configuration key rather than
        #: a context key; absent or None means every edge.
        EDGE_ID: str = "edge_id"

        #: True on the stream that defines the master timeline.
        MASTER_TIMELINE: str = "master_timeline"

        #: Priority the master won its election with, from
        #: MasterPriority. Travels beside MASTER_TIMELINE so that a
        #: process receiving the stream over a Link can re-run the same
        #: election locally on the same rung, rather than guessing one.
        MASTER_PRIORITY: str = "master_priority"

        #: True on a stream whose master was elected in *another*
        #: process, i.e. one that crossed a Link.
        #:
        #: MASTER_TIMELINE cannot say this. It means "a node upstream of
        #: you has already claimed", which is what stops a second Sync
        #: claiming a second time -- true in one process, false across a
        #: Link, where the claimant cannot reach this process's timeline
        #: at all. Forwarding it leaves nobody claiming; dropping it
        #: leaves nobody claiming either, unless the receiver is told to
        #: claim in the absent source's place. That is what this says.
        REMOTE_MASTER: str = "remote_master"

        #: True on a stream whose source will not feed the rate
        #: estimator, so whichever node is master here must do it.
        #:
        #: Sync's numbered path publishes the position and leaves the
        #: estimator alone, because the one source that numbers its
        #: samples in band -- an amplifier -- observes from its own
        #: acquisition callback, **once per notification**. Sync sees
        #: frames, and on a BCI Core those differ four-fold, which is
        #: what makes the source's observation the better one. Not,
        #: as this used to say, because its arrival time has "not been
        #: through a queue": that time is carried in band precisely so
        #: it survives the queue, and measurement put the two within
        #: the 100 us stamp quantum of each other. See
        #: ``Sync._observe_samples`` and ``D-TIME-30``.
        #:
        #: A replay core is a source that does not observe at all:
        #: ``load_from`` reproduces the recorded candidacy, the frames
        #: and their boundaries, but there is no acquisition callback
        #: behind it.
        #:
        #: The consequence is silent and was measured on a BCI Core
        #: recording: the relation is never solved, the numbered stream
        #: replays perfectly anyway -- it carries its own positions --
        #: and every sparse stream beside it is held in
        #: ``Sync._pending`` for the whole run. Not discarded, so not
        #: counted and not logged. 52 keystrokes in, 0 out.
        #:
        #: Distinct from REMOTE_MASTER, which says the master is real
        #: but unreachable. Here the master is in this process and
        #: perfectly reachable; it simply does not observe.
        UNOBSERVED_SOURCE: str = "unobserved_source"

        #: The clock-sync health of the edge that sent this stream:
        #: ``{offset, uncertainty, converged, epoch, edge}``, as the
        #: edge's start converged on it (D-TIME-69). Set by the edge's
        #: sending Link, read by the server's Sync; absent from a stream
        #: whose sender does not sync, such as a 4.0 edge.
        CLOCK_SYNC: str = "clock_sync"

        #: What the server's Sync found wrong with that health, for a
        #: recording to carry as a note: absent while nothing is wrong.
        CLOCK_NOTE: str = "clock_note"

        #: Serial number of the device that produced this stream.
        #:
        #: Provenance, not configuration: it answers "which physical
        #: amplifier is this recording from", which a configuration
        #: cannot. A BLE source discovers its device during start(), by
        #: which point its Configuration already exists and is read-only,
        #: so the serial has nowhere to go there -- but a port context is
        #: built at setup(), after the handle is open, which is exactly
        #: late enough.
        DEVICE_SERIAL: str = "device_serial"

        #: Execution mode of the run this stream belongs to, from
        #: Constants.ExecutionMode. Travels in the context so a node can
        #: declare its output shape accordingly in setup(), which is the
        #: one place the mode legitimately has to be visible.
        EXECUTION_MODE: str = "execution_mode"

        #: Total samples the source will emit, when it knows. A reader
        #: does; an amplifier never will.
        SAMPLE_COUNT: str = "sample_count"

        #: Physical unit per channel, one entry per channel. Absent
        #: means the source does not record units -- which is not the
        #: same as dimensionless, and is why this is absent rather than
        #: defaulted.
        CHANNEL_UNITS: str = "channel_units"

        #: Per-channel calibration, GTC's CHAN field set (D-BATCH-43):
        #: one entry per channel, None where a channel's own value is
        #: not recorded.
        #:
        #: Gain and offset are exact -- ``[numerator, denominator]``,
        #: the same shape as SAMPLING_RATE_EXACT -- never a float, so a
        #: stored calibration round-trips exactly rather than as a
        #: decimal a second conversion cannot undo.
        CHANNEL_GAINS: str = "channel_gains"
        CHANNEL_OFFSETS: str = "channel_offsets"
        #: ``[low, high]`` in the channel's own unit, the range a value
        #: was clipped to before storage.
        CHANNEL_CLIPPING: str = "channel_clipping"
        #: Hardware filters already applied, as a list of descriptions
        #: per channel (empty where none were). Free provenance a
        #: vendor's driver can supply and nothing downstream can
        #: reconstruct once the data has passed through them.
        CHANNEL_FILTERS: str = "channel_filters"

        #: The sampling rate as ``[numerator, denominator]``, when the
        #: source knows it exactly. SAMPLING_RATE stays the float the
        #: engine computes with; this is what a round trip restores, so
        #: 512000/1001 does not become a decimal on the way through.
        SAMPLING_RATE_EXACT: str = "sampling_rate_exact"

        #: Absolute time of the first sample, ISO 8601 with offset. A
        #: string rather than a datetime because a port context crosses
        #: a Link as JSON.
        START_TIME: str = "start_time"

        #: How much the source vouches for the data: whether the file
        #: was sealed, whether anything was recovered. Carried so a
        #: result derived from an unsealed recording can say so.
        TRUST: str = "trust"

        #: Recording interruptions, as ``[[first_missing, n_missing,
        #: reason], ...]`` on the sample grid. A dense block across a
        #: dropout is silently wrong, so a consumer that cuts epochs has
        #: to be able to see them.
        GAPS: str = "gaps"

        #: Events, as ``[[sample, duration, channel, label], ...]``,
        #: with sample relative to the first sample of the block. A
        #: trigger channel is how markers arrive; this is how they are
        #: reasoned about afterwards.
        MARKERS: str = "markers"

        #: Per-trial provenance of a ``(time, channel, trial)`` block,
        #: as ``[[source_sample, label], ...]``, one entry per trial in
        #: the trial axis's order. ``source_sample`` is the marker's
        #: sample in the recording the trial was cut from; ``label`` is
        #: its condition. Published by :class:`~gpype.Epochs`, which
        #: also drops MARKERS from the same context -- the time axis no
        #: longer indexes the source recording, so a marker position
        #: would name a meaningless row.
        TRIALS: str = "trials"

        #: The file a reader read, as ``{name, size, mtime, sha256}``:
        #: the basename, the size in bytes, the modification time as ISO
        #: 8601 in UTC, and the hex SHA-256 of the whole file. The digest
        #: is the identity; a copy keeps it under another name and mtime.
        #: Absent where the reader opened no file: on a server, under
        #: ``load_from``, on an edge it is not assigned to, and where the
        #: path names no regular file.
        INPUT: str = "input"

        #: This run's provenance record: the document (or, when the
        #: pipeline holds a function node, the names of those instead), its
        #: content hash, package versions, execution mode, any fitted
        #: artifacts and what trained them, and this stream's own
        #: ``INPUT`` nested under ``"input"``. Stamped once by
        #: ``Pipeline.start()`` (``common._private.provenance``) and
        #: merged into every source's own output context, so it crosses
        #: a Link as JSON like everything else here (D-BATCH-14).
        #: ``Result.provenance`` reads it.
        PROVENANCE: str = "provenance"

    class TimeBase:
        """What a source's sample positions advance against.

        Two sources with different time bases cannot be merged: one moves
        with the host clock while the other moves through a recording, so
        there is no single instant that a given sample of each refers to.
        Declared rather than detected, because this is a correctness
        check, not a security boundary -- a source that misdeclares gets a
        meaningless pipeline, which is its own author's problem.
        """

        #: Advances with the host clock. Every live source, including a
        #: synthetic one such as Generator.
        WALL_CLOCK: str = "wall_clock"
        #: Advances through recorded data at whatever speed the reader is
        #: asked for, including as fast as the file can be read.
        RECORDED: str = "recorded"

    class ExecutionMode:
        """How a pipeline is driven.

        A pipeline is wholly one or the other -- there are no mixed
        pipelines, and therefore no barrier inside a pipeline and no
        end-of-stream token in the API. Declared by the *source*, not
        derived from the time base: a recording paced to the wall clock
        is a legitimate realtime pipeline, which is what makes those two
        properties independent.
        """

        #: Frames arrive as the clock produces them. Every live source,
        #: and a reader replaying at a pace.
        REALTIME: str = "realtime"
        #: The whole recording arrives in one cycle, already
        #: synchronised, and the run ends when it has been processed.
        BATCH: str = "batch"

    class ChannelRoles:
        """Semantic role of a channel within a port.

        Roles say *how a channel must be treated*, so that multichannel
        operations do not have to infer meaning from channel position.
        Labels say what a channel *is*; the two are independent.
        """

        #: Measured signal; what filters and spatial processing operate on
        SIGNAL: str = "signal"
        #: Position on the master timeline
        INDEX: str = "index"
        #: Event or marker channel; must never be filtered
        TRIGGER: str = "trigger"
        #: Data validity, saturation or link quality indicator
        QUALITY: str = "quality"
        #: Auxiliary sensor, e.g. accelerometer or battery
        AUXILIARY: str = "auxiliary"
        #: Host time at which the sample beside it was acquired,
        #: wrapped as channels.wrap_time describes.
        #:
        #: Consumed by Sync and never forwarded, so it is internal
        #: to the stretch between a source and its Sync. It exists
        #: because that stretch may contain a Link: reading a clock
        #: on the far side of one measures the transport, not the
        #: acquisition, and the master-timeline fit built from those
        #: readings is then a fit to the network.
        TIMESTAMP: str = "timestamp"

    class Residency:
        """Residency mode values for distributed pipeline execution."""

        #: All nodes run locally, link acts as passthrough
        STANDALONE: str = "standalone"
        #: Edge nodes run, link sends to server
        EDGE: str = "edge"
        #: Server nodes run, link receives from edge
        SERVER: str = "server"

    class Bind:
        """Which interfaces a SERVER-residency broker listens on.

        Deliberately separate from the endpoint. An endpoint's host is
        the address a *client* dials, which is not always the address
        the server binds: a container binds a wildcard and advertises
        its own address, and one string cannot carry both. Reading the
        endpoint as a bind address would therefore break exactly the
        deployment that needs the distinction.
        """

        #: Every interface, on both address families. What a LAN or
        #: container deployment needs, and what the broker has always
        #: done.
        ANY: str = "any"

        #: The loopback addresses only, so nothing off this machine can
        #: reach the data socket at all. What a single-machine or
        #: embedded deployment wants, and what keeps a phone's data
        #: socket off the Wi-Fi it happens to be on.
        LOOPBACK: str = "loopback"

    class Defaults(ioc.Constants.Defaults):

        #: Default frame size in samples
        FRAME_SIZE: int = 1
