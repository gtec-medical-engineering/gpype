"""What a batch run hands back."""

from __future__ import annotations

import copy
import datetime
import math
from dataclasses import dataclass
from fractions import Fraction
from typing import NamedTuple, Optional

import numpy as np

from ._private import channels as _channels
from ._private import units as _units
from .constants import Constants

#: How far the exact rate may sit from the float one and still describe
#: it. Float arithmetic on the way moves the float by a few ulp; a
#: fraction left over from before a rate change is off by a factor.
_RATE_TOLERANCE = 1e-9

#: Labels ``repr`` shows before it counts the rest.
_REPR_LABELS = 4

#: Channels ``_repr_html_`` lists before it counts the rest.
_HTML_CHANNELS = 16

#: What ``_repr_html_`` shows for a field the context does not carry.
_NOT_RECORDED = "not recorded"


def _exact(pair) -> Optional[Fraction]:
    """One CHANNEL_GAINS/CHANNEL_OFFSETS entry as a Fraction, or None.

    Args:
        pair: A ``[numerator, denominator]`` entry, or None.

    Returns:
        The value as a Fraction, or None when absent or malformed --
        never taken to be zero or one, which would be believed.
    """
    if pair is None:
        return None
    try:
        numerator, denominator = pair
        return Fraction(int(numerator), int(denominator))
    except (TypeError, ValueError, ZeroDivisionError):
        return None


class Event(NamedTuple):
    """One marker: a time and a label, plus two fields that earn it.

    ``duration`` exists because "seizure from here to here" expressed as
    two markers forces a reader to pair them by matching label text, and
    ``channel`` because "channel 7 went bad at 1200 s" has to be
    machine-readable. Both mirror the GTC marker model rather than a
    smaller one of g.Pype's own.
    """

    #: Sample index, relative to the first sample of the result.
    sample: int
    #: Length in samples; zero for a point event.
    duration: int
    #: Channel the event is about, or None for the whole recording.
    channel: Optional[int]
    #: What it says. A string; no numeric codes, no registry.
    label: str


class Trial(NamedTuple):
    """One trial's provenance, in a ``(time, channel, trial)`` result."""

    #: Sample of the marker this trial was cut around, in the source
    #: recording -- not relative to this trial's own first sample.
    sample: int
    #: The marker's label, i.e. the condition this trial belongs to.
    label: str


class Gap(NamedTuple):
    """A recording interruption, on the sample grid."""

    #: First sample that is missing.
    first_missing: int
    #: How many samples are missing.
    n_missing: int
    #: Why, as the recording states it.
    reason: str


@dataclass(frozen=True)
class Channel:
    """One channel of a result: what it is, how it is treated, its unit.

    A frozen dataclass rather than a tuple, so a field can be added
    later without breaking a caller that unpacks it -- which is how
    gain, offset, clipping and filters arrived:
    additive fields, defaulting to None or empty for a source that
    records none of GTC's CHAN field set.
    """

    #: Label, or None when the channel is unnamed.
    name: Optional[str]
    #: Role from ``Constants.ChannelRoles``. Signal when the context
    #: declares no roles; None when the roles it declares describe a
    #: different number of channels.
    role: Optional[str]
    #: Physical unit, or None when none is recorded.
    unit: Optional[str]
    #: Stored-to-physical gain, exact, or None when not recorded.
    gain: Optional[Fraction] = None
    #: Stored-to-physical offset, exact, or None when not recorded.
    offset: Optional[Fraction] = None
    #: ``(low, high)`` this channel was clipped to before storage, in
    #: its own unit, or None when not recorded.
    clipping: Optional[tuple] = None
    #: Hardware filters already applied, oldest first; empty when none
    #: were, or when the source does not record this.
    filters: tuple = ()

    @property
    def si_scale(self) -> Optional[float]:
        """Factor from ``unit`` to its SI unit, or None.

        None when the unit is not recorded, or is not one g.Pype knows,
        such as a derived unit. The known units are ``V``, ``mV``,
        ``uV`` (also written with a micro sign), ``nV``, ``g``,
        ``deg/s``, ``%``, ``count`` and ``s``.
        """
        return _units.si_scale(self.unit)


class Result:
    """One node's output from a batch run, with what describes it.

    A bare array plus a dictionary is not a deliverable: the pipeline
    computes channel labels, roles and a sampling rate, and handing back
    a tuple makes every caller rebuild that bookkeeping. So this is the
    object a run returns.

    It is a **view over the port context**, not a second metadata store.
    Anything the context gains -- units, calibration, events, gaps, trust
    state -- becomes readable here without a change to how it travels.
    Fields the context does not carry yet are **absent rather than
    faked**, so adding them later is additive and nothing has to guess
    what a default meant. The same holds for a per-channel list whose
    length is not the channel count: a node upstream changed the
    channels and not the list, so it describes other channels.

    Args:
        data: The block the node emitted, shape ``(time, channel)`` or
            ``(time, channel, trial)``.
        context: The input port context the node was set up with.
        name: Name of the node that produced it, if known.
    """

    def __init__(
        self,
        data: np.ndarray,
        context: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        self._data = data
        self._context = dict(context or {})
        self._name = name

    @property
    def data(self) -> np.ndarray:
        """The samples, ``(time, channel[, trial])``."""
        return self._data

    @property
    def context(self) -> dict:
        """The port context this was described by, as a copy."""
        return dict(self._context)

    @property
    def name(self) -> Optional[str]:
        """Name of the node that produced this, if known."""
        return self._name

    @property
    def rate(self) -> Optional[float]:
        """Sampling rate in Hz, or None if the context carries none.

        A float, which is what the engine computes with; ``rate_exact``
        is the fraction, where the source knows it.
        """
        return self._context.get(Constants.Keys.SAMPLING_RATE)

    @property
    def rate_exact(self) -> Optional[Fraction]:
        """Sampling rate as an exact fraction, or None.

        None unless the source knows the rate exactly and the fraction
        still agrees with ``rate``: one describing some other rate is
        worse than none, so a stale one is withheld.
        """
        exact = _channels.exact_rate(self._context)
        rate = self.rate
        if exact is None or not rate:
            return None
        if not math.isclose(
            float(exact), float(rate), rel_tol=_RATE_TOLERANCE
        ):
            return None
        return exact

    def _per_channel(self, key: str) -> Optional[list]:
        """A per-channel context list, or None when absent or stale."""
        values = self._context.get(key)
        if not values or len(values) != self.channel_count:
            return None
        return list(values)

    @property
    def labels(self) -> Optional[list]:
        """Channel labels, or None when the channels are unnamed."""
        return self._per_channel(Constants.Keys.CHANNEL_LABELS)

    @property
    def roles(self) -> Optional[list]:
        """Per-channel roles, or None when none describe these channels.

        No roles declared at all means every channel is a signal
        channel -- the same reading the rest of the framework gives it.
        """
        return self._per_channel(Constants.Keys.CHANNEL_ROLES)

    @property
    def units(self) -> Optional[list]:
        """Physical unit per channel, or None if none are recorded.

        None means the source does not record units, which is not the
        same as dimensionless -- a distinction worth keeping, because
        unit confusion silently produces plots off by a million.
        """
        return self._per_channel(Constants.Keys.CHANNEL_UNITS)

    @property
    def si_scale(self) -> Optional[list]:
        """Factor from each channel's unit to its SI unit, or None.

        Derived from ``units`` alone. None when no units are recorded,
        because absent is not dimensionless. An entry is None where that
        channel's unit is not recorded, or is not one g.Pype knows (see
        :attr:`Channel.si_scale`), such as a derived unit.
        """
        units = self.units
        if units is None:
            return None
        return [_units.si_scale(unit) for unit in units]

    def to_si(self) -> np.ndarray:
        """Return the samples in SI units: ``data`` times ``si_scale``.

        Volts for ``uV``, ``mV`` and ``nV``, m/s^2 for ``g``, rad/s for
        ``deg/s``, and a plain number for ``%`` and ``count``. In the
        data's own precision, which for a run is float32.

        Returns:
            A new array shaped like ``data``.

        Raises:
            ValueError: If the result records no units, or a channel's
                unit is not recorded or not known. A missing unit is
                never taken to be microvolts.
        """
        scale = self.si_scale
        if scale is None:
            raise ValueError(
                "This result records no units, so it cannot be put in SI "
                "units; a missing unit is not taken to be microvolts. "
                "Units come from the source, such as an amplifier, or "
                "from a ChannelLabeler with units=."
            )
        unknown = [
            self._describe(index)
            for index, factor in enumerate(scale)
            if factor is None
        ]
        if unknown:
            raise ValueError(
                f"Cannot put {', '.join(unknown)} in SI units: the unit is "
                f"not recorded or not known. Known units are "
                f"{', '.join(sorted(_units.CANONICAL))}. Select the "
                f"channels that have one, or read si_scale for the rest."
            )
        dtype = np.result_type(self._data.dtype, np.float32)
        factors = np.asarray(scale, dtype=dtype)
        if self._data.ndim > 1:
            shape = [1] * self._data.ndim
            shape[1] = self.channel_count
            factors = factors.reshape(shape)
        return self._data * factors

    def _describe(self, index: int) -> str:
        """Name a channel and its unit, for a message."""
        labels = self.labels
        name = f"channel {index}"
        if labels is not None:
            name += f" ({labels[index]!r})"
        return f"{name} with unit {self.units[index]!r}"

    @property
    def channels(self) -> list:
        """One :class:`Channel` per channel: what a source states about
        it -- name, role, unit, and GTC's calibration field set."""
        names, roles, units = self.labels, self.roles, self.units
        gains = self._per_channel(Constants.Keys.CHANNEL_GAINS)
        offsets = self._per_channel(Constants.Keys.CHANNEL_OFFSETS)
        clipping = self._per_channel(Constants.Keys.CHANNEL_CLIPPING)
        filters = self._per_channel(Constants.Keys.CHANNEL_FILTERS)
        declared = self._context.get(Constants.Keys.CHANNEL_ROLES)
        role = None if declared else Constants.ChannelRoles.SIGNAL
        return [
            Channel(
                name=None if names is None else str(names[index]),
                role=role if roles is None else roles[index],
                unit=None if units is None else units[index],
                gain=None if gains is None else _exact(gains[index]),
                offset=None if offsets is None else _exact(offsets[index]),
                clipping=(
                    None
                    if clipping is None or clipping[index] is None
                    else tuple(clipping[index])
                ),
                filters=(
                    ()
                    if filters is None or filters[index] is None
                    else tuple(filters[index])
                ),
            )
            for index in range(self.channel_count)
        ]

    @property
    def start_time(self) -> Optional[datetime.datetime]:
        """Absolute time of the first sample, or None.

        Parsed here rather than in the context, which carries an ISO
        string because it crosses a Link as JSON.
        """
        stamp = self._context.get(Constants.Keys.START_TIME)
        if not stamp:
            return None
        try:
            return datetime.datetime.fromisoformat(str(stamp))
        except ValueError:
            return None

    @property
    def trust(self) -> Optional[str]:
        """How much the source vouches for the data, or None.

        A result derived from an unsealed or recovered recording says so
        here, and should say so wherever it is displayed.
        """
        return self._context.get(Constants.Keys.TRUST)

    @property
    def provenance(self) -> Optional[dict]:
        """This run's provenance record, or None.

        None before ``Pipeline.start()`` has stamped one -- a context
        built by hand rather than by a running pipeline. Otherwise the
        document (unless the pipeline held a function node, in
        which case ``script_only_nodes`` names it instead), its content
        hash, package versions, execution mode, any already-fitted
        artifacts and what trained them, and this stream's own input
        under ``"input"`` -- nested under ``"input"]["derived_from"]``
        when the file it came from was itself a g.Pype recording:
        reprocessing a g.Pype output yields a lineage chain.
        A :meth:`Pipeline.fit` run adds ``fitted_document``, the
        document after the fit, with its artifacts.

        A deep copy, so editing what this returns cannot reach the
        context another :class:`Result` reading the same run would see.
        """
        record = self._context.get(Constants.Keys.PROVENANCE)
        if record is None:
            return None
        return copy.deepcopy(record)

    def document(self) -> dict:
        """The document that produced this result, ready to rebuild.

        Recovers a pipeline from its own result: pass this to
        :meth:`Pipeline.deserialize`, giving it fresh ids first
        (``common.document.with_fresh_ids``) to rebuild in the same
        process without colliding with a still-live one. The rebuilt
        run is bit-exact against this one only along the ``sosfilt``
        path -- a phase-nonlinear filter and anything
        touching a live clock are not.

        From a :meth:`Pipeline.fit` run, the document after the fit,
        carrying its artifacts, so ``run()`` on the rebuild applies them.

        Returns:
            dict: The recorded document.

        Raises:
            ValueError: If this result carries no provenance record at
                all, the run it describes held a function node
                (``Apply``, or an ``@gp.node``) -- a document could
                never rebuild one, so none was recorded;
                the message names it -- or its pipeline could not be
                saved, and the message says why.
        """
        provenance = self.provenance
        if provenance is None:
            raise ValueError(
                "this result carries no provenance record, so there is "
                "no document to recover from it."
            )
        document = provenance.get("fitted_document") or provenance.get(
            "document"
        )
        if document is not None:
            return document
        reason = provenance.get("not_serializable")
        if reason:
            raise ValueError(
                f"this run's pipeline could not be saved, so no document "
                f"was recorded for it: {reason}"
            )
        names = ", ".join(
            f"{entry.get('name')!r} ({entry.get('class')})"
            for entry in provenance.get("script_only_nodes") or []
        )
        raise ValueError(
            f"this run's pipeline held a function node, {names or '?'}, "
            f"so no document was ever recorded for it. Run "
            f"it again from the script that built it."
        )

    @property
    def gaps(self) -> list:
        """Recording interruptions inside this result.

        Empty when the source records none. A dense block across a
        dropout is silently wrong, so anything cutting epochs has to
        consult this.
        """
        return [
            Gap(int(gap[0]), int(gap[1]), str(gap[2]))
            for gap in self._context.get(Constants.Keys.GAPS, [])
        ]

    @property
    def events(self) -> list:
        """Markers inside this result, relative to its first sample."""
        found = []
        for entry in self._context.get(Constants.Keys.MARKERS, []):
            sample, duration, channel, label = entry
            found.append(
                Event(
                    int(sample),
                    int(duration),
                    None if channel is None else int(channel),
                    str(label),
                )
            )
        return found

    @property
    def trials(self) -> list:
        """Per-trial provenance of a ``(time, channel, trial)`` result.

        Empty when the context records none, which includes every
        two-dimensional result: a trial axis is what this describes.
        """
        return [
            Trial(int(entry[0]), str(entry[1]))
            for entry in self._context.get(Constants.Keys.TRIALS, [])
        ]

    @property
    def channel_count(self) -> int:
        """Number of channels."""
        return int(self._data.shape[1]) if self._data.ndim > 1 else 1

    @property
    def sample_count(self) -> int:
        """Number of samples on the time axis."""
        return int(self._data.shape[0])

    @property
    def duration(self) -> Optional[float]:
        """Length in seconds, or None without a known rate."""
        rate = self.rate
        if not rate:
            return None
        return self.sample_count / float(rate)

    @property
    def times(self) -> Optional[np.ndarray]:
        """Sample times in seconds from the first sample, or None.

        Relative, not absolute: where the source records one,
        ``start_time`` is the absolute time of sample zero.
        """
        rate = self.rate
        if not rate:
            return None
        return np.arange(self.sample_count) / float(rate)

    def to_numpy(self) -> np.ndarray:
        """Return the samples.

        Returns:
            The underlying array, not a copy -- callers that intend to
            modify it should copy first.
        """
        return self._data

    def to_dataframe(self):
        """Return the samples as a pandas DataFrame, time-indexed.

        Returns:
            A DataFrame with one column per channel.

        Raises:
            ImportError: If pandas is not installed. It is not a g.Pype
                dependency; this method exists for callers who have it.
            ValueError: If the data is not two-dimensional.
        """
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - environment
            raise ImportError(
                "Result.to_dataframe() needs pandas, which is not a "
                "g.Pype dependency. Install it with 'pip install pandas' "
                "or use Result.data for the raw array."
            ) from e

        if self._data.ndim != 2:
            raise ValueError(
                f"to_dataframe() needs two-dimensional data; this "
                f"result has shape {self._data.shape}."
            )
        columns = self.labels or [
            f"ch{i + 1}" for i in range(self.channel_count)
        ]
        return pd.DataFrame(self._data, index=self.times, columns=columns)

    def plot(self, ax=None, trial: Optional[int] = None):
        """Plot every channel against time, and mark any events.

        A ``(time, channel)`` result draws one line per channel. A
        ``(time, channel, trial)`` result draws the mean across trials
        by default; pass ``trial`` to draw one trial instead, and its
        line labels name what :attr:`trials` records for it. Either way,
        every event in :attr:`events` is drawn as a labelled vertical
        line.

        Args:
            ax: Axes to draw on. A new figure is created if omitted.
            trial: Index of the one trial to draw. None (the default)
                draws the mean over the trial axis; meaningless, and
                refused, for a two-dimensional result.

        Returns:
            The Axes drawn on, so the caller can label or save it.

        Raises:
            ImportError: If matplotlib is not installed. It is not a
                g.Pype dependency -- the Qt scopes are for realtime, and
                a batch result should not need a widget host.
            ValueError: If the data is neither two- nor
                three-dimensional, if ``trial`` is given for
                two-dimensional data, or if ``trial`` is outside the
                trial axis.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover - environment
            raise ImportError(
                "Result.plot() needs matplotlib, which is not a g.Pype "
                "dependency. Install it with 'pip install matplotlib'."
            ) from e

        if self._data.ndim not in (2, 3):
            raise ValueError(
                f"plot() needs two- or three-dimensional data; this "
                f"result has shape {self._data.shape}."
            )
        if trial is not None and self._data.ndim != 3:
            raise ValueError(
                f"trial is only meaningful for a result with a trial "
                f"axis; this result has shape {self._data.shape}."
            )

        suffix = ""
        if self._data.ndim == 3:
            n_trials = self._data.shape[2]
            if trial is None:
                series = self._data.mean(axis=2)
            else:
                if not 0 <= trial < n_trials:
                    raise ValueError(
                        f"trial={trial} is outside this result's "
                        f"{n_trials} trial(s)."
                    )
                series = self._data[:, :, trial]
                trials = self.trials
                if trial < len(trials):
                    suffix = f" (trial {trial}: {trials[trial].label!r})"
                else:
                    suffix = f" (trial {trial})"
        else:
            series = self._data

        if ax is None:
            _, ax = plt.subplots()
        times = self.times
        x = times if times is not None else np.arange(self.sample_count)
        labels = self.labels or [
            f"ch{i + 1}" for i in range(self.channel_count)
        ]
        for index, label in enumerate(labels):
            ax.plot(x, series[:, index], label=f"{label}{suffix}")
        ax.set_xlabel("time (s)" if times is not None else "sample")

        rate = self.rate
        for event in self.events:
            position = event.sample / rate if rate else event.sample
            ax.axvline(
                position, linestyle="--", color="gray", label=event.label
            )
        return ax

    def __len__(self) -> int:
        """Number of samples, so ``len(result)`` reads as expected."""
        return self.sample_count

    def __repr__(self) -> str:
        """One line saying what this is.

        A result that prints its shape, rate, duration, channels and
        trust state is the difference between a REPL session that
        explains itself and one that prints an object address.
        """
        shape = f"{self.sample_count} samples"
        if self._data.ndim > 1:
            shape += f" x {self.channel_count} ch"
        if self._data.ndim > 2:
            shape += f" x {self._data.shape[2]} trials"

        extra = []
        rate = self.rate
        if rate:
            extra.append(f"{rate:g} Hz")
        duration = self.duration
        if duration is not None:
            extra.append(f"{duration:.3g} s")
        labels = self.labels
        if labels:
            shown = [str(label) for label in labels[:_REPR_LABELS]]
            rest = len(labels) - len(shown)
            if rest:
                shown.append(f"+{rest}")
            extra.append(f"[{', '.join(shown)}]")
        trust = self.trust
        if trust:
            extra.append(f"trust={trust}")

        name = f"'{self._name}' " if self._name else ""
        tail = f", {', '.join(extra)}" if extra else ""
        return f"Result({name}{shape}{tail})"

    def _repr_html_(self) -> str:
        """A compact table for a notebook, which calls this to show one.

        The ``repr`` line, then rate, duration, trials where the data has
        a trial axis, start time, events, gaps and trust, then one row
        per channel with its name, role and unit, the first 16 of them.
        A field the context does not carry reads "not recorded".

        Returns:
            HTML, every value escaped.
        """
        import html

        def cell(value) -> str:
            if value is None:
                return _NOT_RECORDED
            return html.escape(str(value))

        rate = self.rate
        duration = self.duration
        start = self.start_time
        fields = [
            ("rate", f"{rate:g} Hz" if rate else None),
            ("duration", None if duration is None else f"{duration:.6g} s"),
            ("start time", None if start is None else start.isoformat()),
            ("events", len(self.events)),
            ("gaps", len(self.gaps)),
            ("trust", self.trust),
        ]
        if self._data.ndim > 2:
            fields.insert(2, ("trials", int(self._data.shape[2])))
        summary = "".join(
            f"<tr><th>{label}</th><td>{cell(value)}</td></tr>"
            for label, value in fields
        )

        channels = self.channels
        rows = "".join(
            f"<tr><td>{index}</td><td>{cell(ch.name)}</td>"
            f"<td>{cell(ch.role)}</td><td>{cell(ch.unit)}</td></tr>"
            for index, ch in enumerate(channels[:_HTML_CHANNELS])
        )
        rest = len(channels) - _HTML_CHANNELS
        if rest > 0:
            rows += f"<tr><td colspan='4'>+{rest} more</td></tr>"

        return (
            f"<div><code>{html.escape(repr(self))}</code>"
            f"<table>{summary}</table>"
            f"<table><tr><th>#</th><th>channel</th><th>role</th>"
            f"<th>unit</th></tr>{rows}</table></div>"
        )
