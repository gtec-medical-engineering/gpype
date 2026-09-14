"""What a batch run hands back."""

from __future__ import annotations

import datetime
from typing import NamedTuple, Optional

import numpy as np

from .constants import Constants


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


class Gap(NamedTuple):
    """A recording interruption, on the sample grid."""

    #: First sample that is missing.
    first_missing: int
    #: How many samples are missing.
    n_missing: int
    #: Why, as the recording states it.
    reason: str


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
    what a default meant.

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
        """Sampling rate in Hz, or None if the context carries none."""
        return self._context.get(Constants.Keys.SAMPLING_RATE)

    @property
    def labels(self) -> Optional[list]:
        """Channel labels, or None when the channels are unnamed."""
        labels = self._context.get(Constants.Keys.CHANNEL_LABELS)
        return list(labels) if labels else None

    @property
    def roles(self) -> Optional[list]:
        """Per-channel roles, or None when none are declared.

        Absent means every channel is a signal channel -- the same
        reading the rest of the framework gives it.
        """
        roles = self._context.get(Constants.Keys.CHANNEL_ROLES)
        return list(roles) if roles else None

    @property
    def units(self) -> Optional[list]:
        """Physical unit per channel, or None if none are recorded.

        None means the source does not record units, which is not the
        same as dimensionless -- a distinction worth keeping, because
        unit confusion silently produces plots off by a million.
        """
        units = self._context.get(Constants.Keys.CHANNEL_UNITS)
        return list(units) if units else None

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
        """Sample times in seconds from the start, or None.

        Relative to the first sample, because a batch run has no
        absolute start time to offer yet -- that arrives with a format
        that records one.
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

    def plot(self, ax=None):
        """Plot every channel against time.

        Args:
            ax: Axes to draw on. A new figure is created if omitted.

        Returns:
            The Axes drawn on, so the caller can label or save it.

        Raises:
            ImportError: If matplotlib is not installed. It is not a
                g.Pype dependency -- the Qt scopes are for realtime, and
                a batch result should not need a widget host.
            ValueError: If the data is not two-dimensional.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover - environment
            raise ImportError(
                "Result.plot() needs matplotlib, which is not a g.Pype "
                "dependency. Install it with 'pip install matplotlib'."
            ) from e

        if self._data.ndim != 2:
            raise ValueError(
                f"plot() needs two-dimensional data; this result has "
                f"shape {self._data.shape}."
            )
        if ax is None:
            _, ax = plt.subplots()
        times = self.times
        x = times if times is not None else np.arange(self.sample_count)
        labels = self.labels or [
            f"ch{i + 1}" for i in range(self.channel_count)
        ]
        for index, label in enumerate(labels):
            ax.plot(x, self._data[:, index], label=label)
        ax.set_xlabel("time (s)" if times is not None else "sample")
        return ax

    def __len__(self) -> int:
        """Number of samples, so ``len(result)`` reads as expected."""
        return self.sample_count

    def __repr__(self) -> str:
        """One line saying what this is.

        A result that prints its shape, rate and duration is the
        difference between a REPL session that explains itself and one
        that prints an object address.
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

        name = f"'{self._name}' " if self._name else ""
        tail = f", {', '.join(extra)}" if extra else ""
        return f"Result({name}{shape}{tail})"
