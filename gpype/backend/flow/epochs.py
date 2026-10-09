"""Epochs: cuts (time, channel, trial) blocks in a batch run."""

from __future__ import annotations

import numpy as np

from ...common._private.naming import node_label
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Epochs(IONode):
    """Cuts one epoch per marker of a condition, stacked into a trial axis.

    Batch only, like :class:`~gpype.Apply` (D-BATCH-56): a realtime
    pipeline is refused with this node named. Selects on the events
    model rather than on a trigger column (D-BATCH-13) -- every marker
    in the input context whose label equals ``condition`` becomes one
    trial, spanning ``[tmin, tmax)`` seconds around its sample.

    An epoch that straddles a recording gap is refused outright, naming
    the trial and the gap (D-BATCH-12): a dense block across a dropout
    would silently mean less than it claims. One that runs off either
    end of the recording is dropped instead, and the count is logged
    once (D-BATCH-88). An unknown condition is refused, listing the
    labels the recording actually has.

    The output has no samples in common with its input's grid, so its
    context carries no ``markers`` or ``gaps`` (D-BATCH-86); it carries
    ``Constants.Keys.TRIALS`` instead -- ``[[source_sample, label],
    ...]``, one entry per trial -- which :attr:`Result.trials` reads.
    It also publishes ``time_pre``/``time_post`` the way
    :class:`~gpype.Trigger` does, so :class:`~gpype.Baseline` and
    :class:`~gpype.EpochAverage` work downstream of either.

    Args:
        condition: The marker label to cut epochs around.
        tmin: Epoch start, in seconds relative to the marker's sample.
            Negative for time before it.
        tmax: Epoch end, in seconds relative to the marker's sample.
        **kwargs: Additional arguments for the parent IONode.

    Raises:
        ValueError: If ``tmin`` is not earlier than ``tmax``.
    """

    #: Refused outside a batch run, by ``Pipeline._validate`` (see
    #: ``Apply.BATCH_ONLY``): the whole recording has to already be in
    #: hand for a marker's neighbourhood to mean anything.
    BATCH_ONLY: bool = True

    #: Context keys naming where the marker sits in the epoch, in the
    #: exact strings Trigger publishes and Baseline reads -- so either
    #: node can feed a Baseline or an EpochAverage downstream.
    TIME_PRE = "time_pre"
    TIME_POST = "time_post"

    class Configuration(IONode.Configuration):
        """Configuration class for Epochs parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for Epochs."""

            #: Marker label to cut epochs around
            CONDITION = "condition"
            #: Epoch start, seconds relative to the marker
            TMIN = "tmin"
            #: Epoch end, seconds relative to the marker
            TMAX = "tmax"

    def __init__(self, condition: str, tmin: float, tmax: float, **kwargs):
        """Initialize the epoch cutter.

        Args:
            condition: The marker label to cut epochs around.
            tmin: Epoch start in seconds relative to the marker.
            tmax: Epoch end in seconds relative to the marker.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If tmin is not earlier than tmax.
        """
        tmin = float(tmin)
        tmax = float(tmax)
        if tmin >= tmax:
            raise ValueError(
                f"tmin ({tmin}) must be earlier than tmax ({tmax})."
            )
        super().__init__(
            condition=str(condition), tmin=tmin, tmax=tmax, **kwargs
        )
        self._slices: list = []

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Resolve every trial's slice against the input's events model.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.

        Raises:
            ValueError: If the context declares no sampling rate; if no
                marker matches ``condition``, naming the labels present;
                or if an in-range epoch straddles a recording gap,
                naming the trial and the gap.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]
        label = node_label(self)

        rate = context.get(Constants.Keys.SAMPLING_RATE)
        if not rate:
            raise ValueError(
                f"{label} needs a sampling rate in its input context; "
                f"connect it downstream of a source."
            )
        rate = float(rate)

        keys = self.Configuration.Keys
        condition = self.config[keys.CONDITION]
        tmin = self.config[keys.TMIN]
        tmax = self.config[keys.TMAX]

        pre = int(round(-tmin * rate))
        post = int(round(tmax * rate))
        length = pre + post
        if length <= 0:
            raise ValueError(
                f"{label}: [tmin={tmin}, tmax={tmax}) s spans no samples "
                f"at {rate:g} Hz."
            )

        total = context.get(Constants.Keys.FRAME_SIZE) or 0
        markers = context.get(Constants.Keys.MARKERS) or []
        gaps = context.get(Constants.Keys.GAPS) or []

        matched = [m for m in markers if str(m[3]) == condition]
        if not matched:
            present = sorted({str(m[3]) for m in markers})
            raise ValueError(
                f"{label}: no marker labelled {condition!r}. This "
                f"recording has "
                f"{present if present else 'no markers at all'}."
            )

        accepted = []  # (start, end, source_sample, marker_label)
        dropped = 0
        for marker in matched:
            sample = int(marker[0])
            start = sample - pre
            end = sample + post
            if start < 0 or end > total:
                dropped += 1
                continue
            accepted.append((start, end, sample, str(marker[3])))

        if dropped:
            self.log(
                f"{label}: dropped {dropped} epoch(s) of condition "
                f"{condition!r} that ran off the end of the recording.",
                type=Constants.LogTypes.WARNING,
            )

        for index, (start, end, sample, _) in enumerate(accepted):
            for first_missing, n_missing, reason in gaps:
                first_missing = int(first_missing)
                last_missing = first_missing + int(n_missing)
                if start < last_missing and end > first_missing:
                    raise ValueError(
                        f"{label}: trial {index + 1} of condition "
                        f"{condition!r} (marker at sample {sample}) "
                        f"straddles a gap at samples "
                        f"[{first_missing}, {last_missing}) "
                        f"({reason}); refusing rather than returning a "
                        f"dense block across a dropout."
                    )

        if not accepted:
            raise ValueError(
                f"{label}: every epoch of condition {condition!r} ran "
                f"off the end of the recording; there is nothing left "
                f"to cut."
            )

        self._slices = [(start, end) for start, end, _, _ in accepted]
        trials = [
            [sample, marker_label] for _, _, sample, marker_label in accepted
        ]

        out = port_context_out[PORT_OUT]
        out[Constants.Keys.FRAME_SIZE] = length
        out[self.TIME_PRE] = -tmin
        out[self.TIME_POST] = tmax
        out[Constants.Keys.TRIALS] = trials
        out.pop(Constants.Keys.MARKERS, None)
        out.pop(Constants.Keys.GAPS, None)
        return port_context_out

    def process(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Cut every trial and stack them into a trial axis.

        Args:
            data: The complete recording, under key ``PORT_IN``.

        Returns:
            ``(time, channel, trial)`` under key ``PORT_OUT``.
        """
        block = data[PORT_IN]
        epochs = [block[start:end, :] for start, end in self._slices]
        stacked = np.stack(epochs, axis=2)
        return {PORT_OUT: stacked.astype(Constants.DATA_TYPE, copy=False)}
