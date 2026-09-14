from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Reference(IONode):
    """Re-references the signal channels against a chosen reference.

    An EEG channel is a voltage difference, and the electrode it is
    measured against is part of what the number means. Re-referencing
    chooses that electrode after the fact.

    Three forms, in the order they are usually reached for:

    - a common average, the mean over the referenced channels;
    - a single or linked electrode, e.g. one mastoid or the mean of both;
    - explicit bipolar pairs, each output being one difference.

    Only measured-signal channels are referenced. Everything else is
    carried through untouched, so a trigger or an in-band index is not
    quietly turned into a difference against an average.
    """

    #: At most one of these -- "Give either 'reference' or 'pairs', not
    #: both". Both may be absent.
    AT_MOST_ONE_OF = (("reference", "pairs"),)

    class Configuration(IONode.Configuration):
        """Configuration class for Reference parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for Reference settings."""

            pass

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Channels forming the reference, by label or index
            REFERENCE = "reference"
            #: Explicit (active, reference) pairs for bipolar montages
            PAIRS = "pairs"

    def __init__(
        self,
        reference: Optional[Union[str, int, list]] = None,
        pairs: Optional[list] = None,
        **kwargs,
    ):
        """Initialize the reference node.

        Args:
            reference: Channels to reference against, each a label or an
                index. None means the common average over every signal
                channel. A single entry is a single electrode; two are a
                linked pair.
            pairs: Explicit ``[active, reference]`` pairs, each entry a
                label or an index. One output channel per pair.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If both reference and pairs are given, or if
                either is given but empty.
        """
        if reference is not None and pairs is not None:
            raise ValueError("Give either 'reference' or 'pairs', not both.")
        if reference is not None:
            if isinstance(reference, (str, int)):
                reference = [reference]
            if not reference:
                raise ValueError("reference must not be empty.")
            kwargs.setdefault(
                self.Configuration.OptionalKeys.REFERENCE, list(reference)
            )
        if pairs is not None:
            if not pairs:
                raise ValueError("pairs must not be empty.")
            for pair in pairs:
                if len(pair) != 2:
                    raise ValueError(
                        f"Each pair needs exactly two entries, got {pair}."
                    )
            kwargs.setdefault(
                self.Configuration.OptionalKeys.PAIRS,
                [list(pair) for pair in pairs],
            )

        super().__init__(**kwargs)
        self._split = None  # Channels the reference applies to
        self._ref_idx = None  # Reference channel positions
        self._pairs = None  # Resolved (active, reference) positions

    @staticmethod
    def _resolve(entry, context: dict) -> int:
        """Resolve a label or index to a channel position.

        Args:
            entry: A channel label or an integer index.
            context: Input port context.

        Returns:
            The channel position.

        Raises:
            ValueError: If a label is unknown or an index out of range.
        """
        count = channels.channel_count(context)
        if isinstance(entry, str):
            if not channels.has_labels(context):
                raise ValueError(
                    f"Cannot resolve '{entry}': the input declares no "
                    f"channel labels. Use indices, or name the channels "
                    f"with a ChannelLabeler."
                )
            labels = list(channels.labels_of(context))
            if entry not in labels:
                raise ValueError(
                    f"Unknown channel label '{entry}'. Available: {labels}"
                )
            return labels.index(entry)
        if not 0 <= int(entry) < count:
            raise ValueError(
                f"Channel index {entry} is out of range for an input with "
                f"{count} channel(s)."
            )
        return int(entry)

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Resolve the reference and describe the output channels.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]
        opt = self.Configuration.OptionalKeys

        self._split = channels.SignalSplit(context)
        signal = (
            self._split.signal
            if not self._split.all_signal
            else np.arange(channels.channel_count(context))
        )

        pairs = self.config.get(opt.PAIRS)
        if pairs:
            self._pairs = [
                (self._resolve(a, context), self._resolve(r, context))
                for a, r in pairs
            ]
            roles = [Constants.ChannelRoles.SIGNAL] * len(self._pairs)
            labels = None
            if channels.has_labels(context):
                names = list(channels.labels_of(context))
                labels = [f"{names[a]}-{names[r]}" for a, r in self._pairs]
            out = port_context_out[PORT_OUT]
            out[Constants.Keys.CHANNEL_COUNT] = len(self._pairs)
            out.update(channels.describe(roles, labels, None))
            return port_context_out

        reference = self.config.get(opt.REFERENCE)
        if reference is None:
            self._ref_idx = np.asarray(signal, dtype=int)
        else:
            self._ref_idx = np.asarray(
                [self._resolve(entry, context) for entry in reference],
                dtype=int,
            )
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Subtract the reference from the signal channels.

        Args:
            data: Input frame.

        Returns:
            The re-referenced frame.
        """
        frame = data[PORT_IN]

        if self._pairs is not None:
            out = np.empty(
                (frame.shape[0], len(self._pairs)), dtype=frame.dtype
            )
            for i, (active, ref) in enumerate(self._pairs):
                out[:, i] = frame[:, active] - frame[:, ref]
            return {PORT_OUT: out}

        ref = frame[:, self._ref_idx].mean(axis=1, keepdims=True)
        out = frame.copy()
        operated = self._split.take(out)
        out = self._split.merge(out, operated - ref)
        return {PORT_OUT: out}
