"""Per-channel z-score: fit the scale offline, deploy it unchanged online."""

from __future__ import annotations

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core.fittable import Fittable
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Standardize(Fittable, IONode):
    """Per-channel z-score, fitted offline and applied unchanged online.

    The first fittable node (D-BATCH-93): each signal channel's mean and
    standard deviation, learned once from a whole recording via
    :meth:`~gpype.Pipeline.fit`, then subtracted and divided out of every
    frame afterwards -- in a batch run over the same recording, or in a
    realtime deployment reading live from an amplifier. The two produce
    the same numbers because both call the same private transform;
    fitting and deploying are two different methods precisely so that
    neither can drift from the other (see :mod:`gpype.backend.core.fittable`).

    Only the channels the port context calls ``signal`` are touched --
    an index or trigger channel standardized alongside the signal would
    lose the meaning of its own values. Those channels declare no unit
    afterwards: a z-score has none.

    Args:
        ddof: Delta degrees of freedom for the fitted standard
            deviation, as in ``numpy.std``. 0 (the default) is the
            population estimate; 1 is the sample estimate.
        **kwargs: Additional arguments for the parent :class:`IONode`.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for Standardize parameters."""

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Delta degrees of freedom for the fitted standard
            #: deviation.
            DDOF = "ddof"

    def __init__(self, ddof: int = 0, **kwargs):
        super().__init__(ddof=int(ddof), **kwargs)
        self._split: channels.SignalSplit = None
        self._mean: np.ndarray = None
        self._std: np.ndarray = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Resolve which channels are signal, ahead of fitting or applying.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts, unchanged in shape: standardizing
            does not add, remove or reorder channels.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]
        self._split = channels.SignalSplit(context)
        units = channels.units_of(context)
        if units is not None:
            for index in self._split.signal.tolist():
                units[index] = None
            port_context_out[PORT_OUT][Constants.Keys.CHANNEL_UNITS] = units
        return port_context_out

    def fit(self, block: np.ndarray, context: dict) -> dict:
        """Compute each signal channel's mean and standard deviation.

        Args:
            block: The whole recording, shape ``(time, channel)``.
            context: The input context. Read for the channel roles only
                under :meth:`~Fittable.fit_offline`, which runs no
                :meth:`setup`; a context naming no channel count is
                sized by *block*.

        Returns:
            ``{"mean": ..., "std": ...}``, one entry per signal channel.
        """
        if self._split is None:
            described = dict(context)
            described.setdefault(
                Constants.Keys.CHANNEL_COUNT, int(block.shape[1])
            )
            self._split = channels.SignalSplit(described)
        signal = self._split.take(block).astype(np.float64, copy=False)
        ddof = int(self.config[self.Configuration.OptionalKeys.DDOF])
        return {
            "mean": signal.mean(axis=0),
            "std": signal.std(axis=0, ddof=ddof),
        }

    def load_state(self, state: dict) -> None:
        """Adopt a fitted mean and standard deviation.

        Args:
            state: ``{"mean": ..., "std": ...}``.
        """
        self._mean = np.asarray(state["mean"], dtype=Constants.DATA_TYPE)
        std = np.asarray(state["std"], dtype=Constants.DATA_TYPE)
        # A channel fitted with zero variance is constant, not divisible;
        # dividing by zero would turn "no change" into NaN or inf.
        self._std = np.where(std == 0, np.float32(1.0), std).astype(
            Constants.DATA_TYPE
        )

    def _transform(self, frame: np.ndarray) -> np.ndarray:
        """Apply the fitted z-score to one frame or the whole recording."""
        operated = self._split.take(frame)
        z = (operated - self._mean) / self._std
        return self._split.merge(
            frame, z.astype(Constants.DATA_TYPE, copy=False)
        )

    def apply_block(self, block: np.ndarray) -> np.ndarray:
        """Standardize the whole block at once, for a batch run.

        Args:
            block: The complete input.

        Returns:
            The standardized block, same shape.
        """
        return self._transform(block)

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Standardize one frame, for a realtime deployment.

        Args:
            data: Input frame.

        Returns:
            The standardized frame.
        """
        return {PORT_OUT: self._transform(data[PORT_IN])}
