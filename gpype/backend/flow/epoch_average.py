from __future__ import annotations

from typing import Optional

import numpy as np

from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode
from ..core.o_port import OPort

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class EpochAverage(IONode):
    """Averages epochs as they arrive, and says how many it used.

    An evoked response is buried well below the ongoing EEG, so it only
    appears once many trials are averaged. Until now the only averaging
    in the package happened inside a scope's paint routine, on a leaf
    with no output, so an averaged response could be looked at but never
    written to a file, sent to another program, measured, or compared
    against another condition.

    Trials that are obviously contaminated should not be averaged in at
    all, so rejection is part of this node rather than a separate one:
    one rejected trial is a discarded epoch, not a pipeline branch.

    The count travels on its own port, because an average without its
    trial count cannot be interpreted.
    """

    #: Two outputs, so they are declared. Same names the constructor
    #: builds its OPort configurations from.
    OUTPUT_PORT_NAMES = ("out", "count")

    #: Port carrying the number of epochs in the current average.
    PORT_COUNT = "count"

    class Configuration(IONode.Configuration):
        """Configuration class for EpochAverage parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for EpochAverage settings."""

            #: How epochs are combined
            MODE = "mode"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Number of epochs in a moving or exponential average
            COUNT = "count"
            #: Reject an epoch exceeding this absolute amplitude
            REJECT_LEVEL = "reject_level"
            #: Reject an epoch exceeding this peak-to-peak amplitude
            REJECT_PEAK_TO_PEAK = "reject_peak_to_peak"

    #: Average every epoch seen so far.
    CUMULATIVE = "cumulative"
    #: Average the last ``count`` epochs.
    MOVING = "moving"
    #: Exponentially weight recent epochs, with ``count`` as the constant.
    EXPONENTIAL = "exponential"

    def __init__(
        self,
        mode: str = CUMULATIVE,
        count: Optional[int] = None,
        reject_level: Optional[float] = None,
        reject_peak_to_peak: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the epoch average.

        Args:
            mode: CUMULATIVE, MOVING or EXPONENTIAL.
            count: Number of epochs for MOVING, or the weighting constant
                for EXPONENTIAL. Required for both.
            reject_level: Discard an epoch if any sample of any signal
                channel exceeds this absolute amplitude.
            reject_peak_to_peak: Discard an epoch if any channel's
                peak-to-peak amplitude exceeds this.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If the mode is unknown, if a count is needed and
                missing, or if a rejection limit is not positive.
        """
        if mode not in (self.CUMULATIVE, self.MOVING, self.EXPONENTIAL):
            raise ValueError(
                f"mode must be one of '{self.CUMULATIVE}', "
                f"'{self.MOVING}' or '{self.EXPONENTIAL}'."
            )
        if mode != self.CUMULATIVE:
            if count is None or count < 1:
                raise ValueError(f"mode '{mode}' needs a count of at least 1.")
            kwargs.setdefault(
                self.Configuration.OptionalKeys.COUNT, int(count)
            )
        for name, value, key in (
            (
                "reject_level",
                reject_level,
                self.Configuration.OptionalKeys.REJECT_LEVEL,
            ),
            (
                "reject_peak_to_peak",
                reject_peak_to_peak,
                self.Configuration.OptionalKeys.REJECT_PEAK_TO_PEAK,
            ),
        ):
            if value is not None:
                if value <= 0:
                    raise ValueError(f"{name} must be greater than 0.")
                kwargs.setdefault(key, float(value))

        # An epoch arrives when a trigger fires, so both ports are sparse.
        kwargs.setdefault(
            self.Configuration.Keys.INPUT_PORTS,
            [IPort.Configuration(name=PORT_IN, timing=Constants.Timing.ASYNC)],
        )
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [
                OPort.Configuration(
                    name=PORT_OUT, timing=Constants.Timing.ASYNC
                ),
                OPort.Configuration(
                    name=self.PORT_COUNT, timing=Constants.Timing.ASYNC
                ),
            ],
        )

        super().__init__(mode=mode, **kwargs)
        self._accumulator = None  # Running sum, or the average itself
        self._history = None  # Recent epochs, for MOVING
        self._n = 0  # Epochs accepted so far
        self._rejected = 0  # Epochs discarded so far

    @property
    def accepted(self) -> int:
        """Number of epochs averaged in so far."""
        return self._n

    @property
    def rejected(self) -> int:
        """Number of epochs discarded by the rejection limits."""
        return self._rejected

    def reset(self) -> None:
        """Discard the average and start again."""
        self._accumulator = None
        self._history = None
        self._n = 0
        self._rejected = 0

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Describe the averaged output and the count port.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.
        """
        port_context_out = super().setup(data, port_context_in)
        source = dict(port_context_in[PORT_IN])

        # The average has the shape of one epoch.
        port_context_out[PORT_OUT].update(
            {
                k: v
                for k, v in source.items()
                if k
                not in (
                    IPort.Configuration.Keys.ID,
                    IPort.Configuration.Keys.NAME,
                )
            }
        )
        port_context_out[PORT_OUT][
            OPort.Configuration.Keys.TIMING
        ] = Constants.Timing.ASYNC

        count_ctx = port_context_out.get(self.PORT_COUNT, {})
        count_ctx[Constants.Keys.CHANNEL_COUNT] = 1
        count_ctx[Constants.Keys.FRAME_SIZE] = 1
        count_ctx[Constants.Keys.SAMPLING_RATE] = source.get(
            Constants.Keys.SAMPLING_RATE
        )
        count_ctx[OPort.Configuration.Keys.TIMING] = Constants.Timing.ASYNC
        port_context_out[self.PORT_COUNT] = count_ctx

        self.reset()
        return port_context_out

    def _rejects(self, epoch: np.ndarray) -> bool:
        """Whether an epoch exceeds a rejection limit.

        Args:
            epoch: The epoch to judge.

        Returns:
            True if the epoch must be discarded.
        """
        opt = self.Configuration.OptionalKeys
        level = self.config.get(opt.REJECT_LEVEL)
        if level is not None and np.abs(epoch).max() > level:
            return True
        ptp = self.config.get(opt.REJECT_PEAK_TO_PEAK)
        if ptp is not None and np.ptp(epoch, axis=0).max() > ptp:
            return True
        return False

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Average one epoch in and emit the current average.

        Args:
            data: Input epoch.

        Returns:
            The current average and its trial count, or None while there
            is nothing to report.
        """
        epoch = data.get(PORT_IN)
        if epoch is None:
            return None

        epoch = np.asarray(epoch, dtype=Constants.DATA_TYPE)
        if self._rejects(epoch):
            self._rejected += 1
            return None

        mode = self.config[self.Configuration.Keys.MODE]
        count = self.config.get(self.Configuration.OptionalKeys.COUNT)

        if mode == self.CUMULATIVE:
            if self._accumulator is None:
                self._accumulator = np.zeros_like(epoch)
            self._accumulator += epoch
            self._n += 1
            average = self._accumulator / self._n
        elif mode == self.MOVING:
            if self._history is None:
                self._history = []
            self._history.append(epoch)
            if len(self._history) > count:
                self._history.pop(0)
            self._n = len(self._history)
            average = np.mean(self._history, axis=0)
        else:
            # Weight the newest epoch by 1/count, which converges on the
            # same average while forgetting old trials smoothly.
            alpha = 1.0 / count
            if self._accumulator is None:
                self._accumulator = epoch.copy()
            else:
                self._accumulator += alpha * (epoch - self._accumulator)
            self._n += 1
            average = self._accumulator

        return {
            PORT_OUT: average.astype(Constants.DATA_TYPE),
            self.PORT_COUNT: np.array([[self._n]], dtype=Constants.DATA_TYPE),
        }
