from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class RollingStatistic(IONode):
    """One statistic per channel over a sliding window.

    Amplitude features and amplitude-based quality judgements both need
    this, and most of these are reachable by no combination of the other
    nodes: a moving average is the mean and only the mean. Peak-to-peak
    over a window, in particular, is the textbook way to spot a channel
    that has come loose.

    The window is a real window: every emitted value describes the last
    ``window_size`` samples, and the first values are computed against a
    history primed with the first sample rather than against silence, so
    there is no start-up ramp to mistake for signal.
    """

    #: Exactly one of these must be given; the window is either a
    #: sample count or a duration.
    ONE_OF = (("window_size", "window_seconds"),)

    #: Statistics this node can compute.
    STATISTICS = (
        "rms",
        "mean",
        "var",
        "std",
        "min",
        "max",
        "peak_to_peak",
        "median",
    )

    class Configuration(IONode.Configuration):
        """Configuration class for RollingStatistic parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for RollingStatistic settings."""

            #: Which statistic to compute
            STATISTIC = "statistic"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Window length in samples
            WINDOW_SIZE = "window_size"
            #: Window length in seconds, resolved against the rate
            WINDOW_SECONDS = "window_seconds"

    def __init__(
        self,
        statistic: str = "rms",
        window_size: Optional[int] = None,
        window_seconds: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the rolling statistic.

        Args:
            statistic: One of STATISTICS.
            window_size: Window length in samples.
            window_seconds: Window length in seconds, as an alternative
                to window_size; resolved once the sampling rate is known.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If the statistic is unknown, if both or neither
                window form is given, or if a window is not positive.
        """
        if statistic not in self.STATISTICS:
            raise ValueError(
                f"Unknown statistic '{statistic}'. Available: "
                f"{list(self.STATISTICS)}"
            )
        if (window_size is None) == (window_seconds is None):
            raise ValueError(
                "Give exactly one of 'window_size' or 'window_seconds'."
            )
        if window_size is not None:
            if window_size < 1:
                raise ValueError("window_size must be at least 1.")
            kwargs.setdefault(
                self.Configuration.OptionalKeys.WINDOW_SIZE, int(window_size)
            )
        else:
            if window_seconds <= 0:
                raise ValueError("window_seconds must be greater than 0.")
            kwargs.setdefault(
                self.Configuration.OptionalKeys.WINDOW_SECONDS,
                float(window_seconds),
            )

        super().__init__(statistic=statistic, **kwargs)
        self._buffer = None  # Ring of the last window_size samples
        self._index = 0  # Next write position in the ring
        self._primed = False  # Whether the ring holds real history
        self._split = None  # Channels the statistic applies to
        self._window = None  # Resolved window length in samples

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Resolve the window and allocate the history.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.

        Raises:
            ValueError: If a window in seconds was requested and the input
                declares no sampling rate.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]
        opt = self.Configuration.OptionalKeys

        window = self.config.get(opt.WINDOW_SIZE)
        if window is None:
            rate = context.get(Constants.Keys.SAMPLING_RATE)
            if not rate:
                raise ValueError(
                    "window_seconds needs a sampling rate in the context."
                )
            window = max(1, int(round(self.config[opt.WINDOW_SECONDS] * rate)))
        self._window = window

        self._split = channels.SignalSplit(context)
        count = channels.channel_count(context)
        if not self._split.all_signal:
            count = len(self._split.signal)

        self._buffer = np.zeros((window, count), dtype=Constants.DATA_TYPE)
        self._index = 0
        self._primed = False

        # One value per channel per frame -- which means the output rate
        # is the input rate divided by the input frame size, not the
        # input rate.
        #
        # This declared the input's rate unchanged while emitting one row
        # per cycle. At frame_size 1 the two agree and the claim is
        # honest; at any larger frame it overstates the rate by exactly
        # the frame size, and everything downstream that derives a time
        # axis from it -- a scope's window, a filter's cutoff -- is
        # wrong by that factor with nothing to show for it.
        rate_in = context.get(Constants.Keys.SAMPLING_RATE)
        frame_size_in = context.get(Constants.Keys.FRAME_SIZE) or 1
        port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE] = 1
        port_context_out[PORT_OUT][Constants.Keys.CHANNEL_COUNT] = count
        if rate_in:
            port_context_out[PORT_OUT][Constants.Keys.SAMPLING_RATE] = (
                rate_in / frame_size_in
            )
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Update the window and emit the statistic.

        Args:
            data: Input frame.

        Returns:
            One row holding the statistic per channel.
        """
        block = self._split.take(data[PORT_IN])

        if not self._primed:
            # Treat the history before the first sample as that sample,
            # so the first outputs describe the signal rather than the
            # gap between silence and it.
            self._buffer[:] = block[0, :]
            self._primed = True

        for row in block:
            self._buffer[self._index % self._window, :] = row
            self._index += 1

        stat = self.config[self.Configuration.Keys.STATISTIC]
        if stat == "rms":
            value = np.sqrt(np.mean(self._buffer**2, axis=0))
        elif stat == "mean":
            value = np.mean(self._buffer, axis=0)
        elif stat == "var":
            value = np.var(self._buffer, axis=0)
        elif stat == "std":
            value = np.std(self._buffer, axis=0)
        elif stat == "min":
            value = np.min(self._buffer, axis=0)
        elif stat == "max":
            value = np.max(self._buffer, axis=0)
        elif stat == "peak_to_peak":
            value = np.ptp(self._buffer, axis=0)
        else:
            value = np.median(self._buffer, axis=0)

        return {PORT_OUT: value.reshape(1, -1)}
