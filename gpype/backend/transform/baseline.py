from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode
from ..core.o_port import OPort

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Baseline(IONode):
    """Removes an epoch's baseline level.

    An evoked response is a deflection away from what the signal was
    doing beforehand, so the pre-stimulus level has to be subtracted for
    the number to mean anything. This is part of the definition of an
    ERP rather than an optional tidy-up.

    The interval defaults to the whole pre-stimulus span, which the
    Trigger upstream already publishes, so in the usual case there is
    nothing to configure.

    A high-pass filter is not a substitute: it is causal, it distorts the
    slow components that carry much of the response, and it removes a
    standing offset over its own time constant rather than exactly.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for Baseline parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for Baseline settings."""

            #: How the baseline is removed
            MODE = "mode"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Interval start, in seconds relative to the trigger
            START = "start"
            #: Interval end, in seconds relative to the trigger
            STOP = "stop"

    #: Subtract the baseline level.
    SUBTRACT = "subtract"
    #: Express the epoch as a percentage change from the baseline.
    PERCENT = "percent"

    def __init__(
        self,
        start: Optional[float] = None,
        stop: Optional[float] = None,
        mode: str = SUBTRACT,
        **kwargs,
    ):
        """Initialize the baseline node.

        Args:
            start: Interval start in seconds relative to the trigger,
                negative before it. Defaults to the start of the epoch.
            stop: Interval end in seconds relative to the trigger.
                Defaults to the trigger itself, i.e. the whole
                pre-stimulus span.
            mode: SUBTRACT or PERCENT.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If the mode is unknown or the interval is empty.
        """
        if mode not in (self.SUBTRACT, self.PERCENT):
            raise ValueError(
                f"mode must be '{self.SUBTRACT}' or '{self.PERCENT}'."
            )
        if start is not None and stop is not None and start >= stop:
            raise ValueError("start must be earlier than stop.")
        if start is not None:
            kwargs.setdefault(
                self.Configuration.OptionalKeys.START, float(start)
            )
        if stop is not None:
            kwargs.setdefault(
                self.Configuration.OptionalKeys.STOP, float(stop)
            )

        # A baseline is taken over an epoch, and an epoch arrives when a
        # trigger fires -- so both ports are sparse, exactly as
        # EpochAverage declares them.
        #
        # Without this the ports took IPort's SYNC default, and
        # connecting the documented chain raised TypeError: Trigger
        # declares its output ASYNC, and ioiocore refuses a connection
        # between two concrete, unequal timings.
        #
        # There is no continuous-stream Baseline to keep working.
        # `setup()` below requires `time_pre` in the input context, and
        # `Trigger` is the only node in this package that ever publishes
        # it, so a Baseline fed by anything else already failed -- at
        # start() rather than at connect(), which is the later and worse
        # of the two.
        #
        # setdefault rather than a parameter: an author can still
        # override the ports, and the generated catalog is derived from
        # the class attributes, so it does not move.
        kwargs.setdefault(
            self.Configuration.Keys.INPUT_PORTS,
            [IPort.Configuration(name=PORT_IN, timing=Constants.Timing.ASYNC)],
        )
        kwargs.setdefault(
            self.Configuration.Keys.OUTPUT_PORTS,
            [
                OPort.Configuration(
                    name=PORT_OUT, timing=Constants.Timing.ASYNC
                )
            ],
        )

        super().__init__(mode=mode, **kwargs)
        self._slice = None  # Sample range the baseline is taken over
        self._split = None  # Channels the baseline applies to

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Resolve the baseline interval against the epoch layout.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.

        Raises:
            ValueError: If the epoch layout is unknown, or the interval
                falls outside the epoch.
        """
        port_context_out = super().setup(data, port_context_in)
        context = port_context_in[PORT_IN]
        opt = self.Configuration.OptionalKeys

        rate = context.get(Constants.Keys.SAMPLING_RATE)
        frame = context.get(Constants.Keys.FRAME_SIZE)
        time_pre = context.get("time_pre")
        if not rate or not frame:
            raise ValueError(
                "Baseline needs a sampling rate and a frame size in the "
                "context; connect it downstream of a Trigger."
            )
        if time_pre is None:
            raise ValueError(
                "Baseline needs to know where the trigger sits in the "
                "epoch. Connect it downstream of a Trigger, which "
                "publishes that, or the interval cannot be resolved."
            )

        start = self.config.get(opt.START, -time_pre)
        stop = self.config.get(opt.STOP, 0.0)

        # Convert to sample positions within the epoch, where the trigger
        # sits at time_pre seconds from the start.
        i0 = int(round((start + time_pre) * rate))
        i1 = int(round((stop + time_pre) * rate))
        i0 = max(0, min(i0, frame))
        i1 = max(0, min(i1, frame))
        if i1 <= i0:
            raise ValueError(
                f"The baseline interval [{start}, {stop}] s covers no "
                f"samples of an epoch spanning [{-time_pre}, "
                f"{(frame / rate) - time_pre}] s."
            )
        self._slice = slice(i0, i1)
        self._split = channels.SignalSplit(context)
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Remove the baseline from one epoch.

        Args:
            data: Input epoch.

        Returns:
            The corrected epoch.
        """
        epoch = data[PORT_IN]
        operated = self._split.take(epoch)
        base = operated[self._slice, :].mean(axis=0, keepdims=True)

        if self.config[self.Configuration.Keys.MODE] == self.PERCENT:
            # A baseline at zero has no scale to express a change against.
            safe = np.where(base == 0, np.finfo(base.dtype).eps, base)
            corrected = (operated - base) / safe * 100.0
        else:
            corrected = operated - base

        return {PORT_OUT: self._split.merge(epoch, corrected)}
