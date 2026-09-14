from __future__ import annotations

from typing import Optional

import numpy as np

from ...common.constants import Constants
from ..core._private.controllable import controllable
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Threshold(IONode):
    """Turns a continuous signal into a held 0/1 state.

    This is the step from a measurement to a decision, and the two
    parameters that matter are the ones that stop it chattering. A bare
    comparison flips on every sample that grazes the level, which as a
    control signal means an output that switches tens of times a second
    while the user holds still.

    Hysteresis gives the release its own, lower level, so leaving a state
    takes more than the noise that entered it. Dwell requires the new
    state to hold for a number of samples before it is believed.

    The output is 0.0 or 1.0 in the pipeline's own data type, not a
    boolean array, so it flows through the rest of the pipeline like any
    other channel.
    """

    class Configuration(IONode.Configuration):
        """Configuration class for Threshold parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration keys for Threshold settings."""

            #: Level at which the state is entered
            LEVEL = "level"
            #: Whether crossing upwards or downwards enters the state
            DIRECTION = "direction"

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Level at which the state is left; defaults to LEVEL
            RELEASE = "release"
            #: Samples the new state must hold before it is accepted
            DWELL = "dwell"

    #: Enter the state when the signal rises above the level.
    ABOVE = "above"
    #: Enter the state when the signal falls below the level.
    BELOW = "below"

    def __init__(
        self,
        level: float,
        direction: str = ABOVE,
        release: Optional[float] = None,
        dwell: int = 1,
        **kwargs,
    ):
        """Initialize the threshold.

        Args:
            level: Level at which the state is entered.
            direction: ABOVE or BELOW.
            release: Level at which the state is left. Defaults to level,
                i.e. no hysteresis. For direction ABOVE it must not be
                above level, and the other way round for BELOW.
            dwell: Samples the new state must hold before it is accepted.
            **kwargs: Additional arguments for the parent IONode.

        Raises:
            ValueError: If level is missing, the direction is unknown, the
                release sits on the wrong side of the level, or dwell is
                below one.
        """
        if level is None:
            raise ValueError("level must be provided.")
        if direction not in (self.ABOVE, self.BELOW):
            raise ValueError(
                f"direction must be '{self.ABOVE}' or '{self.BELOW}'."
            )
        if dwell < 1:
            raise ValueError("dwell must be at least 1.")
        if release is not None:
            if direction == self.ABOVE and release > level:
                raise ValueError(
                    "For direction 'above' the release level must not be "
                    "above the level, or the state could never be left."
                )
            if direction == self.BELOW and release < level:
                raise ValueError(
                    "For direction 'below' the release level must not be "
                    "below the level, or the state could never be left."
                )
            kwargs.setdefault(
                self.Configuration.OptionalKeys.RELEASE, float(release)
            )
        kwargs.setdefault(self.Configuration.OptionalKeys.DWELL, int(dwell))

        super().__init__(level=float(level), direction=direction, **kwargs)
        self._state = None  # Current accepted state, per channel
        self._pending = None  # Candidate state awaiting its dwell
        self._count = None  # Samples the candidate has held

        # The live levels, seeded from the configuration and thereafter
        # owned here. They cannot live in `self.config`, which is
        # read-only by design -- so a node with a control surface keeps
        # the controllable half as instance state and treats the
        # configuration as the initial value rather than the current one.
        opt = self.Configuration.OptionalKeys
        self._level = float(self.config[self.Configuration.Keys.LEVEL])
        self._release = float(self.config.get(opt.RELEASE, self._level))
        self._dwell = int(self.config.get(opt.DWELL, 1))

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Allocate the per-channel state.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts.

        Returns:
            Output port contexts.
        """
        port_context_out = super().setup(data, port_context_in)
        count = port_context_in[PORT_IN][Constants.Keys.CHANNEL_COUNT]
        if isinstance(count, list):
            count = count[0]
        self._state = np.zeros(count, dtype=bool)
        self._pending = np.zeros(count, dtype=bool)
        self._count = np.zeros(count, dtype=int)
        return port_context_out

    @controllable
    def get_control(self) -> dict:
        """Report the levels, so a caller can see before it sets.

        Returns:
            ``level``, ``release`` and ``dwell`` as they stand.
        """
        return {
            "level": self._level,
            "release": self._release,
            "dwell": self._dwell,
        }

    @controllable
    def set_control(self, arg: dict) -> dict:
        """Move the levels while the pipeline runs.

        This is the node the surface exists for. A threshold is tuned
        against a signal you are watching -- a neurofeedback level is
        found by moving it until the subject can hold it, which cannot
        be done by stopping, editing and restarting, because the thing
        being tuned against is the person in the chair.

        Safe to change mid-run because the level is read on every frame
        rather than cached at setup, and because it is not filter state:
        a new level takes effect on the next sample with no transient.
        The dwell counter is deliberately *not* reset -- a candidate
        state part-way through its dwell keeps its progress, so nudging
        the level does not restart the debounce and make the output feel
        stuck.

        Args:
            arg: The whole resulting state; the framework merges before
                calling, so every key is present.

        Returns:
            The state actually in force, which is what a caller should
            display rather than what it sent.

        Raises:
            ValueError: If the combination cannot run. Refused as a
                whole rather than partly applied -- a release above the
                level inverts the hysteresis and would latch the output
                on, which is worse than a rejected request.
        """
        level = float(arg["level"])
        release = float(arg["release"])
        dwell = int(arg["dwell"])

        if dwell < 1:
            raise ValueError(f"dwell must be at least 1 sample, got {dwell}")
        direction = self.config[self.Configuration.Keys.DIRECTION]
        if direction == self.ABOVE and release > level:
            raise ValueError(
                f"release {release} is above level {level}; for an ABOVE "
                f"threshold the release must be at or below it, or the "
                f"output latches on"
            )
        if direction == self.BELOW and release < level:
            raise ValueError(
                f"release {release} is below level {level}; for a BELOW "
                f"threshold the release must be at or above it, or the "
                f"output latches on"
            )

        # Assigned together, after every check, so a refused request
        # leaves the node exactly as it was rather than half-moved.
        self._level = level
        self._release = release
        self._dwell = dwell
        return self.get_control()

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compare the frame against the level and hold the state.

        Args:
            data: Input frame.

        Returns:
            The state per channel, one row per input row.
        """
        frame = data[PORT_IN]
        # The live values, not the configured ones: set_control moves
        # these while the pipeline runs, and reading them here is what
        # makes a change take effect on the very next frame.
        level = self._level
        release = self._release
        dwell = self._dwell
        keys = self.Configuration.Keys
        above = self.config[keys.DIRECTION] == self.ABOVE

        out = np.empty(frame.shape, dtype=Constants.DATA_TYPE)
        for i, row in enumerate(frame):
            # While in the state the comparison is against the release
            # level, so noise around the entry level cannot flip it back.
            limits = np.where(self._state, release, level)
            raw = row > limits if above else row < limits

            changed = raw != self._state
            self._count = np.where(
                changed & (raw == self._pending), self._count + 1, 1
            )
            self._pending = np.where(changed, raw, self._state)
            accept = changed & (self._count >= dwell)
            self._state = np.where(accept, raw, self._state)
            out[i, :] = self._state.astype(Constants.DATA_TYPE)

        return {PORT_OUT: out}
