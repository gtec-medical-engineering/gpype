"""Base for a source that hands over a whole recording at once."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ....common.constants import Constants
from ...core.o_port import OPort
from .source import Source

#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class BatchSource(Source):
    """A source whose recording arrives in one cycle.

    Deliberately **not** a :class:`FixedRateSource`. That class exists to
    pace a stream against the clock -- it owns a thread, a ``speed`` and
    a drift-compensating sleep loop -- and a batch run has none of that:
    the driver calls ``cycle()`` once and the whole recording is emitted
    in that call. Inheriting the pacing machinery in order to switch it
    off would leave two ways to be a batch source, one of them carrying
    parameters that cannot mean anything.

    A subclass supplies the data and what describes it:

    * :meth:`read_all` returns the samples, ``(time, channel)``;
    * :meth:`describe` returns whatever else the file knows -- channel
      labels, roles, units, gaps, trust state, absolute start time.

    Everything else is here: the frame is the recording, the run ends
    when the block has been handed over, and the mode is declared so a
    pipeline can refuse to drive it against a clock.

    Args:
        sampling_rate: Rate of the recording in Hz.
        channel_count: Number of channels.
        sample_count: Number of samples on the time axis. Becomes the
            frame size, because the frame *is* the recording.
        output_ports: Output port configurations. One by default.
        **kwargs: Additional arguments for the parent Source.

    Raises:
        ValueError: If the rate, channel count or sample count is not
            positive.
    """

    #: One block, one cycle, and the run ends when it is done.
    EXECUTION_MODE: str = Constants.ExecutionMode.BATCH

    #: Positions advance through a recording, not with the host clock.
    #: A batch source's frame *is* the recording -- the whole block in
    #: one call is the point of the mode -- so a rate-derived slice would
    #: be meaningless. See Source.DERIVE_FRAME_SIZE.
    DERIVE_FRAME_SIZE: bool = False

    TIME_BASE: str = Constants.TimeBase.RECORDED

    class Configuration(Source.Configuration):
        """Configuration class for BatchSource parameters."""

        class Keys(Source.Configuration.Keys):
            """Configuration keys for batch source settings."""

            #: Sampling rate of the recording, in Hz
            SAMPLING_RATE = Constants.Keys.SAMPLING_RATE

    def __init__(
        self,
        sampling_rate: float,
        channel_count: int,
        sample_count: int,
        output_ports: Optional[list] = None,
        **kwargs,
    ):
        if sampling_rate is None or float(sampling_rate) <= 0:
            raise ValueError("sampling_rate must be greater than zero.")
        if int(channel_count) < 1:
            raise ValueError("channel_count must be at least one.")
        if int(sample_count) < 1:
            raise ValueError(
                "sample_count must be at least one: a batch source with "
                "no samples has nothing to hand over."
            )

        if output_ports is None:
            output_ports = [OPort.Configuration()]

        self._sample_count = int(sample_count)
        self._emitted = False

        super().__init__(
            output_ports=output_ports,
            channel_count=int(channel_count),
            frame_size=self._sample_count,
            sampling_rate=float(sampling_rate),
            **kwargs,
        )

    # ------------------------------------------------------------ subclass

    def read_all(self) -> np.ndarray:
        """Return the whole recording.

        Returns:
            Samples of shape ``(time, channel)``.

        Raises:
            NotImplementedError: If the subclass does not implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement read_all()."
        )

    def describe(self) -> dict:
        """Return what the recording knows about itself.

        Anything a reader can supply and the port context can carry:
        channel labels and roles, units, gaps, trust state, absolute
        start time, markers. Keys that mean nothing for a given format
        are simply absent -- never defaulted, so a consumer can tell
        "this format does not record it" from "it is zero".

        Every value must be JSON-representable, because a port context
        crosses a Link as JSON: an ISO 8601 string rather than a
        datetime, a list of lists rather than a tuple of records.

        Returns:
            Context entries to publish on the output port.
        """
        return {}

    # --------------------------------------------------------------- engine

    @property
    def sample_count(self) -> int:
        """Number of samples in the recording."""
        return self._sample_count

    @property
    def is_exhausted(self) -> bool:
        """Whether the recording has been handed over."""
        return self._emitted

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Describe the output port.

        Args:
            data: Initial data dictionary.
            port_context_in: Input port contexts, empty for a source.

        Returns:
            Output port contexts, carrying the rate and whatever
            :meth:`describe` supplies.
        """
        port_context_out = super().setup(data, port_context_in)

        # A restart re-emits: the run is over when the block has been
        # handed over *this* time.
        self._emitted = False

        rate = float(self.config[self.Configuration.Keys.SAMPLING_RATE])
        described = self.describe()
        for name, context in port_context_out.items():
            frame_size = context.get(
                Constants.Keys.FRAME_SIZE, self._sample_count
            )
            context[Constants.Keys.SAMPLING_RATE] = rate
            # One frame per recording, so this is the reciprocal of the
            # whole duration. Published for consistency rather than for
            # anyone to pace by.
            context[Constants.Keys.FRAME_RATE] = rate / float(frame_size)
            context.update(described)
            del name
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> Optional[dict]:
        """Hand over the recording, once.

        Args:
            data: Ignored; a source has no inputs.

        Returns:
            The whole recording on the first cycle, None afterwards.

        Raises:
            ValueError: If the subclass returns something that is not a
                2-D block of the declared width.
        """
        if self._emitted:
            return None

        block = np.asarray(self.read_all())
        if block.ndim != 2:
            raise ValueError(
                f"{type(self).__name__}.read_all() must return a 2-D "
                f"block (time x channel); got shape {block.shape}."
            )
        declared = self.scalar(
            self.config[self.Configuration.Keys.CHANNEL_COUNT]
        )
        if block.shape[1] != declared:
            raise ValueError(
                f"{type(self).__name__}.read_all() returned "
                f"{block.shape[1]} channels, but the source declared "
                f"{declared}. The declaration is what every node "
                f"downstream was set up against."
            )

        self._emitted = True
        return {PORT_OUT: block}
