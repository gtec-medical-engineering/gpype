from __future__ import annotations

import numpy as np

from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Framer(IONode):
    """Frame aggregation node for combining single samples into frames.

    Collects individual samples (frame size = 1) and aggregates them into
    larger frames of a specified size. Maintains an internal buffer that
    accumulates samples until a complete frame is assembled, then outputs
    the entire frame. Useful for converting sample-by-sample streams into
    frame-based processing.
    """

    #: See FFT: declared so the catalog can read it.
    DEFAULT_FRAME_SIZE: int = 1

    # Type annotation for the internal buffer
    _buf: np.ndarray

    class Configuration(IONode.Configuration):
        """Configuration class for Framer parameters."""

        class Keys(IONode.Configuration.Keys):
            """Configuration key constants for the Framer."""

            #: Frame size configuration key
            FRAME_SIZE = Constants.Keys.FRAME_SIZE

    def __init__(self, frame_size: int = None, **kwargs):
        """Initialize the Framer node.

        Args:
            frame_size: Size of output frames to generate. Must be a positive
                integer. Defaults to 1 if None.
            **kwargs: Additional configuration parameters passed to IONode.

        Raises:
            ValueError: If frame_size is not an integer or is less than 1.
        """
        # Validate and set default frame size
        if frame_size is None:
            frame_size = self.DEFAULT_FRAME_SIZE
        if not isinstance(frame_size, int):
            raise ValueError("frame_size must be integer.")
        if frame_size < 1:
            raise ValueError("frame_size must be greater or equal 1.")
        decimation_factor = frame_size

        # serialize() writes decimation_factor, so a stored document
        # carries it and it has to be accepted. It is derived from
        # frame_size, though, and a hand-written document that disagrees
        # would tell every downstream node a rate the frames do not
        # have. Accept the value that matches; refuse the one that lies.
        supplied = kwargs.pop(
            self.Configuration.Keys.DECIMATION_FACTOR, decimation_factor
        )
        if supplied != decimation_factor:
            raise ValueError(
                f"decimation_factor is derived from frame_size and must "
                f"be {decimation_factor}, not {supplied}. Set frame_size "
                f"instead."
            )

        # Initialize parent IONode with frame configuration
        # Set decimation_factor = frame_size to output every frame_size steps
        super().__init__(
            frame_size=frame_size,
            decimation_factor=decimation_factor,
            **kwargs,
        )

        # Initialize internal buffer (will be allocated in setup())
        self._buf = None

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Set up the Framer node and allocate the internal buffer.

        Validates input port configuration and initializes the internal buffer
        based on output frame size and channel count. Input must have
        frame_size = 1 for proper single-sample aggregation.

        Args:
            data: Initial data dictionary for port configuration.
            port_context_in: Input port context with frame size and channel
                count specifications.

        Returns:
            Output port context with updated frame size information.

        Raises:
            ValueError: If input frame size is not 1.
        """
        # Initialize parent setup and get output port context
        port_context_out = super().setup(data, port_context_in)

        # Get configuration for output frame setup
        frame_size_out = self.config[self.Configuration.Keys.FRAME_SIZE]

        # Any input frame is accepted, as long as it fits inside an
        # output frame. This node re-blocks a stream: the samples and
        # their rate are unchanged, only how many travel together.
        #
        # It used to demand a single sample per cycle, and behind that
        # guard it kept only the LAST row of whatever arrived and counted
        # cycles rather than samples -- so lifting the guard without
        # rewriting step() would have silently dropped frame_size_in - 1
        # of every frame.
        #
        # It can only aggregate, never split: a cycle emits at most one
        # frame, so turning frames of 4 back into single samples would
        # need four emissions in one cycle.
        frame_size_in = port_context_in[PORT_IN][Constants.Keys.FRAME_SIZE]
        if frame_size_in is None:
            raise ValueError(
                "Framer needs a frame size in the input context; got none."
            )
        if frame_size_in > frame_size_out:
            raise ValueError(
                f"Framer assembles frames of {frame_size_out} but its "
                f"input already carries {frame_size_in}, and a cycle can "
                f"emit only one frame. Ask for at least {frame_size_in}, "
                f"or reduce the source's frame_size."
            )
        channel_count = port_context_out[PORT_OUT][
            Constants.Keys.CHANNEL_COUNT
        ]

        # Update output port context with new frame size
        port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE] = frame_size_out

        # Allocate internal buffer for frame assembly
        self._buf = np.zeros(shape=(frame_size_out, channel_count))
        self._frame_size = frame_size_out
        #: Rows written into the current frame. Reset here because
        #: setup() reruns on every run.
        self._fill = 0

        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Take every incoming sample and emit whole frames.

        Serialises what arrives -- one sample or many -- into a running
        frame, and emits it once it is full. A frame that does not divide
        the input evenly simply carries the remainder into the next one,
        so no sample is dropped or repeated at the boundary.

        Args:
            data: Input data dictionary with PORT_IN, shape
                (rows, channels) for any number of rows.

        Returns:
            Output data dictionary with PORT_OUT holding the completed
            frame, or None while the frame is still filling.
        """
        block = data[PORT_IN]
        rows = block.shape[0]
        out = None

        taken = 0
        while taken < rows:
            room = self._frame_size - self._fill
            take = min(room, rows - taken)
            self._buf[self._fill : self._fill + take] = block[
                taken : taken + take
            ]
            self._fill += take
            taken += take
            if self._fill == self._frame_size:
                # A copy, because the buffer is reused immediately and a
                # consumer holding the frame would otherwise watch it
                # change underneath.
                out = self._buf.copy()
                self._fill = 0

        if out is None:
            return None
        return {PORT_OUT: out}
