from __future__ import annotations

from typing import Optional

import numpy as np

from ...common._private import channels
from ...common.constants import Constants
from ..core.i_port import IPort
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Delay(IONode):
    """Delays the input stream by an exact number of samples.

    ``num_samples`` counts samples and is independent of frame_size. The
    first num_samples samples of output are ``initial_value``, every
    emitted frame keeps the input's shape and ``Constants.DATA_TYPE``,
    and num_samples=0 is a pass-through.

    Declaring both ``channel_count`` and ``sampling_rate`` turns the node
    into a feedback-loop breaker: it asserts its own output context
    instead of deriving it from an input that does not exist yet, pushes
    one priming frame of ``initial_value`` in :meth:`start`, and reports
    ``BREAKS_CYCLES`` True -- which is what lets ``gpype.Pipeline``
    accept a cyclic graph at all. The loop delay is then exactly
    num_samples samples, and must be at least frame_size because a node
    pushes a whole frame at a time. :meth:`setup` re-verifies the
    declaration against what comes back round the loop, and refuses a
    channel with the ``index`` role on that edge.

    Args:
        num_samples: Samples to delay the stream by. Non-negative, and
            at least frame_size when a context is declared.
        initial_value: Value every warm-up sample holds, and what the
            rest of a loop sees as its "sample -1".
        channel_count: Channels the declared edge carries. Declaring
            form only; None derives it from the input.
        sampling_rate: Sampling rate of the declared edge, in Hz.
            Declaring form only; None derives it from the input.
        frame_size: Samples per pushed frame on the declared edge.
            Declaring form only, where it defaults to 1; must be None
            otherwise, since a deriving Delay takes it from its input.
        **kwargs: Additional arguments for parent IONode.

    Raises:
        ValueError: If num_samples is negative; if exactly one of
            channel_count and sampling_rate is given; if frame_size is
            given without them; if a declared channel_count,
            sampling_rate or frame_size is not positive; or if a
            declared num_samples is smaller than frame_size.
    """

    #: Duck-typed marker read by Pipeline's cycle detector so it can
    #: recognise a loop breaker without importing a concrete node class
    #: from a family it otherwise stays agnostic to -- the same reason a
    #: source is found by a TIME_BASE attribute rather than an isinstance
    #: check.
    #:
    #: False here and set per instance in __init__, because only the
    #: declaring form breaks anything: a Delay that derives its context
    #: from its input cannot break the *context* cycle, and accepting one
    #: as a breaker would let the detector pass a pipeline that then stalls
    #: silently at Healthy with an unbounded queue -- the exact failure
    #: the check exists to prevent.
    BREAKS_CYCLES = False

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            #: Configuration key for number of delay samples
            NUM_SAMPLES = "num_samples"
            #: Configuration key for the warm-up fill value
            INITIAL_VALUE = "initial_value"

    def __init__(
        self,
        num_samples: int,
        initial_value: float = 0.0,
        channel_count: Optional[int] = None,
        sampling_rate: Optional[float] = None,
        frame_size: Optional[int] = None,
        **kwargs,
    ):
        """Initialize the delay and, optionally, declare its own context.

        Args:
            num_samples: Number of samples to delay the signal by.
            initial_value: Fill value for the warm-up samples.
            channel_count: Channels on the declared edge, or None.
            sampling_rate: Sampling rate of the declared edge, or None.
            frame_size: Frame size of the declared edge, or None.
            **kwargs: Additional arguments for parent IONode.

        Raises:
            ValueError: See the class docstring.
        """
        if num_samples < 0:
            raise ValueError("Number of samples must be non-negative.")

        declared = channel_count is not None or sampling_rate is not None
        if declared and (channel_count is None or sampling_rate is None):
            raise ValueError(
                f"a Delay that breaks a feedback loop must declare BOTH "
                f"channel_count and sampling_rate (got "
                f"channel_count={channel_count!r}, "
                f"sampling_rate={sampling_rate!r}). The loop's context "
                f"cannot be derived from an input that does not exist "
                f"yet: this node's input arrives from the same loop its "
                f"output feeds, so one side has to be asserted outright. "
                f"Give both to break a loop, or neither to delay an "
                f"ordinary stream."
            )
        if not declared and frame_size is not None:
            raise ValueError(
                f"frame_size={frame_size!r} only means something for a "
                f"Delay that declares its own context. A Delay deriving "
                f"its context takes the frame size from its input and "
                f"delays by num_samples samples whatever it is. Declare "
                f"channel_count and sampling_rate as well if this Delay "
                f"is meant to break a feedback loop, or drop frame_size."
            )

        if declared:
            if channel_count < 1:
                raise ValueError("channel_count must be a positive integer.")
            if sampling_rate <= 0:
                raise ValueError("sampling_rate must be positive.")
            if frame_size is None:
                frame_size = Constants.Defaults.FRAME_SIZE
            if frame_size < 1:
                raise ValueError("frame_size must be a positive integer.")
            if num_samples < frame_size:
                raise ValueError(
                    f"num_samples={num_samples} is shorter than one "
                    f"frame (frame_size={frame_size}), and one frame is "
                    f"the floor for a declared loop delay. A node pushes "
                    f"once per cycle with its whole frame (see _cycle in "
                    f"ioiocore/imp/node_imp.py), so every sample of a "
                    f"frame already exists before any of them can leave "
                    f"the node: a loop delay shorter than one frame "
                    f"cannot exist across a node boundary. A sample-rate "
                    f"recursion has to live inside a single node's "
                    f"step() instead."
                )

        # No port configuration is passed explicitly here, on purpose.
        # Deserialization hands the stored input_ports/output_ports back
        # inside kwargs, and passing either one again as an explicit
        # keyword alongside **kwargs collides with itself -- "got
        # multiple values for keyword argument 'output_ports'", which is
        # exactly the blocker this node shipped with once. IONode takes
        # both as named parameters, so letting them travel inside kwargs
        # is the whole fix; a node that really needs a non-default port
        # must pop it out of kwargs (see flow/trigger.py) instead of
        # passing it as well.
        super().__init__(
            num_samples=num_samples,
            initial_value=initial_value,
            channel_count=channel_count,
            sampling_rate=sampling_rate,
            frame_size=frame_size,
            **kwargs,
        )
        #: Total delay, in samples, from this node's input to what the
        #: next node reads -- including the primed frame in the declaring
        #: form, which is why it is not the same as the ring's depth.
        self._num_samples: int = int(num_samples)
        #: Fill value for the warm-up samples and the priming frame
        self._initial_value: float = float(initial_value)
        #: Whether this instance declares its own output context
        self._declared: bool = declared
        #: Declared channel count, or None when derived
        self._channel_count: Optional[int] = (
            int(channel_count) if declared else None
        )
        #: Declared sampling rate in Hz, or None when derived
        self._sampling_rate: Optional[float] = (
            float(sampling_rate) if declared else None
        )
        #: Declared frame size, or None when derived
        self._declared_frame_size: Optional[int] = (
            int(frame_size) if declared else None
        )
        # Only the declaring form may report itself to the cycle
        # detector as a breaker; see the class attribute above.
        self.BREAKS_CYCLES: bool = declared
        #: Whether the priming frame has been pushed for this run --
        #: reset in stop() so a second start() primes again.
        self._primed: bool = False
        #: Samples this node's own buffer holds back. Equal to
        #: num_samples when the context is derived; num_samples minus one
        #: frame when it is declared, because the frame pushed in start()
        #: already supplies that much of the delay.
        self._buffered_samples: int = (
            num_samples - frame_size if declared else num_samples
        )
        #: Ring buffer of exactly _buffered_samples rows -- the samples
        #: not yet emitted. Sized to the buffered depth only, never to
        #: frame_size, so its per-cycle upkeep cost does not grow with
        #: frame_size; see step() for how it is advanced without
        #: shifting.
        self._ring: np.ndarray = None
        #: Index of the oldest (next-to-read) row currently in the ring.
        self._write_pos: int = 0
        #: Frame size the ring was sized for, fixed at setup time (known
        #: already at construction in the declaring form).
        self._frame_size: int = self._declared_frame_size
        #: Whether history alone (True) covers a whole output frame, or
        #: the frame that just arrived has to contribute part of the
        #: output too (False). Fixed by frame_size vs the buffered depth,
        #: both constant for the life of a setup() call, so this is
        #: decided once there instead of every step().
        self._frame_fits_in_history: bool = None

    def _declared_context(self) -> dict:
        """The context this node asserts for the edge it declares.

        Built fresh on every call, never cached: the same dict must not
        be handed out twice, because a later merge downstream (e.g.
        :meth:`IONode.setup` combining several input contexts) mutates
        the copy it receives, and a shared dict would let that reach
        back and corrupt the declaration itself.

        Returns:
            The port context this node's output declares, and its input
            must match once the loop completes its first round trip.
        """
        context = {
            Constants.Keys.CHANNEL_COUNT: self._channel_count,
            Constants.Keys.SAMPLING_RATE: self._sampling_rate,
            Constants.Keys.FRAME_SIZE: self._declared_frame_size,
            Constants.Keys.FRAME_RATE: (
                self._sampling_rate / self._declared_frame_size
            ),
            IPort.Configuration.Keys.TIMING: Constants.Timing.SYNC,
            IPort.Configuration.Keys.TYPE: "Any",
        }
        context.update(
            channels.describe(
                [Constants.ChannelRoles.SIGNAL] * self._channel_count
            )
        )
        return context

    def start(self) -> None:
        """Start the node and, if it declares a context, prime the loop.

        The priming push runs after ``super().start()``: it sets the
        output port's context directly, which only makes sense once that
        port exists. Safe to call again after a stop() -- see stop().
        """
        super().start()
        if not self._declared or self._primed:
            return
        output_port = self.get_output_port(PORT_OUT)
        # The engine's only context-propagation path is push() copying
        # an OPort's own _context into whichever connected IPorts still
        # have none (see OPortImp.push) -- so the context has to be set
        # here, directly, before the first push.
        output_port._context = self._declared_context()
        frame = np.full(
            (self._declared_frame_size, self._channel_count),
            self._initial_value,
            dtype=Constants.DATA_TYPE,
        )
        output_port.push(frame)
        self._primed = True

    def stop(self) -> None:
        """Stop the node and forget that it has primed a loop.

        Without this, a second start() would see ``_primed`` already
        True and push no fresh token. Pipeline.start() clears every
        input queue before any node starts, so with no new token nothing
        would ever again reach the other side of the junction the loop
        closes at -- the loop would go silent, not error.
        """
        self._primed = False
        super().stop()

    def _verify_declaration(self, arrived: dict) -> dict:
        """Check what came back round the loop against the declaration.

        Args:
            arrived: Context of this node's input port.

        Returns:
            The declared context, to be published on the output port.

        Raises:
            ValueError: If the arriving channel_count, sampling_rate or
                frame_size disagrees with what was declared, or if the
                edge carries a channel with the INDEX role.
        """
        declared = self._declared_context()
        for key in (
            Constants.Keys.CHANNEL_COUNT,
            Constants.Keys.SAMPLING_RATE,
            Constants.Keys.FRAME_SIZE,
        ):
            if arrived.get(key) != declared.get(key):
                raise ValueError(
                    f"this Delay declared {key}={declared.get(key)!r} "
                    f"for the edge it breaks, but the value coming back "
                    f"round the loop has {key}={arrived.get(key)!r}. The "
                    f"rest of the loop was already built from the "
                    f"declaration, so this cannot be corrected now -- "
                    f"fix the mismatch at its source instead."
                )
        if len(
            channels.channels_with_role(arrived, Constants.ChannelRoles.INDEX)
        ):
            raise ValueError(
                "the loop edge carries a channel with the 'index' role "
                "(a master-timeline position). Nothing strips a "
                "position channel from a SYNC edge, so it would travel "
                "the loop band-passed and summed like ordinary signal. "
                "Keep a position channel out of any loop a Delay closes."
            )
        return declared

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Publish the output context and size the sample-domain ring.

        In the deriving form this is the ordinary IONode merge. In the
        declaring form ``super().setup()`` is deliberately not called:
        the output context was asserted at construction, not derived
        from the input the way an ordinary IONode's is, so the generic
        multi-port merge would be answering a question already answered
        -- and the arriving context is re-verified against the
        declaration instead.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts containing channel count
                and frame size.

        Returns:
            Output port contexts -- the parent's merge, or the
            declaration.

        Raises:
            ValueError: If channel count is not provided in context (or,
                via the parent's own check below, frame size isn't); or,
                in the declaring form, if the arriving context disagrees
                with the declaration.
        """
        md = port_context_in[PORT_IN]
        if self._declared:
            port_context_out = {PORT_OUT: self._verify_declaration(md)}
            channel_count = self._channel_count
            frame_size = self._declared_frame_size
        else:
            channel_count = md.get(Constants.Keys.CHANNEL_COUNT)
            if channel_count is None:
                raise ValueError("Channel count must be provided in context.")

            # frame_size is validated by IONode.setup() below rather than
            # a second, separately worded check here: this class used to
            # raise its own "Frame size must be provided in context.", so
            # the same missing-context condition read differently for
            # Delay than for every other IONode. Letting the parent
            # report it once removes that drift instead of just matching
            # its wording by hand.
            port_context_out = super().setup(data, port_context_in)
            frame_size = md[Constants.Keys.FRAME_SIZE]

        self._frame_size = frame_size
        # The frame pushed in start() already carries frame_size samples
        # of the declared delay, so the ring only has to hold back the
        # rest. Together they make the total exactly num_samples -- see
        # the class docstring.
        buffered = self._num_samples - (frame_size if self._declared else 0)
        self._buffered_samples = buffered
        if buffered > 0:
            # Sized to the buffered depth alone (not depth + frame_size,
            # as the previous shifting-buffer version needed): a
            # write-index ring means advancing history is an index
            # update, not a copy of the whole history. Measured against
            # the old O(num_samples) shift, float32: up to ~107us/step at
            # num_samples=2500, frame_size=1, channel_count=64 -- 13% of
            # the 833us cycle budget at 1200 Hz, for a 10s delay on 64
            # channels at the frame_size the shipped example already
            # uses.
            #
            # Pre-filled with initial_value rather than zeros: that fill
            # *is* the warm-up output, and in a loop it is what the
            # junction reads as its own past.
            self._ring = np.full(
                (buffered, channel_count),
                self._initial_value,
                dtype=Constants.DATA_TYPE,
            )
            self._frame_fits_in_history = frame_size <= buffered
        else:
            self._ring = None
            self._frame_fits_in_history = None
        self._write_pos = 0
        return port_context_out

    def step(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Process one step of delay.

        Args:
            data: Input data dictionary containing signal to delay.

        Returns:
            Dictionary with the delayed signal, one frame_size-shaped
            frame per call, filled with initial_value while still in the
            warm-up period. A pass-through when this node buffers
            nothing: num_samples=0 in the deriving form, or
            num_samples == frame_size in the declaring form, where the
            whole delay is the frame pushed in start().
        """
        data_in = data[PORT_IN]
        buffered = self._buffered_samples

        if buffered == 0:
            return {PORT_OUT: data_in.astype(Constants.DATA_TYPE, copy=False)}

        frame_size = self._frame_size
        ring = self._ring
        write_pos = self._write_pos

        # Always a fresh array, never a view into the ring: the ring
        # keeps getting mutated on every later call, and downstream is
        # expected to still hold a valid value from this cycle after
        # that. OPortImp.push happens to deep-copy at the producer
        # boundary today, which would make handing out a ring view
        # "work" by accident -- but that guarantee belongs to the
        # producer, not to this node, and a real-time node should not
        # depend on a copy strategy owned three layers away.
        out = np.empty((frame_size, ring.shape[1]), dtype=Constants.DATA_TYPE)

        if self._frame_fits_in_history:
            # History alone covers the whole output frame: read the
            # oldest frame_size rows out of the ring, then evict them by
            # writing this cycle's frame into those same slots -- in a
            # ring buffer, "shift the history" is just letting the next
            # write land where the oldest data used to be.
            end = write_pos + frame_size
            if end <= buffered:
                out[:] = ring[write_pos:end]
                ring[write_pos:end] = data_in
            else:
                # The window straddles the end of the ring: split both
                # the read and the write into the tail and the wrapped
                # head.
                tail = buffered - write_pos
                out[:tail] = ring[write_pos:]
                out[tail:] = ring[: end - buffered]
                ring[write_pos:] = data_in[:tail]
                ring[: end - buffered] = data_in[tail:]
            self._write_pos = end % buffered
        else:
            # Less history than a frame holds: the output tail reaches
            # into the frame that just arrived, shifted back by the
            # buffered depth -- the same case the old combined-buffer
            # version handled by concatenating history and the new
            # frame. Read the whole ring first, oldest row first.
            if write_pos == 0:
                out[:buffered] = ring
            else:
                tail = buffered - write_pos
                out[:tail] = ring[write_pos:]
                out[tail:buffered] = ring[:write_pos]
            out[buffered:] = data_in[: frame_size - buffered]
            # An output frame this wide flushes history completely; what
            # is left for next cycle is just the newest `buffered` rows
            # of the frame that just arrived.
            ring[:] = data_in[-buffered:]
            self._write_pos = 0

        return {PORT_OUT: out}
