from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi

from ...common._private import channels
from ...common.constants import Constants
from ..core.io_node import IONode

#: Default input port identifier
PORT_IN = Constants.Defaults.PORT_IN
#: Default output port identifier
PORT_OUT = Constants.Defaults.PORT_OUT


class Interpolator(IONode):
    """Raises the data rate of a stream by an integer factor L.

    The mirror image of :class:`Decimator`, and the replacement for the
    former ``Hold`` node: it raises the rate for real, so every
    downstream node still sees a ``sampling_rate`` and a ``frame_size``
    that mean what they say. The rate rises because L frames are emitted
    per input frame, not because frames grow -- ``frame_size`` is passed
    through unchanged, which is what lets the output be joined with a
    stream already running at the higher rate. The cost is burstiness:
    the L frames arrive together, once per input frame, so a joined
    result lags by up to one input period.

    ``method`` chooses how the missing samples are filled. ``"hold"``
    (the default) repeats the last computed value, which is exactly what
    a windowed estimate means; ``"filter"`` zero-stuffs and low-passes
    for a smooth reconstruction, at the price of overshoot and about two
    input samples of group delay -- some 400 ms at a 5 Hz feature rate.
    Use "hold" for a feature stream, "filter" for a genuine signal.

    Trigger and auxiliary channels are always held, in both methods:
    filtering a trigger turns a step into a ramp, which can cross a
    Trigger threshold on a row where nothing happened.

    A master-timeline index or a quality channel on the input is refused
    at setup -- both are consumed by Sync, so seeing one means this node
    sits upstream of Sync, where it must not sit. Only integer-factor
    upsampling is in scope.
    """

    #: Cutoff as a fraction of the *input* Nyquist frequency. Below one so
    #: that the transition band fits below that Nyquist rather than
    #: straddling it, which is where the zero-stuffed spectral images
    #: this filter exists to remove would leak back in.
    CUTOFF_RATIO = 0.8
    #: Order of the reconstruction low-pass.
    FILTER_ORDER = 8
    #: Repeat the last computed value across the L output rows.
    METHOD_HOLD = "hold"
    #: Zero-stuff and low-pass to reconstruct a smooth trajectory.
    METHOD_FILTER = "filter"

    class Configuration(IONode.Configuration):
        class Keys(IONode.Configuration.Keys):
            pass

        class OptionalKeys(IONode.Configuration.OptionalKeys):
            """Optional configuration keys."""

            #: Integer factor L by which the data rate is raised.
            INTERPOLATION_FACTOR = "interpolation_factor"
            #: Which of METHOD_HOLD / METHOD_FILTER to apply to signal
            #: channels.
            METHOD = "method"

    def __init__(
        self,
        interpolation_factor: int = 1,
        method: str = METHOD_HOLD,
        **kwargs,
    ):
        """Initialize interpolator with interpolation factor and method.

        Args:
            interpolation_factor: Factor L by which to raise the data
                rate. Must be a positive integer. Value of 1 means no
                interpolation (a pure pass-through).
            method: METHOD_HOLD or METHOD_FILTER, applied to signal
                channels only -- see the class docstring for the
                trade-off. Trigger and auxiliary channels are always
                held, regardless of this setting.
            **kwargs: Additional arguments for parent IONode.

        Raises:
            ValueError: If interpolation_factor is not a positive
                integer, or method is not one of METHOD_HOLD or
                METHOD_FILTER.
        """
        if type(interpolation_factor) is not int or interpolation_factor < 1:
            raise ValueError(
                "interpolation_factor must be a positive integer, got "
                f"{interpolation_factor!r}."
            )
        if method not in (self.METHOD_HOLD, self.METHOD_FILTER):
            raise ValueError(
                f"method must be one of {self.METHOD_HOLD!r}, "
                f"{self.METHOD_FILTER!r}, got {method!r}."
            )
        super().__init__(
            interpolation_factor=interpolation_factor,
            method=method,
            **kwargs,
        )
        # decimation_factor (ioiocore's engine-cadence key, distinct from
        # this node's own interpolation_factor above) is deliberately
        # never passed, so it stays at ONode.Configuration's own default
        # of 1. That is what makes is_decimation_step() true on every
        # cycle -- see step()'s docstring for why that is the only cadence
        # this node can have.
        self._l = None  # Resolved interpolation factor, set in setup()
        self._method = None  # Resolved method, set in setup()
        self._split = None  # Which channels the chosen method may touch
        self._sos = None  # Reconstruction filter coefficients, if built
        self._zi = None  # Streaming filter state, one set per channel
        self._primed = False  # Whether _zi has been anchored yet

    def setup(
        self, data: dict[str, np.ndarray], port_context_in: dict[str, dict]
    ) -> dict[str, dict]:
        """Setup output context with raised sampling rate and frame size.

        Args:
            data: Input data arrays.
            port_context_in: Input port contexts containing frame size,
                sampling rate, and per-channel role information.

        Returns:
            Output port contexts with the interpolated sampling rate and
            a proportionally larger frame size. Channel count, labels,
            roles and montage are carried through unchanged -- this node
            changes rows, not columns.

        Raises:
            ValueError: If frame_size is not provided in the input
                context, or if the input carries a master-timeline index
                or a quality channel.
        """
        port_context_out = super().setup(data, port_context_in)
        context_in = port_context_in[PORT_IN]
        roles = Constants.ChannelRoles

        # Both refusals below mean the same thing: this node sits
        # upstream of Sync, where it must not sit, because Sync is what
        # consumes these two channel kinds (see _private/sync.py).
        if channels.channels_with_role(context_in, roles.INDEX).size:
            raise ValueError(
                "Interpolator found a master_index channel on its "
                "input, which means it sits upstream of Sync. "
                "Interpolating a timeline position invents positions "
                "that were never measured, and holding it repeats one "
                "position across L samples. Place the Interpolator "
                "after Sync."
            )
        if channels.channels_with_role(context_in, roles.QUALITY).size:
            raise ValueError(
                "Interpolator found a quality channel on its input. "
                "Quality is consumed by Sync, so seeing it here means "
                "this node sits upstream of Sync, where it must not "
                "sit."
            )

        frame_size_in = port_context_out[PORT_OUT][Constants.Keys.FRAME_SIZE]
        if frame_size_in is None:
            raise ValueError("frame_size must be provided in context.")

        L = self.config[self.Configuration.OptionalKeys.INTERPOLATION_FACTOR]
        method = self.config[self.Configuration.OptionalKeys.METHOD]

        sr_key = Constants.Keys.SAMPLING_RATE
        sampling_rate_in = context_in[sr_key]
        sampling_rate_out = sampling_rate_in * L
        port_context_out[PORT_OUT][sr_key] = sampling_rate_out
        # frame_size is deliberately NOT scaled. The rate rises because
        # this node emits L frames per input frame (see step()), not
        # because the frames grow -- and that is exactly what lets the
        # output be combined with a stream already running at the higher
        # rate, since a Router or Equation requires every input to agree
        # on frame_size as well as on sampling_rate. Measured both ways:
        # growing the frame fails the join with "All ports must have the
        # same frame size", while emitting L frames joins a raw 250 Hz
        # branch cleanly -- Healthy, exactly 250 rows/s, backlog bounded
        # at 3. FRAME_RATE is left untouched: only a source or fft.py
        # ever writes it and nothing gates on it.

        self._l = L
        self._method = method
        self._split = channels.SignalSplit(context_in)
        self._sos = None
        self._zi = None
        self._primed = False

        if L > 1 and method == self.METHOD_FILTER:
            n_signal = len(self._split.signal)
            if self._split.all_signal:
                n_signal = port_context_out[PORT_OUT][
                    Constants.Keys.CHANNEL_COUNT
                ]
            if n_signal:
                cutoff = self.CUTOFF_RATIO * sampling_rate_in / 2.0
                sos = butter(
                    self.FILTER_ORDER,
                    cutoff,
                    btype="lowpass",
                    output="sos",
                    fs=sampling_rate_out,
                )
                # Zero-stuffing divides the signal's mean energy by L (L
                # - 1 out of every L stuffed samples are 0), so a
                # unity-gain filter would leave a reconstructed constant
                # c at c / L instead of c. Baking a gain of L into the
                # first section's numerator restores it: standard
                # multirate theory, not a repo-specific measurement.
                sos[0, :3] *= L
                self._sos = sos
                # (sections, 2, channels) for filtering along axis 0. The
                # template describes the filter's own steady state for a
                # unit step input -- which, now that the gain above is
                # baked in, settles at L, not 1 -- and step() rescales it
                # by the first real sample once that sample is known, so
                # a stream that starts at a constant offset does not
                # ramp towards it from that assumed unit step.
                self._zi = np.repeat(
                    sosfilt_zi(self._sos)[:, :, None], n_signal, axis=2
                )
        return port_context_out

    def _emit(self, frame: np.ndarray) -> None:
        """Push one output frame the way the framework would have.

        ioiocore's ``node_imp._cycle`` runs the post-step hook and then
        pushes at most once. An upsampler has to emit more often than
        that to raise the rate without changing the frame shape, so it
        emits from inside ``step`` -- and does both halves the framework
        would have done, in the same order, for every frame rather than
        only the last. The hook is a no-op lambda unless something
        installed one (only the ``gp.Node(target=...)`` wrapper does),
        but relying on that would make this node quietly wrong the day
        it stops being true.

        Args:
            frame: One output frame, already in its final shape.
        """
        out = {PORT_OUT: frame}
        self._imp.post_step_handler(out)
        self._imp._push_data(out)

    def step(self, data: dict) -> dict | None:
        """Expand one input frame into L frames at L times the rate.

        L == 1 is a pure pass-through: the frame is forwarded unchanged,
        with no filter ever built or run. Otherwise the expanded rows go
        out as L frames of the input's own frame size, so the frame shape
        is preserved and only the emission rate rises.

        Args:
            data: Input data dictionary containing one frame to expand.

        Returns:
            The frame itself when L == 1, otherwise None -- the L frames
            have already been emitted by then.
        """
        block = data[PORT_IN]
        L = self._l
        if L == 1:
            return {PORT_OUT: block}

        signal_in = self._split.take(block)

        if self._method == self.METHOD_FILTER and self._sos is not None:
            n_in, n_signal = signal_in.shape
            # Insert L - 1 zeros after each input sample: the real
            # sample lands on the first row of the L-row block it
            # expands into, exactly as Trigger's held edge does (see the
            # class docstring), so both methods agree on where in the
            # block the observation itself sits.
            stuffed = np.zeros((n_in * L, n_signal), dtype=Constants.DATA_TYPE)
            stuffed[0::L, :] = signal_in
            if not self._primed:
                self._zi = self._zi * signal_in[0, :]
                self._primed = True
            signal_out, self._zi = sosfilt(
                self._sos, stuffed, axis=0, zi=self._zi
            )
        else:
            signal_out = np.repeat(signal_in, L, axis=0)

        if self._split.all_signal:
            expanded = signal_out
        else:
            # channels.SignalSplit.merge() cannot be reused here: it
            # assumes the operated block has no more rows than the frame
            # it is merged back into (frame[-operated.shape[0]:]), which
            # is true for a Decimator shrinking rows but false here,
            # where signal_out already has L times the input's rows.
            # Every carried (trigger, auxiliary) channel is always held
            # here too, regardless of method, so it is repeated the same
            # way rather than merged against the original row count.
            held = np.repeat(block[:, self._split.other], L, axis=0)
            expanded = np.empty(
                (signal_out.shape[0], block.shape[1]),
                dtype=Constants.DATA_TYPE,
            )
            expanded[:, self._split.signal] = signal_out
            expanded[:, self._split.other] = held

        # Hand the expanded rows out as L frames of the input's own
        # frame size, in time order, and return None so _cycle does not
        # push a duplicate. Returning None to emit nothing from _cycle is
        # an established pattern here -- Decimator and Trigger both do
        # it.
        n_in = block.shape[0]
        for i in range(L):
            self._emit(expanded[i * n_in : (i + 1) * n_in, :])
        return None
